"""
The following measurements were made using vLLM version 0.6.2.
This code may not function correctly for other versions of vLLM.

Run using the command below after logging into HuggingFace with `huggingface-cli login`.
The `transformers` version must be compatible with Llama v3.1 for this example.

```bash
export VLLM_SKIP_WARMUP=true
export PT_HPU_LAZY_MODE=1
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true

python -m fire prefill/vllm_llama.py main \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --seq_len $((8 * 1024)) \
    --num_steps 32
```

For power measurements, use one of the following commands on the host.
The container will likely not have `ipmitool` available.

```bash
sudo ipmitool dcmi power reading 5_sec
sudo ipmitool sensor get Total_Power
```

Note that the true throughput is better than the measurement because
we are unable to remove the overhead from the `generate` method.
"""
import torch
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.hpu import Event
import habana_frameworks.torch.distributed.hccl  # noqa
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def approx_llama_forward_macs(
        num_decoder_blocks: int,
        sequence_length: int,
        vocabulary_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        exclude_causal_mask: bool = False,
        gated_ffn_act: bool = True,
        head_dim: int | None = None,
) -> int:
    assert hidden_size % num_attention_heads == 0
    assert num_attention_heads % num_key_value_heads == 0
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    # Query, Key, Value linear projection with Group Query Attention.
    num_qkv_heads = num_attention_heads + 2 * num_key_value_heads
    qkv_macs = sequence_length * hidden_size * head_dim * num_qkv_heads
    # Matrix multiply QK^T to get the self-attention matrix.
    qkt_macs = sequence_length * (head_dim * num_attention_heads) * sequence_length
    # Self-attention with the value tensor.
    sav_macs = sequence_length * sequence_length * (head_dim * num_attention_heads)
    # Post-attention projection with the output tensor.
    pap_macs = sequence_length * (head_dim * num_attention_heads) * hidden_size
    # Total number of MACs in attention.
    attn_macs = qkv_macs + qkt_macs + sav_macs + pap_macs
    # Exclude causal attention mask MACs from the total MAC count if desired.
    mask_shape = (sequence_length * (sequence_length - 1)) // 2
    mask_macs = head_dim * num_attention_heads * mask_shape
    attn_macs -= int(exclude_causal_mask) * 2 * mask_macs
    ffn_macs = sequence_length * hidden_size * intermediate_size
    # SwiGLU and other gated FFNs have another matrix multiply.
    ffn_macs *= 2 + int(gated_ffn_act)
    # Final matrix multiply over the vocabulary head,
    # which is sometimes tied to the input embedding weights.
    head_macs = sequence_length * hidden_size * vocabulary_size
    macs = head_macs + num_decoder_blocks * (attn_macs + ffn_macs)
    return macs  # MAC count.


def main(
        model_name: str,
        seq_len: int,
        num_steps: int,
        batch_size: int = 1,
        warmup_steps: int = 4,
        tensor_parallel_size: int = 8,
        exclude_causal_mask: bool = False,
        block_size: int = 128,
):
    config = AutoConfig.from_pretrained(model_name)

    flops = approx_llama_forward_macs(
        num_decoder_blocks=config.num_hidden_layers,
        sequence_length=seq_len,
        vocabulary_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        gated_ffn_act=True,
        exclude_causal_mask=exclude_causal_mask,
    ) * 2  # 1 MAC is approximately 2 FLOPs.

    model = LLM(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        # enable_chunked_prefill=False,  # Enabled by default if the input is 32K+
        # max_seq_len_to_capture=seq_len,  # Use HPU graphs.
        block_size=block_size,
    )
    sampling_params = SamplingParams(max_tokens=1)

    tic = Event(enable_timing=True)
    toc = Event(enable_timing=True)

    xss = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(warmup_steps, batch_size, seq_len),
        dtype=torch.int64,
    ).tolist()
    tps = [[TokensPrompt(prompt_token_ids=x) for x in xs] for xs in xss]

    for i in range(warmup_steps):  # Warmup
        model.generate(tps[i], sampling_params=sampling_params, use_tqdm=True)

    xss = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(num_steps, batch_size, seq_len),
        dtype=torch.int64,
    ).tolist()
    tps = [[TokensPrompt(prompt_token_ids=x) for x in xs] for xs in xss]

    tic.wait()
    tic.record()
    for i in range(num_steps):
        model.generate(tps[i],sampling_params=sampling_params, use_tqdm=True)
    toc.record()
    ht.synchronize()
    ms = tic.elapsed_time(toc)

    tfps = flops * batch_size * num_steps * 1e-9 / tensor_parallel_size / ms
    flop = flops * batch_size * num_steps * 1e-12 / tensor_parallel_size
    print(f"\n\nTFLOPS/HPU: {tfps:.1f}. TFLOPs/HPU: {flop:.1f}. Time: {ms:.1f} ms.\n\n")
