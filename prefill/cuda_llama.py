"""
The following measurements were made using vLLM version 0.6.2.
This code may not function correctly for other versions of vLLM.

Run using the command below after logging into HuggingFace with `huggingface-cli login`.
The `transformers` version must be compatible with Llama v3.1 for this example.

```bash
python -m fire prefill/cuda_llama.py main \
    --model_name meta-llama/Llama-3.1-70B \
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
    qkv_macs = sequence_length * hidden_size * head_dim * (num_attention_heads + 2 * num_key_value_heads)
    # Matrix multiply QK^T to get the self-attention matrix.
    qkt_macs = sequence_length * (head_dim * num_attention_heads) * sequence_length
    # Self-attention with the value tensor.
    sav_macs = sequence_length * sequence_length * (head_dim * num_attention_heads)
    # Post-attention projection with the output tensor.
    pap_macs = sequence_length * (head_dim * num_attention_heads) * hidden_size
    # Total number of MACs in attention.
    attn_macs = qkv_macs + qkt_macs + sav_macs + pap_macs
    # Exclude causal attention mask MACs from the total MAC count if desired.
    causal_macs = (head_dim * num_attention_heads) * sequence_length * (sequence_length - 1) // 2
    attn_macs -= int(exclude_causal_mask) * 2 * causal_macs
    ffn_macs = sequence_length * hidden_size * intermediate_size
    # SwiGLU and other gated FFNs have another matrix multiply.
    ffn_macs *= 2 + int(gated_ffn_act)
    # Final matrix multiply over the vocabulary head,
    # which is sometimes tied to the input embedding weights.
    head_macs = sequence_length * hidden_size * vocabulary_size
    macs = head_macs + num_decoder_blocks * (attn_macs + ffn_macs)
    return macs  # MAC count.


def main(model_name: str, seq_len, num_steps, tensor_parallel_size: int = 8):
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    hs = config.hidden_size
    vs = config.vocab_size
    nl = config.num_hidden_layers
    it = config.intermediate_size
    nh = config.num_attention_heads
    kv = config.num_key_value_heads

    flops = approx_llama_forward_macs(
        num_decoder_blocks=nl,
        sequence_length=seq_len,
        vocabulary_size=vs,   
        hidden_size=hs,
        intermediate_size=it,
        num_attention_heads=nh,
        num_key_value_heads=kv,
        gated_ffn_act=True,
    ) * 2  # 1 MAC is approximately 2 FLOPs.

    model = LLM(
        model_name, 
        tensor_parallel_size=tensor_parallel_size, 
        dtype=torch.bfloat16, 
        enable_chunked_prefill=False,  # Enabled by default when the input is 32K+.
        max_seq_len_to_capture=seq_len,  # Use CUDA graphs.
    )
    sampling_params = SamplingParams(max_tokens=1)

    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    for _ in range(16):  # Warmup
        x = torch.randint(low=0, high=4096, size=(seq_len,), dtype=torch.int64).tolist()
        model.generate(TokensPrompt(prompt_token_ids=x), sampling_params=sampling_params, use_tqdm=False)

    tic.wait()
    tic.record()
    for _ in range(num_steps):
        x = torch.randint(low=0, high=4096, size=(seq_len,), dtype=torch.int64).tolist()
        model.generate(TokensPrompt(prompt_token_ids=x), sampling_params=sampling_params, use_tqdm=False)
    toc.record()
    torch.cuda.synchronize()
    ms = tic.elapsed_time(toc)

    tfps = flops * num_steps * 1e-9 / tensor_parallel_size / ms
    print(f"\n\nTFLOPS/GPU: {tfps:.1f}\n\n")
