## Notes
The equation used to compute FLOPS in this code is more exact than the one in 
https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#training-testing-methodology-gpt1-5b-llama-8b-llama-70b-mistral
because we calculate the actual MACs from the matrix multiplies involved in a
Llama forward pass. The MAC counts from the `approx_llama_forward_macs` function
have been verified to (almost) match the results provided by the
`torch.utils.flop_counter.FlopCounterMode` function.

## Results for Llama v3.1 70B (BF16)

| Model Name | Batch Size | Input Sequence Length |  Latency (ms)  | TP Degree | Synapse AI Version | Mean TFLOPS | Peak TFLOPS | MFU | Mean TFLOPS/HPU |
|:---:|---:|---:|---:|:---:|:---:|---:|---:|---|---:|
| Llama v3.1 70B | 1 | 1024 |    295  | 2 | 1.17.0.495 | 557.2 | 432 | 64% | 278.6 |
| Llama v3.1 70B | 1 | 1024 |    170  | 4 | 1.17.0.495 | 969.0 | 432 | 56% | 242.3 |
| Llama v3.1 70B | 1 | 1024 |    152  | 8 | 1.17.0.495 | 1083.5 | 432 | 31% | 135.4 |
| Llama v3.1 70B | 1 | 2048 |    599  | 2 | 1.17.0.495 | 558.0 | 432 | 65% | 279.0 |
| Llama v3.1 70B | 1 | 2048 |    308  | 4 | 1.17.0.495 | 1084.8 | 432 | 63% | 271.2 |
| Llama v3.1 70B | 1 | 2048 |    173  | 8 | 1.17.0.495 | 1936.1 | 432 | 56% | 242.0 |
| Llama v3.1 70B | 1 | 4096 |  1,256  | 2 | 1.17.0.495 | 549.5 | 432 | 64% | 274.8 |
| Llama v3.1 70B | 1 | 4096 |    653  | 4 | 1.17.0.495 | 1057.4 | 432 | 61% | 264.4 |
| Llama v3.1 70B | 1 | 4096 |    350  | 8 | 1.17.0.495 | 1973.2 | 432 | 57% | 246.7 |
| Llama v3.1 70B | 1 | 8192 |  2,659  | 2 | 1.17.0.495 | 552.3 | 432 | 64% | 276.1 |
| Llama v3.1 70B | 1 | 8192 |  1,385  | 4 | 1.17.0.495 | 1060.3 | 432 | 61% | 265.1 |
| Llama v3.1 70B | 1 | 8192 |    777  | 8 | 1.17.0.495 | 1890.9 | 432 | 55% | 236.4 |
| Llama v3.1 70B | 1 | 16384 |  6,080 | 2 | 1.17.0.495 | 540.9 | 432 | 63% | 270.5 |
| Llama v3.1 70B | 1 | 16384 |  3,143 | 4 | 1.17.0.495 | 1046.6 | 432 | 61% | 261.6 |
| Llama v3.1 70B | 1 | 16384 |  1,682 | 8 | 1.17.0.495 | 1955.5 | 432 | 57% | 244.4 |
| Llama v3.1 70B | 1 | 32768 | 15,999 | 2 | 1.17.0.495 | 499.1 | 432 | 58% | 249.6 |
| Llama v3.1 70B | 1 | 32768 |  8,065 | 4 | 1.17.0.495 | 990.1 | 432 | 57% | 247.5 |
| Llama v3.1 70B | 1 | 32768 |  4,367 | 8 | 1.17.0.495 | 1828.4 | 432 | 53% | 228.6 |
