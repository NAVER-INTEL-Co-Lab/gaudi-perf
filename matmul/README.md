## Notes

I do not agree with the argument made in
https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#popular-gemm-benchmark-isnt-accurate
that the cache should be flushed prior to each iteration.
For actual models, the input tensor will be
the output of the previous layer, and it is reasonable to expect it
to be on the cache already. As for weight tensors, they can be
prefetched since their values are known before the computation occurs.

Because of this, the throughput measurements for both NVIDIA GPUs
and Gaudi HPUs are made without flushing the cache.
However, different tensors are used for each repeat in a step,
which may cause an OOM if the number of repeats is set too high.

Note that for small matrices below 4Kx4K,
the Python host overhead becomes visible during GEMM.
Because of this, the GEMM throughput measurements implement
repeated GEMMs in a single function.

## Results for BF16 matrix multiplication

| M     | K     | N     | Device  | Mean TFLOPS | Peak TFLOPS | MFU   |
|:-----:|:-----:|:-----:|:--------|------------:|------------:|------:|
| 16384 |  8192 |  1280 | Gaudi 2 | 395.4       | 432         | 91.5% |
|       |       |       | A100    | 264.9       | 312         | 84.9% |
|       |       |       | H100    | 754.0       | 989.4       | 76.2% |
| 16384 |  1024 |  8192 | Gaudi 2 | 417.9       | 432         | 96.7% |
|       |       |       | A100    | 235.9       | 312         | 75.6% |
|       |       |       | H100    | 657.1       | 989.4       | 66.4% |
| 16384 |  8192 |  7168 | Gaudi 2 | 425.2       | 432         | 98.4% |
|       |       |       | A100    | 261.7       | 312         | 83.9% |
|       |       |       | H100    | 700.7       | 989.4       | 70.8% |
| 16384 |  3584 |  8192 | Gaudi 2 | 422.9       | 432         | 97.9% |
|       |       |       | A100    | 257.1       | 312         | 82.4% |
|       |       |       | H100    | 665.5       | 989.4       | 67.3% |
|  4096 |  4096 |  4096 | Gaudi 2 | 408.2       | 432         | 94.5% |
|       |       |       | A100    | 252.4       | 312         | 80.9% |
|       |       |       | H100    | 748.4       | 989.4       | 75.7% |
|  8192 |  8192 |  8192 | Gaudi 2 | 423.9       | 432         | 98.1% |
|       |       |       | A100    | 263.6       | 312         | 84.5% |
|       |       |       | H100    | 693.4       | 989.4       | 70.1% |
| 16384 | 16384 | 16384 | Gaudi 2 | 379.8       | 432         | 87.9% |
|       |       |       | A100    | 267.1       | 312         | 85.6% |
|       |       |       | H100    | 688.9       | 989.4       | 69.7% |
