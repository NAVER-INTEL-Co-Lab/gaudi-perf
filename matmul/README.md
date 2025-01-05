## Notes

I do not agree with the argument made in 
https://semianalysis.com/2024/12/22/mi300x-vs-h100-vs-h200-benchmark-part-1-training/#popular-gemm-benchmark-isnt-accurate
that the cache should be flushed prior to each iteration.
For actual models, the input tensor will be
the output of the previous layer, and it is reasonable to expect it
to be on the cache already. As for weight tensors, they can be
prefetched since their values are known before the computation occurs.

Because of this, the throughput measurements for both NVIDIA GPUs and Gaudi HPUs
are made without flushing the cache.

Note that for small matrices below 4Kx4K, the Python host overhead becomes visible during GEMM.
Because of this, the GEMM throughput measurements implement repeated GEMMs in a single function.
The implementations are sligtly suboptimal, since it would be best to get `einsum("bmk,bnk->bmn")`
instead of `einsum("bmk,bnk->mn")` as is implemented in the code. However, this was the best that
could be done without complicating the code too much. Maybe this can be fixed later.


## Results for FP8 matrix multiplication.

|   M   |   K   |   N   | Scaling Granularity |  Device  | Peak TFLOPS | TFLOPS | MFU (%) |
|:-----:|:-----:|:-----:|:-------------------:|:---------|------------:|-------:|--------:|
| 16384 |  8192 |  1280 |      Per Tensor     | Gaudi v2 | 865         | 648.6  | 75.0%   |
|       |       |       |                     | H100     | 1978.9      | 1532.3 | 77.4%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 687.0  | 79.4%   |
|       |       |       |                     | H100     | 1978.9      | 1258.3 | 63.6%   |
| 16384 |  1024 |  8192 |      Per Tensor     | Gaudi v2 | 865         | 787.9  | 91.1%   |
|       |       |       |                     | H100     | 1978.9      | 1190.6 | 60.2%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 697.4  | 80.6%   |
|       |       |       |                     | H100     | 1978.9      | 706.0  | 35.7%   |
| 16384 |  8192 |  7168 |      Per Tensor     | Gaudi v2 | 865         | 799.7  | 92.5%   |
|       |       |       |                     | H100     | 1978.9      | 1361.9 | 68.8%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 799.4  | 92.4%   |
|       |       |       |                     | H100     | 1978.9      | 1185.6 | 59.9%   |
| 16384 |  3584 |  8192 |      Per Tensor     | Gaudi v2 | 865         | 801.4  | 92.6%   |
|       |       |       |                     | H100     | 1978.9      | 1321.6 | 66.8%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 790.3  | 91.4%   |
|       |       |       |                     | H100     | 1978.9      | 1259.7 | 63.7%   |
|  4096 |  4096 |  4096 |      Per Tensor     | Gaudi v2 | 865         | 751.7  | 86.9%   |
|       |       |       |                     | H100     | 1978.9      | 1312.4 | 66.3%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 730.8  | 84.5%   |
|       |       |       |                     | H100     | 1978.9      | 1377.7 | 69.6%   |
|  8192 |  8192 |  8192 |      Per Tensor     | Gaudi v2 | 865         | 811.0  | 93.8%   |
|       |       |       |                     | H100     | 1978.9      | 1363.6 | 68.9%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 684.7  | 79.2%   |
|       |       |       |                     | H100     | 1978.9      | 1179.2 | 59.6%   |
| 16384 | 16384 | 16384 |      Per Tensor     | Gaudi v2 | 865         | 842.7  | 97.4%   |
|       |       |       |                     | H100     | 1978.9      | 1342.6 | 67.8%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 820.4  | 94.8%   |
|       |       |       |                     | H100     | 1978.9      | 1122.0 | 56.7%   |
| 32768 | 32768 | 32768 |      Per Tensor     | Gaudi v2 | 865         | 739.7  | 85.5%   |
|       |       |       |                     | H100     | 1978.9      | 1337.9 | 67.6%   |
|       |       |       |       Rowwise       | Gaudi v2 | 865         | 708.2  | 81.9%   |
|       |       |       |                     | H100     | 1978.9      | 1024.5 | 51.8%   |


## Results for BF16 matrix multiplication

| M     | K     | N     | Device    | Mean TFLOPS | Peak TFLOPS | MFU   |
|:-----:|:-----:|:-----:|:----------|------------:|------------:|------:|
| 16384 |  8192 |  1280 |  Gaudi v2 | 395.4       | 432         | 91.5% |
|       |       |       |      A100 | 264.9       | 312         | 84.9% |
|       |       |       |      H100 | 754.0       | 989.4       | 76.2% |
| 16384 |  1024 |  8192 |  Gaudi v2 | 417.9       | 432         | 96.7% |
|       |       |       |      A100 | 235.9       | 312         | 75.6% |
|       |       |       |      H100 | 657.1       | 989.4       | 66.4% |
| 16384 |  8192 |  7168 |  Gaudi v2 | 425.2       | 432         | 98.4% |
|       |       |       |      A100 | 261.7       | 312         | 83.9% |
|       |       |       |      H100 | 700.7       | 989.4       | 70.8% |
| 16384 |  3584 |  8192 |  Gaudi v2 | 422.9       | 432         | 97.9% |
|       |       |       |      A100 | 257.1       | 312         | 82.4% |
|       |       |       |      H100 | 665.5       | 989.4       | 67.3% |
|  4096 |  4096 |  4096 |  Gaudi v2 | 408.2       | 432         | 94.5% |
|       |       |       |      A100 | 252.4       | 312         | 80.9% |
|       |       |       |      H100 | 748.4       | 989.4       | 75.7% |
|  8192 |  8192 |  8192 |  Gaudi v2 | 423.9       | 432         | 98.1% |
|       |       |       |      A100 | 263.6       | 312         | 84.5% |
|       |       |       |      H100 | 693.4       | 989.4       | 70.1% |
| 16384 | 16384 | 16384 |  Gaudi v2 | 379.8       | 432         | 87.9% |
|       |       |       |      A100 | 267.1       | 312         | 85.6% |
|       |       |       |      H100 | 688.9       | 989.4       | 69.7% |
| 32768 | 32768 | 32768 |  Gaudi v2 | 363.8       | 432         | 84.2% |
|       |       |       |      A100 | 250.2       | 312         | 80.2% |
|       |       |       |      H100 | 671.3       | 989.4       | 67.9% |
