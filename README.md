# Demo Repository for the Intel Gaudi

## Introduction

This repository contains code to reproduce performance numbers for the Intel Gaudi.

Specifically, code is available to measure the throughput for matrix multiplication
(BF16 and FP8) and the prefill stage of Llama models.

In addition, we also provide code for users to reproduce throughput numbers for NVIDIA
GPUs such as the A100 and the H100. However, setting up the necessary
development environments is left to the user.

## Setup

Visit https://github.com/NAVER-INTEL-Co-Lab/gaudi-cresset for detailed setup instructions.

1. Run `make env` to create a `.env` file. This need only be done once per directory.
2. Run `make build` to build the Docker image and start the container.
Run this command when you wish to rebuild the Docker image.
3. Run `make exec` to enter an existing Docker container.


## Getting started

For instructions on matrix multiplication throughput measurements,
visit the `matmul` directory. Commands are described in their respective files.

To measure prefill throughput for Llama models, visit the `prefill` directory.

Single-node training throughput is available in the `train` directory.
