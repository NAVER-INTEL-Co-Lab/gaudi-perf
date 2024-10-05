# Demo Repository for the Intel Gaudi

## Introduction

This repository contains code to reproduce performance numbers for the Intel Gaudi.

Specifically, code is available to measure the throughput for matrix multiplication
(BF16 and FP8) and the prefill stage of Llama models.

## Setup

Visit https://github.com/NAVER-INTEL-Co-Lab/gaudi-cresset for detailed setup instructions.

1. Run `make env` to create a `.env` file. This need only be done once per directory.
2. Run `make build` to build the Docker image and start the container.
Run this command when you wish to rebuild the Docker image.
3. Run `make exec` to enter an existing Docker container.
