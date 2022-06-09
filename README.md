# PULP-TrainLib

PULP-TrainLib is the first Deep Neural Network training library for the PULP Platform. PULP-TrainLib features a wide set of performance-tunable DNN layer primitives for training, together with optimizers, losses and activation functions.
To enable on-device training, PULP-TrainLib is equipped with AutoTuner, a pre-deployment tool to select the fastest configuration for each DNN layer, according to the training step to be performed and the shapes of the layer tensors.
To facilitate the deployment of training tasks on the target PULP device, PULP-TrainLib is equipped with the TrainLib Deployer, a code generator capable of generating a project folder containing all the files and the code to run a DNN training task on PULP.

If you use any part of PULP-TrainLib , please cite:
```
D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"

BibTeX citation:
(TBD)
```

This repository is released under the [Apache License Version 2.0](./LICENSE).

## PULP-TrainLib's training library

PULP-TrainLib is the first open-source training library for RISC-V-based multicore MCUs, including a set of performance-tunable DNN layer primitives to enable DNN training on ultra-low-power devices.


## PULP-TrainLib's AutoTuner

## The TrainLib Deployer



# Repository overview

This library provides primitives to execute DNN training tasks on PULP Platform (e.g GVSoc, Vega, ..). All the primitives feature CHW format.

All the library files are included in the `lib/` folder ([lib's README](lib/README.md)).

The `tests/` folder provides useful tests to try out and verify PULP-TrainLib's layers and functions (tests are performed against a PyTorch Golden model).
Each test can be customized according to the user specifications and profiles the execution of the layer's primitives with PULP's performance counters. 
If in need of further info, please refer to the [test's README](tests/README.md).

The `tools/` folder contains useful tools to complement the library, like the AutoTuner (to optimize the DNN primitives according to the DNN structure) 
and the TrainLib Deployer (to easily generate a given DNN architecture to be deployed on a specific target). For further info, please refer to [tools' README](tools/README.md).



# Installation and requirements

## PULP-SDK

PULP-TrainLib requires [PULP-SDK](https://github.com/pulp-platform/pulp-sdk) and the [RISC-V GNU GCC TOOLCHAIN](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) to be used and compiled.
Please refer to the links to correctly setup your working environment.

## Python - PyTorch requirements

To successfully run the tests, Python 3.6 is needed, together with PyTorch 1.9.0. To install the dependencies (with CPU only), run:

```
python3.6 -m pip install argparse 
python3.6 -m pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3.6 -m pip install torchsummary
```

If you require the GPU (CUDA 10.2) version for your applications, instead run:

```
python3.6 -m pip install argparse 
python3.6 -m pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3.6 -m pip install torchsummary
```

It is recommended to run the tests with torch version "1.9.0+cpu".

## PULP-TrainLib

To get started with PULP-TrainLib, just clone this repository.

Before compiling any project, source `pulp-sdk/configs/pulp_open.sh` from the terminal from which you intend to compile your project. 
The `configs/` folder is located inside the path to your pulp-sdk directory.

When generating a DNN for PULP with the TrainLib Deployer, make sure to launch the python task from a terminal in which you did not source the `pulp_open.sh` (otherwise you'll get an error).



# Branches

PULP-TrainLib's repository is organized with these branches:
- `pulp-trainlib-stable`: stable version 
- `pulp-trainlib-dev`: this is the work-in-progress version (more features, less stable)
- `pulp-trainlib-paper`: this is the version with the results shown in the paper "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"



# Status log

## Available features

- Forward passes for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32)
- Weight gradients for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32)
- Input gradients for DepthWise and PointWise Convolution, Fully-Connected, Conv2D (FP32)
- Matmul test (FP32, FP16)
- Memory profiling tool for estimating the memory occupation of a given layer/tile
- Autotuner to profile tile/mm_optimization

## To be completed/fixed

- Verification of all layer steps and adaptation of the tests to the new features (stride, padding)
- Dependency of the tensor sizes on padding and stride in TrainLib Deployer (fix the computation of memory occupation in the internal function of the deployer)
- DW, PW, Conv2D and all other primitives in FP16 (they need to be updated in the same way as FP32, which has the latest version)
- Conv2D test in FP16
- fp16 mm_dw SIMD kernels
- sporadic bugs in "mm_M_.." matmuls and "mm_u2" in FP32 (mostly on leftovers)
- fp16 DepthWise and PointWise test
- padding operators for each primitive (asymmetrical padding)
- stride operators for each primitive (implemented in im2col, but not tested)
- functions that take "mm_manager_list" files as inputs, other than mm_manager
- AutoTuner working with "NUM_TILING_SOLUTIONS = 1"



# References

D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"
