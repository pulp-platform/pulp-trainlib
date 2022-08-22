# PULP-TrainLib

PULP-TrainLib is the first Deep Neural Network training library for the PULP Platform. PULP-TrainLib features a wide set of performance-tunable DNN layer primitives for training, together with optimizers, losses and activation functions.
To enable on-device training, PULP-TrainLib is equipped with AutoTuner, a pre-deployment tool to select the fastest configuration for each DNN layer, according to the training step to be performed and the shapes of the layer tensors.
To facilitate the deployment of training tasks on the target PULP device, PULP-TrainLib is equipped with the TrainLib Deployer, a code generator capable of generating a project folder containing all the files and the code to run a DNN training task on PULP.

If you use any part of PULP-TrainLib , please cite:
```
@inproceedings{NadaliniPULPTrainLib22,
  author    = {Davide Nadalini and 
               Manuele Rusci and
               Giuseppe Tagliavini and
               Leonardo Ravaglia and
               Luca Benini and
               Francesco Conti},
  title     = {{{PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning}}},
  booktitle = {Embedded Computer Systems: Architectures, Modeling, and Simulation
               - 22nd International Conference, {SAMOS} 2022, Samos, Greece, July
               3-7, 2022, Proceedings (to appear)},
  series    = {Lecture Notes in Computer Science}
}
```

This repository is released under the [Apache License Version 2.0](./LICENSE).

## PULP-TrainLib's training library

PULP-TrainLib is the first open-source training library for RISC-V-based multicore MCUs, including a set of performance-tunable DNN layer primitives to enable DNN training on ultra-low-power devices.



# Repository overview - STM32 version

This library provides primitives to execute DNN training tasks on STM32 boards (forked from the PULP version). All the primitives feature CHW format.

All the library files are included in the `lib/` folder ([lib's README](lib/README.md)).



# Installation and requirements

## STM32CUBEIDE

The native environment for STM32 boards is [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html), which you can use to include the STM32 version of PULP-TrainLib and run DNN training on your MCU model. 

## PULP-TrainLib

To get started with PULP-TrainLib (STM32 version) clone this repository and copy it to your project's repository.

Please refer to the PULP version of the library to furhter details on the primitives (apart from the PULP-specific instructions, the structure of the library is the same). 

To correctly compile the library, just include `stm32_train.h`.



# Branches

PULP-TrainLib's repository is organized with these branches:
- `pulp-trainlib-stable`: stable version 
- `pulp-trainlib-dev`: this is the work-in-progress version (more features, less stable)
- `pulp-trainlib-paper`: this is the version with the results shown in the paper "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"
- `pulp-trainlib-stm32`: this is a version (initially forked from `pulp-trainlib-dev`) compatible with STM32, with FP32 format only



# Available features status log

- [x] Forward passes for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32)
- [x] Weight gradients for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32)
- [x] Input gradients for DepthWise and PointWise Convolution, Fully-Connected, Conv2D (FP32)
- [x] Matmul test (FP32, FP16)
- [x] Memory profiling tool for estimating the memory occupation of a given layer/tile
- [x] Autotuner to profile tile/mm_optimization
- [X] padding operators for each primitive (asymmetrical padding)
- [x] stride operators for conv2d and DW
- [ ] Verification of all layer steps and adaptation of the tests to the new features (stride, padding)
- [ ] Dependency of the tensor sizes on padding and stride in TrainLib Deployer (fix the computation of memory occupation in the internal function of the deployer)
- [ ] DW, PW, Conv2D and all other primitives in FP16 (they need to be updated in the same way as FP32, which has the latest version)
- [ ] Conv2D test in FP16
- [ ] fp16 mm_dw SIMD kernels
- [ ] sporadic bugs in "mm_M_.." matmuls and "mm_u2" in FP32 (mostly on leftovers)
- [ ] fp16 DepthWise and PointWise test
- [ ] functions that take "mm_manager_list" files as inputs, other than mm_manager
- [ ] AutoTuner working with "NUM_TILING_SOLUTIONS = 1"
- [ ] im2col with DMA (only partially implemented, to move L2 to L1 operands directly)
- [ ] bug fix in im2col with padding with kernel size > 2 (error in padding conditions)
- [ ] bug fix in im2col with DMA (FW/WG) when putting opposite paddings (left-right, up-down)



# References

D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"
