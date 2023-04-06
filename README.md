# PULP-TrainLib

PULP-TrainLib is the first Deep Neural Network (DNN) training library for multi-core RISC-V MCUs (as PULP), enabling On-Device Learning on ultra-low-power devices of this class.

PULP-TrainLib features a variety of training primitives and support functions to enable backpropagation-based training on multicore MCUs. More in depth:

- A set of performance-tunable DNN layer primitives for training, based on Matrix Multiplication (MM). 
- Several common loss functions, like MSE and CrossEntropy. 
- SGD-based optimizers.
- Activation (ReLU, etc) and support functions.

[PULP-TrainLib](./lib/) is fully released as open-source under [Apache License Version 2.0](./LICENSE).

To ease the deployment of DNN training tasks on MCUs, PULP-TrainLib is equipped with additional tools:

- [TrainLib_Deployer](./tools/TrainLib_Deployer/), an automated code-generation tool to generate the C code to validate and train a user-specified DNN model on a PULP architecture. 
- [AutoTuner](./tools/AutoTuner/), a pre-deployment tool to select the fastest configuration of each layer of a DNN model, according to the shapes of the layer, the training step and the tiling strategy.

If you use any part of PULP-TrainLib , please cite:
```
@InProceedings{10.1007/978-3-031-15074-6_13,
author="Nadalini, Davide
and Rusci, Manuele
and Tagliavini, Giuseppe
and Ravaglia, Leonardo
and Benini, Luca
and Conti, Francesco",
editor="Orailoglu, Alex
and Reichenbach, Marc
and Jung, Matthias",
title="PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-core MCUs Through Performance-Driven Autotuning",
booktitle="Embedded Computer Systems: Architectures, Modeling, and Simulation",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="200--216",
abstract="An open challenge in making Internet-of-Things sensor nodes ``smart'' and self-adaptive is to enable on-chip Deep Neural Network (DNN) training on Ultra-Low-Power (ULP) microcontroller units (MCUs). To this aim, we present a framework, based on PULP-TrainLib, to deploy DNN training tasks on RISC-V-based Parallel-ULP (PULP) MCUs. PULP-TrainLib is a library of parallel software DNN primitives enabling the execution of forward and backward steps on PULP MCUs. To optimize PULP-TrainLib's kernels, we propose a strategy to automatically select and configure (autotune) the fastest among a set of tiling options and optimized floating-point matrix multiplication kernels, according to the tensor shapes of every DNN layer. Results on an 8-core RISC-V MCU show that our auto-tuned primitives improve MAC/clk by up to 2.4{\$}{\$}{\backslash}times {\$}{\$}{\texttimes}compared to ``one-size-fits-all'' matrix multiplication, achieving up to 4.39 MAC/clk - 36.6{\$}{\$}{\backslash}times {\$}{\$}{\texttimes}better than a commercial STM32L4 MCU executing the same DNN layer training workload. Furthermore, our strategy proves to be 30.7{\$}{\$}{\backslash}times {\$}{\$}{\texttimes}faster than AIfES, a state-of-the-art training library for MCUs, while training a complete TinyML model.",
isbn="978-3-031-15074-6"
}

```

This repository is released under the [Apache License Version 2.0](./LICENSE).

## PULP-TrainLib's training library

PULP-TrainLib is the first open-source training library for RISC-V-based multicore MCUs, including a set of performance-tunable DNN layer primitives to enable DNN training on ultra-low-power devices. The training flow of PULP-TrainLib's primitives follows the canonic approach for the backpropagation algorithm, considering a streaming approach (batch size = 1). I.e., first we compute the Forward (FW) step to compute the prediction for a given input data. Then, a Backward Step is called to compute, layer-by-layer, the gradient of the loss function with respect to the weights (WG-BW) and the gradient of the loss function with respect to the input (IG-BW). The structure of the training flow of a single layer (e.g. a Fully-Connected) is depicted as follows:

![PULP-TrainLib's Primitives](./assets/img/pulp-trainlib-primitives.png)

Note that every training step for each layer is implemented as a Matrix Multiplication (MM) between tensor data. For a Conv2D and Fully-Connected Layer, the structure and sizes of the involved matrices can be represented as follows:

![MM-based training primitives](./assets/img/pulp-trainlib-mm-flow.png)

Convolutions are implemented as Image-to-Column (or Image-to-Row) pre-processed data + MM. The sizes of the tensors are denoted as (CI, HI, WI) for the input feature map, (CO, CI, Hk, Wk) for the weights, and (CO, HO, WO) for the output feature map. To tune the performances of the training primitives, specific optimizations can be selected case-by-case for the MM algorithm. 


## The TrainLib_Deployer

The development of C code for running On-Device Learning can be a time-consuming process. To make deployment easier, PULP-TrainLib provides TrainLib_Deployer, a code generation tool which creates all the necessary files to run DNN validation and training on PULP. To minimize the memory occupation, the TrainLib_Deployer adopts a data-reuse approach to store tensors in C arrays. The flow of the TrainLib_Deployer is illustrated as follows:

![TrainLib_Deployer](./assets/img/trainlib-deployer.png)

The input arguments of the TrainLib_Deployer are the architecture of the model to be trained on an MCU and the setup (memory and number of cores) of the target device. While running, the tool takes care of verifying if the model fits the memory. As output, the tool generates a project folder containing all the code to run a verification task on the target model (PyTorch Golden Model, or GM, C code, Makefile). 


## PULP-TrainLib's AutoTuner

PULP-TrainLib optimizes the core computational kernel of DNN training primitives - the Matrix Multiplication (or MM) - with various unrolling and parallelization schemes. To select the best optimization for a given training step and tile size, PULP-TrainLib provides an Autotuner, which exhaustively searches for the fastest kernel among the [library of available optimized MM kernels](lib/include/pulp_matmul_fp32.h). AutoTuner's flow can be represented as follows:

![AutoTuner](./assets/img/autotuner.png)

Given the properties of the target device and the layer/training step informations on a generic layer (e.g. 8 cores, 64kB, Conv2D, Forward), AutoTuner exhaustively searches for the fastest tile shape which best fits the specified memory amount and the fastest MM kernel which minimizes the latency on the given tile shape. For further info, readers may refer to "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf).




# Repository overview

PULP-TrainLib's library files are located under the `lib/` folder ([lib's README](lib/README.md)).

The `tests/` folder provides useful tests to try out and verify PULP-TrainLib's layers and functions (tests are performed against PyTorch Golden models).
Each test can be customized according to the user specifications and profiles the execution of the layer's primitives with PULP's performance counters.
If further info are needed, please refer to the [test's README](tests/README.md).

The `tools/` folder contains useful tools which ease the usage of PULP-TrainLib, as the TrainLib_Deployer and AutoTuner. For further info, please refer to [tools' README](tools/README.md).



# Installation and requirements

## PULP-SDK

PULP-TrainLib requires [PULP-SDK](https://github.com/pulp-platform/pulp-sdk) and the [RISC-V GNU GCC TOOLCHAIN](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) to be used and compiled.
Please refer to the links to correctly setup your working environment.

## Python - PyTorch requirements

To successfully run the tests, Python (>= 3.6) is needed, together with PyTorch 1.9.0. To install the dependencies (with CPU only), run:

```
python -m pip install argparse 
python -m pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install torchsummary
```

If you require the GPU (CUDA 10.2) version for your applications, instead run:

```
python -m pip install argparse 
python -m pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install torchsummary
```

It is recommended to run the tests with torch version "1.9.0+cpu".


## PULP-TrainLib

To get started with PULP-TrainLib, just clone this repository on your local PC.

Before compiling any project, source `pulp-sdk/configs/pulp_open.sh` from the terminal from which you intend to compile your project. 
The `configs/` folder is located inside the path to your pulp-sdk directory.

When generating a DNN for PULP with the TrainLib Deployer, make sure to launch the python task from a terminal in which you did not source the `pulp_open.sh`.



# Branches

PULP-TrainLib's repository is organized with these branches:
- `pulp-trainlib-stable`: stable branch 
- `pulp-trainlib-dev`: work-in-progress branch (more features, less stable)
- `pulp-trainlib-paper`: branch to generate the results provided in the paper ["PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning"](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf)
- `pulp-trainlib-stm32`: this is a PULP-TrainLib port compatible with STM32 and other MCUs (FP32 format only).



# Available features status log

> Note: checked are complete, unchecked are ongoing

- [X] Forward passes for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32, FP16)
- [X] Weight gradients for DepthWise, PointWise and 2D Convolution, Fully-Connected (FP32, FP16)
- [X] Input gradients for DepthWise and PointWise Convolution, Fully-Connected, Conv2D (FP32, FP16)
- [X] CWH data layout for DepthWise, PointWise and 2D Convolutions (FP32, FP16)
- [X] ReLU activation function (FP32, FP16)
- [X] Gradient Descent optimizer (FP32, FP16)
- [X] Max and Average Pooling (FP32, FP16)
- [X] Padding operators for DepthWise and 2D Convolution (Forward & Weight Grad, FP32 and FP16)
- [ ] Padding operators for DepthWise and 2D Convolution (Input Grad, FP32 and FP16)
- [ ] HWC data layout for PointWise and 2D Convolutions (FP32, FP16)
- [ ] HWC data layout for DepthWise Convolutions (FP32, FP16)
- [ ] Stride operators for 2D Convolutions and DepthWise
- [ ] Verification of all layer steps and adaptation of the tests to the new features (stride, padding)
- [ ] RNN training primitives (FP32, FP16)
- [ ] Multihead Self Attention training primitives (FP32, FP16)
- [ ] Residual connection

# Known bugs / issues (open for contributions)

- FP32 HWC Conv2D primitives (Input Grad)
- FP16 HWC Conv2D primitives (Input Grad)
- AutoTuner working with "NUM_TILING_SOLUTIONS = 1"
- Sporadic bugs in "mm_u2" in FP32 (mostly on leftovers)
- Performance bugs in im2col/im2row with DMA loading (performances tend to be less than im2col/im2row with cores)
- Missing test for residual connections
- Missing integration for both residual connections and RNN / MHSE in TrainLib_Deployer


# Contributors

## Currently active

- Davide Nadalini (d.nadalini@unibo.it, davide.nadalini@polito.it)
- Alberto Dequino (alberto.dequino@unibo.it, alberto.dequino@polito.it)
- Manuele Rusci (manuele.rusci@kuleuven.be)
- Francesco Conti (f.conti@unibo.it)

## Past Contributors

- Francesco Conoscenti (francesco.conoscenti@studio.unibo.it)
- Leonardo Ravaglia (leonardo.ravaglia2@unibo.it)


# References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf)
