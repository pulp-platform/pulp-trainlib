# PULP-TrainLib - Training Library sources

PULP-TrainLib is organized as follows:
- `include/`: folder containing the header files of the library (together with all of the definitions of the structures and the function primitives)
- `sources/`: folder contatining all the implementations

To use PULP-TrainLib in your project, just include `pulp_train.h`.

## Structure definitions and data types

PULP-TrainLib is designed to work with different data types. For each data format, a set of source files is designed (for example, `pulp_xxx_fp32.h` refers to a header file with functions and data in fp32).

Many of the functions inside PULP-TrainLib require specific data structures to work, passing the operands implicitly (as void* args) for compatibility with PULP PMSIS. These data structures, for each data type, are defined inside `pulp_train_utils_fpxx.h`.

## Training primitives

With the same naming convention, each DNN layer has its own Forward and Backward functions, for each of the available data types. E.g: for a CONV2D layer in fp16, its training primitives are defined inside `pulp_conv2d_fp16.h` and `pulp_conv2d_fp16.c`. Inside each of these files, you can find both the forward and the backward functions. The same holds for the activation functions.

## Define the fp16 format 

The fp16 format refers to PULP native data formats. To select the format you need, please refer to `pulp_train_defines.h`, in which the format to be used is defined (`float16` refers to the FP16 Sign-Exponent-Mantissa 1-5-10 format, while `float16alt` to the Bfloat16 1-8-7 format).

## Other general defines

`pulp_train_defines.h` contains different kinds of definitions, as the available macros, the data type definitions etc.

## Building a DNN training workload

To build the code to run a DNN training on a PULP-based platform with PULP-TrainLib, you need to:
- define the arrays which will contain the data of the tensors
- define the sizes of each tensor
- define a `blob` structure to wrap the tensor and link the data array and the tensor sizes to it
- define the training algorithm (loss, optimizer, etc)
- run the computation on the PULP cluster, since each of the training functions is able to parallelize the execution on the `NUM_CORES` parameter.

Please refer to the [tests/](../tests/) folder for examples.

To speed up the design process, you can make use of the [TrainLib Deployer](../tools/TrainLib_Deployer/TrainLib_Deployer.py).


