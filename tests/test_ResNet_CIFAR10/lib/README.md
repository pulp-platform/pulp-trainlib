# PULP-TrainLib - Training Library sources

PULP-TrainLib is organized as follows:
- `include/`: contains the header files. The structures to configure the support functions and their function headers are defined in `pulp_train_utils_fpXX.h`. The layer configuration structures and primitives are defined in each layers' .h file.
- `sources/`: contains the body of the functions.

To use PULP-TrainLib in your project, include `pulp_train.h`.

## Structure definitions and data types

PULP-TrainLib is designed to work with different data types. For each type, PULP-TrainLib provides a set of files (for example, `pulp_xxx_fp32.h` contains fp32 primitives).

Most of PULP-TrainLib's primitives and functions require specific data structures as arguments, passing the operands implicitly (as void* args). For each data type, the data structures are defined inside `pulp_train_utils_fpxx.h` (for support functions) and inside the related file for primitives (e.g. `pulp_linear_fp32.h` for a fp32 fully-connected layer).

## Training primitives

The same naming convention holds for each DNN layer. In each layers' files, the Forward and Backward functions are defined for each of the available data types. E.g: the primitives of a CONV2D layer in fp16 are defined inside `pulp_conv2d_fp16.h` and `pulp_conv2d_fp16.c`. Inside each of these files, you can find both the forward and the backward functions. The same holds for the activation functions and others.

## Define the fp16 format 

The PULP Platform supports multiple fp16 data formats. To select the one you need, please refer to `pulp_train_defines.h`. In this file, you can select either `float16` (fp16 1-5-10 - Sign-Exponent-Mantissa), or `float16alt` (Bfloat16 1-8-7).

## Other general defines

`pulp_train_defines.h` contains useful defines and macros used to support the library.

## Building a DNN training workload

To automatically generate the C deployment code of your DNN model, make use of the [TrainLib Deployer](../tools/TrainLib_Deployer/TrainLib_Deployer.py).

To manually develop the deployment code for your on-device learning task, you need to:

- define the arrays which will contain tensor data and gradients;
- define the sizes of each tensor;
- for each tensor, define a `blob` structure to wrap the data, gradient and sizes;
- write the code for forward and backward steps, comprising the loss and optimizer. Note that the optimizer has to be applied to all the weight tensors of each layer;
- schedule the training task from PULP's fabric controller to PULP's cluster to exploit parallel computations on `NUM_CORES`.

Please refer to the [tests/](../tests/) folder for examples.
