/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pmsis.h"
#include "pulp_train_defines.h"

/**
 * =====> STRUCTURES <=====
 */

/**
 * @brief "Bunch of data" structure, grouping a tensor and its gradient and sizes.
 * @param data pointer to the input data array
 * @param diff pointer to the input diff array
 * @param dim size of data as a 1-D array on memory
 * @param W width of data
 * @param H height of data
 * @param C number of channels of data
 */ 
struct blob_fp16 {
   fp16 * data;
   fp16 * diff;
   int dim;
   int W;
   int H;
   int C;
};

/**
 * @brief Arguments for im2col function
 * @param input input of the conv layer
 * @param c weight matrix of the conv layer
 * @param output output of the conv layer
 * @param pBuffer im2col buffer which will contain the transformed version of the data to be tranformed
 * @param pad  padding parameter
 * @param mod  0 stands for forward (im2col of the input feature map), 1 for backward (im2col and flip of output feature map)
 * @param tile_start  im2col starting element of current processed tile
 * @param tile_h  size of the tile
 * @param large sets the amount of padding as large-pad (to be fixed, im2col_temporary only)
 * @param stride_w sets the amount of horizontal stride
 * @param stride_h sets the amount of vertical stride
 * @param DW if == 1, notifies that the convolution is a DepthWise
 */
struct im2col_args_fp16
{
  struct blob_fp16 * input;
  struct blob_fp16 * c;
  struct blob_fp16 * output;
  fp16 * pBuffer;
  int pad;
  int mod;
  int tile_start;
  int tile_h;
  int large;
  int stride_w;
  int stride_h;
  int DW;
};

/**
 * @brief Arguments for ReLU
 * @param compare input of the activation function
 * @param out output array for the activation
 * @param size size of the arrays to be processed
 **/
struct relu_args_fp16 {
  fp16 * compare;
  fp16 * out;
  int size;
};


struct maxPool_args_fp16 {
  struct blob_fp16 * top;
  struct blob_fp16 * bottom;
  unsigned int * inMap;
  fp16 * tempBuffer;
};


struct softMax_args_fp16 {
  fp16 * input;
  fp16 * output;
  fp16 * sum;
  int size;
};

/**
 * @brief Arguments for the copy function
 * @param from source array
 * @param to array in which to copy 
 * @param size size of the arrays
 **/
struct copy_args_fp16 {
  fp16 * from;
  fp16 * to;
  int size;
};

/**
 * @brief Arguments for the set_to_value function
 * @param to target array to set to a single value
 * @param value value to be used to fill the array
 * @param size size of the array
 **/
struct set_to_value_args_fp16 {
  fp16 * to;
  fp16 value;
  int size;
};

/**
 * @brief Arguments for standard matrix multiplication C=A*B (A=N*K, B=K*M, result is C=N*M)
 * @param A  pointer to input matrix A
 * @param B  pointer to input matrix B
 * @param C  pointer to output matrix C
 * @param N  rows of A
 * @param M  columns of B
 * @param K  columns of A / rows of B
 * @param trans_B  if set to 1, compute C=A*Bt
 * @param H for Conv2D in grad: input width
 * @param W for Conv2D in grad: input height
 * @param pW for Conv2D in grad: kernel width
 * @param pH for Conv2D in grad: kernel height
 * @param pCin for Conv2D in grad: kernel in channels
 * @param pCout for Conv2D in grad: kernel out channels (number of blocks of filters with pCin channels each)
 */
struct matMul_args_fp16 {
  fp16 * __restrict__ A;
  fp16 * __restrict__ B;
  fp16 * __restrict__ C;
  int N;
  int M;
  int K;
  int trans_B;
  // For Conv2D in grad
  int H;
  int W;
  int pW;
  int pH;
  int pCin;
  int pCout;
};

/**
 * @brief Arguments for depthwise matrix multiplication (A=N*K, B=K*M, result is C=N*M)
 * @param A  pointer to input matrix A
 * @param B  pointer to input matrix B
 * @param C  pointer to output matrix C
 * @param N  rows of A
 * @param M  columns of B
 * @param K  columns of A / rows of B
 * @param ker_size  size of the kernel involved in the matrix multiplication
 */
struct matMul_DW_args_fp16 {
  fp16 * __restrict__ A;
  fp16 * __restrict__ B;
  fp16 * __restrict__ C;
  int N;
  int M;
  int K;
  int ker_size;
};

/**
 * @brief Arguments for mm_manager function, which selects which matmul to be executed.
 * @param mm_args The pointer to the structure to be used by the matmul to be chosen (not for DW convolution)
 * @param mm_dw_args The pointer to the structure to be used by the matmul to be chosen (DW convolution only)
 * @param layer_type The type of layer in which to select the correct matmul. Can be targeted by using defines of type "LAYER_LINEAR" (groupdef inside pulp_train_utils).
 * @param step_type The step to be performed (forward, weigth grad or input grad). Can be targeted by using defines of type "STEP_FW".
 * @param matmul_type The type of matmul to be selected for the chosen pass.
 */
struct mm_manager_args_fp16 {
  struct matMul_args_fp16 * mm_args;
  struct matMul_DW_args_fp16 * mm_dw_args;
  int layer_type;
  int step_type;
  int matmul_type;
};


/**
 * =====> FUNCTIONS <=====
 */

/**
 * @brief Copies an array of size "size" into another destination array. Set up the arguments by using a "struct copy_args_fp16" structure.
 * @param (void * ) (struct copy_args_fp16 void_args)
 */
void copy_fp16 (void * void_args);

/**
 * @brief Sets an array of size "size" to a value "value". Set up the arguments by using a "struct set_to_value_args_fp16" structure.
 * @param (void * ) (struct set_to_value_args_fp16 void_args)
 */
void set_to_value_fp16 (void * void_args);

/**
 * @brief Selects the matmul to be executed in the selected layer. Can be launched by using "pi_cl_team_fork", as well as any kind of matmul function.
 * @param (void *) (struct mm_manager_args_fp16 void_args)
 */
void mm_manager_fp16 (void * void_args);
