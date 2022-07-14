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
struct blob {
   float * data;
   float * diff;
   int dim;
   int W;
   int H;
   int C;
};

/**
 * @brief Arguments for im2col function
 * @param input input blob of the conv layer
 * @param c weight matrix blob of the conv layer
 * @param output output blob of the conv layer
 * @param pBuffer im2col buffer which will contain the transformed version of the data to be tranformed
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param mod  0 stands for forward (im2col of the input feature map), 1 for backward (im2col and flip of output feature map)
 * @param tile_start  im2col starting element of current processed tile
 * @param tile_h  horizontal size of the tile
 * @param stride_w sets the amount of horizontal stride
 * @param stride_h sets the amount of vertical stride
 * @param DW if == 1, notifies that the convolution is a DepthWise
 * @param USE_DMA set this to 1 if your tensor data is in L2 and you want to im2col that data into local L1 stored im2colbuffer, using cluster DMA
 */
struct im2col_args
{
  struct blob * input;
  struct blob * c;
  struct blob * output;
  float * pBuffer;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
  int mod;
  int tile_start;
  int tile_h;
  int stride_w;
  int stride_h;
  int DW;
  int USE_DMA;
};

/**
 * @brief Arguments for ReLU
 * @param compare input of the activation function
 * @param out output array for the activation
 * @param size size of the arrays to be processed
 **/
struct relu_args {
  float * compare;
  float * out;
  int size;
};

/**
 * @brief Arguments for pooling functions
 * @param input input blob for the function
 * @param output output blob for the function
 * @param Hker vertical size of the pooling kernel
 * @param Wker horizontal size of the pooling kernel
 * @param Hstride controls the vertical stride of the kernel
 * @param Wstride controls the horizontal stride of the kernel
 */
struct pool_args {
  struct blob * input;
  struct blob * output;
  int Hker;
  int Wker;
  int Hstride;
  int Wstride;
};

struct softMax_args {
  float * input;
  float * output;
  float * sum;
  int size;
};

/**
 * @brief Parameters for optimizer fucntions for every single layer
 * @param weights blob of the weights (with their gradient inside)
 * @param learning_rate the learning rate of the optimizer
 */
struct optim_args {
  struct blob * weights;
  float learning_rate;
};

/**
 * @brief Transposes an array containing a matrix (of sizes N and M) into another target array
 * @param matrix Matrix to be transposed
 * @param transp_matrix Output tranposed matrix
 * @param N Number of rows of the matrix
 * @param M Number of columns of the matrix
 */
struct transp_args {
  float * matrix;
  float * transp_matrix;
  int N;
  int M;
};


/**
 * @brief Arguments for pulp_blocktransp_fp32 to block-transpose a weight matrix (for conv2d in grad)
 * @param weights weights to be transposed 
 * @param Cin input channels of the convolutional layer
 * @param Cout output channels of the convolutional layer
 * @param Hk height of the convolutional kernel
 * @param Wk width of the convolutional kernel
 */
struct blocktransp_args {
  float * weights;
  float * bt_weights;
  int Cin;
  int Cout;
  int Hk;
  int Wk;
};

/**
 * @brief Arguments for the copy function
 * @param from source array
 * @param to array in which to copy 
 * @param size size of the arrays
 **/
struct copy_args {
  float * from;
  float * to;
  int size;
};

/**
 * @brief Arguments for the set_to_value function
 * @param to target array to set to a single value
 * @param value value to be used to fill the array
 * @param size size of the array
 **/
struct set_to_value_args {
  float * to;
  float value;
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
struct matMul_args {
  float * __restrict__ A;
  float * __restrict__ B;
  float * __restrict__ C;
  int N;
  int M;
  int K;
  int trans_B;
  // For Conv2D in grad & naive
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
struct matMul_DW_args {
  float * __restrict__ A;
  float * __restrict__ B;
  float * __restrict__ C;
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
struct mm_manager_args {
  struct matMul_args * mm_args;
  struct matMul_DW_args * mm_dw_args;
  int layer_type;
  int step_type;
  int matmul_type;
};


/**
 * =====> FUNCTIONS <=====
 */

/**
 * @brief Checks if a tensor is equal to a reference one and notifies the index and the value of the incorrect values. If tensor_out contains errors, a flag is also raised as return value.
 * 
 * @param tensor_out tensor to be checked
 * @param tensor_ref reference tensor
 * @param size number of elements of the tensors to be compared
 * @param tolerance tolerance on the difference between the tensors
 * @return int 0, 1: flag that notifies if the checked tensor contains errors
 */
int verify_tensor(float * tensor_out, float * tensor_ref, int size, float tolerance);

/**
 * @brief Transpose a matrix with specified N, M sizes into another matrix array. Use pi_cl_team_fork(NUM_CORES, transpose, &args) to parallelize.
 * @param void_args (void *) (struct transp_args void_args)
 */
void transpose(void * void_args);

/**
 * @brief Copies an array of size "size" into another destination array. Set up the arguments by using a "struct copy_args" structure. Use pi_cl_team_fork(NUM_CORES, copy, &args) to parallelize.
 * @param (void * ) (struct copy_args void_args)
 */
void copy (void * void_args);

/**
 * @brief Sets an array of size "size" to a value "value". Set up the arguments by using a "struct set_to_value_args" structure. Use pi_cl_team_fork(NUM_CORES, set_to_value, &args) to parallelize.
 * @param (void * ) (struct set_to_value_args void_args)
 */
void set_to_value (void * void_args);

/**
 * @brief Selects the matmul to be executed in the selected layer. Use pi_cl_team_fork(NUM_CORES, mm_manager, &args) to parallelize.
 * @param (void *) (struct mm_manager_args void_args)
 */
void mm_manager (void * void_args);
