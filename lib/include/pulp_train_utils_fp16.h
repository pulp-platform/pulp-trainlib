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
 * =====> BACKEND STRUCTURES <=====
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
 * @param input input blob of the conv layer
 * @param c weight matrix blob of the conv layer
 * @param output output blob of the conv layer
 * @param pBuffer im2col buffer which will contain the transformed version of the data to be tranformed
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param mod  0 stands for forward (im2col of the input feature map), 1 for backward (im2col and flip of output feature map)
 * @param stride_w sets the amount of horizontal stride
 * @param stride_h sets the amount of vertical stride
 * @param HWC sets if the format of the input (mod=0) or output grad (mod=1) is CHW (HWC=0) or HWC (HWC=1). In case of HWC, channels of the same "pixel" are adjacent, while in CHW the width elements are adjacent. Set this according to the format of your own input or output format (check format!) 
 * @param USE_DMA set this to 1 if your tensor data is in L2 and you want to im2col that data into local L1 stored im2colbuffer, using cluster DMA
 */
struct im2col_args_fp16
{
  struct blob_fp16 * input;
  struct blob_fp16 * c;
  struct blob_fp16 * output;
  fp16 * pBuffer;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
  int mod;
  int stride_w;
  int stride_h;
  int HWC;
  int USE_DMA;
};

/**
 * @brief Transposes an array containing a matrix (of sizes N and M) into another target array
 * @param matrix Matrix to be transposed
 * @param transp_matrix Output tranposed matrix
 * @param N Number of rows of the matrix
 * @param M Number of columns of the matrix
 */
struct transp_args_fp16 {
  fp16 * matrix;
  fp16 * transp_matrix;
  int N;
  int M;
};

/**
 * @brief Arguments for pulp_blocktransp_fp16 to block-transpose a weight matrix (for conv2d in grad)
 * @param weights weights to be transposed 
 * @param Cin input channels of the convolutional layer
 * @param Cout output channels of the convolutional layer
 * @param Hk height of the convolutional kernel
 * @param Wk width of the convolutional kernel
 */
struct blocktransp_args_fp16 {
  fp16 * weights;
  fp16 * bt_weights;
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
 * @brief Arguments for the cast_fp32_tensor_to_fp16 function
 * @param source pointer to a fp32 tensor to be cast in float 
 * @param destination pointer to the cast buffer
 * @param size number of elements of the tensor to be cast
 */
struct cast_32t16_args {
  float * source;
  fp16 * destination;
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
 * @brief Checks if a tensor is equal to a reference one and notifies the index and the value of the incorrect values. If tensor_out contains errors, a flag is also raised as return value.
 * 
 * @param tensor_out tensor to be checked
 * @param tensor_ref reference tensor
 * @param size number of elements of the tensors to be compared
 * @param tolerance tolerance on the difference between the tensors
 * @return int 0, 1: flag that notifies if the checked tensor contains errors
 */
int verify_tensor_fp16(fp16 * tensor_out, fp16 * tensor_ref, int size, fp16 tolerance);

/**
 * @brief Transpose a matrix with specified N, M sizes into another matrix array. Use pi_cl_team_fork(NUM_CORES, transpose_fp16, &args) to parallelize.
 * @param void_args (void *) (struct transp_args_fp16 void_args)
 */
void transpose_fp16(void * void_args);

/**
 * @brief Copies an array of size "size" into another destination array. Set up the arguments by using a "struct copy_args_fp16" structure. Use pi_cl_team_fork(NUM_CORES, copy_fp16, &args) to parallelize.
 * @param (void * ) (struct copy_args_fp16 void_args)
 */
void copy_fp16 (void * void_args);

/**
 * @brief Sets an array of size "size" to a value "value". Set up the arguments by using a "struct set_to_value_args_fp16" structure. Use pi_cl_team_fork(NUM_CORES, set_to_value_fp16, &args) to parallelize.
 * @param (void * ) (struct set_to_value_args_fp16 void_args)
 */
void set_to_value_fp16 (void * void_args);

/**
 * @brief Cast a FP32 tensor to FP16. Set up the arguments by using a "struct cast_32t16_args" structure. Use pi_cl_team_fork(NUM_CORES, cast_fp32_tensor_to_fp16, &args) to parallelize.
 * @param (void *) (struct cast_32t16_args cast_args)
 */
void cast_fp32_tensor_to_fp16 (void * cast_32t16_args);

/**
 * @brief Selects the matmul to be executed in the selected layer. Use pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &args) to parallelize.
 * @param (void *) (struct mm_manager_args_fp16 void_args)
 */
void mm_manager_fp16 (void * void_args);


/**
 * =====> ASSEMBLY CALLS <=====
 */

/**
 * @brief Assembly call to vfdotp (dot product of two v2f16 vectors)
 * @param a first v2f16 vector
 * @param b second v2f16 vector
 * @return fp16 result of the dot product
 */
fp16 vfdotp(v2f16 a, v2f16 b);

/**
 * @brief Packs two fp16 elements into a v2f16 vector
 * @param a MSB packed element
 * @param b LSB packed element
 * @return v2f16 vector of packed element
 */
v2f16 vfpack(fp16 a, fp16 b);