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
 * @param stride_w sets the amount of horizontal stride
 * @param stride_h sets the amount of vertical stride
 * @param HWC sets if the format of the input (mod=0) or output grad (mod=1) is CHW (HWC=0) or HWC (HWC=1). In case of HWC, channels of the same "pixel" are adjacent, while in CHW the width elements are adjacent. Set this according to the format of your own input or output format (check format!) 
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
struct transp_args {
  float * matrix;
  float * transp_matrix;
  int N;
  int M;
};

/**
 * @brief Args used to change the data layout of a tensor (CHW to HWC or vice versa)
 * @param tensor tensor whose layout needs to be changed
 * @param transp_buffer buffer of the size of the tensor's data/gradient to be used to change the format
 * @param transpose_data set this to 1 if you need to change the layout of tensor's data
 * @param transpose_grad set this to 1 if you need to change the layout of tensor's grad
 */
struct layout_args {
  struct blob * tensor;
  float * transp_buffer;
  int transpose_data;
  int transpose_grad;
};

/**
 * @brief Arguments for pulp_blocktransp_fp32 to block-transpose a weight matrix (for conv2d in grad)
 * @param weights weights to be transposed 
 * @param Cin input channels of the convolutional layer
 * @param Cout output channels of the convolutional layer
 * @param Hk height of the convolutional kernel
 * @param Wk width of the convolutional kernel
 * @param HWC sets if the format of the input (mod=0) or output grad (mod=1) is CHW (HWC=0) or HWC (HWC=1). In case of HWC, channels of the same "pixel" are adjacent, while in CHW the width elements are adjacent. Set this according to the format of your own input or output format (check format!) 
 */
struct blocktransp_args {
  float * weights;
  float * bt_weights;
  int Cin;
  int Cout;
  int Hk;
  int Wk;
  int HWC;
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
 * @brief Arguments for the vect_copy function (sums two arrays)
 * @param op_1 first array to be summed of size "size"
 * @param op_2 second array to be summed of size "size"
 * @param dest third array which contains op_1 + op_2
 * @param size size of all the arrays
 */
struct vect_sum_args {
  float * op_1;
  float * op_2;
  float * dest;
  int size;
};

/**
 * @brief Arguments for the cast_fp16_tensor_to_fp32 function
 * @param source pointer to a fp16 tensor to be cast in float 
 * @param destination pointer to the cast buffer
 * @param size number of elements of the tensor to be cast
 */
struct cast_16t32_args {
  fp16 * source;
  float * destination;
  int size;
};

/**
 * @brief Arguments for the pad_tensor
 * @param source Tensor to be padded
 * @param dest Padded tensor
 * @param C Channels of the tensor
 * @param H Height of the tensor
 * @param W Width of the tensor
 * @param RPAD Right padding
 * @param LPAD Left padding
 * @param UPAD Upper padding
 * @param DPAD Lower padding
 * @param HWC_lay Set to 0 if CHW layout, 1 if HWC
*/
struct pad_args {
  float * source;
  float * dest;
  int C;
  int H;
  int W;
  int T_RPAD;
  int T_LPAD;
  int T_UPAD;
  int T_DPAD;
  int HWC_lay;
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
 * @param stride_w sets the amount of horizontal stride
 * @param stride_h sets the amount of vertical stride
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
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
  int stride_h;
  int stride_w;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
};

/**
 * @brief Arguments for the naive core kernel of DepthWise Convolution (forward and backward)
 * @param input pointer to the input blob
 * @param weight pointer to the weight blob
 * @param output pointer to the output blob
*/
struct kernel_DW_args {
  struct blob * input;
  struct blob * weights;
  struct blob * output;
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
 * @brief Arguments for tanh in parallel output=tanh(input)
 * @param input   pointer to input vector
 * @param dim     dimension vector
 * @param output  pointer to output vector
*/
struct tanh_args{
  float* input;
  int dim;
  float* output;
};

/**
 * @brief Arguments weight updates output=output + gradient
 * @param accum    pointer to weight gradient accumulators
 * @param grad    pointer to weight gradient of the current timestep
 * @param dim       dimension vector
*/
struct update_weight_args{
  float* accum;
  float* grad;
  int dim;
};

/**
 * @brief Arguments for implementing parallelized max on an input vector
 * @param input   input vector on which we want to find the max
 * @param maxes   vector on which each core saves the max they have found
 * @param dim     dimension of input
*/
struct max_args{
  float* input;
  float* maxes;
  int dim;
};

/**
 * @brief Arguments for implementing parallelized exponential and sum on an input vector
 * @param input   input vector on which we want to calculate the exponential and summatory
 * @param sums    vector on which each core saves their sum
 * @param output  vector where the exponential is saved
 * @param dim     dimension of input
 * @param max     maximum value of the input map
*/
struct exp_sum_args{
  float* input;
  float* sums;
  float* output;
  int dim;
  float max;
};

/**
 * @brief Arguments for implementing parallelized division of an input vector and a scalar
 * @param input   input vector we want to divide
 * @param n       scalar value we want to divide the vector with
 * @param dim     dimension of input
*/
struct div_args{
  float* input;
  float n;
  int dim;
};

/**
 * @brief Arguments for implementing parallelized multiplication of an input vector and a scalar
 * @param input   input vector we want to multiply
 * @param scalar  scalar value we want to divide the vector with
 * @param dim     dimension of input
*/
struct scalar_mul_args{
  float* input;
  float scalar;
  int dim;
};


/**
 * @brief Arguments for calculating mean, variance and standard deviation of a vector
 * @param input   input vector
 * @param mean    calculated mean
 * @param var    calculated var
 * @param std    calculated std
 * @param epsilon small number used to avoid division by zero
 * @param dim     dimension of input
*/
struct mean_std_args{
  float* input;
  float* mean;
  float* var;
  float* std;
  float epsilon;
  int dim;
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
 * @brief Sums two arrays of size "size" into a third one. Set up the arguments by using a "struct vect_sum_args" structure. Use pi_cl_team_fork(NUM_CORES, vect_sum, &args) to parallelize.
 * @param vect_sum_args (void *) (struct vect_sum_args vect_sum_args)
 */
void vect_sum (void * vect_sum_args);

/**
 * @brief Cast a FP16 tensor to FP32. Set up the arguments by using a "struct cast_16t32_args" structure. Use pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &args) to parallelize.
 * @param (void *) (struct cast_16t32_args cast_args)
 */
void cast_fp16_tensor_to_fp32 (void * cast_16t32_args);

/**
 * @brief Transforms the data layout of data/grad of a given tensor to CHW from HWC
 * @param layout_args (void *) (struct layout_args layout_args) 
 */
void HWC_to_CHW (void * layout_args);

/**
 * @brief Transforms the data layout of data/grad of a given tensor to HWC from CHW
 * @param layout_args (void *) (struct layout_args layout_args) 
 */
void CHW_to_HWC (void * layout_args);

/**
 * @brief Pad a tensor into a destination buffer specifying its size and the spatial sizes of the padding. Parallelize with pi_cl_team_fork(NUM_CORES, pad_tensor, &args).
 * @param (void *) (struct pad_args pad_args)
*/
void pad_tensor (void * pad_args);

/**
 * @brief Selects the matmul to be executed in the selected layer. Use pi_cl_team_fork(NUM_CORES, mm_manager, &args) to parallelize.
 * @param (void *) (struct mm_manager_args void_args)
 */
void mm_manager (void * void_args);

/**
 * @brief Calculates the exponential value of each element in the input vector/matrix.
 * @param (void *) (struct softmax_args void_args)
 */
void exponential (void * void_args);

/**
 * @brief Divides each output vector element by their sum.
 * @param (void *) (struct softmax_args void_args)
 */
void softmax (void * void_args);

/**
 * @brief Calculate the maxes of a vector in parallelized fashion
 * @param (void *)  (struct max_args void_args)
 */
void pulp_max_fp32_cl(void * void_args);

/**
 * @brief Calculate the exponential of each element and sum them
 * @param (void *)  (struct exp_sum_args void_args)
 */
void pulp_exp_sum_fp32_cl(void* void_args);

/**
 * @brief Element-wise division of vector with a single constant
 * @param (void *)  (struct div_args void_args)
 */
void pulp_div_fp32_cl(void* void_args);

/**
 * @brief Element-wise multiplication of vector with a single constant
 * @param (void *)  (struct scalar_mul_args void_args)
 */
void pulp_scalar_mul_fp32_cl(void* void_args);

static inline float
fasterexp (float p);

static inline float
fasterpow2 (float p);

/**
 * @brief Mean, Variance and standard deviation calculation of a vector
 * @param (void *)  (struct mean_std_args void_args)
 */
void pulp_mean_std_fp32_cl(void * mean_std_args);
