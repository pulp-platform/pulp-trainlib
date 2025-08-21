/*
 * Copyright (C) 2021-2025 ETH Zurich and University of Bologna
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

 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
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
    float *data;
    float *diff;
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
 * @param pBuffer im2col buffer which will contain the transformed version of the data to be transformed
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
struct im2col_args {
    struct blob *input;
    struct blob *c;
    struct blob *output;
    float *pBuffer;
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
 * @brief Transposes an n-dimensional array, according to the required axis reordering,
 * outputting to another target array, similar to the NumPy procedure.
 *
 * @param in_array Input array of floats to be transposed
 * @param out_array Output transposed array of floats
 * @param dim Array of integers representing the dimensions of the input array
 * @param transposed_axes Array of integers representing the permutation of the axes
 * @param n_dim Integer representing the number of dimensions of the input array
 * @param N In case of a matrix transposition, number of rows
 * @param M In case of a matrix transposition, number of columns
 */
struct transp_args {
    float *in_matrix;
    float *out_matrix;
    int N;
    int M;
    int *dim;
    int *transposed_axes;
    int n_dim;
};


/**
 * @brief Args used to change the data layout of a tensor (CHW to HWC or vice versa)
 * @param tensor tensor whose layout needs to be changed
 * @param transp_buffer buffer of the size of the tensor's data/gradient to be used to change the format
 * @param transpose_data set this to 1 if you need to change the layout of tensor's data
 * @param transpose_grad set this to 1 if you need to change the layout of tensor's grad
 */
struct layout_args {
    struct blob *tensor;
    float *transp_buffer;
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
    float *weights;
    float *bt_weights;
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
    float *from;
    float *to;
    int size;
};


/**
 * @brief Arguments for the set_to_value function
 * @param to target array to set to a single value
 * @param value value to be used to fill the array
 * @param size size of the array
 **/
struct set_to_value_args {
    float *to;
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
    float *op_1;
    float *op_2;
    float *dest;
    int size;
};


/**
 * @brief Multi-dimensional array sum with NumPy-style broadcasting.
 *
 * @param op_1 First array to be summed
 * @param op_2 Second array to be summed
 * @param dest Destination array of the sum result
 * @param op_1_dims Dimensions of the first operand
 * @param op_2_dims Dimensions of the second operand
 * @param op_1_dims_len Number of dimensions of the first operand
 * @param op_2_dims_len Number of dimensions of the second operand
 */
struct array_broadcast_sum_fp32_args {
    float *op_1;
    float *op_2;
    float *dest;
    int *op_1_dims;
    int *op_2_dims;
    int op_1_dims_len;
    int op_2_dims_len;
};


/**
 * @brief Arguments for the cast_fp16_tensor_to_fp32 function
 * @param source pointer to a fp16 tensor to be cast in float 
 * @param destination pointer to the cast buffer
 * @param size number of elements of the tensor to be cast
 */
struct cast_16t32_args {
    fp16 *source;
    float *destination;
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
    float *source;
    float *dest;
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
 * @param bias pointer to bias vector
 * @param bias_dim dimension of bias (should be equal to C_out of layer)
 * @param USE_BIASES Set to 0 if not using biases, 1 if using biases
 * @param bias_transposed Set to 1 if you want to do column-wise bias add
 * @param HWC Set to 0 if CHW layout, 1 if HWC
 */
struct matMul_args {
    float *__restrict__ A;
    float *__restrict__ B;
    float *__restrict__ C;
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

    // For bias handling
    float *__restrict__ bias;
    int bias_dim;
    int USE_BIASES;
    int bias_transposed;
    int HWC;
};


/**
 * @brief Arguments for the matrix multiplication with NumPy-style broadcast support.
 * @param A pointer to the input matrix A
 * @param B pointer to the input matrix B
 * @param C pointer to the output matrix C
 * @param A_dims dimensions of the input matrix A
 * @param B_dims dimensions of the input matrix B
 * @param A_dims_len number of dimensions of the input matrix A
 * @param B_dims_len number of dimensions of the input matrix B
 */
struct broadcastMatMul_args_fp32 {
    float *__restrict__ A;
    float *__restrict__ B;
    float *__restrict__ C;

    int *__restrict__ A_dims;
    int *__restrict__ B_dims;

    int A_dims_len;
    int B_dims_len;
};


/**
 * @brief Arguments for the naive core kernel of DepthWise Convolution (forward and backward)
 * @param input pointer to the input blob
 * @param weight pointer to the weight blob
 * @param output pointer to the output blob
 *
 * @param stride_h horizontal stride
 * @param stride_w vertical stride
 *
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
*/
struct kernel_DW_args {
    struct blob *input;
    struct blob *weights;
    struct blob *output;

    int stride_h;
    int stride_w;

    int Lpad;
    int Rpad;
    int Upad;
    int Dpad;
};


/**
 * @brief Arguments for mm_manager function, which selects which matmul to be executed.
 * @param mm_args The pointer to the structure to be used by the matmul to be chosen (not for DW convolution)
 * @param mm_dw_args The pointer to the structure to be used by the matmul to be chosen (DW convolution only)
 * @param layer_type The type of layer in which to select the correct matmul. Can be targeted by using defines of type "LAYER_LINEAR" (groupdef inside pulp_train_utils).
 * @param step_type The step to be performed (forward, weight grad or input grad). Can be targeted by using defines of type "STEP_FW".
 * @param matmul_type The type of matmul to be selected for the chosen pass.
 */
struct mm_manager_args {
    struct matMul_args *mm_args;
    struct matMul_DW_args *mm_dw_args;
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
struct tanh_args {
    float *input;
    int dim;
    float *output;
};


/**
 * @brief Arguments weight updates output=output + gradient
 * @param accum    pointer to weight gradient accumulators
 * @param grad    pointer to weight gradient of the current timestep
 * @param dim       dimension vector
*/
struct update_weight_args {
    float *accum;
    float *grad;
    int dim;
};


/**
 * @brief Arguments for implementing parallelized max on an input vector
 * @param input     input vector on which we want to find the max
 * @param H         height of input
 * @param W         width of input
 * @param maxes     vector on which each core saves the max they have found
*/
struct max_args {
    float *input;
    int H;
    int W;
    float *maxes;
};


/**
 * @brief Arguments for implementing parallelized exponential and sum on an input vector
 * @param input     input vector on which we want to calculate the exponential and the summation
 * @param output    vector where the exponential is saved
 * @param H         height of input
 * @param W         width of input
 * @param maxes     maximum value of the input map
 * @param sums      vector on which each core saves their sum
*/
struct exp_sum_args {
    float *input;
    float *output;
    int H;
    int W;
    float *maxes;
    float *sums;
};


/**
 * @brief Arguments for implementing parallelized exponential and sum on an input vector
 * @param input   input vector on which we want to calculate the exponential and summatory
 * @param sums    vector on which each core saves their sum
 * @param output  vector where the exponential is saved
 * @param dim     dimension of input
 * @param dim     dimension of input^2
 * @param maxes   maximum value for each row of the input map
*/
struct shift_sum_args {
    float *input;
    float *sums;
    float *output;
    int dim;
    int dim2;
    float *maxes;
};


/**
 * @brief Arguments for implementing parallelized division of an input vector and a scalar
 * @param input   input vector we want to divide
 * @param n       scalar value we want to divide the vector with
 * @param dim     dimension of input
*/
struct div_args {
    float *input;
    float n;
    int dim;
};


/**
 * @brief Arguments for implementing parallelized division of an input vector and a vector
 * @param input     input vector we want to divide
 * @param H         height of input
 * @param W         width of input
 * @param sums      values we want to divide the vector with
*/
struct row_div_args {
    float *input;
    int H;
    int W;
    float *sums;
};


/**
 * @brief Arguments for implementing parallelized multiplication of an input vector and a scalar
 * @param input   input vector we want to multiply
 * @param scalar  scalar value we want to divide the vector with
 * @param dim     dimension of input
*/
struct scalar_mul_args {
    float *input;
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
struct mean_std_args {
    float *input;
    float *mean;
    float *var;
    float *std;
    float epsilon;
    int dim;
};


/**
 * @brief Arguments for calculating the sum(exp(input - max(input)))
 * @param input   input vector
 * @param sums    support buffer of the sums (1 for each core)
 * @param output  exp(input - max(input))
 * @param dim     dimension of input
 * @param max     max(input)
*/
struct vector_exp_sum_args {
    float *input;
    float *sums;
    float *output;
    int dim;
    float max;
};


/**
 * @brief Arguments for the first operation of the softmax backward pass.
 * @param A     *float: input matrix A [H x W]
 * @param B     *float: input matrix B [H x W]
 * @param S     *float: output vector S [H]
 * @param H     int: height of input matrices, length of output array
 * @param W     int: width of input matrices
 */
struct sm_bw_op_1_args {
    float *A;
    float *B;
    float *S;
    int H;
    int W;
};


/**
 * @brief Arguments for the first operation of the softmax backward pass.
 * @param A         *float: input matrix A [H x W]
 * @param B         *float: input matrix B [H x W]
 * @param S         *float: input vector S [H]
 * @param output    *float: output matrix [H x W]
 * @param H         int: height of input matrices, length of output array
 * @param W         int: width of input matrices
 */
struct sm_bw_op_2_args {
    float *A;
    float *B;
    float *S;
    float *output;
    int H;
    int W;
};


/**
 * @brief Arguments for the mat mul bias addition operation
 * @param mat       *float: input matrix mat [H x W]
 * @param bias      *float: bias [W]
 * @param H         int: height of input matrix
 * @param W         int: width of input matrix and length of bias
 * @param t         int: 1 if you want column-based bias add
 */
struct mm_bias_add_args {
    float *mat;
    float *bias;
    int H;
    int W;
    int t;
};


/**
 * @brief Arguments for the reduce mean operation in fp32
 * @param input         *float: input array
 * @param output        *float: output array
 * @param dims          *int: array containing the dimensions sizes of the input array
 * @param dims_len      int: number of dimensions of the input array
 * @param reduce_axis   int: axis along which to reduce the mean
 */
struct reduce_mean_args_fp32 {
    float *input;
    float *output;
    int *dims;
    int dims_len;
    int reduce_axis;
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
int verify_tensor(float *tensor_out, float *tensor_ref, int size, float tolerance);


/**
 * @brief Transpose a matrix with specified N, M sizes into another matrix array. Use pi_cl_team_fork(NUM_CORES, transpose_matrix, &args) to parallelize.
 * @param void_args (void *) (struct transp_args void_args)
 */
void transpose_matrix(void *void_args);


/**
 * @brief Transpose two dims of a multi-dimensional tensor. Use pi_cl_team_fork(NUM_CORES, transpose, &args) to parallelize.
 * @param void_args (void *) (struct transp_args void_args)
 */
void transpose(void *void_args);


/**
 * @brief Copies an array of size "size" into another destination array. Set up the arguments by using a "struct copy_args" structure. Use pi_cl_team_fork(NUM_CORES, copy, &args) to parallelize.
 * @param (void * ) (struct copy_args void_args)
 */
void copy(void *void_args);


/**
 * @brief Sets an array of size "size" to a value "value". Set up the arguments by using a "struct set_to_value_args" structure. Use pi_cl_team_fork(NUM_CORES, set_to_value, &args) to parallelize.
 * @param (void * ) (struct set_to_value_args void_args)
 */
void set_to_value(void *void_args);


/**
 * @brief Sums two arrays of size "size" into a third one. Set up the arguments by using a "struct vect_sum_args" structure. Use pi_cl_team_fork(NUM_CORES, vect_sum, &args) to parallelize.
 * @param vect_sum_args (void *) (struct vect_sum_args vect_sum_args)
 */
void vect_sum(void *vect_sum_args);


/**
 * @brief Sums two arrays of different but compatible sizes, with NumPy-style broadcasting.
 * @param arr_bc_args
 */
void array_broadcast_sum_fp32(void *arr_bc_args);


/**
 * @brief Cast a FP16 tensor to FP32. Set up the arguments by using a "struct cast_16t32_args" structure. Use pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &args) to parallelize.
 * @param (void *) (struct cast_16t32_args cast_args)
 */
void cast_fp16_tensor_to_fp32(void *cast_16t32_args);


/**
 * @brief Transforms the data layout of data/grad of a given tensor to CHW from HWC
 * @param layout_args (void *) (struct layout_args layout_args) 
 */
void HWC_to_CHW(void *layout_args);


/**
 * @brief Transforms the data layout of data/grad of a given tensor to HWC from CHW
 * @param layout_args (void *) (struct layout_args layout_args) 
 */
void CHW_to_HWC(void *layout_args);


/**
 * @brief Pad a tensor into a destination buffer specifying its size and the spatial sizes of the padding. Parallelize with pi_cl_team_fork(NUM_CORES, pad_tensor, &args).
 * @param pad_args (void *) (struct pad_args pad_args)
*/
void pad_tensor(void *pad_args);


/**
 * @brief Selects the matmul to be executed in the selected layer. Use pi_cl_team_fork(NUM_CORES, mm_manager, &args) to parallelize.
 * @param (void *) (struct mm_manager_args void_args)
 */
void mm_manager(void *void_args);


/**
 * @brief Calculates the exponential value of each element in the input vector/matrix.
 * @param (void *) (struct softmax_args void_args)
 */
void exponential(void *void_args);


/**
 * @brief Divides each output vector element by their sum.
 * @param (void *) (struct softmax_args void_args)
 */
void softmax(void *void_args);


/**
 * @brief Calculate the maxes of a vector in parallelized fashion. Set up the arguments by using a "struct max_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_max_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct max_args void_args)
 */
void pulp_max_fp32_cl(void *void_args);


/**
 * @brief Calculate the 1/2^diff of each element and sum them. Set up the arguments by using a "struct shift_sum_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_shift_sum_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct shift_sum_args void_args)
 */
void pulp_shift_sum_fp32_cl(void *void_args);


/**
 * @brief Element-wise division of vector with a single constant. Set up the arguments by using a "struct div_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_div_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct div_args void_args)
 */
void pulp_div_fp32_cl(void *void_args);


/**
 * @brief Element-wise multiplication of vector with a single constant. Set up the arguments by using a "struct scalar_mul_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct scalar_mul_args void_args)
 */
void pulp_scalar_mul_fp32_cl(void *void_args);


/**
 * @brief Simple thresholding function. x > threshold ? threshold : x. Also applies Taylor's series approximation of exponential to it.
 * @param float x
 */
float threshold(float x);


/**
 * @brief Function that clamps an input value between a minimum and a maximum (both for integers and float)
 * @param value the value to be clamped
 * @param min minimum value to be clamped to 
 * @param max maximum value to be clamped to
 */
int clamp_int(int value, int min, int max);
float clamp(float value, float min, float max);


static inline float
fasterexp(float p);


static inline float
fasterpow2(float p);


/**
 * @brief Mean, Variance and standard deviation calculation of a vector
 * @param (void *)  (struct mean_std_args void_args)
 */
void pulp_mean_std_fp32_cl(void *mean_std_args);


/**
 * @brief Approximated version of exponential using bit manipulation of mantissa and exponent. Returns the exponential of x.
 * @param x floating-point number to be exponentiated
 */
float fastexp_gist(float x);


/**
 * @brief Approximated version of the inverse square root, AKA the "fastinversesquareroot" (https://en.wikipedia.org/wiki/Fast_inverse_square_root).
 * @param x floating-point number to be fast-inverse-square-rooted
 */
float q_rsqrt(float number);


/**
 * @brief sum(exp(input - max(input))). Set up the arguments by using a "struct vector_exp_sum_args" structure. Use pi_cl_team_fork(NUM_CORES, vector_exp_sum_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct mean_std_args void_args)
 */
void vector_exp_sum_fp32_cl(void *vector_exp_sum_args);


/**
 * @brief CORDIC's sin and cos approximate calculator of input angle.
 * @param angle value in radians
 * @param cos pointer to the value to save the angle's cosine
 * @param sin pointer to the value to save the angle's sin
 */
void cordic_cos_sin_fp32(float angle, float *cos, float *sin);


// ~~~~~~~~~~~~~~~~~~ SOFTMAX FUNCTIONS ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~      FORWARD      ~~~~~~~~~~~~~~~~~~
/**
 * @brief Calculate the maxes for each row of a square matrix in parallelized fashion. Set up the arguments by using a "struct max_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct max_args void_args)
 */
void pulp_row_max_fp32_cl(void *void_args);


/**
 * @brief Calculate the exponential of each element and sum them. Set up the arguments by using a "struct exp_sum_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct exp_sum_args void_args)
 */
void pulp_exp_sum_fp32_cl(void *void_args);


/**
 * @brief Element-wise division of vector with values obtained by shit_sum. Set up the arguments by using a "struct row_div_args" structure. Use pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct div_args void_args)
 */
void pulp_row_div_fp32_cl(void *void_args);


// ~~~~~~~~~~~~~~~~~~      BACKWARD     ~~~~~~~~~~~~~~~~~~
/**
 * @brief The first operation of the backward pass of softmax. It receives 2 matrices, A and B, of the same size,
 * and returns a vector S with the same length as the height of either of the 2 input matrices. Each unit of this
 * output vector will be the sum of all the element-wise products of the corresponding row (element S[i] will contain
 * the sum for row i).
 * @param (void *)  (struct sm_bw_op_1_args void_args)
 */
void pulp_sm_bw_op_1(void *void_args);


/**
 * @brief The second operation of the backward pass of softmax. It receives 2 matrices, A and B, of the same size,
 * and a vector S with the same length as the height of either of the 2 input matrices, and an output matrix of the size
 * of either of the inputs. Each unit of this output matrix will have the value equal to (a - s) * b, where a and b are
 * the equivalent elements from matrices A and B and s is the current row-th element of S.
 * @param (void *)  (struct sm_bw_op_2_args void_args)
 */
void pulp_sm_bw_op_2(void *void_args);
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


/**
 * @brief Bias addition for matrix multiplication. Element-wise addition between bias and each column of the given matrix.
 * @param (void *) (struct mm_bias_add_args void_args)
 */
void mm_bias_add_transposed(void *void_args);


/**
 * @brief Reduce mean operation in fp32, similar to NumPy's np.mean() function.
 * Set up the arguments by using a "struct reduce_mean_args_fp32" structure.
 * Use pi_cl_team_fork(NUM_CORES, reduce_mean, &args) to parallelize.
 *
 * @param (void *) (struct reduce_mean_args_fp32 void_args)
 */
void reduce_mean_fp32(void *void_args);
