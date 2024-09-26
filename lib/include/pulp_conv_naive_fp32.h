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

/** DEPTHWISE CONVOLUTION KERNELS **/

/**
 * @brief Naive core kernel for Depthwise Convolution (forward). Parallelizes on the channels.
 * @param matMul_DW_args  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp32.h)
*/
void dw_kernel_forward(
    void * matMul_DW_args
);

/**
 * @brief Naive core kernel for Depthwise Convolution (weight gradient). Parallelizes on the channels.
 * @param matMul_DW_args  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp32.h)
*/
void dw_kernel_weight_grad(
    void * matMul_DW_args
);

/**
 * @brief Naive core kernel for Depthwise Convolution (input gradient). Parallelizes on the channels.
 * @param matMul_DW_args  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp32.h)
*/
void dw_kernel_input_grad(
    void * matMul_DW_args
);


/** CONV2D KERNELS **/

/**
 * @brief Naive conv2d kernel for forward propagation (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_fw_kernel_CHW(
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the weight gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_param_grad_kernel_CHW(
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the input gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_in_grad_kernel_CHW(
    void * matMul_args
);



/** CONV2D OPTIMIZED VERSIONS **/

/**
 * @brief Naive conv2d kernel for forward propagation (CHW format), optimized for the case of 3x3 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_fw_kernel_CHW_k3x3_s2_p1 (
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the weight gradient (CHW format), optimized for the case of 3x3 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_param_grad_kernel_CHW_k3x3_s2_p1 (
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the input gradient (CHW format), optimized for the case of 3x3 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure
 */
void naive_conv2d_in_grad_kernel_CHW_k3x3_s2_p1 (
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for forward propagation (CHW format), optimized for the case of 5x5 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure
 */
void naive_conv2d_fw_kernel_CHW_k5x5_s2_p1 (
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the weight gradient (CHW format), optimized for the case of 5x5 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure
 */
void naive_conv2d_param_grad_kernel_CHW_k5x5_s2_p1 (
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the input gradient (CHW format), optimized for the case of 5x5 kernel with stride 2 and padding 1 on all sides
 * @param matMul_args pointer to a matMul_args structure
 */
void naive_conv2d_in_grad_kernel_CHW_k5x5_s2_p1 (
    void * matMul_args
);



/** TRANSPOSED CONV2D KERNELS **/

/**
 * @brief Naive transposed conv2d kernel for forward propagation (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_transp_conv2d_fw_kernel_CHW(
    void * matMul_args
);

/**
 * @brief Naive transposed conv2d kernel for the computation of the weight gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_transp_conv2d_param_grad_kernel_CHW(
    void * matMul_args
);

/**
 * @brief Naive transposed conv2d kernel for the computation of the input gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_transp_conv2d_in_grad_kernel_CHW(
    void * matMul_args
);