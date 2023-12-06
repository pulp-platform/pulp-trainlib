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


/**
 * Collection of different matrix multiply functions for different purposes
 * Use pi_cl_team_fork(NUM_CORES, MM_NAME, &args) to parallelize.
 */

/**
 * @brief Naive matrix multiply algorithm, performing C=A*B (C is N*M, A is N*K, B is K*M). Parallelizes on N.
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void mm_fp16(
    void * void_args
);

/**
 * @brief Naive matrix multiply algorithm, performing C=A*B (C is N*M, A is N*K, B is K*M). Parallelizes on M.
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void mm_M_fp16(
	void * void_args
);

/**
 * @brief Naive core kernel for Depthwise Convolution (forward). Parallelizes on the channels.
 * @param matMul_DW_args_fp16  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp16.h)
*/
void dw_kernel_forward_fp16(
    void * matMul_DW_args_fp16
);

/**
 * @brief Naive core kernel for Depthwise Convolution (weight gradient). Parallelizes on the channels.
 * @param matMul_DW_args_fp16  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp16.h)
*/
void dw_kernel_weight_grad_fp16(
    void * matMul_DW_args_fp16
);

/**
 * @brief Naive core kernel for Depthwise Convolution (input gradient). Parallelizes on the channels.
 * @param matMul_DW_args_fp16  pointer to a matMul_DW_args structure (please refer to pulp_train_utils_fp16.h)
*/
void dw_kernel_input_grad_fp16(
    void * matMul_DW_args_fp16
);

/**
 * @brief Matrix multiply for 2D convolution (input grads calculation, flips A matrix).
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void mm_conv2d_in_grad_fp16(
    void * void_args
);

/**
 * @brief Naive conv2d kernel for forward propagation (CWH format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_fw_kernel_CHW_fp16(
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the weight gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_param_grad_kernel_CHW_fp16(
    void * matMul_args
);

/**
 * @brief Naive conv2d kernel for the computation of the input gradient (CHW format)
 * @param matMul_args pointer to a matMul_args structure  
 */
void naive_conv2d_in_grad_kernel_CHW_fp16(
    void * matMul_args
);




/**
 * Optimized versions
 */

// =====> PARALLELISM ON N <=====

/**
 * @brief SIMD matmul which unrolls 2 elements of A and 4 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_SIMD_2x4 (
    void * void_args
);


/**
 * @brief SIMD matmul which unrolls 4 elements of A and 8 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_SIMD_4x8 (
    void * void_args
);



// =====> PARALLELISM ON M <=====

/**
 * @brief SIMD matmul which unrolls 2 elements of A and 4 of B. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_M_fp16_SIMD_2x4 (
    void * void_args
);

/**
 * @brief SIMD matmul which unrolls 4 elements of A and 8 of B. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_M_fp16_SIMD_4x8 (
    void * void_args
);
