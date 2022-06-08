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
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm(
    void * void_args
);

/**
 * @brief Naive matrix multiply algorithm, performing C=A*B (C is N*M, A is N*K, B is K*M). Parallelizes on M.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M(
	void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix). Parallelizes on K/ker_dim.
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad(
    void * void_args
);

/**
 * @brief Matrix multiply for 2D convolution (input grads calculation, flips A matrix).
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_conv2d_in_grad(
    void * void_args
);



/**
 * Optimized versions
 */

// =====> PARALLELISM ON N <=====

/**
 * @brief Standard matmul with unrolling factor of 2 in the inner loop on K, parallelizes on N.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_u2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 1 row of A, 2 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_1x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 1 row of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_1x4(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 1 row of A, 8 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_1x8(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 2 row of A, 1 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_2x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 4 row of A, 1 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_4x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 8 row of A, 1 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_8x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 2 rows of A, 2 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_2x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 2 rows of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_2x4(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 4 rows of A, 2 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_4x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on N. Unrolls 4 rows of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_unroll_4x4(
    void * void_args
);



// =====> PARALLELISM ON M <=====

/**
 * @brief Naive matmul with unrolling factor of 2 in the inner loop on K, parallelizes on M.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_u2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 2 rows of A, 1 column of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_2x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 4 rows of A, 1 column of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_4x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 8 rows of A, 1 column of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_8x1(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 1 row of A, 2 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_1x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 1 row of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_1x4(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 1 row of A, 8 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_1x8(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 2 rows of A, 2 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_2x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 2 rows of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_4x2(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 2 rows of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_2x4(
    void * void_args
);

/**
 * @brief Standard matmul with unrolling, parallelizes on M. Unrolls 4 rows of A, 4 columns of B.
 * @param void_args pointer to a matMul_args structure (please refer to this to setup the args)
 */
void mm_M_unroll_4x4(
    void * void_args
);



// =====> Matmuls for depthwise convolution <=====

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 2 in the inner loop. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_u2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 3 in the inner loop. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_u3(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 2 in M. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_unroll_1x2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 4 in M. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_unroll_1x4(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 2 in M and 2 in the inner loop. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_unroll_1x2_u2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution, with unrolling of 4 in M and 2 in the inner loop. Parallelizes on N. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_unroll_1x4_u2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 2 in the inner loop. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_u2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 3 in the inner loop. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_u3(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 2 in M. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_unroll_1x2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 4 in M. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_unroll_1x4(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 2 in M and 2 in the inner loop. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_unroll_1x2_u2(
    void * void_args
);

/**
 * @brief Matrix multiply for depthwise convolution (input grads calculation, flips A matrix), with unrolling of 4 in M and 2 in the inner loop. Parallelizes on K/ker_dim. 
 * @param void_args pointer to a matMul_DW_args structure (please refer to this to setup the args)
 */
void mm_dw_in_grad_unroll_1x4_u2(
    void * void_args
);
