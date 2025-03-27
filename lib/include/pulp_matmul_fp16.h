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

 * Collection of different matrix multiply functions for different purposes
 * Use pi_cl_team_fork(NUM_CORES, MM_NAME, &args) to parallelize.
 */


/**
 * @brief Matrix multiplication algorithm, supporting multiple-sized arrays, with NumPy-style broadcasting.
 * @param broadcastMatMul_args_fp16 pointer to a broadcastMatMul_args_fp32 structure
 */
void mm_broadcast_fp16(void *broadcastMatMul_args_fp16);

/**
 * @brief Naive matrix multiply algorithm, performing C=A*B (C is N*M, A is N*K, B is K*M). Parallelizes on N.
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void mm_fp16(
        void *void_args
);

/**
 * @brief Naive matrix multiply algorithm, performing C=A*B (C is N*M, A is N*K, B is K*M). Parallelizes on M.
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void mm_M_fp16(
        void *void_args
);


/**
 * Optimized versions
 */

// =====> PARALLELISM ON N <=====

/**
 * @brief SIMD matmul which unrolls 2 elements of A and 4 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_SIMD_2x4(
        void *void_args
);

/**
 * @brief SIMD matmul which unrolls 4 elements of A and 8 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_SIMD_4x8(
        void *void_args
);

/**
 * @brief matmul which unrolls 8 elements of A and 1 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_unroll_8x1(
        void *void_args
);

/**
 * @brief matmul which unrolls 4 elements of A and 1 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_unroll_4x1(
        void *void_args
);

/**
 * @brief matmul which unrolls 2 elements of A and 1 of B. Parallelizes on N. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_fp16_unroll_2x1(
        void *void_args
);


// =====> PARALLELISM ON M <=====

/**
 * @brief SIMD matmul which unrolls 2 elements of A and 4 of B. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_M_fp16_SIMD_2x4(
        void *void_args
);

/**
 * @brief SIMD matmul which unrolls 4 elements of A and 8 of B. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mm_M_fp16_SIMD_4x8(
        void *void_args
);


//  =====> MATRIX X VECTOR OPERATIONS <=====

/**
 * @brief SIMD matrix * vector operation which unrolls 2 elements of A. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mv_fp16_SIMD_2x1(
        void *void_args
);

/**
 * @brief SIMD matrix * vector operation which unrolls 4 elements of A. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mv_fp16_SIMD_4x1(
        void *void_args
);

/**
 * @brief SIMD matrix * vector operation which unrolls 8 elements of A. Parallelizes on M. 
 * @param void_args pointer to a matMul_args_fp16 structure (please refer to this to setup the args)
 */
void __attribute__((noinline)) mv_fp16_SIMD_8x1(
        void *void_args
);
