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
 *
 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
*/ 


#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"

#include "pmsis.h"


/**
 * NAIVE VERSIONS
 */

void mm_broadcast_fp16(void *broadcastMatMul_args) {
    // Extract arguments
    struct broadcastMatMul_args_fp16 *args = (struct broadcastMatMul_args_fp16 *) broadcastMatMul_args;

    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    int *__restrict__ A_dims = args->A_dims;
    int *__restrict__ B_dims = args->B_dims;

    const uint32_t A_dims_len = args->A_dims_len;
    const uint32_t B_dims_len = args->B_dims_len;

    // Compute number of output matrices
    uint32_t max_dims_len;
    if (A_dims_len > B_dims_len) max_dims_len = A_dims_len - 2;
    else max_dims_len = B_dims_len - 2;

    uint32_t out_dims[max_dims_len];
    uint32_t total_matrices = 1;

    const uint32_t start_A = A_dims_len - 3;
    const uint32_t start_B = B_dims_len - 3;

    if (A_dims_len > B_dims_len) {
        for (uint32_t i = 0; i < B_dims_len - 2; i++) {
            if (args->A_dims[start_A - i] > args->B_dims[start_B - i]) {
                out_dims[max_dims_len - i - 1] = args->A_dims[start_A - i];
            } else {
                out_dims[max_dims_len - i - 1] = args->B_dims[start_B - i];
            }

            total_matrices *= out_dims[max_dims_len - i - 1];
        }

        for (uint32_t i = 0; i < A_dims_len - B_dims_len; i++) {
            out_dims[i] = args->A_dims[i];
            total_matrices *= out_dims[i];
        }
    } else {
        for (uint32_t i = 0; i < A_dims_len - 2; i++) {
            if (args->A_dims[start_A - i] > args->B_dims[start_B - i]) {
                out_dims[max_dims_len - i - 1] = args->A_dims[start_A - i];
            } else {
                out_dims[max_dims_len - i - 1] = args->B_dims[start_B - i];
            }

            total_matrices *= out_dims[max_dims_len - i - 1];
        }

        for (uint32_t i = 0; i < B_dims_len - A_dims_len; i++) {
            out_dims[i] = args->B_dims[i];
            total_matrices *= out_dims[i];
        }
    }

    // Prepare look-up tables for new indexes computation
    // A
    int prod_A[A_dims_len - 2];
    int prod_so_far = A_dims[A_dims_len - 2] * A_dims[A_dims_len - 1];

    if (A_dims[A_dims_len - 3] == 1)
        prod_A[A_dims_len - 3] = 0;
    else
        prod_A[A_dims_len - 3] = prod_so_far;

    for (int i = A_dims_len - 4; i >= 0; i--) {
        prod_so_far *= A_dims[i + 1];

        if (A_dims[i] == 1)
            prod_A[i] = 0;
        else
            prod_A[i] = prod_so_far;
    }

    // B
    int prod_B[B_dims_len - 2];
    prod_so_far = B_dims[B_dims_len - 2] * B_dims[B_dims_len - 1];

    if (B_dims[B_dims_len - 3] == 1)
        prod_B[B_dims_len - 3] = 0;
    else
        prod_B[B_dims_len - 3] = prod_so_far;

    for (int i = B_dims_len - 4; i >= 0; i--) {
        prod_so_far *= B_dims[i + 1];

        if (B_dims[i] == 1)
            prod_B[i] = 0;
        else
            prod_B[i] = prod_so_far;
    }

    // Iterate through matrices and compute MatMul result
    for (uint32_t i = 0; i < total_matrices; i++) {
        // Compute current starting matrix indices
        uint32_t idx_A = 0;
        uint32_t idx_B = 0;

        uint32_t idx = i;

        if (A_dims_len < B_dims_len) {
            for (uint32_t j = 0; j < A_dims_len - 2; j++) {
                idx_A += (idx % out_dims[max_dims_len - j - 1]) * prod_A[A_dims_len - j - 3];
                idx_B += (idx % out_dims[max_dims_len - j - 1]) * prod_B[B_dims_len - j - 3];

                idx /= out_dims[max_dims_len - j - 1];
            }
        } else {
            for (uint32_t j = 0; j < B_dims_len - 2; j++) {
                idx_A += (idx % out_dims[max_dims_len - j - 1]) * prod_A[A_dims_len - j - 3];
                idx_B += (idx % out_dims[max_dims_len - j - 1]) * prod_B[B_dims_len - j - 3];

                idx /= out_dims[max_dims_len - j - 1];
            }
        }

        if (A_dims_len < B_dims_len) {
            for (uint32_t j = A_dims_len - 2; j < B_dims_len - 2; j++) {
                idx_B += (idx % out_dims[max_dims_len - j - 1]) * prod_B[B_dims_len - j - 3];
                idx /= out_dims[max_dims_len - j - 1];
            }
        } else {
            for (uint32_t j = B_dims_len - 2; j < A_dims_len - 2; j++) {
                idx_A += (idx % out_dims[max_dims_len - j - 1]) * prod_A[A_dims_len - j - 3];
                idx /= out_dims[max_dims_len - j - 1];
            }
        }

        const uint32_t idx_C = i * A_dims[A_dims_len - 2] * B_dims[B_dims_len - 1];

        // Compute MatMul
        struct matMul_args_fp16 current_matMul_args;

        current_matMul_args.A = A + idx_A;
        current_matMul_args.B = B + idx_B;
        current_matMul_args.C = C + idx_C;

        current_matMul_args.N = A_dims[A_dims_len - 2];
        current_matMul_args.K = A_dims[A_dims_len - 1];
        current_matMul_args.M = B_dims[B_dims_len - 1];

        current_matMul_args.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &current_matMul_args);
        #else
        struct mm_manager_args_fp16 man_current_matMul_args;

        man_current_matMul_args.mm_args = &current_matMul_args;

        man_current_matMul_args.layer_type = LAYER_LINEAR;
        man_current_matMul_args.step_type = STEP_FW;

        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_current_matMul_args);
        #endif
    }
}


void mm_fp16(void * void_args) {
    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;

    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    const uint32_t N = args->N;
    const uint32_t M = args->M;
    const uint32_t K = args->K;

    uint32_t transp = args->trans_B;

    const uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > N ? N : start + blockSize;

    #ifdef DEBUG
    // Output tracking
    printf("mm_fp16 OUTPUT DATA (size=%d, addr=0x%x):\n", N, (unsigned int)&C);
    for (int i=0; i<N; i++) {
        printf("%f ", C[i]);
    } printf("\n");
    #endif

    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) {
        if (K == 1) {
            for (uint32_t i = start; i < stop; i++) {
                for (uint32_t j = 0; j < M; j++) {
                    C[i * M + j] = A[i * K] * B[j];
#ifdef DEBUG
                    printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K, j, C[i*M+j], A[i*K], B[j]);
#endif
                }
            }
        } else if (K > 0) {
            for (uint32_t i = start; i < stop; i++) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        temp += A[i * K + k] * B[j + k * M];
#ifdef DEBUG
                        printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, j+k*M, temp, A[i*K+k], B[j+k*M]);
#endif
                    }
                    C[i * M + j] = temp;
                }
            }
        }
    }

        // =====> B IS TRANSPOSED <=====
    else {
        if (K == 1) {
            for (uint32_t i = start; i < stop; i++) {
                for (uint32_t j = 0; j < M; j++) {
                    C[i * M + j] = A[i * K] * B[j * K];
#ifdef DEBUG
                    printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K, j*K, C[i*M+j], A[i*K], B[j*K]);
#endif
                }
            }
        } else if (K > 0) {
            for (uint32_t i = start; i < stop; i++) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        temp += A[i * K + k] * B[k + j * K];
#ifdef DEBUG
                        printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
#endif
                    }
                    C[i * M + j] = temp;
                }
            }
        }
    }
}


// Naive matmul with parallelism on M
void mm_M_fp16(void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    const uint32_t N = args->N;
    const uint32_t M = args->M;
    const uint32_t K = args->K;

    uint32_t transp = args->trans_B;

    const uint32_t blockSize = (M + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > M ? M : start + blockSize;

    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) {
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = start; j < stop; j++) {
                fp16 temp = 0;
                for (uint32_t k = 0; k < K; k++) {
                    temp += A[i * K + k] * B[j + k * M];
#ifdef DEBUG
                    printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
#endif
                }
                C[i * M + j] = temp;
            }
        }
    }

        // =====> B IS TRANSPOSED <=====
    else {
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t j = start; j < stop; j++) {
                fp16 temp = 0;
                for (uint32_t k = 0; k < K; k++) {
                    temp += A[i * K + k] * B[j * K + k];
#ifdef DEBUG
                    printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
#endif
                }
                C[i * M + j] = temp;
            }
        }
    }
}


/**
 * Optimized versions
 */
void __attribute__((noinline)) mm_fp16_SIMD_2x4 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;
    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t transp = args->trans_B;

    uint32_t indexA, indexB;
    v2f16 Av;
    v2f16 Bv0, Bv1;
    v2f16 *Cv;

    // Optimized looping variables
    uint32_t M_loop = (M & 0xfffffffe);
    uint32_t K_loop = (K & 0xfffffffe);

    if (M < 2) mm_fp16(args);
    else if (K < 2) mm_fp16(args);
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
#if NUM_CORES > 1
            const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = pi_core_id()*blockSize;
            const uint32_t stop = start+blockSize < N? start+blockSize: N;

            for (uint32_t i = start; i < stop; i++) {

#else
            const uint32_t start = 0;
            const uint32_t stop = N;

            for (uint32_t i = start; i < stop; i++) {
#endif

                for (uint32_t j = 0; j < M_loop; j += 2) {
                    v2f16 temp = (v2f16) {0, 0};
                    indexA = i * K;
                    indexB = j;

                    for (uint32_t k = 0; k < K_loop; k += 2) {
                        Av = *((v2f16 * ) & A[indexA/*i*K+k*/]);
                        Bv0 = *((v2f16 * ) & B[indexB/*k*M+j*/]);
                        Bv1 = *((v2f16 * ) & B[indexB + M/*k*M+j+M*/]);
                        temp += (v2f16)(__builtin_shuffle(Av, (v2s) {0, 0})) * Bv0;
                        temp += (v2f16)(__builtin_shuffle(Av, (v2s) {1, 1})) * Bv1;

                        indexA += 2;
                        indexB += 2 * M;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av = (v2f16) {A[i * K + (K - 1)], A[i * K + (K - 1)]};
                        Bv0 = *((v2f16 * ) & B[(K - 1) * M + j]);
                        temp += Av * Bv0;
                    }
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;
                }
            }
            // Leftover on M
            if (M & 1) {
                for (uint32_t i = start; i < stop; i++) {
                    fp16 val = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        val += A[i * K + k] * B[k * M + (M - 1)];
                    }
                    C[i * M + (M - 1)] = val;
                }
            }

        }


            // =====> B IS TRANSPOSED <=====
        else {
#if NUM_CORES > 1
            const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = pi_core_id()*blockSize;
            const uint32_t stop = start+blockSize < N? start+blockSize: N;

            for (uint32_t i = start; i < stop; i++) {

#else
            const uint32_t start = 0;
            const uint32_t stop = N;

            for (uint32_t i = start; i < stop; i++) {
#endif

                for (uint32_t j = 0; j < M_loop; j += 2) {
                    // Global accumulator
                    v2f16 temp = (v2f16) {0, 0};
                    // Dot product accumulators
                    v2f16 tmp0 = (v2f16) {0, 0};
                    v2f16 tmp1 = (v2f16) {0, 0};
                    // K leftover accumulator
                    v2f16 vtmp = (v2f16) {0, 0};
                    // Scalar accumulators for final result
                    fp16 a = 0;
                    fp16 b = 0;
                    // Indices
                    indexA = i * K;
                    indexB = j * K; //j*M;

                    for (uint32_t k = 0; k < K_loop; k += 2) {
                        Av = *((v2f16 * ) & A[indexA/*i*K+k*/]);

                        Bv0 = *((v2f16 * ) & B[indexB]);
                        Bv1 = *((v2f16 * ) & B[indexB + K]);
                        tmp0 += (v2f16)(Av * Bv0);
                        tmp1 += (v2f16)(Av * Bv1);

                        indexA += 2;
                        indexB += 2;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av = (v2f16) {A[indexA], A[indexA]};
                        Bv0 = (v2f16) {B[indexB], B[indexB + K]};
#ifdef DEBUG
                        printf("------ MM K left (indexA=%d, indexB=%d) => Av = {0x%x, 0x%x}, Bv0 = {0x%x, 0x%x}\n", indexA, indexB,
                          ((unsigned int) Av)&(0xffff0000)>>16, ((unsigned int) Av)&(0x0000ffff),
                          ((unsigned int) Bv0)&(0xffff0000)>>16, ((unsigned int) Bv0)&(0x0000ffff));
#endif
                        vtmp += (v2f16)(Av * Bv0);
                        a += vtmp[0];
                        b += vtmp[1];
#ifdef DEBUG
                        printf("------ MM K left => temp = {0x%x, 0x%x}\n", ((unsigned int) vtmp[0]), ((unsigned int) vtmp[1]));
#endif
                    }
                    // Complete dot product
                    a += tmp0[0] + tmp0[1];
                    b += tmp1[0] + tmp1[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;
                }
            }

            // Leftover on N
            if (M & 1) {
                for (uint32_t i = start; i < stop; i++) {
                    fp16 val = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        val += A[i * K + k] * B[(M - 1) * K + k];
                    }
                    C[i * M + (M - 1)] = val;
                }
            }
        }
    }
}


void __attribute__((noinline)) mm_fp16_SIMD_4x8 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t transp = args->trans_B;

    uint32_t indexA0, indexA1;
    uint32_t indexB;
    v2f16 Av0, Av1;
    v2f16 Bv0, Bv1, Bv2, Bv3;
    v2f16 *Cv;

    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N ? N : start + blockSize;

    blockSize = stop - start;
    uint32_t blockSize_par = blockSize & 0xfffffffe;
    uint32_t blockSize_left = blockSize - blockSize_par;

    // Looping variables for leftovers (TO REMOVE)
    uint32_t N_loop = N & 0xfffffffe;
    uint32_t N_left = N - N_loop;

    // Integrity barrier for oversized unrolling
    uint32_t N_bound = (N_loop) / NUM_CORES;
    uint32_t M_bound = (M - (M & 0x00000003));
    uint32_t K_bound = (K - (K & 0x00000001));

    if (M_bound < 4) mm_fp16_SIMD_2x4(args);
    else if (K_bound < 2) mm_fp16_SIMD_2x4(args);
    else if (N_bound < 2) mm_fp16_SIMD_2x4(args);
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
            uint32_t i;
            for (i = start; i < stop - 1; i += 2) {
                for (uint32_t j = 0; j < (M & 0xfffffffc); j += 4) {
                    v2f16 temp0 = (v2f16) {0, 0};
                    v2f16 temp1 = (v2f16) {0, 0};
                    v2f16 temp2 = (v2f16) {0, 0};
                    v2f16 temp3 = (v2f16) {0, 0};

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        Av1 = *(v2f16 * ) & A[(i + 1) * K + k];
                        // B vectors
                        Bv0 = *(v2f16 * ) & B[k * M + j];
                        Bv1 = *(v2f16 * ) & B[(k + 1) * M + j];
                        Bv2 = *(v2f16 * ) & B[k * M + j + 2];
                        Bv3 = *(v2f16 * ) & B[(k + 1) * M + j + 2];

                        // Ci,j, Ci,j+1
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv0;
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv1;
                        // Ci,j+2, Ci,j+3
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv2;
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv3;
                        // Ci+1,j, Ci+1,j+1
                        temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s) {0, 0})) * Bv0;
                        temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s) {1, 1})) * Bv1;
                        // Ci+1,j+2, Ci+1,j+3
                        temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s) {0, 0})) * Bv2;
                        temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s) {1, 1})) * Bv3;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av0 = (v2f16) {A[i * K + (K - 1)], A[i * K + (K - 1)]};
                        Av1 = (v2f16) {A[(i + 1) * K + (K - 1)], A[(i + 1) * K + (K - 1)]};
                        Bv0 = *((v2f16 * ) & B[(K - 1) * M + j]);
                        Bv1 = *((v2f16 * ) & B[(K - 1) * M + j + 2]);
                        temp0 += Av0 * Bv0;
                        temp1 += Av0 * Bv1;
                        temp2 += Av1 * Bv0;
                        temp3 += Av1 * Bv1;
                    }
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp0;

                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp1;

                    Cv = (v2f16 * ) & C[(i + 1) * M + j];
                    *Cv = temp2;

                    Cv = (v2f16 * ) & C[(i + 1) * M + j + 2];
                    *Cv = temp3;
                }
                // Leftover in M
                if (M & 0x00000003) {
                    for (uint32_t ii = i; ii < i + 2; ii++) {
                        for (uint32_t j = (M - (M & 0x00000003)); j < M; j++) {
                            fp16 left_temp = 0;
                            for (uint32_t k = 0; k < K; k++) {
                                left_temp += A[ii * K + k] * B[k * M + j];
                            }
                            C[ii * M + j] = left_temp;
                        }
                    }
                }
            }
            // Leftover in block
            if (blockSize_left > 0) {
                for (uint32_t j = 0; j < (M & 0xfffffffc); j += 4) {
                    v2f16 temp0 = (v2f16) {0, 0};
                    v2f16 temp1 = (v2f16) {0, 0};

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        // B vectors
                        Bv0 = *(v2f16 * ) & B[k * M + j];
                        Bv1 = *(v2f16 * ) & B[(k + 1) * M + j];
                        Bv2 = *(v2f16 * ) & B[k * M + j + 2];
                        Bv3 = *(v2f16 * ) & B[(k + 1) * M + j + 2];

                        // Ci,j, Ci,j+1
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv0;
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv1;
                        // Ci,j+2, Ci,j+3
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv2;
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv3;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av0 = (v2f16) {A[i * K + (K - 1)], A[i * K + (K - 1)]};
                        Bv0 = *((v2f16 * ) & B[(K - 1) * M + j]);
                        Bv1 = *((v2f16 * ) & B[(K - 1) * M + j + 2]);
                        temp0 += Av0 * Bv0;
                        temp1 += Av0 * Bv1;
                    }
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp0;

                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp1;
                }
                // Leftover in M
                if (M & 0x00000003) {
                    for (uint32_t j = (M - (M & 0x00000003)); j < M; j++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[i * K + k] * B[k * M + j];
                        }
                        C[i * M + j] = left_temp;
                    }
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
            uint32_t i;
            for (i = start; i < stop - 1; i += 2) {
                for (uint32_t j = 0; j < (M & 0xfffffffc); j += 4) {
                    // Global accumulator
                    v2f16 temp = (v2f16) {0, 0};
                    // Dot product accumulators
                    v2f16 tmp0 = (v2f16) {0, 0};
                    v2f16 tmp1 = (v2f16) {0, 0};
                    v2f16 tmp2 = (v2f16) {0, 0};
                    v2f16 tmp3 = (v2f16) {0, 0};
                    v2f16 tmp4 = (v2f16) {0, 0};
                    v2f16 tmp5 = (v2f16) {0, 0};
                    v2f16 tmp6 = (v2f16) {0, 0};
                    v2f16 tmp7 = (v2f16) {0, 0};
                    // Scalar accumulators
                    fp16 a = 0;
                    fp16 b = 0;

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        Av1 = *(v2f16 * ) & A[(i + 1) * K + k];
                        // B vectors (transposed matrix)
                        Bv0 = *(v2f16 * ) & B[j * K + k];
                        Bv1 = *(v2f16 * ) & B[(j + 1) * K + k];
                        Bv2 = *(v2f16 * ) & B[(j + 2) * K + k];
                        Bv3 = *(v2f16 * ) & B[(j + 3) * K + k];

                        // Products in Ci,j and successive with Av0
                        tmp0 += Av0 * Bv0;
                        tmp1 += Av0 * Bv1;
                        tmp2 += Av0 * Bv2;
                        tmp3 += Av0 * Bv3;
                        // Products in Ci+1,j and successive with Av1
                        tmp4 += Av1 * Bv0;
                        tmp5 += Av1 * Bv1;
                        tmp6 += Av1 * Bv2;
                        tmp7 += Av1 * Bv3;
                    }
                    // Leftover on K
                    if (K & 1) {
                        // A elements
                        fp16 A0 = A[i * K + (K - 1)];
                        fp16 A1 = A[(i + 1) * K + (K - 1)];
                        // B elements (transposed matrix)
                        fp16 B0 = B[j * K + (K - 1)];
                        fp16 B1 = B[(j + 1) * K + (K - 1)];
                        fp16 B2 = B[(j + 2) * K + (K - 1)];
                        fp16 B3 = B[(j + 3) * K + (K - 1)];

                        // Products in Ci,j and successive with Av0
                        tmp0[0] += A0 * B0;
                        tmp1[0] += A0 * B1;
                        tmp2[0] += A0 * B2;
                        tmp3[0] += A0 * B3;
                        // Products in Ci+1,j and successive with Av1
                        tmp4[0] += A1 * B0;
                        tmp5[0] += A1 * B1;
                        tmp6[0] += A1 * B2;
                        tmp7[0] += A1 * B3;
                    }
                    // Accumulate to compute dot product and store
                    // Row 1
                    a = tmp0[0] + tmp0[1];
                    b = tmp1[0] + tmp1[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;

                    a = tmp2[0] + tmp2[1];
                    b = tmp3[0] + tmp3[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp;

                    // Row 2
                    a = tmp4[0] + tmp4[1];
                    b = tmp5[0] + tmp5[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[(i + 1) * M + j];
                    *Cv = temp;

                    a = tmp6[0] + tmp6[1];
                    b = tmp7[0] + tmp7[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[(i + 1) * M + j + 2];
                    *Cv = temp;
                }
                // Leftover in M
                if (M & 0x00000003) {
                    for (uint32_t ii = i; ii < i + 2; ii++) {
                        for (uint32_t j = (M - (M & 0x00000003)); j < M; j++) {
                            fp16 left_temp = 0;
                            for (uint32_t k = 0; k < K; k++) {
                                left_temp += A[ii * K + k] * B[j * K + k];
                            }
                            C[ii * M + j] = left_temp;
                        }
                    }
                }
            }
            // Leftover in block
            if (blockSize_left > 0) {
                for (uint32_t j = 0; j < (M & 0xfffffffc); j += 4) {
                    // Global accumulator
                    v2f16 temp = (v2f16) {0, 0};
                    // Dot product accumulators
                    v2f16 tmp0 = (v2f16) {0, 0};
                    v2f16 tmp1 = (v2f16) {0, 0};
                    v2f16 tmp2 = (v2f16) {0, 0};
                    v2f16 tmp3 = (v2f16) {0, 0};
                    // Scalar accumulators
                    fp16 a = 0;
                    fp16 b = 0;

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        // B vectors (transposed matrix)
                        Bv0 = *(v2f16 * ) & B[j * K + k];
                        Bv1 = *(v2f16 * ) & B[(j + 1) * K + k];
                        Bv2 = *(v2f16 * ) & B[(j + 2) * K + k];
                        Bv3 = *(v2f16 * ) & B[(j + 3) * K + k];

                        // Products in Ci,j and successive with Av0
                        tmp0 += Av0 * Bv0;
                        tmp1 += Av0 * Bv1;
                        tmp2 += Av0 * Bv2;
                        tmp3 += Av0 * Bv3;
                    }
                    // Leftover on K
                    if (K & 1) {
                        // A elements
                        fp16 A0 = A[i * K + (K - 1)];
                        // B elements (transposed matrix)
                        fp16 B0 = B[j * K + (K - 1)];
                        fp16 B1 = B[(j + 1) * K + (K - 1)];
                        fp16 B2 = B[(j + 2) * K + (K - 1)];
                        fp16 B3 = B[(j + 3) * K + (K - 1)];

                        // Products in Ci,j and successive with Av0
                        tmp0[0] += A0 * B0;
                        tmp1[0] += A0 * B1;
                        tmp2[0] += A0 * B2;
                        tmp3[0] += A0 * B3;
                    }
                    // Accumulate to compute dot product and store
                    // Row 1
                    a = tmp0[0] + tmp0[1];
                    b = tmp1[0] + tmp1[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;

                    a = tmp2[0] + tmp2[1];
                    b = tmp3[0] + tmp3[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp;
                }
                // Leftover in M
                if (M & 0x00000003) {
                    for (uint32_t j = (M - (M & 0x00000003)); j < M; j++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[i * K + k] * B[j * K + k];
                        }
                        C[i * M + j] = left_temp;
                    }
                }
            }
        }
    }
}


void mm_fp16_unroll_8x1 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;

    uint32_t transp = args->trans_B;
    uint32_t N_par = N & 0xfffffff8;
    uint32_t N_left = N - N_par;
    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N_par + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N_par ? N_par : start + blockSize;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if ((N_par / NUM_CORES) < 8) { mm_fp16_unroll_4x1(args); }
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
            for (uint32_t i = start; i < stop; i = i + 8) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;
                    fp16 temp2 = 0;
                    fp16 temp3 = 0;
                    fp16 temp4 = 0;
                    fp16 temp5 = 0;
                    fp16 temp6 = 0;
                    fp16 temp7 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k * M + j];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                        temp2 += A[idx + 2 * K] * Bsh;
                        temp3 += A[idx + 3 * K] * Bsh;
                        temp4 += A[idx + 4 * K] * Bsh;
                        temp5 += A[idx + 5 * K] * Bsh;
                        temp6 += A[idx + 6 * K] * Bsh;
                        temp7 += A[idx + 7 * K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                    C[(i + 2) * M + j] = temp2;
                    C[(i + 3) * M + j] = temp3;
                    C[(i + 4) * M + j] = temp4;
                    C[(i + 5) * M + j] = temp5;
                    C[(i + 6) * M + j] = temp6;
                    C[(i + 7) * M + j] = temp7;
                }
            }
            // Leftover on N (parallel on M)
            if (N_left > 0) {
                uint32_t j_block = (M + NUM_CORES - 1) / NUM_CORES;
                uint32_t j_start = core_id * j_block;
                uint32_t j_stop = j_start + j_block > M ? M : j_start + j_block;

                for (uint32_t j = j_start; j < j_stop; j++) {
                    for (uint32_t i = (N - N_left); i < N; i++) {
                        fp16 temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            temp += A[i * K + k] * B[k * M + j];
                        }
                        C[i * M + j] = temp;
                    }
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
            for (uint32_t i = start; i < stop; i = i + 8) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;
                    fp16 temp2 = 0;
                    fp16 temp3 = 0;
                    fp16 temp4 = 0;
                    fp16 temp5 = 0;
                    fp16 temp6 = 0;
                    fp16 temp7 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k + j * K];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                        temp2 += A[idx + 2 * K] * Bsh;
                        temp3 += A[idx + 3 * K] * Bsh;
                        temp4 += A[idx + 4 * K] * Bsh;
                        temp5 += A[idx + 5 * K] * Bsh;
                        temp6 += A[idx + 6 * K] * Bsh;
                        temp7 += A[idx + 7 * K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                    C[(i + 2) * M + j] = temp2;
                    C[(i + 3) * M + j] = temp3;
                    C[(i + 4) * M + j] = temp4;
                    C[(i + 5) * M + j] = temp5;
                    C[(i + 6) * M + j] = temp6;
                    C[(i + 7) * M + j] = temp7;
                }
            }
            // Leftover on N (parallel on M)
            if (N_left > 0) {
                uint32_t j_block = (M + NUM_CORES - 1) / NUM_CORES;
                uint32_t j_start = core_id * j_block;
                uint32_t j_stop = j_start + j_block > M ? M : j_start + j_block;

                for (uint32_t j = j_start; j < j_stop; j++) {
                    for (uint32_t i = (N - N_left); i < N; i++) {
                        fp16 temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            temp += A[i * K + k] * B[k + j * K];
                        }
                        C[i * M + j] = temp;
                    }
                }
            }
        }
    }
}


void mm_fp16_unroll_4x1 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;

    uint32_t transp = args->trans_B;
    uint32_t N_par = N & 0xfffffffc;
    uint32_t N_left = N - N_par;
    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N_par + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N_par ? N_par : start + blockSize;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if ((N_par / NUM_CORES) < 4) { mm_fp16_unroll_2x1(args); }
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
            for (uint32_t i = start; i < stop; i = i + 4) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;
                    fp16 temp2 = 0;
                    fp16 temp3 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k * M + j];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                        temp2 += A[idx + 2 * K] * Bsh;
                        temp3 += A[idx + 3 * K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                    C[(i + 2) * M + j] = temp2;
                    C[(i + 3) * M + j] = temp3;
                }
            }
            // Leftover on N (parallel on M)
            if (N_left > 0) {
                uint32_t j_block = (M + NUM_CORES - 1) / NUM_CORES;
                uint32_t j_start = core_id * j_block;
                uint32_t j_stop = j_start + j_block > M ? M : j_start + j_block;

                for (uint32_t j = j_start; j < j_stop; j++) {
                    for (uint32_t i = (N - N_left); i < N; i++) {
                        fp16 temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            temp += A[i * K + k] * B[k * M + j];
                        }
                        C[i * M + j] = temp;
                    }
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
            for (uint32_t i = start; i < stop; i = i + 4) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;
                    fp16 temp2 = 0;
                    fp16 temp3 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k + j * K];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                        temp2 += A[idx + 2 * K] * Bsh;
                        temp3 += A[idx + 3 * K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                    C[(i + 2) * M + j] = temp2;
                    C[(i + 3) * M + j] = temp3;
                }
            }
            // Leftover on N (parallel on M)
            if (N_left > 0) {
                uint32_t j_block = (M + NUM_CORES - 1) / NUM_CORES;
                uint32_t j_start = core_id * j_block;
                uint32_t j_stop = j_start + j_block > M ? M : j_start + j_block;

                for (uint32_t j = j_start; j < j_stop; j++) {
                    for (uint32_t i = (N - N_left); i < N; i++) {
                        fp16 temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            temp += A[i * K + k] * B[k + j * K];
                        }
                        C[i * M + j] = temp;
                    }
                }
            }
        }
    }
}


void mm_fp16_unroll_2x1 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t transp = args->trans_B;

    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N ? N : start + blockSize;

    blockSize = stop - start;
    uint32_t blockSize_par = blockSize & 0xfffffffe;
    uint32_t blockSize_left = blockSize - blockSize_par;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if ((N / NUM_CORES) < 2) { mm_fp16(args); }
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
            uint32_t i;
            for (i = start; i < stop - 1; i = i + 2) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k * M + j];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                }
            }
            // Leftover in block
            if (blockSize_left > 0) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k * M + j];
                        temp0 += A[idx] * Bsh;
                    }
                    C[i * M + j] = temp0;
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
            uint32_t i;
            for (i = start; i < stop - 1; i = i + 2) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;
                    fp16 temp1 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k + j * K];
                        temp0 += A[idx] * Bsh;
                        temp1 += A[idx + K] * Bsh;
                    }
                    C[i * M + j] = temp0;
                    C[(i + 1) * M + j] = temp1;
                }
            }
            // Leftover in block
            if (blockSize_left > 0) {
                for (uint32_t j = 0; j < M; j++) {
                    fp16 temp0 = 0;

                    for (uint32_t k = 0; k < K; k++) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k + j * K];
                        temp0 += A[idx] * Bsh;
                    }
                    C[i * M + j] = temp0;
                }
            }
        }
    }
}


void mv_fp16_SIMD_2x1 (void * void_args) {
    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;

    if (M > 1) {
        uint32_t transp = args->trans_B;
        if (transp) {
            K = M;
        } else {
            printf("\nError: second tensor is not a vector.\n");
            exit(1);
        }
    }

    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N ? N : start + blockSize;

    blockSize = stop - start;
    uint32_t blockSize_par = blockSize & 0xfffffffe;
    uint32_t blockSize_left = blockSize - blockSize_par;
    uint32_t K_par = K & 0xfffffffe;
    uint32_t k_left = K - K_par;
    uint32_t k = 0;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if (blockSize < 2) { mm_fp16(args); }
    else {
        uint32_t i;
        for (i = start; i < stop - 1; i += 2) {
            // Temporary sum buffers
            v2f16 temp0 = (v2f16) {0, 0};
            v2f16 temp1 = (v2f16) {0, 0};

            // Cycle on middle dimension
            for (k = 0; k < K - 1; k += 2) {
                uint32_t idx = i * K + k;
                v2f16 Bv = *((v2f16 * ) & B[k]);
                v2f16 Av1 = *((v2f16 * ) & A[idx]);
                v2f16 Av2 = *((v2f16 * ) & A[idx + K]);
                temp0 += Av1 * Bv;
                temp1 += Av2 * Bv;
            }
            // Leftover in K
            if (k_left > 0) {
                while (k < K) {
                    uint32_t idx = i * K + k;
                    fp16 Bsh = B[k];
                    fp16 A1 = A[idx];
                    fp16 A2 = A[idx + K];
                    temp0[0] += A1 * Bsh;
                    temp1[0] += A2 * Bsh;
                    k++;
                }
            }
            // Sum the buffers to store result
            C[i] = temp0[0] + temp0[1];
            C[i + 1] = temp1[0] + temp1[1];
        }
        // Leftover in block
        if (blockSize_left > 0) {
            v2f16 temp0 = (v2f16) {0, 0};
            for (k = 0; k < K; k += 2) {
                uint32_t idx = i * K + k;
                v2f16 Bv = *((v2f16 * ) & B[k]);
                v2f16 Av = *((v2f16 * ) & A[idx]);
                temp0 += Av * Bv;
            }
            // Leftover in K
            if (k_left > 0) {
                while (k < K) {
                    uint32_t idx = i * K + k;
                    fp16 Bsh = B[k];
                    fp16 A1 = A[idx];
                    temp0[0] += A1 * Bsh;
                    k++;
                }
            }
            C[i] = temp0[0] + temp0[1];
        }
    }
}


void mv_fp16_SIMD_4x1 (void * void_args) {
    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;

    if (M > 1) {
        uint32_t transp = args->trans_B;
        if (transp) {
            K = M;
        } else {
            printf("\nError: second tensor is not a vector.\n");
            exit(1);
        }
    }

    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N ? N : start + blockSize;

    blockSize = stop - start;
    uint32_t blockSize_par = blockSize & 0xfffffffc;
    uint32_t blockSize_left = blockSize - blockSize_par;
    uint32_t K_par = K & 0xfffffffe;
    uint32_t k_left = K - K_par;
    uint32_t k = 0;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if (blockSize < 4) { mv_fp16_SIMD_2x1(args); }
    else {
        uint32_t i;
        for (i = start; i < stop - 1; i += 4) {
            // Temporary sum buffers
            v2f16 temp0 = (v2f16) {0, 0};
            v2f16 temp1 = (v2f16) {0, 0};
            v2f16 temp2 = (v2f16) {0, 0};
            v2f16 temp3 = (v2f16) {0, 0};

            // Cycle on middle dimension
            for (k = 0; k < K - 1; k += 2) {
                uint32_t idx = i * K + k;
                v2f16 Bv = *((v2f16 * ) & B[k]);
                v2f16 Av1 = *((v2f16 * ) & A[idx]);
                v2f16 Av2 = *((v2f16 * ) & A[idx + K]);
                v2f16 Av3 = *((v2f16 * ) & A[idx + 2 * K]);
                v2f16 Av4 = *((v2f16 * ) & A[idx + 3 * K]);
                temp0 += Av1 * Bv;
                temp1 += Av2 * Bv;
                temp2 += Av3 * Bv;
                temp3 += Av4 * Bv;
            }
            // Leftover in K
            if (k_left > 0) {
                while (k < K) {
                    uint32_t idx = i * K + k;
                    fp16 Bsh = B[k];
                    fp16 A1 = A[idx];
                    fp16 A2 = A[idx + K];
                    fp16 A3 = A[idx + 2 * K];
                    fp16 A4 = A[idx + 3 * K];
                    temp0[0] += A1 * Bsh;
                    temp1[0] += A2 * Bsh;
                    temp2[0] += A3 * Bsh;
                    temp3[0] += A4 * Bsh;
                    k++;
                }
            }
            // Sum the buffers to store result
            C[i] = temp0[0] + temp0[1];
            C[i + 1] = temp1[0] + temp1[1];
            C[i + 2] = temp2[0] + temp2[1];
            C[i + 3] = temp3[0] + temp3[1];
        }
        // Leftover in block
        if (blockSize_left > 0) {
            while (i < stop) {
                v2f16 temp0 = (v2f16) {0, 0};
                for (k = 0; k < K; k += 2) {
                    uint32_t idx = i * K + k;
                    v2f16 Bv = *((v2f16 * ) & B[k]);
                    v2f16 Av = *((v2f16 * ) & A[idx]);
                    temp0 += Av * Bv;
                }
                // Leftover in K
                if (k_left > 0) {
                    while (k < K) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k];
                        fp16 A1 = A[idx];
                        temp0[0] += A1 * Bsh;
                        k++;
                    }
                }
                C[i] = temp0[0] + temp0[1];
                i++;
            }
        }
    }
}


void mv_fp16_SIMD_8x1 (void * void_args) {
    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;

    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;

    if (M > 1) {
        uint32_t transp = args->trans_B;
        if (transp) {
            K = M;
        } else {
            printf("\nError: second tensor is not a vector.\n");
            exit(1);
        }
    }

    uint32_t core_id = pi_core_id();

    uint32_t blockSize = (N + NUM_CORES - 1) / NUM_CORES;
    uint32_t start = core_id * blockSize;
    uint32_t stop = start + blockSize > N ? N : start + blockSize;

    blockSize = stop - start;
    uint32_t blockSize_par = blockSize & 0xfffffff8;
    uint32_t blockSize_left = blockSize - blockSize_par;
    uint32_t K_par = K & 0xfffffffe;
    uint32_t k_left = K - K_par;
    uint32_t k = 0;

    // Check if sizes are smaller than the unrolling, and take countermeasures
    if (blockSize < 8) { mv_fp16_SIMD_4x1(args); }
    else {
        uint32_t i;
        for (i = start; i < stop - 1; i += 8) {
            // Temporary sum buffers
            v2f16 temp0 = (v2f16) {0, 0};
            v2f16 temp1 = (v2f16) {0, 0};
            v2f16 temp2 = (v2f16) {0, 0};
            v2f16 temp3 = (v2f16) {0, 0};
            v2f16 temp4 = (v2f16) {0, 0};
            v2f16 temp5 = (v2f16) {0, 0};
            v2f16 temp6 = (v2f16) {0, 0};
            v2f16 temp7 = (v2f16) {0, 0};

            // Cycle on middle dimension
            for (k = 0; k < K - 1; k += 2) {
                uint32_t idx = i * K + k;
                v2f16 Bv = *((v2f16 * ) & B[k]);
                v2f16 Av1 = *((v2f16 * ) & A[idx]);
                v2f16 Av2 = *((v2f16 * ) & A[idx + K]);
                v2f16 Av3 = *((v2f16 * ) & A[idx + 2 * K]);
                v2f16 Av4 = *((v2f16 * ) & A[idx + 3 * K]);
                v2f16 Av5 = *((v2f16 * ) & A[idx + 4 * K]);
                v2f16 Av6 = *((v2f16 * ) & A[idx + 5 * K]);
                v2f16 Av7 = *((v2f16 * ) & A[idx + 6 * K]);
                v2f16 Av8 = *((v2f16 * ) & A[idx + 7 * K]);
                temp0 += Av1 * Bv;
                temp1 += Av2 * Bv;
                temp2 += Av3 * Bv;
                temp3 += Av4 * Bv;
                temp4 += Av5 * Bv;
                temp5 += Av6 * Bv;
                temp6 += Av7 * Bv;
                temp7 += Av8 * Bv;
            }
            // Leftover in K
            if (k_left > 0) {
                while (k < K) {
                    uint32_t idx = i * K + k;
                    fp16 Bsh = B[k];
                    fp16 A1 = A[idx];
                    fp16 A2 = A[idx + K];
                    fp16 A3 = A[idx + 2 * K];
                    fp16 A4 = A[idx + 3 * K];
                    fp16 A5 = A[idx + 4 * K];
                    fp16 A6 = A[idx + 5 * K];
                    fp16 A7 = A[idx + 6 * K];
                    fp16 A8 = A[idx + 7 * K];
                    temp0[0] += A1 * Bsh;
                    temp1[0] += A2 * Bsh;
                    temp2[0] += A3 * Bsh;
                    temp3[0] += A4 * Bsh;
                    temp4[0] += A5 * Bsh;
                    temp5[0] += A6 * Bsh;
                    temp6[0] += A7 * Bsh;
                    temp7[0] += A8 * Bsh;
                    k++;
                }
            }
            // Sum the buffers to store result
            C[i] = temp0[0] + temp0[1];
            C[i + 1] = temp1[0] + temp1[1];
            C[i + 2] = temp2[0] + temp2[1];
            C[i + 3] = temp3[0] + temp3[1];
            C[i + 4] = temp4[0] + temp4[1];
            C[i + 5] = temp5[0] + temp5[1];
            C[i + 6] = temp6[0] + temp6[1];
            C[i + 7] = temp7[0] + temp7[1];
        }
        // Leftover in block
        if (blockSize_left > 0) {
            while (i < stop) {
                v2f16 temp0 = (v2f16) {0, 0};
                for (k = 0; k < K; k += 2) {
                    uint32_t idx = i * K + k;
                    v2f16 Bv = *((v2f16 * ) & B[k]);
                    v2f16 Av = *((v2f16 * ) & A[idx]);
                    temp0 += Av * Bv;
                }
                // Leftover in K
                if (k_left > 0) {
                    while (k < K) {
                        uint32_t idx = i * K + k;
                        fp16 Bsh = B[k];
                        fp16 A1 = A[idx];
                        temp0[0] += A1 * Bsh;
                        k++;
                    }
                }
                C[i] = temp0[0] + temp0[1];
                i++;
            }
        }
    }
}


void __attribute__((noinline)) mm_M_fp16_SIMD_2x4 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;
    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t transp = args->trans_B;

    uint32_t indexA, indexB;
    v2f16 Av;
    v2f16 Bv0, Bv1;
    v2f16 *Cv;

    uint32_t M_par = M & 0xfffffffe;
    uint32_t M_left = M - M_par;
    uint32_t core_id = pi_core_id();

    if (M_par < 2) mm_fp16(args);
    else if (K < 2) mm_fp16(args);
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
#if NUM_CORES > 1
            const uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = core_id*blockSize;
            const uint32_t stop = start+blockSize < M_par? start+blockSize: M_par;

            for (uint32_t j = start; j < stop; j+=2)
#else
            const uint32_t start = 0;
            const uint32_t stop = M_par;

            for (uint32_t j = start; j < stop; j += 2)
#endif
            {
                for (uint32_t i = 0; i < N; i++) {
                    v2f16 temp = (v2f16) {0, 0};
                    indexA = i * K;
                    indexB = j;

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        Av = *((v2f16 * ) & A[indexA/*i*K+k*/]);
                        Bv0 = *((v2f16 * ) & B[indexB/*k*M+j*/]);
                        Bv1 = *((v2f16 * ) & B[indexB + M/*k*M+j+M*/]);
                        temp += (v2f16)(__builtin_shuffle(Av, (v2s) {0, 0})) * Bv0;
                        temp += (v2f16)(__builtin_shuffle(Av, (v2s) {1, 1})) * Bv1;

                        indexA += 2;
                        indexB += 2 * M;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av = (v2f16) {A[i * K + (K - 1)], A[i * K + (K - 1)]};
                        Bv0 = *((v2f16 * ) & B[(K - 1) * M + j]);
                        temp += Av * Bv0;
                    }
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;
                }
            }
            // Leftover on M
            if (M_left > 0) {
                uint32_t i_block = (N + NUM_CORES - 1) / NUM_CORES;
                uint32_t i_start = core_id * i_block;
                uint32_t i_stop = i_start + i_block > N ? N : i_start + i_block;

                for (uint32_t i = i_start; i < i_stop; i++) {
                    fp16 val = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        val += A[i * K + k] * B[k * M + (M - 1)];
                    }
                    C[i * M + (M - 1)] = val;
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
#if NUM_CORES > 1
            const uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = core_id*blockSize;
            const uint32_t stop = start+blockSize < M_par? start+blockSize: M_par;

            for (uint32_t j = start; j < stop; j+=2)
#else
            const uint32_t start = 0;
            const uint32_t stop = M_par;

            for (uint32_t j = start; j < stop; j += 2)
#endif
            {
                for (uint32_t i = 0; i < N; i++) {
                    // Global accumulator
                    v2f16 temp = (v2f16) {0, 0};
                    // Dot product accumulators
                    v2f16 tmp0 = (v2f16) {0, 0};
                    v2f16 tmp1 = (v2f16) {0, 0};
                    // K leftover accumulator
                    v2f16 vtmp = (v2f16) {0, 0};
                    // Scalar accumulators for final result
                    fp16 a = 0;
                    fp16 b = 0;
                    // Indices
                    indexA = i * K;
                    indexB = j * K; //j*M;

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        Av = *((v2f16 * ) & A[indexA/*i*K+k*/]);

                        Bv0 = *((v2f16 * ) & B[indexB]);
                        Bv1 = *((v2f16 * ) & B[indexB + K]);
                        tmp0 += (v2f16)(Av * Bv0);
                        tmp1 += (v2f16)(Av * Bv1);

                        indexA += 2;
                        indexB += 2;
                    }
                    // Leftover in K
                    if (K & 1) {
                        Av = (v2f16) {A[indexA], A[indexA]};
                        Bv0 = (v2f16) {B[indexB], B[indexB + K]};
#ifdef DEBUG
                        printf("------ MM K left (indexA=%d, indexB=%d) => Av = {0x%x, 0x%x}, Bv0 = {0x%x, 0x%x}\n", indexA, indexB,
                          ((unsigned int) Av)&(0xffff0000)>>16, ((unsigned int) Av)&(0x0000ffff),
                          ((unsigned int) Bv0)&(0xffff0000)>>16, ((unsigned int) Bv0)&(0x0000ffff));
#endif
                        vtmp += (v2f16)(Av * Bv0);
                        a += vtmp[0];
                        b += vtmp[1];
#ifdef DEBUG
                        printf("------ MM K left => temp = {0x%x, 0x%x}\n", ((unsigned int) vtmp[0]), ((unsigned int) vtmp[1]));
#endif
                    }
                    // Complete dot product
                    a += tmp0[0] + tmp0[1];
                    b += tmp1[0] + tmp1[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;
                }
            }
            // Leftover on M
            if (M_left > 0) {
                uint32_t i_block = (N + NUM_CORES - 1) / NUM_CORES;
                uint32_t i_start = core_id * i_block;
                uint32_t i_stop = i_start + i_block > N ? N : i_start + i_block;

                for (uint32_t i = i_start; i < i_stop; i++) {
                    fp16 val = 0;
                    for (uint32_t k = 0; k < K; k++) {
                        val += A[i * K + k] * B[(M - 1) * K + k];
                    }
                    C[i * M + (M - 1)] = val;
                }
            }
        }
    }
}


void __attribute__((noinline)) mm_M_fp16_SIMD_4x8 (void * void_args) {

    struct matMul_args_fp16 *args = (struct matMul_args_fp16 *) void_args;
    fp16 *__restrict__ A = args->A;
    fp16 *__restrict__ B = args->B;
    fp16 *__restrict__ C = args->C;
    uint32_t N = args->N;
    uint32_t M = args->M;
    uint32_t K = args->K;
    uint32_t transp = args->trans_B;

    uint32_t indexA, indexB;
    v2f16 Av0, Av1;
    v2f16 Bv0, Bv1, Bv2, Bv3;
    v2f16 *Cv;

    uint32_t M_par = M & 0xfffffffc;
    uint32_t M_left = M - M_par;
    uint32_t core_id = pi_core_id();

    uint32_t N_par = N & 0xfffffffe;
    uint32_t N_left = N - N_par;

    // Integrity barrier for oversized unrolling
    uint32_t N_bound = (N - (N & 0x00000001));
    uint32_t M_bound = (M_par) / NUM_CORES;
    uint32_t K_bound = (K - (K & 0x00000001));

    if (M_bound < 4) mm_M_fp16_SIMD_2x4(args);
    else if (K_bound < 2) mm_M_fp16_SIMD_2x4(args);
    else if (N_bound < 2) mm_M_fp16_SIMD_2x4(args);
    else {
        // =====> B NOT TRANSPOSED <=====
        if (transp == 0) {
#if NUM_CORES > 1
            const uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = core_id*blockSize;
            const uint32_t stop = start+blockSize < M_par? start+blockSize: M_par;

            for (uint32_t j = start; j < stop; j+=4)
#else
            const uint32_t start = 0;
            const uint32_t stop = M_par;

            for (uint32_t j = start; j < stop; j += 4)
#endif
            {
                for (uint32_t i = 0; i < N_par; i += 2) {
                    v2f16 temp0 = (v2f16) {0, 0};
                    v2f16 temp1 = (v2f16) {0, 0};
                    v2f16 temp2 = (v2f16) {0, 0};
                    v2f16 temp3 = (v2f16) {0, 0};

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        Av1 = *(v2f16 * ) & A[(i + 1) * K + k];
                        // B vectors
                        Bv0 = *(v2f16 * ) & B[k * M + j];
                        Bv1 = *(v2f16 * ) & B[(k + 1) * M + j];
                        Bv2 = *(v2f16 * ) & B[k * M + j + 2];
                        Bv3 = *(v2f16 * ) & B[(k + 1) * M + j + 2];

                        // Ci,j, Ci,j+1
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv0;
                        temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv1;
                        // Ci,j+2, Ci,j+3
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {0, 0})) * Bv2;
                        temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s) {1, 1})) * Bv3;
                        // Ci+1,j, Ci+1,j+1
                        temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s) {0, 0})) * Bv0;
                        temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s) {1, 1})) * Bv1;
                        // Ci+1,j+2, Ci+1,j+3
                        temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s) {0, 0})) * Bv2;
                        temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s) {1, 1})) * Bv3;
                    }
                    // Leftover on K
                    if (K & 1) {
                        Av0 = (v2f16) {A[i * K + (K - 1)], A[i * K + (K - 1)]};
                        Av1 = (v2f16) {A[(i + 1) * K + (K - 1)], A[(i + 1) * K + (K - 1)]};
                        Bv0 = *((v2f16 * ) & B[(K - 1) * M + j]);
                        Bv1 = *((v2f16 * ) & B[(K - 1) * M + j + 2]);
                        temp0 += Av0 * Bv0;
                        temp1 += Av0 * Bv1;
                        temp2 += Av1 * Bv0;
                        temp3 += Av1 * Bv1;
                    }
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp0;

                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp1;

                    Cv = (v2f16 * ) & C[(i + 1) * M + j];
                    *Cv = temp2;

                    Cv = (v2f16 * ) & C[(i + 1) * M + j + 2];
                    *Cv = temp3;
                }
                // Leftover on N
                if (N & 0x00000001) {
                    for (uint32_t jj = j; jj < j + 4; jj++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[(N - 1) * K + k] * B[jj + k * M];
                        }
                        C[(N - 1) * M + jj] = left_temp;
                    }
                }
            }
            // Leftover on M (parallel on N)
            if (M_left > 0) {
                uint32_t i_block = (N + NUM_CORES - 1) / NUM_CORES;
                uint32_t i_start = core_id * i_block;
                uint32_t i_stop = i_start + i_block > N ? N : i_start + i_block;

                for (uint32_t i = i_start; i < i_stop; i++) {
                    for (uint32_t j = M - M_left; j < M; j++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[i * K + k] * B[j + k * M];
                        }
                        C[i * M + j] = left_temp;
                    }
                }
            }
        }

            // =====> B IS TRANSPOSED <=====
        else {
#if NUM_CORES > 1
            const uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
            const uint32_t start = core_id*blockSize;
            const uint32_t stop = start+blockSize < M_par? start+blockSize: M_par;

            for (uint32_t j = start; j < stop; j+=4)
#else
            const uint32_t start = 0;
            const uint32_t stop = M_par;

            for (uint32_t j = start; j < stop; j += 4)
#endif
            {
                for (uint32_t i = 0; i < (N & 0xfffffffe); i += 2) {
                    // Global accumulator
                    v2f16 temp = (v2f16) {0, 0};
                    // Dot product accumulators
                    v2f16 tmp0 = (v2f16) {0, 0};
                    v2f16 tmp1 = (v2f16) {0, 0};
                    v2f16 tmp2 = (v2f16) {0, 0};
                    v2f16 tmp3 = (v2f16) {0, 0};
                    v2f16 tmp4 = (v2f16) {0, 0};
                    v2f16 tmp5 = (v2f16) {0, 0};
                    v2f16 tmp6 = (v2f16) {0, 0};
                    v2f16 tmp7 = (v2f16) {0, 0};
                    // Scalar accumulators
                    fp16 a = 0;
                    fp16 b = 0;

                    for (uint32_t k = 0; k < (K & 0xfffffffe); k += 2) {
                        // A vectors
                        Av0 = *(v2f16 * ) & A[i * K + k];
                        Av1 = *(v2f16 * ) & A[(i + 1) * K + k];
                        // B vectors (transposed matrix)
                        Bv0 = *(v2f16 * ) & B[j * K + k];
                        Bv1 = *(v2f16 * ) & B[(j + 1) * K + k];
                        Bv2 = *(v2f16 * ) & B[(j + 2) * K + k];
                        Bv3 = *(v2f16 * ) & B[(j + 3) * K + k];

                        // Products in Ci,j and successive with Av0
                        tmp0 += Av0 * Bv0;
                        tmp1 += Av0 * Bv1;
                        tmp2 += Av0 * Bv2;
                        tmp3 += Av0 * Bv3;
                        // Products in Ci+1,j and successive with Av1
                        tmp4 += Av1 * Bv0;
                        tmp5 += Av1 * Bv1;
                        tmp6 += Av1 * Bv2;
                        tmp7 += Av1 * Bv3;
                    }
                    if (K & 1) {
                        // A elements
                        fp16 A0 = A[i * K + (K - 1)];
                        fp16 A1 = A[(i + 1) * K + (K - 1)];
                        // B elements (transposed matrix)
                        fp16 B0 = B[j * K + (K - 1)];
                        fp16 B1 = B[(j + 1) * K + (K - 1)];
                        fp16 B2 = B[(j + 2) * K + (K - 1)];
                        fp16 B3 = B[(j + 3) * K + (K - 1)];

                        // Products in Ci,j and successive with Av0
                        tmp0[0] += A0 * B0;
                        tmp1[0] += A0 * B1;
                        tmp2[0] += A0 * B2;
                        tmp3[0] += A0 * B3;
                        // Products in Ci+1,j and successive with Av1
                        tmp4[0] += A1 * B0;
                        tmp5[0] += A1 * B1;
                        tmp6[0] += A1 * B2;
                        tmp7[0] += A1 * B3;
                    }
                    // Accumulate to compute dot product and store
                    // Row 1
                    a = tmp0[0] + tmp0[1];
                    b = tmp1[0] + tmp1[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j];
                    *Cv = temp;

                    a = tmp2[0] + tmp2[1];
                    b = tmp3[0] + tmp3[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[i * M + j + 2];
                    *Cv = temp;

                    // Row 2
                    a = tmp4[0] + tmp4[1];
                    b = tmp5[0] + tmp5[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[(i + 1) * M + j];
                    *Cv = temp;

                    a = tmp6[0] + tmp6[1];
                    b = tmp7[0] + tmp7[1];
                    temp = (v2f16) {a, b};
                    Cv = (v2f16 * ) & C[(i + 1) * M + j + 2];
                    *Cv = temp;
                }
                // Leftover on N
                if (N & 0x00000001) {
                    for (uint32_t jj = j; jj < j + 4; jj++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[(N - 1) * K + k] * B[jj * K + k];
                        }
                        C[(N - 1) * M + jj] = left_temp;
                    }
                }
            }
            // Leftover on M (parallel on N)
            if (M_left > 0) {
                uint32_t i_block = (N + NUM_CORES - 1) / NUM_CORES;
                uint32_t i_start = core_id * i_block;
                uint32_t i_stop = i_start + i_block > N ? N : i_start + i_block;

                for (uint32_t i = i_start; i < i_stop; i++) {
                    for (uint32_t j = M - M_left; j < M; j++) {
                        fp16 left_temp = 0;
                        for (uint32_t k = 0; k < K; k++) {
                            left_temp += A[i * K + k] * B[j * K + k];
                        }
                        C[i * M + j] = left_temp;
                    }
                }
            }
        }
    }
}
