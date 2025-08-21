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

#include "pmsis.h"
#include "pulp_train.h"

#include "stats.h"
#include "output_eval.h"

#include "net_args.h"
#include "matmul_data.h"


// DATA DEFINITION

#ifdef FLOAT32
// General purpose matmuls
#ifdef STANDARD
PI_L1 float result[IN_CH*OUT_CH];
#endif
#endif


#ifdef FLOAT16
// General purpose matmuls
#ifdef STANDARD
PI_L1 fp16 result[IN_CH*OUT_CH];
#endif
#endif


PI_L1 float zero_init = 0.0f;


// Function to null tensor
#ifdef FLOAT32
static inline void null_tensor (float * tensor, int size) 
#endif
#ifdef FLOAT16
static inline void null_tensor (fp16 * tensor, int size) 
#endif
{
    for (int idx=0; idx<size; idx++) {
        tensor[idx] = zero_init;
    }
}



// Matmul test core function
#ifdef STANDARD
static inline void multiply () 
{
    #ifdef FLOAT32
    struct matMul_args mm_args;
    #endif

    #ifdef FLOAT16
    struct matMul_args_fp16 mm_args;
    #endif

    // General setup for matmuls
    mm_args.A = A;
    mm_args.B = B;
    mm_args.C = result; 
    mm_args.N = IN_CH;
    mm_args.K = MID_CH;
    mm_args.M = OUT_CH;
    mm_args.trans_B = TRANSPOSE_B;
    // End of general setup

    if (mm_args.trans_B == 1) printf("Running matmuls with transposed B matrix.\n");
    else printf("Running matmuls with no transpositions.\n");

    printf("Running matmuls on %d cores.\n", NUM_CORES);

    #ifdef FLOAT32
    printf("\n=====> PROFILING MATMULS WITH PARALLELISM ON N <=====\n");
    
    printf("\n-----> Profiling mm:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);


    printf("\n-----> Profiling mm_u2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_u2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_1x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_1x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_1x8:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_1x8, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);
    

    printf("\n-----> Profiling mm_unroll_2x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    
    printf("\n-----> Profiling mm_unroll_2x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_4x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_8x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_8x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_2x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_2x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_4x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_unroll_4x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n=====> PROFILING MATMULS WITH PARALLELISM ON M <=====\n");    

    printf("\n-----> Profiling mm_M:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_u2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_u2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_2x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_4x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_8x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_8x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_1x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_1x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_1x8:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_1x8, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_2x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_4x2:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x2, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_2x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_2x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_unroll_4x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);
    
    #endif



    #ifdef FLOAT16
    printf("\n-----> Profiling mm_fp16:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_fp16_unroll_8x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16_unroll_8x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_fp16_unroll_4x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16_unroll_4x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_fp16_unroll_2x1:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16_unroll_2x1, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_M_fp16:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_fp16, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);

    printf("\n-----> Profiling mm_fp16_SIMD_2x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16_SIMD_2x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);    

    printf("\n-----> Profiling mm_fp16_SIMD_4x8:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_fp16_SIMD_4x8, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);    

    printf("\n-----> Profiling mm_M_fp16_SIMD_2x4:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_fp16_SIMD_2x4, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH);    

    printf("\n-----> Profiling mm_M_fp16_SIMD_4x8:\n");
    START_STATS();
    pi_cl_team_fork(NUM_CORES, mm_M_fp16_SIMD_4x8, &mm_args);
    STOP_STATS();
    check_tensor(result, C, IN_CH*OUT_CH);
    compare_tensors(result, C, IN_CH*OUT_CH);
    null_tensor(result, IN_CH*OUT_CH); 
    #endif



}
#endif



void net_step () {

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    #ifdef STANDARD
    printf("Matmul sizes are:\nN: %d, K: %d, M: %d\n", IN_CH, MID_CH, OUT_CH);
    #endif

    #ifdef FLOAT32
    printf("Data type is float32.\n");
    #endif
    #ifdef FLOAT16
    printf("Data type is bfloat16.\n");
    #endif

    multiply();

    return;
}