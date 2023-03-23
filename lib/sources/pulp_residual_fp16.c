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
 * Authors: Davide Nadalini
*/ 

#include "pmsis.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_residual_fp16.h"


// FORWARD PRIMITIVES

void pulp_sumnode_fp16_fw( void * SkipConn_args_fp16 )
{
    struct SkipConn_args_fp16 * args = (struct SkipConn_args_fp16 *) SkipConn_args_fp16;
    struct blob_fp16 act0 = args->activation0;
    struct blob_fp16 act1 = args->activation1;
    struct blob_fp16 out = args->activation2;

    if (act0.dim != act1.dim || act1.dim != out.dim) {
        printf("[pulp_sumnode_fp16_fw]: Sizes of input and output activations not matching!!"); return;
    }

    struct vect_sum_args_fp16 args;
    args.op_1 = act0.data;
    args.op_2 = act1.data;
    args.dest = out.data;
    args.size = out.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &args);
}



void pulp_residualconn_fp16_fw( void * SkipConn_args_fp16 )
{
    struct SkipConn_args_fp16 * args = (struct SkipConn_args_fp16 *) SkipConn_args_fp16;
    struct blob_fp16 act0 = args->activation0;
    struct blob_fp16 act1 = args->activation1;
    struct blob_fp16 out = args->activation2;

    if (act0.dim != act1.dim || act1.dim != out.dim) {
        printf("[pulp_residualconn_fp16_fw]: Sizes of input and output activations not matching!!"); return;
    }

    struct vect_sum_args_fp16 args;
    args.op_1 = act0.data;
    args.op_2 = act1.data;
    args.dest = out.data;
    args.size = out.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &args);
}





// BACKWARD PRIMITIVES

void pulp_sumnode_fp16_bw( void * SkipConn_args_fp16 )
{
    struct SkipConn_args_fp16 * args = (struct SkipConn_args_fp16 *) SkipConn_args_fp16;
    struct blob_fp16 act0 = args->activation0;
    struct blob_fp16 act1 = args->activation1;
    struct blob_fp16 indiff = args->activation2;
    
    if (act0.dim != act1.dim || act1.dim != indiff.dim) {
        printf("[pulp_sumnode_fp16_bw]: Sizes of input and output gradients not matching!!"); return;
    }

    struct vect_sum_args_fp16 args;
    args.op_1 = act0.diff;
    args.op_2 = act1.diff;
    args.dest = indiff.diff;
    args.size = indiff.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum_fp16, &args);
}



void pulp_residualconn_fp16_bw( void * SkipConn_args_fp16 )
{
    struct SkipConn_args_fp16 * args = (struct SkipConn_args_fp16 *) SkipConn_args_fp16;
    struct blob_fp16 act0 = args->activation0;
    struct blob_fp16 act1 = args->activation1;
    struct blob_fp16 outdiff = args->activation2;
    
    if (act0.dim != act1.dim || act1.dim != outdiff.dim) {
        printf("[pulp_residualconn_fp16_bw]: Sizes of input and output gradients not matching!!"); return;
    }

    // Copy gradient into the input
    struct copy_args_fp16 cpy_args;
    cpy_args.from = outdiff.diff;
    cpy_args.to = act0.diff;
    cpy_args.size = outdiff.dim;
    pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);
    cpy_args.to = act1.diff;
    pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);
}
