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
#include "pulp_train_utils_fp32.h"
#include "pulp_residual_fp32.h"


// FORWARD PRIMITIVES

void pulp_sumnode_fp32_fw( void * SkipConn_args )
{
    struct SkipConn_args * args = (struct SkipConn_args *) SkipConn_args;
    struct blob act0 = args->activation0;
    struct blob act1 = args->activation1;
    struct blob out = args->activation2;

    if (act0.dim != act1.dim || act1.dim != out.dim) {
        printf("[pulp_sumnode_fp32_fw]: Sizes of input and output activations not matching!!"); return;
    }

    struct vect_sum_args args;
    args.op_1 = act0.data;
    args.op_2 = act1.data;
    args.dest = out.data;
    args.size = out.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum, &args);
}



void pulp_residualconn_fp32_fw( void * SkipConn_args )
{
    struct SkipConn_args * args = (struct SkipConn_args *) SkipConn_args;
    struct blob act0 = args->activation0;
    struct blob act1 = args->activation1;
    struct blob out = args->activation2;

    if (act0.dim != act1.dim || act1.dim != out.dim) {
        printf("[pulp_residualconn_fp32_fw]: Sizes of input and output activations not matching!!"); return;
    }

    struct vect_sum_args args;
    args.op_1 = act0.data;
    args.op_2 = act1.data;
    args.dest = out.data;
    args.size = out.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum, &args);
}





// BACKWARD PRIMITIVES

void pulp_sumnode_fp32_bw( void * SkipConn_args )
{
    struct SkipConn_args * args = (struct SkipConn_args *) SkipConn_args;
    struct blob act0 = args->activation0;
    struct blob act1 = args->activation1;
    struct blob indiff = args->activation2;
    
    if (act0.dim != act1.dim || act1.dim != indiff.dim) {
        printf("[pulp_sumnode_fp32_bw]: Sizes of input and output gradients not matching!!"); return;
    }

    struct vect_sum_args args;
    args.op_1 = act0.diff;
    args.op_2 = act1.diff;
    args.dest = indiff.diff;
    args.size = indiff.dim;

    pi_cl_team_fork(NUM_CORES, vect_sum, &args);
}



void pulp_residualconn_fp32_bw( void * SkipConn_args )
{
    struct SkipConn_args * args = (struct SkipConn_args *) SkipConn_args;
    struct blob act0 = args->activation0;
    struct blob act1 = args->activation1;
    struct blob outdiff = args->activation2;
    
    if (act0.dim != act1.dim || act1.dim != outdiff.dim) {
        printf("[pulp_residualconn_fp32_bw]: Sizes of input and output gradients not matching!!"); return;
    }

    // Copy gradient into the input
    struct copy_args cpy_args;
    cpy_args.from = outdiff.diff;
    cpy_args.to = act0.diff;
    cpy_args.size = outdiff.dim;
    pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    cpy_args.to = act1.diff;
    pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
}
