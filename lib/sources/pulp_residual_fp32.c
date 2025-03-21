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
 * Authors: Davide Nadalini, Giacomo Saporetti, Calin Diaconu
*/

#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_residual_fp32.h"


// FORWARD PRIMITIVES

void pulp_residualconn_fp32_fw(void *SkipConn_args) {
    struct SkipConn_args *args = (struct SkipConn_args *) SkipConn_args;
    struct blob *skip = args->skip;
    struct blob *lout = args->lout;
    struct blob *out = args->output;

    if (skip->dim != lout->dim || lout->dim != out->dim) {
        printf("\n[pulp_residualconn_fp16_fw]: Sizes of input and output activations not matching!!, "
               "got %d , %d and %d (%d, %d, %d), (%d, %d, %d) and (%d, %d, %d)\n",
               skip->dim, lout->dim, out->dim, skip->C, skip->H, skip->W, lout->C, lout->H, lout->W, out->C, out->H,
               out->W);
        return;
    }

    int dims[] = {out->dim};

    struct array_broadcast_sum_fp32_args args_sum;

    args_sum.op_1 = skip->data;
    args_sum.op_2 = lout->data;
    args_sum.dest = out->data;

    args_sum.op_1_dims = dims;
    args_sum.op_2_dims = dims;

    args_sum.op_1_dims_len = 1;
    args_sum.op_2_dims_len = 1;

    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &args_sum);
}


// BACKWARD PRIMITIVES
void pulp_sumnode_fp32_bw(void *SkipConn_args) {
    struct SkipConn_args *args = (struct SkipConn_args *) SkipConn_args;
    struct blob *skip = args->skip;
    struct blob *lout = args->lout;
    struct blob *out = args->output;

    if (args->skip_in_grad == 0) {
        if (skip->dim != out->dim) {
            printf("[pulp_sumnode_fp32_bw]: Sizes of input and output activations not matching!!");
            printf("\ngot (NCHW) Skip: %d,%d,%d,%d Lout: %d,%d,%d,%d Out:%d,%d,%d,%d\n", skip->dim, skip->C, skip->H,
                   skip->W, lout->dim, lout->C, lout->H, lout->W, out->dim, out->C, out->H, out->W);
            return;
        }

        int dims[] = {skip->dim};

        struct array_broadcast_sum_fp32_args args_sum;

        args_sum.op_1 = out->diff;
        args_sum.op_2 = skip->diff;
        args_sum.dest = skip->diff;

        args_sum.op_1_dims = dims;
        args_sum.op_2_dims = dims;

        args_sum.op_1_dims_len = 1;
        args_sum.op_2_dims_len = 1;

        pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &args_sum);
    }
}


void pulp_residualconn_fp32_bw(void *SkipConn_args) {
    struct SkipConn_args *args = (struct SkipConn_args *) SkipConn_args;
    //struct blob * skip = args->skip;
    struct blob *lout = args->lout;
    struct blob *out = args->output;

    if (lout->dim != out->dim) {
        printf("[pulp_residualconn_fp32_bw]: Sizes of input and output activations not matching!!");
        printf("\ngot (NCHW)  Lout: %d,%d,%d,%d Out:%d,%d,%d,%d\n", lout->dim, lout->C, lout->H, lout->W, out->dim,
               out->C, out->H, out->W);
        return;
    }

    // Copy gradient into the input
    struct copy_args cpy_args;
    cpy_args.from = out->diff;
    //cpy_args.to = skip->diff;
    cpy_args.size = out->dim;
    //pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    cpy_args.to = lout->diff;
    pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
}
