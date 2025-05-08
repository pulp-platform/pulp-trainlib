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

#include "pulp_train.h"

#include "init-defines.h"
#include "input-sequence.h"
#include "mhsa-grads.h"
#include "mhsa-output.h"
#include "attention_scores.h"
#include "stats.h"

#include "step-check.h"

#include "net.h"


// DATA DEFINITION
// MHSA
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;
PI_L1 struct Mhsa_args mhsa_args;
PI_L1 struct blob layer0_in, layer0_wgt_in, layer0_wgt_out, layer0_qkv, layer0_att_map, layer0_h_buffer, layer0_softmax_buffer, layer0_out;

// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;


#ifdef FORWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_ker_in[Tin_W_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_ker_out[Tatt_dim_l1 * Tin_W_l1];
PI_L1 float l0_qkv[Tin_H_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_att_map[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_h_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_softmax_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_out[Tin_H_l1 * Tin_W_l1];
// TODO: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
PI_L1 float l0_temp[Tin_H_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_partial_exp_sum[NUM_CORES * Tin_H_l1];
PI_L1 float l0_global_max[NUM_CORES * Tin_H_l1];
#endif


#ifdef BACKWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_in_diff[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_ker_in[Tin_W_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_ker_out[Tatt_dim_l1 * Tin_W_l1];
PI_L1 float l0_ker_in_diff[Tin_W_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_ker_out_diff[Tatt_dim_l1 * Tin_W_l1];
PI_L1 float l0_qkv[Tin_H_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_qkv_diff[Tin_H_l1 * Tatt_dim_l1 * 3];
PI_L1 float l0_att_map[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_att_map_diff[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_out[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_out_diff[Tin_H_l1 * Tin_W_l1];
// TODO: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
PI_L1 float l0_temp[Tin_H_l1 * Tatt_dim_l1 * 3];
// Buffer containing the pre-softmax head buffer gradient, necessary in the backward process
PI_L1 float l0_grad[Tin_H_l1 * Tin_H_l1];
PI_L1 float l0_h_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_h_buffer_diff[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_softmax_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
#endif


#ifdef FORWARD
static inline void tensor_init() {
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in[i] = INPUT[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1 * 3; i++) l0_ker_in[i] = INPUT_WEIGHTS[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++) l0_ker_out[i] = OUTPUT_WEIGHTS[i];
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_out[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1 * 3; i++) l0_qkv[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_att_map[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_h_buffer[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_softmax_buffer[i] = zero_init;
    // TODO: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1 * 3; i++) l0_temp[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * NUM_CORES; i++) l0_partial_exp_sum[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * NUM_CORES; i++) l0_global_max[i] = min_float;
}


static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;

    layer0_wgt_in.data = l0_ker_in;
    layer0_wgt_in.dim = Tin_W_l1 * Tatt_dim_l1 * 3;
    layer0_wgt_in.H = Tin_W_l1;
    layer0_wgt_in.W = Tatt_dim_l1 * 3;
    layer0_wgt_in.C = Tout_C_l1;

    layer0_wgt_out.data = l0_ker_out;
    layer0_wgt_out.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_out.H = Tin_W_l1;
    layer0_wgt_out.W = Tatt_dim_l1;
    layer0_wgt_out.C = Tin_C_l1;

    layer0_qkv.data = l0_qkv;
    layer0_qkv.dim = Tatt_dim_l1 * 3 * Tin_H_l1;
    layer0_qkv.H = Tin_H_l1;
    layer0_qkv.W = Tatt_dim_l1 * 3;
    layer0_qkv.C = Tin_C_l1;

    layer0_out.data = l0_out;
    layer0_out.dim = Tin_W_l1 * Tin_H_l1;
    layer0_out.H = Tin_H_l1;
    layer0_out.W = Tin_W_l1;
    layer0_out.C = Tin_C_l1;

    layer0_att_map.data = l0_att_map;
    layer0_att_map.dim = Tin_H_l1 * Tatt_dim_l1;
    layer0_att_map.H = Tin_H_l1;
    layer0_att_map.W = Tatt_dim_l1;
    layer0_att_map.C = Tin_C_l1;

    layer0_h_buffer.data = l0_h_buffer;
    layer0_h_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_h_buffer.H = Tn_heads_l1;
    layer0_h_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_h_buffer.C = Tin_C_l1;

    layer0_softmax_buffer.data = l0_softmax_buffer;
    layer0_softmax_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_softmax_buffer.H = Tn_heads_l1;
    layer0_softmax_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_softmax_buffer.C = Tin_C_l1;

    mhsa_args.input = &layer0_in;
    mhsa_args.n_heads = Tn_heads_l1;
    mhsa_args.qkv = &layer0_qkv;
    mhsa_args.output = &layer0_out;
    mhsa_args.coeff_in = &layer0_wgt_in;
    mhsa_args.coeff_out = &layer0_wgt_out;
    mhsa_args.attention_map = &layer0_att_map;
    mhsa_args.head_buffer = &layer0_h_buffer;
    mhsa_args.softmax_buffer = &layer0_softmax_buffer;
    mhsa_args.temp_buffer = l0_temp;
    mhsa_args.partial_exp_sum = l0_partial_exp_sum;
    mhsa_args.global_max = l0_global_max;
    mhsa_args.opt_matmul_type_fw = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_wg = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_ig = MATMUL_TYPE;
}


static inline void compute_memory_occupation() {
    // Input
    L1_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV
    L1_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * 3 * sizeof(float);
    // Output
    L1_memocc_bytes += Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map
    L1_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Scores
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Heads Softmax Output
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Tmp buffer
    L1_memocc_bytes += Tin_H_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // partial_exp_sum buffer
    L1_memocc_bytes += NUM_CORES * Tin_H_l1 * sizeof(float);
    // global_max buffer
    L1_memocc_bytes += NUM_CORES * Tin_H_l1 * sizeof(float);

    // Input
    L2_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L2_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output
    L2_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV
    L2_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * 3 * sizeof(float);
    // Output
    L2_memocc_bytes += Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map
    L2_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Scores
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Heads Softmax Output
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Tmp buffer
    L2_memocc_bytes += Tin_H_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // partial_exp_sum buffer
    L2_memocc_bytes += NUM_CORES * Tin_H_l1 * sizeof(float);
    // global_max buffer
    L2_memocc_bytes += NUM_CORES * Tin_H_l1 * sizeof(float);
}
#endif


#ifdef BACKWARD
static inline void tensor_init() {
    // Backward grad
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in[i] = INPUT[i];
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in_diff[i] = zero_init;

    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1 * 3; i++) l0_ker_in[i] = INPUT_WEIGHTS[i];
    // Initialization to zero, then it is overwritten the result in the function
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1 * 3; i++) l0_ker_in_diff[i] = zero_init;

    for (int i = 0; i < Tatt_dim_l1 * Tin_W_l1; i++) l0_ker_out[i] = OUTPUT_WEIGHTS[i];
    for (int i = 0; i < Tatt_dim_l1 * Tin_W_l1; i++) l0_ker_out_diff[i] = zero_init;

    for (int i = 0; i < Tin_W_l1 * Tin_H_l1; i++) l0_out_diff[i] = OUTPUT_GRAD[i];
    for (int i = 0; i < Tin_W_l1 * Tin_H_l1; i++) l0_out[i] = OUTPUT[i];

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1 * 3; i++) l0_temp[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tin_H_l1; i++) l0_grad[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1 * 3; i++) l0_qkv_diff[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1 * 3; i++) l0_qkv[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_att_map[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_att_map_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_h_buffer[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_h_buffer_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_softmax_buffer[i] = zero_init;
}


static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;
    layer0_in.diff = l0_in_diff;

    layer0_wgt_in.data = l0_ker_in;
    layer0_wgt_in.dim = Tin_W_l1 * Tatt_dim_l1 * 3;
    layer0_wgt_in.H = Tin_W_l1;
    layer0_wgt_in.W = Tatt_dim_l1 * 3;
    layer0_wgt_in.C = Tout_C_l1;
    layer0_wgt_in.diff = l0_ker_in_diff;

    layer0_wgt_out.data = l0_ker_out;
    layer0_wgt_out.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_out.H = Tin_W_l1;
    layer0_wgt_out.W = Tatt_dim_l1;
    layer0_wgt_out.C = Tin_C_l1;
    layer0_wgt_out.diff = l0_ker_out_diff;

    layer0_qkv.data = l0_qkv;
    layer0_qkv.dim = Tatt_dim_l1 * 3 * Tin_H_l1;
    layer0_qkv.H = Tin_H_l1;
    layer0_qkv.W = Tatt_dim_l1 * 3;
    layer0_qkv.C = Tin_C_l1;
    layer0_qkv.diff = l0_qkv_diff;

    layer0_out.data = l0_out;
    layer0_out.dim = Tin_W_l1 * Tin_H_l1;
    layer0_out.H = Tin_H_l1;
    layer0_out.W = Tin_W_l1;
    layer0_out.C = Tin_C_l1;
    layer0_out.diff = l0_out_diff;

    layer0_att_map.data = l0_att_map;
    layer0_att_map.dim = Tin_H_l1 * Tatt_dim_l1;
    layer0_att_map.H = Tin_H_l1;
    layer0_att_map.W = Tatt_dim_l1;
    layer0_att_map.C = Tin_C_l1;
    layer0_att_map.diff = l0_att_map_diff;

    layer0_h_buffer.data = l0_h_buffer;
    layer0_h_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_h_buffer.H = Tn_heads_l1;
    layer0_h_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_h_buffer.C = Tin_C_l1;
    layer0_h_buffer.diff = l0_h_buffer_diff;

    layer0_softmax_buffer.data = l0_softmax_buffer;
    layer0_softmax_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_softmax_buffer.H = Tn_heads_l1;
    layer0_softmax_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_softmax_buffer.C = Tin_C_l1;

    mhsa_args.input = &layer0_in;
    mhsa_args.qkv = &layer0_qkv;
    mhsa_args.output = &layer0_out;
    mhsa_args.coeff_in = &layer0_wgt_in;
    mhsa_args.coeff_out = &layer0_wgt_out;
    mhsa_args.temp_buffer = l0_temp;
    mhsa_args.grad = l0_grad;
    mhsa_args.attention_map = &layer0_att_map;
    mhsa_args.head_buffer = &layer0_h_buffer;
    mhsa_args.softmax_buffer = &layer0_softmax_buffer;
    mhsa_args.n_heads = Tn_heads_l1;
    mhsa_args.opt_matmul_type_fw = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_wg = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_ig = MATMUL_TYPE;
}


static inline void compute_memory_occupation() {
    // Input
    L1_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV + grad
    L1_memocc_bytes += 2 * Tatt_dim_l1 * Tin_H_l1 * 3 * sizeof(float);
    // Output + grad
    L1_memocc_bytes += 2 * Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map + grad
    L1_memocc_bytes += 2 * Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Scores + grad
    L1_memocc_bytes += 2 * Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Tmp buffer
    L1_memocc_bytes += Tin_H_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Gradient buffer
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * sizeof(float);
    // Heads Softmax Output
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);

    // Input + grad
    L2_memocc_bytes += 2 * Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input + grad
    L2_memocc_bytes += 2 * Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output + grad
    L2_memocc_bytes += 2 * Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV + grad
    L2_memocc_bytes += 2 * Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Output + grad
    L2_memocc_bytes += 2 * Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map + grad
    L2_memocc_bytes += 2 * Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Scores + grad
    L2_memocc_bytes += 2 * Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Tmp buffer
    L2_memocc_bytes += Tin_H_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Gradient buffer
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * sizeof(float);
    // Heads Softmax Output
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
}
#endif


static inline void compare_tensors(float *A, float *B, int length) {
    float mean_err_rel = 0.0f;
    float diff = 0.0f;

    for (int i = 0; i < length; i++) {
        diff = A[i] - B[i];
        if (diff > 0) diff = diff;
        else diff = -diff;
        mean_err_rel = mean_err_rel + diff / length;
    }
    if (mean_err_rel < ERROR_TOLERANCE) printf("\n>>>TENSOR MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
    else printf("\n>>>TENSOR NOT MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
}


// Elementwise checker
int check_tensor(float *tensor_out, float *tensor_ref, int size) {

    int error_flag = 0;
    for (int i = 0; i < size; i++) {
        if (ABS(tensor_out[i] - tensor_ref[i]) > CHECK_TOLERANCE) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i,
                   tensor_ref[i], *(unsigned int *) &tensor_ref[i], tensor_out[i], *(unsigned int *) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}


static inline void train() {
#ifdef PROF_FWD
    printf("\nForward stats\n");
    START_STATS();
#endif


#ifdef FORWARD
    pulp_mhsa_fp32_fw_cl_2(&mhsa_args);
#endif


#ifdef PROF_FWD
    STOP_STATS();
#endif


#ifdef BACKWARD
    pulp_mhsa_fp32_fw_cl(&mhsa_args);

    printf("\nFORWARD CHECK: \n");
    compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
    check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);

    printf("\nATTENTION SCORE CHECK: \n");
    compare_tensors(l0_att_map, ATTENTION_SCORES, ATTENTION_S_LENGTH);
    check_tensor(l0_att_map, ATTENTION_SCORES, ATTENTION_S_LENGTH);

#ifdef PROF_BCKWD
    printf("\nBackward stats\n");
    START_STATS();
#endif

    pulp_mhsa_fp32_bw_cl(&mhsa_args);

#ifdef PROF_BCKWD
    STOP_STATS();
#endif
#endif


#ifdef FORWARD
    printf("\nFORWARD CHECK: \n");
    compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
    check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);
#endif


#ifdef BACKWARD
    printf("\nFINAL WEIGHTS GRADIENT CHECK: \n");
    printf("\nINPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(l0_ker_in_diff, INPUT_WGT_GRAD, G_INPUT_WGT_SIZE);
    check_tensor(l0_ker_in_diff, INPUT_WGT_GRAD, G_INPUT_WGT_SIZE);

    printf("\nOUTPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(l0_ker_out_diff, OUTPUT_WGT_GRAD, G_OUTPUT_WGT_SIZE);
    check_tensor(l0_ker_out_diff, OUTPUT_WGT_GRAD, G_OUTPUT_WGT_SIZE);

    printf("\nINPUT GRADIENT CHECK: \n");
    compare_tensors(l0_in_diff, INPUT_GRAD, G_IN_SIZE);
    check_tensor(l0_in_diff, INPUT_GRAD, G_IN_SIZE);
#endif
}


// Most important function: it connects each passage to step the net and perform training
void net_step() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

#ifdef MEMOCC_COMP
    compute_memory_occupation();
    printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
    printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
#endif

    tensor_init();

    connect_blobs();

    train();

    return;
}
