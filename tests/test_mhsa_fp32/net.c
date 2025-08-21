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

 * Authors: Alberto Dequino (alberto.dequino@unibo.it), Calin Diaconu (calin.diaconu@studio.unibo.it)
 */

// Required to be first library to be included!
#include "pulp_train.h"

#include "attention_scores.h"
#include "init-defines.h"
#include "input-sequence.h"
#include "mhsa-grads.h"
#include "mhsa-output.h"
#include "net.h"
#include "stats.h"
#include "step-check.h"


// ~~~~~~~~~~ VARIABLE DEFINITION ~~~~~~~~~~
// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

// Variables definition
PI_L1 struct Mhsa_args mhsa_args;
PI_L1 struct blob layer0_in,
        layer0_wgt_in_q, layer0_wgt_in_k, layer0_wgt_in_v,
        layer0_bias_in_q, layer0_bias_in_k, layer0_bias_in_v,
        layer0_wgt_out,
        layer0_q, layer0_k, layer0_v,
        layer0_att_map, layer0_softmax_buffer, layer0_out;

// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

// Forward step variables
#ifdef FORWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];

PI_L1 float l0_ker_in_q[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_k[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_v[Tin_W_l1 * Tatt_dim_l1];

PI_L1 float l0_bias_in_q[Tatt_dim_l1];
PI_L1 float l0_bias_in_k[Tatt_dim_l1];
PI_L1 float l0_bias_in_v[Tatt_dim_l1];

PI_L1 float l0_ker_out[Tatt_dim_l1 * Tin_W_l1];
PI_L1 float l0_q[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_k[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_v[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_att_map[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_softmax_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_out[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_temp[Ttemp_max];
PI_L1 float l0_sums[Tin_H_l1];
PI_L1 float l0_maxes[Tin_H_l1];
#endif

// Backward step variables
#ifdef BACKWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_in_diff[Tin_H_l1 * Tin_W_l1];

PI_L1 float l0_ker_in_q[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_k[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_v[Tin_W_l1 * Tatt_dim_l1];

PI_L1 float l0_ker_in_q_diff[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_k_diff[Tin_W_l1 * Tatt_dim_l1];
PI_L1 float l0_ker_in_v_diff[Tin_W_l1 * Tatt_dim_l1];

PI_L1 float l0_bias_in_q[Tatt_dim_l1];
PI_L1 float l0_bias_in_k[Tatt_dim_l1];
PI_L1 float l0_bias_in_v[Tatt_dim_l1];

PI_L1 float l0_ker_out[Tatt_dim_l1 * Tin_W_l1];
PI_L1 float l0_ker_out_diff[Tatt_dim_l1 * Tin_W_l1];

PI_L1 float l0_q[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_q_diff[Tin_H_l1 * Tatt_dim_l1];

PI_L1 float l0_k[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_k_diff[Tin_H_l1 * Tatt_dim_l1];

PI_L1 float l0_v[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_v_diff[Tin_H_l1 * Tatt_dim_l1];

PI_L1 float l0_att_map[Tin_H_l1 * Tatt_dim_l1];
PI_L1 float l0_att_map_diff[Tin_H_l1 * Tatt_dim_l1];

PI_L1 float l0_softmax_buffer[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];
PI_L1 float l0_softmax_buffer_diff[Tin_H_l1 * Tin_H_l1 * Tn_heads_l1];

PI_L1 float l0_out[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_out_diff[Tin_H_l1 * Tin_W_l1];

PI_L1 float l0_temp[Ttemp_max];

// Buffer containing the pre-softmax head buffer gradient, necessary in the backward process
PI_L1 float l0_grad[Tin_H_l1 * Tin_H_l1];

PI_L1 float l0_sums[Tin_H_l1];
PI_L1 float l0_maxes[Tin_H_l1];
#endif


// ~~~~~~~~~~~~~~~~~~~~ INITIALIZATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~
#ifdef FORWARD

static inline void tensor_init() {
    printf("Initializing the things\n");
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in[i] = INPUT[i];

    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++) l0_ker_in_q[i] = INPUT_WEIGHTS_Q[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++) l0_ker_in_k[i] = INPUT_WEIGHTS_K[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++) l0_ker_in_v[i] = INPUT_WEIGHTS_V[i];

    for (int i = 0; i < Tatt_dim_l1; i++) l0_bias_in_q[i] = INPUT_BIASES_Q[i];
    for (int i = 0; i < Tatt_dim_l1; i++) l0_bias_in_k[i] = INPUT_BIASES_K[i];
    for (int i = 0; i < Tatt_dim_l1; i++) l0_bias_in_v[i] = INPUT_BIASES_V[i];

    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++) l0_ker_out[i] = OUTPUT_WEIGHTS[i];
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_out[i] = zero_init;
    for (int i = 0; i < Ttemp_max; i++) l0_temp[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_q[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_k[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_v[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++) l0_att_map[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++) l0_softmax_buffer[i] = zero_init;
    for (int i = 0; i < Tin_H_l1; i++) l0_sums[i] = zero_init;
    for (int i = 0; i < Tin_H_l1; i++) l0_maxes[i] = min_float;
    printf("Finished initializing the things\n");
}


static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;

    layer0_wgt_in_q.data = l0_ker_in_q;
    layer0_wgt_in_q.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_q.H = Tatt_dim_l1;
    layer0_wgt_in_q.W = Tin_W_l1;
    layer0_wgt_in_q.C = Tout_C_l1;

    layer0_bias_in_q.data = l0_bias_in_q;
    layer0_bias_in_q.dim = Tatt_dim_l1;
    layer0_bias_in_q.H = 1;
    layer0_bias_in_q.W = Tatt_dim_l1;
    layer0_bias_in_q.C = Tout_C_l1;

    layer0_wgt_in_k.data = l0_ker_in_k;
    layer0_wgt_in_k.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_k.H = Tatt_dim_l1;
    layer0_wgt_in_k.W = Tin_W_l1;
    layer0_wgt_in_k.C = Tout_C_l1;

    layer0_bias_in_k.data = l0_bias_in_k;
    layer0_bias_in_k.dim = Tatt_dim_l1;
    layer0_bias_in_k.H = 1;
    layer0_bias_in_k.W = Tatt_dim_l1;
    layer0_bias_in_k.C = Tout_C_l1;

    layer0_wgt_in_v.data = l0_ker_in_v;
    layer0_wgt_in_v.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_v.H = Tatt_dim_l1;
    layer0_wgt_in_v.W = Tin_W_l1;
    layer0_wgt_in_v.C = Tout_C_l1;

    layer0_bias_in_v.data = l0_bias_in_v;
    layer0_bias_in_v.dim = Tatt_dim_l1;
    layer0_bias_in_v.H = 1;
    layer0_bias_in_v.W = Tatt_dim_l1;
    layer0_bias_in_v.C = Tout_C_l1;

    layer0_wgt_out.data = l0_ker_out;
    layer0_wgt_out.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_out.H = Tin_W_l1;
    layer0_wgt_out.W = Tatt_dim_l1;
    layer0_wgt_out.C = Tin_C_l1;

    layer0_q.data = l0_q;
    layer0_q.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_q.H = Tin_H_l1;
    layer0_q.W = Tatt_dim_l1;
    layer0_q.C = Tin_C_l1;

    layer0_k.data = l0_k;
    layer0_k.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_k.H = Tin_H_l1;
    layer0_k.W = Tatt_dim_l1;
    layer0_k.C = Tin_C_l1;

    layer0_v.data = l0_v;
    layer0_v.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_v.H = Tin_H_l1;
    layer0_v.W = Tatt_dim_l1;
    layer0_v.C = Tin_C_l1;

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

    layer0_softmax_buffer.data = l0_softmax_buffer;
    layer0_softmax_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_softmax_buffer.H = Tn_heads_l1;
    layer0_softmax_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_softmax_buffer.C = Tin_C_l1;

    mhsa_args.input = &layer0_in;
    mhsa_args.n_heads = Tn_heads_l1;
    mhsa_args.q = &layer0_q;
    mhsa_args.k = &layer0_k;
    mhsa_args.v = &layer0_v;
    mhsa_args.output = &layer0_out;

    mhsa_args.coeff_in_q = &layer0_wgt_in_q;
    mhsa_args.coeff_in_k = &layer0_wgt_in_k;
    mhsa_args.coeff_in_v = &layer0_wgt_in_v;

    mhsa_args.bias_in_q = &layer0_bias_in_q;
    mhsa_args.bias_in_k = &layer0_bias_in_k;
    mhsa_args.bias_in_v = &layer0_bias_in_v;

    mhsa_args.coeff_out = &layer0_wgt_out;
    mhsa_args.attention_map = &layer0_att_map;
    mhsa_args.softmax_buffer = &layer0_softmax_buffer;
    mhsa_args.temp_buffer = l0_temp;
    mhsa_args.sums = l0_sums;
    mhsa_args.maxes = l0_maxes;
    mhsa_args.opt_matmul_type_fw = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_wg = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_ig = MATMUL_TYPE;
}


static inline void compute_memory_occupation() {
    // Input
    L1_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Bias input
    L1_memocc_bytes += Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output
    L1_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV
    L1_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * 3 * sizeof(float);
    // Output
    L1_memocc_bytes += Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map
    L1_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Softmax Output
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Temp buffer
    L1_memocc_bytes += Ttemp_max * sizeof(float);
    // sums buffer
    L1_memocc_bytes += Tin_H_l1 * sizeof(float);
    // maxes buffer
    L1_memocc_bytes += Tin_H_l1 * sizeof(float);

    // Input
    L2_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L2_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * 3 * sizeof(float);
    // Bias input
    L2_memocc_bytes += Tatt_dim_l1 * 3 * sizeof(float);
    // Kernel output
    L2_memocc_bytes += Tin_W_l1 * Tatt_dim_l1 * sizeof(float);
    // QKV
    L2_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * 3 * sizeof(float);
    // Output
    L2_memocc_bytes += Tin_W_l1 * Tin_H_l1 * sizeof(float);
    // Attention Map
    L2_memocc_bytes += Tatt_dim_l1 * Tin_H_l1 * sizeof(float);
    // Heads Softmax Output
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Temp buffer
    L2_memocc_bytes += Ttemp_max * sizeof(float);
    // sums buffer
    L2_memocc_bytes += Tin_H_l1 * sizeof(float);
    // maxes buffer
    L2_memocc_bytes += Tin_H_l1 * sizeof(float);
}

#endif


#ifdef BACKWARD
static inline void tensor_init() {
    // Backward grad
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++)                   l0_in[i] = INPUT[i];
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++)                   l0_in_diff[i] = zero_init;

    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_q[i] = INPUT_WEIGHTS_Q[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_k[i] = INPUT_WEIGHTS_K[i];
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_v[i] = INPUT_WEIGHTS_V[i];

    for (int i = 0; i < Tatt_dim_l1; i++)                           l0_bias_in_q[i] = INPUT_BIASES_Q[i];
    for (int i = 0; i < Tatt_dim_l1; i++)                           l0_bias_in_k[i] = INPUT_BIASES_K[i];
    for (int i = 0; i < Tatt_dim_l1; i++)                           l0_bias_in_v[i] = INPUT_BIASES_V[i];

    // Initialization to zero, then it is overwritten the result in the function
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_q_diff[i] = zero_init;
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_k_diff[i] = zero_init;
    for (int i = 0; i < Tin_W_l1 * Tatt_dim_l1; i++)                l0_ker_in_v_diff[i] = zero_init;

    for (int i = 0; i < Tatt_dim_l1 * Tin_W_l1; i++)                l0_ker_out[i] = OUTPUT_WEIGHTS[i];
    for (int i = 0; i < Tatt_dim_l1 * Tin_W_l1; i++)                l0_ker_out_diff[i] = zero_init;

    for (int i = 0; i < Tin_W_l1 * Tin_H_l1; i++)                   l0_out[i] = OUTPUT[i];
    for (int i = 0; i < Tin_W_l1 * Tin_H_l1; i++)                   l0_out_diff[i] = OUTPUT_GRAD[i];

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_q[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_q_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_k[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_k_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_v[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_v_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_att_map[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tatt_dim_l1; i++)                l0_att_map_diff[i] = zero_init;

    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++)     l0_softmax_buffer[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1 * Tn_heads_l1; i++)     l0_softmax_buffer_diff[i] = zero_init;

    for (int i = 0; i < Ttemp_max; i++)                             l0_temp[i] = zero_init;
    for (int i = 0; i < Tin_H_l1 * Tin_H_l1; i++)                   l0_grad[i] = zero_init;
    for (int i = 0; i < Tin_H_l1; i++)                              l0_sums[i] = zero_init;
    for (int i = 0; i < Tin_H_l1; i++)                              l0_maxes[i] = min_float;
}


static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;
    layer0_in.diff = l0_in_diff;

    layer0_wgt_in_q.data = l0_ker_in_q;
    layer0_wgt_in_q.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_q.H = Tatt_dim_l1;
    layer0_wgt_in_q.W = Tin_W_l1;
    layer0_wgt_in_q.C = Tout_C_l1;
    layer0_wgt_in_q.diff = l0_ker_in_q_diff;

    layer0_bias_in_q.data = l0_bias_in_q;
    layer0_bias_in_q.dim = Tatt_dim_l1;
    layer0_bias_in_q.H = 1;
    layer0_bias_in_q.W = Tatt_dim_l1;
    layer0_bias_in_q.C = Tout_C_l1;

    layer0_wgt_in_k.data = l0_ker_in_k;
    layer0_wgt_in_k.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_k.H = Tatt_dim_l1;
    layer0_wgt_in_k.W = Tin_W_l1;
    layer0_wgt_in_k.C = Tout_C_l1;
    layer0_wgt_in_k.diff = l0_ker_in_k_diff;

    layer0_bias_in_k.data = l0_bias_in_k;
    layer0_bias_in_k.dim = Tatt_dim_l1;
    layer0_bias_in_k.H = 1;
    layer0_bias_in_k.W = Tatt_dim_l1;
    layer0_bias_in_k.C = Tout_C_l1;

    layer0_wgt_in_v.data = l0_ker_in_v;
    layer0_wgt_in_v.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_in_v.H = Tatt_dim_l1;
    layer0_wgt_in_v.W = Tin_W_l1;
    layer0_wgt_in_v.C = Tout_C_l1;
    layer0_wgt_in_v.diff = l0_ker_in_v_diff;

    layer0_bias_in_v.data = l0_bias_in_v;
    layer0_bias_in_v.dim = Tatt_dim_l1;
    layer0_bias_in_v.H = 1;
    layer0_bias_in_v.W = Tatt_dim_l1;
    layer0_bias_in_v.C = Tout_C_l1;

    layer0_wgt_out.data = l0_ker_out;
    layer0_wgt_out.dim = Tin_W_l1 * Tatt_dim_l1;
    layer0_wgt_out.H = Tin_W_l1;
    layer0_wgt_out.W = Tatt_dim_l1;
    layer0_wgt_out.C = Tin_C_l1;
    layer0_wgt_out.diff = l0_ker_out_diff;

    layer0_q.data = l0_q;
    layer0_q.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_q.H = Tin_H_l1;
    layer0_q.W = Tatt_dim_l1;
    layer0_q.C = Tin_C_l1;
    layer0_q.diff = l0_q_diff;

    layer0_k.data = l0_k;
    layer0_k.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_k.H = Tin_H_l1;
    layer0_k.W = Tatt_dim_l1;
    layer0_k.C = Tin_C_l1;
    layer0_k.diff = l0_k_diff;

    layer0_v.data = l0_v;
    layer0_v.dim = Tatt_dim_l1 * Tin_H_l1;
    layer0_v.H = Tin_H_l1;
    layer0_v.W = Tatt_dim_l1;
    layer0_v.C = Tin_C_l1;
    layer0_v.diff = l0_v_diff;

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

    layer0_softmax_buffer.data = l0_softmax_buffer;
    layer0_softmax_buffer.dim = Tin_H_l1 * Tin_H_l1 * Tn_heads_l1;
    layer0_softmax_buffer.H = Tn_heads_l1;
    layer0_softmax_buffer.W = Tin_H_l1 * Tin_H_l1;
    layer0_softmax_buffer.C = Tin_C_l1;
    layer0_softmax_buffer.diff = l0_softmax_buffer_diff;

    mhsa_args.input = &layer0_in;
    mhsa_args.n_heads = Tn_heads_l1;
    mhsa_args.q = &layer0_q;
    mhsa_args.k = &layer0_k;
    mhsa_args.v = &layer0_v;
    mhsa_args.output = &layer0_out;

    mhsa_args.coeff_in_q = &layer0_wgt_in_q;
    mhsa_args.coeff_in_k = &layer0_wgt_in_k;
    mhsa_args.coeff_in_v = &layer0_wgt_in_v;

    mhsa_args.bias_in_q = &layer0_bias_in_q;
    mhsa_args.bias_in_k = &layer0_bias_in_k;
    mhsa_args.bias_in_v = &layer0_bias_in_v;

    mhsa_args.coeff_out = &layer0_wgt_out;
    mhsa_args.attention_map = &layer0_att_map;
    mhsa_args.softmax_buffer = &layer0_softmax_buffer;
    mhsa_args.temp_buffer = l0_temp;
    mhsa_args.sums = l0_sums;
    mhsa_args.maxes = l0_maxes;
    mhsa_args.opt_matmul_type_fw = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_wg = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_ig = MATMUL_TYPE;
    mhsa_args.grad = l0_grad;
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
    // Heads Softmax Output + Grad
    L1_memocc_bytes += 2 * Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Temp buffer
    L1_memocc_bytes += Ttemp_max * sizeof(float);
    // Sums buffer
    L1_memocc_bytes += Tin_H_l1 * sizeof(float);
    // Maxes buffer
    L1_memocc_bytes += Tin_H_l1 * sizeof(float);
    // Gradient buffer
    L1_memocc_bytes += Tin_H_l1 * Tin_H_l1 * sizeof(float);

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
    // Heads Softmax Output + Grad
    L2_memocc_bytes += 2 * Tin_H_l1 * Tin_H_l1 * Tn_heads_l1 * sizeof(float);
    // Sums buffer
    L2_memocc_bytes += Tin_H_l1 * sizeof(float);
    // Maxes buffer
    L2_memocc_bytes += Tin_H_l1 * sizeof(float);
    // Temp buffer
    L2_memocc_bytes += Ttemp_max * sizeof(float);
    // Gradient buffer
    L2_memocc_bytes += Tin_H_l1 * Tin_H_l1 * sizeof(float);
}
#endif


// ~~~~~~~~~~~~~~~~~~~~ UTILITY FUNCTIONS ~~~~~~~~~~~~~~~~~~~~
// Mean error checker
static inline void compare_tensors(float *A, float *B, int length) {
    float mean_err_rel = 0.0f;
    float diff;

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


// ~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTIONS ~~~~~~~~~~~~~~~~~~~~
// Function that performs the actual forward and backward steps
static inline void train() {
    // Start profiling
    pi_perf_conf((1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR) | (1 << PI_PERF_LD) | (1 << PI_PERF_ACTIVE_CYCLES));
    pi_perf_stop();
    pi_perf_reset();
    pi_perf_start();

    // Start forward statistics watcher
#ifdef PROF_FWD
    printf("\nForward stats\n");
    START_STATS();
#endif

    // Perform actual forward step
#ifdef FORWARD
    pulp_mhsa_fp32_fw_cl(&mhsa_args);
#endif

    // Stop forward statistics watcher
#ifdef PROF_FWD
    STOP_STATS();
#endif

#ifdef BACKWARD
    // Perform forward pass
    pulp_mhsa_fp32_fw_cl(&mhsa_args);

    // Check results of forward pass
    printf("\nFORWARD CHECK: \n");
    compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
    check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);

    printf("\nATTENTION SCORE CHECK: \n");
    compare_tensors(l0_att_map, ATTENTION_SCORES, ATTENTION_S_LENGTH);
    check_tensor(l0_att_map, ATTENTION_SCORES, ATTENTION_S_LENGTH);

    // Start backward statistics watcher
#ifdef PROF_BCKWD
    printf("\nBackward stats\n");
    START_STATS();
#endif

    // Perform backward pass
    pulp_mhsa_fp32_bw_cl(&mhsa_args);

    // Stop backward statistics watcher
#ifdef PROF_BCKWD
    STOP_STATS();
#endif
#endif

    // Stop profiling and print performance results
    pi_perf_stop();

    int instr_count = pi_perf_read(PI_PERF_INSTR);
    int cycles_count = pi_perf_read(PI_PERF_CYCLES);
    int load_count = pi_perf_read(PI_PERF_LD);
    int active_cycles_count = pi_perf_read(PI_PERF_ACTIVE_CYCLES);

    printf("\nperformance\n");
    printf("\nCycles count %d \n", cycles_count);
    printf("Instruction Count %d\n", instr_count);
    printf("Active Cycles Count %d\n", active_cycles_count);
    printf("Load Count %d\n", load_count);
    printf("Cycles/Instruction %f\n", (float) cycles_count / instr_count);

    // Perform check of forward results
#ifdef FORWARD
    printf("\nFORWARD CHECK: \n");
    compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
    check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);
#endif

    // Perform check of backward results
#ifdef BACKWARD
    printf("\nFINAL WEIGHTS GRADIENT CHECK: \n");
    printf("\nINPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(l0_ker_in_q_diff, INPUT_WGT_GRAD_Q, G_INPUT_WGT_SIZE);
    compare_tensors(l0_ker_in_k_diff, INPUT_WGT_GRAD_K, G_INPUT_WGT_SIZE);
    compare_tensors(l0_ker_in_v_diff, INPUT_WGT_GRAD_V, G_INPUT_WGT_SIZE);

    check_tensor(l0_ker_in_q_diff, INPUT_WGT_GRAD_Q, G_INPUT_WGT_SIZE);
    check_tensor(l0_ker_in_k_diff, INPUT_WGT_GRAD_K, G_INPUT_WGT_SIZE);
    check_tensor(l0_ker_in_v_diff, INPUT_WGT_GRAD_V, G_INPUT_WGT_SIZE);

    printf("\nOUTPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(l0_ker_out_diff, OUTPUT_WGT_GRAD, G_OUTPUT_WGT_SIZE);
    check_tensor(l0_ker_out_diff, OUTPUT_WGT_GRAD, G_OUTPUT_WGT_SIZE);

    printf("\nINPUT GRADIENT CHECK: \n");
    compare_tensors(l0_in_diff, INPUT_GRAD, G_IN_SIZE);
    check_tensor(l0_in_diff, INPUT_GRAD, G_IN_SIZE);
#endif
}


// This function connects each passage to step the net and perform training
void net_step() {
    // Initialize statistics watchers
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

    // Call memory occupation computation functions
#ifdef MEMOCC_COMP
    compute_memory_occupation();
    printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
    printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
#endif

    // Call initialization functions
    tensor_init();
    connect_blobs();

    // Call the main function to perform the step
    train();

    return;
}
