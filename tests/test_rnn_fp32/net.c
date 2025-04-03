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
#include "rnn-grads.h"
#include "rnn-output.h"
#include "stats.h"

#include "step-check.h"
#include "stats.h"

#include "net.h"


// DATA DEFINITION
// RNN
PI_L1 float zero_init = 0.0f;
PI_L1 struct Rnn_args rnn_args;
PI_L1 struct blob layer0_in, layer0_wgt_in, layer0_wgt_h, layer0_state, layer0_out;

// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;


#ifdef FORWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_ker_in[Tin_W_l1 * Tout_W_l1];
PI_L1 float l0_ker_h[Tout_W_l1 * Tout_W_l1];
PI_L1 float l0_state[Tout_W_l1 * Tin_H_l1];
PI_L1 float l0_out[Tout_W_l1 * Tin_H_l1];
#endif


#ifdef BACKWARD
PI_L1 float l0_in[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_in_diff[Tin_H_l1 * Tin_W_l1];
PI_L1 float l0_ker_in[Tin_W_l1 * Tout_W_l1];
PI_L1 float l0_ker_h[Tout_W_l1 * Tout_W_l1];
PI_L1 float l0_ker_in_diff[Tin_W_l1 * Tout_W_l1];
PI_L1 float l0_ker_h_diff[Tout_W_l1 * Tout_W_l1];
PI_L1 float l0_state[Tout_W_l1 * Tin_H_l1];
PI_L1 float l0_out[Tout_W_l1 * Tin_H_l1];
PI_L1 float l0_out_diff[Tout_W_l1 * Tin_H_l1];
// Inaccuracy, dimension should be sequence length (Tin_H_l1) times the longest between Tout_W_l1 and Tin_W_l1
PI_L1 float l0_temp[Tin_H_l1 * Tout_W_l1];
#endif


#ifdef FORWARD
static inline void tensor_init() {
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in[i] = INPUT[i];
    for (int i = 0; i < Tin_W_l1 * Tout_W_l1; i++) l0_ker_in[i] = INPUT_WEIGHTS[i];
    for (int i = 0; i < Tout_W_l1 * Tout_W_l1; i++) l0_ker_h[i] = STATE_WEIGHTS[i];
    for (int i = 0; i < Tout_W_l1 * Tin_H_l1; i++) l0_state[i] = STATE[i];
    for (int i = 0; i < Tout_W_l1 * Tin_H_l1; i++) l0_out[i] = zero_init;
}

static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;

    layer0_wgt_in.data = l0_ker_in;
    layer0_wgt_in.dim = Tin_W_l1 * Tout_W_l1;
    layer0_wgt_in.H = Tin_W_l1;
    layer0_wgt_in.W = Tout_W_l1;
    layer0_wgt_in.C = Tout_C_l1;

    layer0_wgt_h.data = l0_ker_h;
    layer0_wgt_h.dim = Tout_W_l1 * Tout_W_l1;
    layer0_wgt_h.H = Tout_W_l1;
    layer0_wgt_h.W = Tout_W_l1;
    layer0_wgt_h.C = Tout_C_l1;

    layer0_state.data = l0_state;
    layer0_state.dim = Tout_W_l1 * Tin_H_l1;
    layer0_state.H = Tin_H_l1;
    layer0_state.W = Tout_W_l1;
    layer0_state.C = Tout_C_l1;

    layer0_out.data = l0_out;
    layer0_out.dim = Tout_W_l1 * Tin_H_l1;
    layer0_out.H = Tin_H_l1;
    layer0_out.W = Tout_W_l1;
    layer0_out.C = Tout_C_l1;

    rnn_args.input = &layer0_in;
    rnn_args.state = &layer0_state;
    rnn_args.output = &layer0_out;
    rnn_args.coeff_x = &layer0_wgt_in;
    rnn_args.coeff_s = &layer0_wgt_h;
}


static inline void compute_memory_occupation() {
    // Input
    L1_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input
    L1_memocc_bytes += Tin_W_l1 * Tout_W_l1 * sizeof(float);
    // Kernel state
    L1_memocc_bytes += Tout_W_l1 * Tout_W_l1 * sizeof(float);
    //hidden states
    L1_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Output
    L1_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);

    // Input data
    L2_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Weights input
    L2_memocc_bytes += Tin_W_l1 * Tout_W_l1 * sizeof(float);
    // Weights state
    L2_memocc_bytes += Tout_W_l1 * Tout_W_l1 * sizeof(float);
    // States
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Output
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
}
#endif


#ifdef BACKWARD
static inline void tensor_init() {
    // Backward grad
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in[i] = INPUT[i];
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1; i++) l0_in_diff[i] = zero_init;

    for (int i = 0; i < Tin_W_l1 * Tout_W_l1; i++) l0_ker_in[i] = INPUT_WEIGHTS[i];
    // Initialization to zero, then it is overwritten the result in the function
    for (int i = 0; i < Tin_W_l1 * Tout_W_l1; i++) l0_ker_in_diff[i] = zero_init;

    for (int i = 0; i < Tout_W_l1 * Tout_W_l1; i++) l0_ker_h[i] = STATE_WEIGHTS[i];
    for (int i = 0; i < Tout_W_l1 * Tout_W_l1; i++) l0_ker_h_diff[i] = zero_init;

    for (int i = 0; i < Tout_W_l1 * Tin_H_l1; i++) l0_state[i] = STATE[i];

    for (int i = 0; i < Tout_W_l1 * Tin_H_l1; i++) l0_out_diff[i] = OUTPUT_GRAD[i];
    for (int i = 0; i < Tout_W_l1 * Tin_H_l1; i++) l0_out[i] = OUTPUT[i];

    for (int i = 0; i < Tin_H_l1 * Tout_W_l1; i++) l0_temp[i] = zero_init;
}

static inline void connect_blobs() {
    layer0_in.data = l0_in;
    layer0_in.dim = Tin_H_l1 * Tin_W_l1;
    layer0_in.W = Tin_W_l1;
    layer0_in.H = Tin_H_l1;
    layer0_in.C = Tin_C_l1;
    layer0_in.diff = l0_in_diff;

    layer0_wgt_in.data = l0_ker_in;
    layer0_wgt_in.dim = Tin_W_l1 * Tout_W_l1;
    layer0_wgt_in.H = Tin_W_l1;
    layer0_wgt_in.W = Tout_W_l1;
    layer0_wgt_in.C = Tout_C_l1;
    layer0_wgt_in.diff = l0_ker_in_diff;

    layer0_wgt_h.data = l0_ker_h;
    layer0_wgt_h.dim = Tout_W_l1 * Tout_W_l1;
    layer0_wgt_h.H = Tout_W_l1;
    layer0_wgt_h.W = Tout_W_l1;
    layer0_wgt_h.C = Tout_C_l1;
    layer0_wgt_h.diff = l0_ker_h_diff;

    layer0_state.data = l0_state;
    layer0_state.dim = Tout_W_l1 * Tin_H_l1;
    layer0_state.H = Tin_H_l1;
    layer0_state.W = Tout_W_l1;
    layer0_state.C = Tout_C_l1;

    layer0_out.data = l0_out;
    layer0_out.dim = Tout_W_l1 * Tin_H_l1;
    layer0_out.H = Tin_H_l1;
    layer0_out.W = Tout_W_l1;
    layer0_out.C = Tout_C_l1;
    layer0_out.diff = l0_out_diff;

    rnn_args.input = &layer0_in;
    rnn_args.state = &layer0_state;
    rnn_args.output = &layer0_out;
    rnn_args.coeff_x = &layer0_wgt_in;
    rnn_args.coeff_s = &layer0_wgt_h;
    rnn_args.temp_buffer = l0_temp;
    rnn_args.grad_buffer = l0_out_diff;
}


static inline void compute_memory_occupation() {
    // Input
    L1_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Kernel input grad
    L1_memocc_bytes += Tin_W_l1 * Tout_W_l1 * sizeof(float);
    // Kernel state grad
    L1_memocc_bytes += Tout_W_l1 * Tout_W_l1 * sizeof(float);
    //hidden states
    L1_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Output grad
    L1_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);

    // Input data
    L2_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
    // Weights input
    L2_memocc_bytes += Tin_W_l1 * Tout_W_l1 * sizeof(float);
    // Weights state
    L2_memocc_bytes += Tout_W_l1 * Tout_W_l1 * sizeof(float);
    // States
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Output
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Output gradient
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // Weight input gradient
    L2_memocc_bytes += Tin_W_l1 * Tout_W_l1 * sizeof(float);
    // Weight state gradient
    L2_memocc_bytes += Tout_W_l1 * Tin_H_l1 * sizeof(float);
    // States gradient DON'T THINK I NEED THOSE
    //L2_memocc_bytes += L0_OUT_CH*(RICORS+1)*sizeof(float);
    // Input gradient
    L2_memocc_bytes += Tin_H_l1 * Tin_W_l1 * sizeof(float);
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
    pi_perf_conf((1 << PI_PERF_CYCLES) | (1 << PI_PERF_INSTR) | (1 << PI_PERF_LD) | (1 << PI_PERF_ACTIVE_CYCLES));

    pi_perf_stop();
    pi_perf_reset();
    pi_perf_start();

#ifdef PROF_FWD
    printf("\nForward stats\n");
    START_STATS();
#endif

#ifdef FORWARD
    pulp_rnn_fp32_fw_cl(&rnn_args);
#endif

#ifdef PROF_FWD
    STOP_STATS();
#endif

#ifdef PROF_BCKWD
    printf("\nBackward stats\n");
    START_STATS();
#endif

#ifdef BACKWARD
    pulp_rnn_fp32_bw_cl(&rnn_args);
#endif

#ifdef PROF_BCKWD
    STOP_STATS();
#endif

    pi_perf_stop();

    int instr_count = pi_perf_read(PI_PERF_INSTR);
    int cycles_count = pi_perf_read(PI_PERF_CYCLES);
    int load_count = pi_perf_read(PI_PERF_LD);
    int active_cycles_count = pi_perf_read(PI_PERF_ACTIVE_CYCLES);

    printf("performance");
    printf("\n%d \n", cycles_count);
    printf("%d\n", instr_count);
    printf("%d\n", active_cycles_count);
    printf("%d\n", load_count);
    printf("%f\n", (float) cycles_count / instr_count);

#ifdef FORWARD
    printf("\nFORWARD CHECK: \n");
    compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
    check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);
#endif

#ifdef BACKWARD
    printf("\nFINAL WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(l0_ker_in_diff, IH_WGT_GRAD, G_IH_WGT_SIZE);
    check_tensor(l0_ker_in_diff, IH_WGT_GRAD, G_IH_WGT_SIZE);

    compare_tensors(l0_ker_h_diff, HH_WGT_GRAD, G_HH_WGT_SIZE);
    check_tensor(l0_ker_h_diff, HH_WGT_GRAD, G_HH_WGT_SIZE);

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
