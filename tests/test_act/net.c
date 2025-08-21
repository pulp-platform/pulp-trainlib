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

#include "net.h"
#include "stats.h"
#include "init_defines.h"
#include "act_data.h"


#if DATA_TYPE == FP32
// Tanh data
PI_L1 struct act_args act_args;
PI_L1 struct leakyrelu_args leakyrelu_args;
PI_L1 struct tanh_args tanh_args;

// ReLU
PI_L1 struct blob relu_in_blob;
PI_L1 struct blob relu_out_blob;
PI_L1 float relu_out[OUT_SIZE];
PI_L1 float relu_out_grad[OUT_SIZE];
PI_L1 float relu_in_grad[IN_SIZE];

// Softmax
PI_L1 struct softmax_args softmax_args;

PI_L1 struct blob softmax_in_blob;
PI_L1 struct blob softmax_out_blob;

PI_L1 float softmax_out[SOFTMAX_OUT_SIZE];

PI_L1 float softmax_out_grad[SOFTMAX_OUT_SIZE];
PI_L1 float softmax_in_grad[SOFTMAX_IN_SIZE];

PI_L1 float softmax_maxes[Tin_H];
PI_L1 float softmax_sums[Tin_H];

// Sigmoid
PI_L1 struct blob sigmoid_in_blob;
PI_L1 struct blob sigmoid_out_blob;
PI_L1 float sigmoid_out[OUT_SIZE];
PI_L1 float sigmoid_out_grad[OUT_SIZE];
PI_L1 float sigmoid_in_grad[IN_SIZE];

// GELU
PI_L1 struct blob gelu_in_blob;
PI_L1 struct blob gelu_out_blob;
PI_L1 float gelu_out[OUT_SIZE];

PI_L1 float tanh_out[OUT_SIZE];

PI_L1 struct blob leakyrelu_in_blob;
PI_L1 struct blob leakyrelu_out_blob;
PI_L1 float leakyrelu_out[OUT_SIZE];
PI_L1 float leakyrelu_out_grad[OUT_SIZE];
PI_L1 float leakyrelu_in_grad[IN_SIZE];

#elif DATA_TYPE == FP16
// Inout data
PI_L1 struct act_args_fp16 act_args;
PI_L1 struct leakyrelu_args_fp16 leakyrelu_args;
PI_L1 struct softmax_args_fp16 softmax_args;
PI_L1 struct tanh_args_fp16 tanh_args;

PI_L1 struct blob_fp16 relu_in_blob;
PI_L1 struct blob_fp16 relu_out_blob;
PI_L1 fp16 relu_out[OUT_SIZE];
PI_L1 fp16 relu_out_grad[OUT_SIZE];
PI_L1 fp16 relu_in_grad[IN_SIZE];

PI_L1 struct blob_fp16 softmax_in_blob;
PI_L1 struct blob_fp16 softmax_out_blob;
PI_L1 fp16 softmax_out[SOFTMAX_OUT_SIZE];
PI_L1 fp16 softmax_out_grad[SOFTMAX_OUT_SIZE];
PI_L1 fp16 softmax_in_grad[SOFTMAX_IN_SIZE];
PI_L1 fp16 softmax_maxes[Tin_H];
PI_L1 fp16 softmax_sums[Tin_H];

PI_L1 struct blob_fp16 sigmoid_in_blob;
PI_L1 struct blob_fp16 sigmoid_out_blob;
PI_L1 fp16 sigmoid_out[OUT_SIZE];
PI_L1 fp16 sigmoid_out_grad[OUT_SIZE];
PI_L1 fp16 sigmoid_in_grad[IN_SIZE];

PI_L1 struct blob_fp16 gelu_in_blob;
PI_L1 struct blob_fp16 gelu_out_blob;
PI_L1 fp16 gelu_out[OUT_SIZE];

PI_L1 fp16 tanh_out[OUT_SIZE];

PI_L1 struct blob_fp16 leakyrelu_in_blob;
PI_L1 struct blob_fp16 leakyrelu_out_blob;
PI_L1 fp16 leakyrelu_out[OUT_SIZE];
PI_L1 fp16 leakyrelu_out_grad[OUT_SIZE];
PI_L1 fp16 leakyrelu_in_grad[IN_SIZE];

#else
#endif


void prepare_data() {
    // Initialize to 0
    for (int i = 0; i < OUT_SIZE; i++) {
        relu_out[i] = 0;
        relu_in_grad[i] = 0;

        sigmoid_out[i] = 0;
        sigmoid_in_grad[i] = 0;
    }

    for (int i = 0; i < SOFTMAX_OUT_SIZE; i++) {
        softmax_out[i] = 0;
        softmax_in_grad[i] = 0;
    }

    for (int i = 0; i < Tin_H; i++) {
        softmax_maxes[i] = 0;
        softmax_sums[i] = 0;
    }

    // ReLU args
    relu_in_blob.data = RELUIN;
    relu_in_blob.diff = relu_in_grad;
    relu_in_blob.dim = Tin_C * Tin_H * Tin_W;
    relu_in_blob.H = Tin_H;
    relu_in_blob.W = Tin_W;
    relu_in_blob.C = Tin_C;

    relu_out_blob.data = relu_out;
    relu_out_blob.diff = RELUOUTPUT_GRAD;
    relu_out_blob.dim = Tout_C * Tout_H * Tout_W;
    relu_out_blob.H = Tout_H;
    relu_out_blob.W = Tout_W;
    relu_out_blob.C = Tout_C;

    // Softmax args
    softmax_in_blob.data = SOFTMIN;
    softmax_in_blob.diff = softmax_in_grad;
    softmax_in_blob.dim = Tin_H * Tin_W;
    softmax_in_blob.H = Tin_H;
    softmax_in_blob.W = Tin_W;
    softmax_in_blob.C = Tin_C;

    softmax_out_blob.data = softmax_out;
    softmax_out_blob.diff = SOFTMOUTPUT_GRAD;
    softmax_out_blob.dim = Tout_H * Tout_W;
    softmax_out_blob.H = Tout_H;
    softmax_out_blob.W = Tout_W;
    softmax_out_blob.C = Tout_C;

    // Sigmoid args
    sigmoid_in_blob.data = SIGMOIDIN;
    sigmoid_in_blob.diff = sigmoid_in_grad;
    sigmoid_in_blob.dim = Tin_C * Tin_H * Tin_W;
    sigmoid_in_blob.H = Tin_H;
    sigmoid_in_blob.W = Tin_W;
    sigmoid_in_blob.C = Tin_C;

    sigmoid_out_blob.data = sigmoid_out;
    sigmoid_out_blob.diff = SIGMOIDOUTPUT_GRAD;
    sigmoid_out_blob.dim = Tout_C * Tout_H * Tout_W;
    sigmoid_out_blob.H = Tout_H;
    sigmoid_out_blob.W = Tout_W;
    sigmoid_out_blob.C = Tout_C;


    // GELU args
    gelu_in_blob.data = GELU_IN;
    gelu_in_blob.dim = Tin_C * Tin_H * Tin_W;
    gelu_in_blob.H = Tin_H;
    gelu_in_blob.W = Tin_W;
    gelu_in_blob.C = Tin_C;

    gelu_out_blob.data = gelu_out;
    gelu_out_blob.dim = Tout_C * Tout_H * Tout_W;
    gelu_out_blob.H = Tout_H;
    gelu_out_blob.W = Tout_W;
    gelu_out_blob.C = Tout_C;

    // LeakyReLU args
    leakyrelu_in_blob.data = LEAKYRELUIN;
    leakyrelu_in_blob.diff = leakyrelu_in_grad;
    leakyrelu_in_blob.dim = Tin_C * Tin_H * Tin_W;
    leakyrelu_in_blob.H = Tin_H;
    leakyrelu_in_blob.W = Tin_W;
    leakyrelu_in_blob.C = Tin_C;

    leakyrelu_out_blob.data = leakyrelu_out;
    leakyrelu_out_blob.diff = LEAKYRELUOUTPUT_GRAD;
    leakyrelu_out_blob.dim = Tout_C * Tout_H * Tout_W;
    leakyrelu_out_blob.H = Tout_H;
    leakyrelu_out_blob.W = Tout_W;
    leakyrelu_out_blob.C = Tout_C;
}


void net_step() {
    // Initialize profiler
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

    // Initialize the data
    prepare_data();

    // ~~~~~~~~~~ Verify ReLU activation ~~~~~~~~~~
    printf("\n----- RELU RESULTS -----\n");

    // Prepare ReLU struct
    act_args.input = &relu_in_blob;
    act_args.output = &relu_out_blob;

    // Print statistics for forward pass
#ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
#endif

    // Apply ReLU activation
#if DATA_TYPE == FP32
    pulp_relu_fp32_fw_cl(&act_args);
#elif DATA_TYPE == FP16
    pulp_relu_fp16_fw_cl(&act_args);
#else
#endif

    // Stop the statistics for the forward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check output match
    printf("\nChecking output..\n");
#if DATA_TYPE == FP32
    verify_tensor(relu_out, RELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(relu_out, RELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
#else
#endif

    // Initialize profiler for backward pass
#ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
#endif

    // Compute gradient for ReLU
#if DATA_TYPE == FP32
    pulp_relu_fp32_bw_cl(&act_args);
#elif DATA_TYPE == FP16
    pulp_relu_fp16_bw_cl(&act_args);
#else
#endif

    // Stop statistics for backward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check gradient match
    printf("\nChecking in grad..\n");
#if DATA_TYPE == FP32
    verify_tensor(relu_in_grad, RELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(relu_in_grad, RELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
#else
#endif

    // ~~~~~~~~~~ Verify softmax activation ~~~~~~~~~~
    printf("\n----- SOFTMAX RESULTS -----\n");

    // Prepare softmax struct
    softmax_args.input_data = SOFTMIN;
    softmax_args.input_diff = softmax_in_grad;
    softmax_args.output_data = softmax_out;
    softmax_args.output_diff = SOFTMOUTPUT_GRAD;
    softmax_args.H = Tin_H;
    softmax_args.W = Tin_W;
    softmax_args.maxes = softmax_maxes;
    softmax_args.sums = softmax_sums;

    // Print statistics for forward pass
#ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
#endif

    // Apply softmax activation
#if DATA_TYPE == FP32
    pulp_softmax_fp32_fw_cl(&softmax_args);
#elif DATA_TYPE == FP16
    pulp_softmax_fp16_fw_cl(&softmax_args);
#else
#endif

    // Stop the statistics for the forward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check output match
    printf("\nChecking output..\n");
#if DATA_TYPE == FP32
    verify_tensor(softmax_out, SOFTMOUTPUT, SOFTMAX_OUT_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(softmax_out, SOFTMOUTPUT, SOFTMAX_OUT_SIZE, ERROR_TOLERANCE);
#else
#endif

    // Initialize profiler for backward pass
#ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
#endif

    // Compute gradient for softmax
#if DATA_TYPE == FP32
    pulp_softmax_fp32_bw_cl(&softmax_args);
#elif DATA_TYPE == FP16
    pulp_softmax_fp16_bw_cl(&softmax_args);
#else
#endif

    // Stop statistics for backward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check gradient match
    printf("\nChecking in grad..\n");
#if DATA_TYPE == FP32
    verify_tensor(softmax_in_grad, SOFTMIN_GRAD, SOFTMAX_IN_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(softmax_in_grad, SOFTMIN_GRAD, SOFTMAX_IN_SIZE, ERROR_TOLERANCE);
#else
#endif


    // ~~~~~~~~~~ Verify sigmoid activation ~~~~~~~~~~
    printf("\n----- SIGMOID RESULTS -----\n");

    // Prepare sigmoid struct
    act_args.input = &sigmoid_in_blob;
    act_args.output = &sigmoid_out_blob;

    // Print statistics for forward pass
#ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
#endif

    // Apply sigmoid activation
#if DATA_TYPE == FP32
    pulp_sigmoid_fp32_fw_cl(&act_args);
#elif DATA_TYPE == FP16
    pulp_sigmoid_fp16_fw_cl(&act_args);
#else
#endif

    // Stop the statistics for the forward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check output match
    printf("\nChecking output..\n");
#if DATA_TYPE == FP32
    verify_tensor(sigmoid_out, SIGMOIDOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(sigmoid_out, SIGMOIDOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
#else
#endif

    // Initialize profiler for backward pass
#ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
#endif

    // Compute gradient for softmax
#if DATA_TYPE == FP32
    pulp_sigmoid_fp32_bw_cl(&act_args);
#elif DATA_TYPE == FP16
    pulp_sigmoid_fp16_bw_cl(&act_args);
#else
#endif

    // Stop statistics for backward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check gradient match
    printf("\nChecking in grad..\n");
#if DATA_TYPE == FP32
    verify_tensor(sigmoid_in_grad, SIGMOIDIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
#elif DATA_TYPE == FP16
    verify_tensor_fp16(sigmoid_in_grad, SIGMOIDIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
#else
#endif


    // ~~~~~~~~~~ Verify GELU with tanh approximation activation ~~~~~~~~~~
    printf("\n----- GELU - TANH APPROX - RESULTS -----\n");

    // Prepare GELU struct
    act_args.input = &gelu_in_blob;
    act_args.output = &gelu_out_blob;

    // Print statistics for forward pass
#ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
#endif

    // Apply GELU activation
#if DATA_TYPE == FP32
    pulp_gelu_tanh_approx_fp32_fw_cl(&act_args);
#endif

    // Stop the statistics for the forward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check output match
    printf("\nChecking output..\n");
#if DATA_TYPE == FP32
    verify_tensor(gelu_out, GELU_OUTPUT, OUT_SIZE, GELU_TANH_APPROX_ERROR_TOLERANCE);
#endif

    // ~~~~~~~~~~ Verify tanh activation ~~~~~~~~~~
    printf("\n----- TANH RESULTS -----\n");

    // Prepare TANH struct
    tanh_args.input = TANH_IN;
    tanh_args.output = tanh_out;
    tanh_args.dim = OUT_SIZE;

    // Print statistics for forward pass
#ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
#endif

    // Apply TANH activation
#if DATA_TYPE == FP32
    pi_cl_team_fork(NUM_CORES, tanh_prll, &tanh_args);
#endif

    // Stop the statistics for the forward pass
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Check output match
    printf("\nChecking output..\n");
#if DATA_TYPE == FP32
    verify_tensor(tanh_out, TANH_OUTPUT, OUT_SIZE, TANH_ERROR_TOLERANCE);
#endif

    // ~~~~~~~~~~ Verify LeakyReLU activation ~~~~~~~~~~
    printf("\n----- LEAKYRELU RESULTS -----\n");

    // Prepare ReLU struct
    leakyrelu_args.input = &leakyrelu_in_blob;
    leakyrelu_args.output = &leakyrelu_out_blob;
    leakyrelu_args.negative_slope = 0.01;

    // Print statistics for forward pass
    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    // Apply ReLU activation
    #if DATA_TYPE == FP32
    pulp_leakyrelu_fp32_fw_cl(&leakyrelu_args);
    #elif DATA_TYPE == FP16
    pulp_leakyrelu_fp16_fw_cl(&leakyrelu_args);
    #else
    #endif

    // Stop the statistics for the forward pass
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    // Check output match
    printf("\nChecking output..\n");
    #if DATA_TYPE == FP32
    verify_tensor(leakyrelu_out, LEAKYRELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(leakyrelu_out, LEAKYRELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #else
    #endif

    // Initialize profiler for backward pass
    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif

    // Compute gradient for ReLU
    #if DATA_TYPE == FP32
    pulp_leakyrelu_fp32_bw_cl(&leakyrelu_args);
    #elif DATA_TYPE == FP16
    pulp_leakyrelu_fp16_bw_cl(&leakyrelu_args);
    #else
    #endif

    // Stop statistics for backward pass
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    // Check gradient match
    printf("\nChecking in grad..\n");
    #if DATA_TYPE == FP32
    verify_tensor(leakyrelu_in_grad, LEAKYRELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(leakyrelu_in_grad, LEAKYRELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #else
    #endif

    return;
}
