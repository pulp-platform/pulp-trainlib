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
// Inout data
PI_L1 struct act_args act_args;
PI_L1 struct leakyrelu_args lkrel_args;

PI_L1 struct blob reluin_blob;
PI_L1 struct blob reluout_blob;
PI_L1 float reluout[OUT_SIZE];
PI_L1 float reluout_grad[OUT_SIZE];
PI_L1 float reluin_grad[IN_SIZE];

PI_L1 struct blob lkreluin_blob;
PI_L1 struct blob lkreluout_blob;
PI_L1 float lkreluout[OUT_SIZE];
PI_L1 float lkreluout_grad[OUT_SIZE];
PI_L1 float lkreluin_grad[IN_SIZE];

// PI_L1 struct blob softmin_blob;
// PI_L1 struct blob softmout_blob;
// PI_L1 float softmout[OUT_SIZE];
// PI_L1 float softmout_grad[OUT_SIZE];
// PI_L1 float softmin_grad[IN_SIZE];

PI_L1 struct blob sigmoidin_blob;
PI_L1 struct blob sigmoidout_blob;
PI_L1 float sigmoidout[OUT_SIZE];
PI_L1 float sigmoidout_grad[OUT_SIZE];
PI_L1 float sigmoidin_grad[IN_SIZE];

#elif DATA_TYPE == FP16
// Inout data
PI_L1 struct act_args_fp16 act_args;
PI_L1 struct leakyrelu_args_fp16 lkrel_args;

PI_L1 struct blob_fp16 reluin_blob;
PI_L1 struct blob_fp16 reluout_blob;
PI_L1 fp16 reluout[OUT_SIZE];
PI_L1 fp16 reluout_grad[OUT_SIZE];
PI_L1 fp16 reluin_grad[IN_SIZE];

PI_L1 struct blob_fp16 lkreluin_blob;
PI_L1 struct blob_fp16 lkreluout_blob;
PI_L1 fp16 lkreluout[OUT_SIZE];
PI_L1 fp16 lkreluout_grad[OUT_SIZE];
PI_L1 fp16 lkreluin_grad[IN_SIZE];

// PI_L1 struct blob_fp16 softmin_blob;
// PI_L1 struct blob_fp16 softmout_blob;
// PI_L1 fp16 softmout[OUT_SIZE];
// PI_L1 fp16 softmout_grad[OUT_SIZE];
// PI_L1 fp16 softmin_grad[IN_SIZE];

PI_L1 struct blob_fp16 sigmoidin_blob;
PI_L1 struct blob_fp16 sigmoidout_blob;
PI_L1 fp16 sigmoidout[OUT_SIZE];
PI_L1 fp16 sigmoidout_grad[OUT_SIZE];
PI_L1 fp16 sigmoidin_grad[IN_SIZE];

#else

#endif



void prepare_data ()
{
    for (int i=0; i<OUT_SIZE; i++) 
    {
        reluout[i] = 0;
        reluin_grad[i] = 0;
        lkreluout[i] = 0;
        lkreluin_grad[i] = 0;
        //softmout[i] = 0;
        //softmin_grad[i] = 0;
    }

    // ReLU args
    reluin_blob.data = RELUIN;
    reluin_blob.diff = reluin_grad;
    reluin_blob.dim = Tin_C*Tin_H*Tin_W;
    reluin_blob.H = Tin_H;
    reluin_blob.W = Tin_W;
    reluin_blob.C = Tin_C;

    reluout_blob.data = reluout;
    reluout_blob.diff = RELUOUTPUT_GRAD;
    reluout_blob.dim = Tout_C*Tout_H*Tout_W;
    reluout_blob.H = Tout_H;
    reluout_blob.W = Tout_W;
    reluout_blob.C = Tout_C;

    // LeakyReLU args
    lkreluin_blob.data = LEAKYRELUIN;
    lkreluin_blob.diff = lkreluin_grad;
    lkreluin_blob.dim = Tin_C*Tin_H*Tin_W;
    lkreluin_blob.H = Tin_H;
    lkreluin_blob.W = Tin_W;
    lkreluin_blob.C = Tin_C;

    lkreluout_blob.data = lkreluout;
    lkreluout_blob.diff = LEAKYRELUOUTPUT_GRAD;
    lkreluout_blob.dim = Tout_C*Tout_H*Tout_W;
    lkreluout_blob.H = Tout_H;
    lkreluout_blob.W = Tout_W;
    lkreluout_blob.C = Tout_C;

    // // Softmax args
    // softmin_blob.data = SOFTMIN;
    // softmin_blob.diff = softmin_grad;
    // softmin_blob.dim = Tin_C*Tin_H*Tin_W;
    // softmin_blob.H = Tin_H;
    // softmin_blob.W = Tin_W;
    // softmin_blob.C = Tin_C;

    // softmout_blob.data = softmout;
    // softmout_blob.diff = SOFTMOUTPUT_GRAD;
    // softmout_blob.dim = Tout_C*Tout_H*Tout_W;
    // softmout_blob.H = Tout_H;
    // softmout_blob.W = Tout_W;
    // softmout_blob.C = Tout_C;

    // Sigmoid args
    sigmoidin_blob.data = SIGMOIDIN;
    sigmoidin_blob.diff = sigmoidin_grad;
    sigmoidin_blob.dim = Tin_C*Tin_H*Tin_W;
    sigmoidin_blob.H = Tin_H;
    sigmoidin_blob.W = Tin_W;
    sigmoidin_blob.C = Tin_C;

    sigmoidout_blob.data = sigmoidout;
    sigmoidout_blob.diff = SIGMOIDOUTPUT_GRAD;
    sigmoidout_blob.dim = Tout_C*Tout_H*Tout_W;
    sigmoidout_blob.H = Tout_H;
    sigmoidout_blob.W = Tout_W;
    sigmoidout_blob.C = Tout_C;
}



void net_step () {

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    prepare_data();

    printf("\n----- RELU RESULTS -----\n");

    // Prepare ReLU struct
    act_args.input = &reluin_blob;
    act_args.output = &reluout_blob;

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    #if DATA_TYPE == FP32
    pulp_relu_fp32_fw_cl(&act_args);
    #elif DATA_TYPE == FP16
    pulp_relu_fp16_fw_cl(&act_args);
    #else 

    #endif


    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    #if DATA_TYPE == FP32
    verify_tensor(reluout, RELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(reluout, RELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #else 

    #endif


    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    #if DATA_TYPE == FP32
    pulp_relu_fp32_bw_cl(&act_args);
    #elif DATA_TYPE == FP16
    pulp_relu_fp16_bw_cl(&act_args);
    #else

    #endif


    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    #if DATA_TYPE == FP32
    verify_tensor(reluin_grad, RELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(reluin_grad, RELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #else 

    #endif





    printf("\n----- LEAKY RELU RESULTS -----\n");

    // Prepare ReLU struct
    lkrel_args.input = &lkreluin_blob;
    lkrel_args.output = &lkreluout_blob;
    lkrel_args.negative_slope = 0.01;

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    #if DATA_TYPE == FP32
    pulp_leakyrelu_fp32_fw_cl(&lkrel_args);
    #elif DATA_TYPE == FP16
    pulp_leakyrelu_fp16_fw_cl(&lkrel_args);
    #else 

    #endif


    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    #if DATA_TYPE == FP32
    verify_tensor(lkreluout, LEAKYRELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(lkreluout, LEAKYRELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #else 

    #endif


    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    #if DATA_TYPE == FP32
    pulp_relu_fp32_bw_cl(&lkrel_args);
    #elif DATA_TYPE == FP16
    pulp_relu_fp16_bw_cl(&lkrel_args);
    #else

    #endif


    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    #if DATA_TYPE == FP32
    verify_tensor(lkreluin_grad, LEAKYRELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(lkreluin_grad, LEAKYRELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #else 

    #endif



    // printf("\n----- SOFTMAX RESULTS -----\n");

    // // Prepare ReLU struct
    // act_args.input = &softmin_blob;
    // act_args.output = &softmout_blob;

    // #ifdef PROF_NET
    // printf("Forward stats: \n");
    // START_STATS();
    // #endif

    // #if DATA_TYPE == FP32
    // pulp_softmax_fp32_fw_cl(&act_args);
    // #elif DATA_TYPE == FP16
    // pulp_softmax_fp16_fw_cl(&act_args);
    // #else

    // #endif
    

    // #ifdef PROF_NET
    // STOP_STATS();
    // #endif

    // printf("\nChecking output..\n");
    // #if DATA_TYPE == FP32
    // verify_tensor(softmout, SOFTMOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    // #elif DATA_TYPE == FP16
    // verify_tensor_fp16(softmout, SOFTMOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    // #else

    // #endif


    // #ifdef PROF_NET
    // printf("\nBackward stats: \n");
    // START_STATS();
    // #endif
    
    // #if DATA_TYPE == FP32
    // pulp_softmax_fp32_bw_cl(&act_args);
    // #elif DATA_TYPE == FP16
    // pulp_softmax_fp16_bw_cl(&act_args);
    // #else

    // #endif


    // #ifdef PROF_NET
    // STOP_STATS();
    // #endif

    // printf("\nChecking in grad..\n");
    // #if DATA_TYPE == FP32
    // verify_tensor(softmin_grad, SOFTMIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    // #elif DATA_TYPE == FP16
    // verify_tensor_fp16(softmin_grad, SOFTMIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    // #else 

    // #endif





    printf("\n----- SIGMOID RESULTS -----\n");

    // Prepare sigmoid struct
    act_args.input = &sigmoidin_blob;
    act_args.output = &sigmoidout_blob;

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    #if DATA_TYPE == FP32
    pulp_sigmoid_fp32_fw_cl(&act_args);
    #elif DATA_TYPE == FP16
    pulp_sigmoid_fp16_fw_cl(&act_args);
    #else

    #endif
    

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    #if DATA_TYPE == FP32
    verify_tensor(sigmoidout, SIGMOIDOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(sigmoidout, SIGMOIDOUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    #else

    #endif


    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    #if DATA_TYPE == FP32
    pulp_sigmoid_fp32_bw_cl(&act_args);
    #elif DATA_TYPE == FP16
    pulp_sigmoid_fp16_bw_cl(&act_args);
    #else

    #endif


    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    #if DATA_TYPE == FP32
    verify_tensor(sigmoidin_grad, SIGMOIDIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #elif DATA_TYPE == FP16
    verify_tensor_fp16(sigmoidin_grad, SIGMOIDIN_GRAD, IN_SIZE, ERROR_TOLERANCE);
    #else 

    #endif

    return;
}