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

// Inout data
PI_L1 struct blob reluin_blob;
PI_L1 struct blob reluout_blob;
PI_L1 float reluout[OUT_SIZE];
PI_L1 float reluout_grad[OUT_SIZE];
PI_L1 float reluin_grad[IN_SIZE];

PI_L1 struct blob softmin_blob;
PI_L1 struct blob softmout_blob;
PI_L1 float softmout[OUT_SIZE];
PI_L1 float softmout_grad[OUT_SIZE];
PI_L1 float softmin_grad[IN_SIZE];



void prepare_data ()
{
    for (int i=0; i<OUT_SIZE; i++) 
    {
        reluout[i] = 0;
        reluin_grad[i] = 0;
        softmout[i] = 0;
        softmin_grad[i] = 0;
    }

    // Maxpool args
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

    // Avgpool args
    softmin_blob.data = SOFTMIN;
    softmin_blob.diff = softmin_grad;
    softmin_blob.dim = Tin_C*Tin_H*Tin_W;
    softmin_blob.H = Tin_H;
    softmin_blob.W = Tin_W;
    softmin_blob.C = Tin_C;

    softmout_blob.data = softmout;
    softmout_blob.diff = SOFTMOUTPUT_GRAD;
    softmout_blob.dim = Tout_C*Tout_H*Tout_W;
    softmout_blob.H = Tout_H;
    softmout_blob.W = Tout_W;
    softmout_blob.C = Tout_C;
}



void net_step () {

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    prepare_data();

    printf("\n----- RELU RESULTS -----\n");

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    pulp_relu_fp32_fw_cl(&reluin_blob, &reluout_blob);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    verify_tensor(reluout, RELUOUTPUT, OUT_SIZE, ERROR_TOLERANCE);

    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    pulp_relu_fp32_bw_cl(&reluin_blob, &reluout_blob);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    verify_tensor(reluin_grad, RELUIN_GRAD, IN_SIZE, ERROR_TOLERANCE);





    printf("\n----- SOFTMAX RESULTS -----\n");

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    pulp_softmax_fp32_fw_cl(&softmin_blob, &softmout_blob);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    verify_tensor(softmout, SOFTMOUTPUT, OUT_SIZE, ERROR_TOLERANCE);

    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    pulp_softmax_fp32_bw_cl(&softmin_blob, &softmout_blob);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    verify_tensor(softmin_grad, SOFTMIN_GRAD, IN_SIZE, ERROR_TOLERANCE);



    return;
}