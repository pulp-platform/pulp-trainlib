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
#include "pool_data.h"

// Inout data
PI_L1 struct pool_args maxargs; 
PI_L1 struct pool_args avgargs;

PI_L1 struct blob maxin_blob;
PI_L1 struct blob maxout_blob;
PI_L1 float maxout[OUT_SIZE];
PI_L1 float maxout_grad[OUT_SIZE];
PI_L1 float maxin_grad[IN_SIZE];

PI_L1 struct blob avgin_blob;
PI_L1 struct blob avgout_blob;
PI_L1 float avgout[OUT_SIZE];
PI_L1 float avgout_grad[OUT_SIZE];
PI_L1 float avgin_grad[IN_SIZE];



void prepare_data ()
{
    for (int i=0; i<OUT_SIZE; i++) 
    {
        maxout[i] = 0;
        maxin_grad[i] = 0;
        avgout[i] = 0;
        avgin_grad[i] = 0;
    }

    // Maxpool args
    maxin_blob.data = MAXIN;
    maxin_blob.diff = maxin_grad;
    maxin_blob.dim = Tin_C*Tin_H*Tin_W;
    maxin_blob.H = Tin_H;
    maxin_blob.W = Tin_W;
    maxin_blob.C = Tin_C;

    maxout_blob.data = maxout;
    maxout_blob.diff = MAXOUTPUT_GRAD;
    maxout_blob.dim = Tout_C*Tout_H*Tout_W;
    maxout_blob.H = Tout_H;
    maxout_blob.W = Tout_W;
    maxout_blob.C = Tout_C;

    maxargs.input = &maxin_blob;
    maxargs.output = &maxout_blob; 
    maxargs.Hker = Tker_H;
    maxargs.Wker = Tker_W;
    maxargs.Hstride = H_STR;
    maxargs.Wstride = W_STR;

    // Avgpool args
    avgin_blob.data = AVGIN;
    avgin_blob.diff = avgin_grad;
    avgin_blob.dim = Tin_C*Tin_H*Tin_W;
    avgin_blob.H = Tin_H;
    avgin_blob.W = Tin_W;
    avgin_blob.C = Tin_C;

    avgout_blob.data = avgout;
    avgout_blob.diff = AVGOUTPUT_GRAD;
    avgout_blob.dim = Tout_C*Tout_H*Tout_W;
    avgout_blob.H = Tout_H;
    avgout_blob.W = Tout_W;
    avgout_blob.C = Tout_C;

    avgargs.input = &avgin_blob;
    avgargs.output = &avgout_blob;
    avgargs.Hker = Tker_H;
    avgargs.Wker = Tker_W;
    avgargs.Hstride = H_STR;
    avgargs.Wstride = W_STR;
}



void net_step () {

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    prepare_data();

    printf("\n----- MAXPOOL RESULTS -----\n");

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &maxargs);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    verify_tensor(maxout, MAXOUTPUT, OUT_SIZE, ERROR_TOLERANCE);

    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &maxargs);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    verify_tensor(maxin_grad, MAXIN_GRAD, IN_SIZE, ERROR_TOLERANCE);





    printf("\n----- AVGPOOL RESULTS -----\n");

    #ifdef PROF_NET
    printf("Forward stats: \n");
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_fw_cl, &avgargs);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking output..\n");
    verify_tensor(avgout, AVGOUTPUT, OUT_SIZE, ERROR_TOLERANCE);

    #ifdef PROF_NET
    printf("\nBackward stats: \n");
    START_STATS();
    #endif
    
    pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_bw_cl, &avgargs);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nChecking in grad..\n");
    verify_tensor(avgin_grad, AVGIN_GRAD, IN_SIZE, ERROR_TOLERANCE);



    return;
}