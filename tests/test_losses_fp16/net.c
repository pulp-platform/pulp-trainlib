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
#include "loss_values.h"


// DATA DEFINITION
#ifndef FLOAT16
#define FLOAT16
#endif

#ifdef FLOAT16
# if LOSS_FN == berHuLoss
PI_L1 struct berHu_loss_args_fp16 loss_args;
# else
PI_L1 struct loss_args_fp16 loss_args;
# endif 
PI_L1 fp16 out[OUT_SIZE];
PI_L1 fp16 out_diff[OUT_SIZE];
PI_L1 fp16 loss = 0;
PI_L1 struct blob_fp16 out_blob;
#endif


void prepare_data ()
{
    for (int i=0; i<OUT_SIZE; i++) 
    {
        out[i] = OUTPUT[i];
        out_diff[i] = 0;
    }

    out_blob.data = out;
    out_blob.diff = out_diff;
    out_blob.dim = OUT_SIZE;

    loss_args.output = &out_blob;
    loss_args.target = LABEL;
    loss_args.wr_loss = &loss;
    # if LOSS_FN == berHuLoss
    loss_args.alpha = 0.2;
    # endif
}


void compute_loss ()
{
    #if LOSS_FN == L1Loss
    pulp_L1Loss_fp16(&loss_args);
    pulp_L1Loss_backward_fp16(&loss_args);
    #elif LOSS_FN == MSE
    pulp_MSELoss_fp16(&loss_args);
    pulp_MSELoss_backward_fp16(&loss_args);
    #elif LOSS_FN == CrossEntropy
    pulp_CrossEntropyLoss_fp16(&loss_args);
    pulp_CrossEntropyLoss_backward_fp16(&loss_args);
    #elif LOSS_FN == berHuLoss
    pulp_berHuLoss_fp16(&loss_args);
    pulp_berHuLoss_backward_fp16(&loss_args);
    #else 
    printf("\nInvalid Loss Function selection!!\n");
    #endif
}


void print_tensors () 
{
    printf("\nOutput:\n");
    for (int i=0; i<OUT_SIZE; i++) printf("%f ", out[i]);
    printf("\n\nOutput grad:\n");
    for (int i=0; i<OUT_SIZE; i++) printf("%f ", out_diff[i]);
    printf("\n\nLoss:\n");
    printf("%f\n", loss);
}



void net_step () {

    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    #ifdef FLOAT32
    printf("Data type is float16.\nOUT_SIZE = %d.\n", OUT_SIZE);
    #endif

    prepare_data();

    #ifdef PROF_NET
    START_STATS();
    #endif

    //print_tensors();

    compute_loss();

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    print_tensors();
    printf("\nEvaluating loss type %d\n", LOSS_FN);
    printf("\nLoss is %f, expected loss is %f\n", loss, LOSS);
    printf("\nChecking output..\n");
    verify_tensor_fp16(out, OUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    printf("Check complete.");
    printf("\nChecking out diff..\n");
    verify_tensor_fp16(out_diff, OUTPUT_GRAD, OUT_SIZE, ERROR_TOLERANCE);
    printf("Check complete.");

    return;
}