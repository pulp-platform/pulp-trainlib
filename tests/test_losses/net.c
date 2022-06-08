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
#ifndef FLOAT32
#define FLOAT32
#endif

#ifdef FLOAT32
PI_L1 float out[OUT_SIZE];
PI_L1 float out_diff[OUT_SIZE];
PI_L1 float loss = 0;
PI_L1 struct blob out_blob;
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
}


void compute_loss ()
{
    pulp_MSELoss(&out_blob, LABEL, &loss);
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
    printf("Data type is float32.\nOUT_SIZE = %d.\n", OUT_SIZE);
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
    printf("\nLoss is %f, expected loss is %f\n", loss, LOSS);
    printf("\nChecking output..\n");
    verify_tensor(out, OUTPUT, OUT_SIZE, ERROR_TOLERANCE);
    printf("Check complete.");
    printf("\nChecking out diff..\n");
    verify_tensor(out_diff, OUTPUT_GRAD, OUT_SIZE, ERROR_TOLERANCE);
    printf("Check complete.");

    return;
}