/*
 * Copyright (C) 2021-2024 ETH Zurich and University of Bologna
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

/**
 * Authors: Alberto Dequino
*/ 

#include "pulp_train_utils_fp32.h"
#include "pulp_nonorm_fp32.h"

void pulp_nonorm_fp32_fw_cl( void * Nonorm_args )
{
    struct Nonorm_args * NN_args = (struct Nonorm_args *) Nonorm_args;
    float *coeffData = NN_args->coeff->data;
    float *biasData = NN_args->bias->data;
    float *outData = NN_args->output->data;  
    float *inputData = NN_args->input->data;
    float temp;

    int N = (NN_args->input)->H; // Sequence Length (we parallelize on this)
    int W = (NN_args->input)->W; // Embedding size

    const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > N ? N : start+blockSize;

    for(uint32_t i = start; i < stop; i++){
        int row = i * W;
        for(uint32_t j = 0; j < W; j++){
            temp = inputData[row + j] * coeffData[j]; 
            outData[row + j] = temp + biasData[j];
        }
    }
}

