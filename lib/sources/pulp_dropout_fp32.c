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

#include <stdio.h>
#include <stdint.h>
#include "pulp_random.h"
#include "pulp_dropout_fp32.h"
#include "pmsis.h"

void pulp_dropout_fp32_cl(void * dropout_args){
    struct dropout_args_fp32 *args = (struct dropout_args *) dropout_args;
    float prob = args->probability;
    if(prob==0.0f){
        return;
    }
    float scale = 1.0f / (1.0f - prob);
    float* input = args->input;
    int use_mask = args->use_mask;
    float* mask = args->mask;
    int size = args->size;
    int seed = args->seed;

    const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > size ? size : start+blockSize;

    seed = seed + start;

    float rand;

    for (uint32_t i=start; i < stop; i++){
        rand = pulp_generate_float_seed(seed);
        seed++;
        if(use_mask){
            input[i] = mask[i] * input[i];
        }
        else{
            if(rand > prob){
                input[i] = input[i] * scale;
            }
            else{
                input[i] = 0.0f;
            }
        }  
    }
}

