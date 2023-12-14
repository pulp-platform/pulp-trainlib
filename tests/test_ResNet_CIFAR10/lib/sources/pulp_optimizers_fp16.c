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

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pmsis.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_optimizers_fp16.h"


void pulp_gradient_descent_fp16 (void * optim_args_fp16) 
{
    struct optim_args_fp16 * args = (struct optim_args_fp16 *) optim_args_fp16;
    fp16 * __restrict__ weights = args->weights->data; 
    fp16 * __restrict__ weight_grad = args->weights->diff;
    const int wgt_size = args->weights->dim; 
    fp16 lr = args->learning_rate;

    #ifdef DEBUG
    printf("\n*** WEIGHTS ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weights[i]);  
    printf("\n*** WEIGHT GRAD ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weight_grad[i]);
    printf("\n\n");
    #endif

    int blockSize = (wgt_size+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > wgt_size ? wgt_size : start+blockSize;

    for (int i=start; i<stop; i++) 
    {   
        weights[i] -= lr * weight_grad[i];
    }    

    #ifdef DEBUG
    printf("\n*** WEIGHTS ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weights[i]);  
    printf("\n\n");
    #endif
}
