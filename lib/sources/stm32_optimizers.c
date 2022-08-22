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

#include "stm32_train_utils.h"
#include "stm32_optimizers.h"


void stm32_gradient_descent_fp32 (void * void_args) 
{
    struct optim_args * args = (struct optim_args *) void_args;
    float * __restrict__ weights = args->weights->data; 
    float * __restrict__ weight_grad = args->weights->diff;
    const int wgt_size = args->weights->dim; 
    float lr = args->learning_rate;

    #ifdef DEBUG
    printf("\n*** WEIGHTS ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weights[i]);  
    printf("\n*** WEIGHT GRAD ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weight_grad[i]);
    printf("\n\n");
    #endif

    for (int i=0; i<wgt_size; i++) 
    {   
        weights[i] -= lr * weight_grad[i];
    }    

    #ifdef DEBUG
    printf("\n*** WEIGHTS ***\n");
    for (int i=0; i<wgt_size; i++)  printf("%f ", weights[i]);  
    printf("\n\n");
    #endif
}
