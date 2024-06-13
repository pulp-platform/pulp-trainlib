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
#include "pulp_train_utils_fp32.h"
#include "pulp_optimizers_fp32.h"


void pulp_gradient_descent_fp32 (void * optim_args) 
{
    struct optim_args * args = (struct optim_args *) optim_args;
    float * __restrict__ weights = args->weights->data; 
    float * __restrict__ weight_grad = args->weights->diff;
    int use_biases = args->use_biases;
    const int wgt_size = args->weights->dim; 
    float lr = args->learning_rate;

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
    
    if (use_biases == 1) {
        float * __restrict__ biases = args->biases->data; 
        float * __restrict__ bias_grad = args->biases->diff;
        const int bias_size = args->biases->dim; 

        #ifdef DEBUG
        printf("\n*** BIASES ***\n");
        for (int i=0; i<bias_size; i++)  printf("%f ", biases[i]);  
        printf("\n*** BIAS GRAD ***\n");
        for (int i=0; i<bias_size; i++)  printf("%f ", bias_grad[i]);
        printf("\n\n");
        #endif

        int blockSize_bias = (bias_size+NUM_CORES-1) / NUM_CORES;
        int start_bias = pi_core_id()*blockSize_bias;
        int stop_bias = start_bias+blockSize_bias > bias_size ? bias_size : start_bias+blockSize_bias;

        for (int i=start_bias; i<stop_bias; i++) 
        {   
            biases[i] -= lr * bias_grad[i];
        }    

        #ifdef DEBUG
        printf("\n*** BIASES ***\n");
        for (int i=0; i<bias_size; i++)  printf("%f ", biases[i]);
        printf("\n\n");
        #endif
    }
}