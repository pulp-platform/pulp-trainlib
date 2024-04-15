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
 * Authors: Giacomo Saporetti, Davide Nadalini
*/ 

#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_instnorm_fp32.h"
#include "pulp_train_defines.h"
#include <math.h>

void pulp_instnorm_fp32_fw_cl( void * InstNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_fw_cl, InstNorm_args);
}

// Real forward function that parallelize on multicore 
void pulp_instnorm_parallelized_fp32_fw_cl( void * InstNorm_args )
{
    struct InstNorm_args * IN_args = (struct InstNorm_args *) InstNorm_args;

    struct blob * in = IN_args->input;
    struct blob * out = IN_args->output;
    struct blob * coeff = IN_args->coeff;
    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    float * in_data = in->data;
    float * out_data = out->data;
    float mean; 
    float std; 
    float var;

    float gamma = 0.0f; 
    float b = 0.0f;

    for(int ch=start; ch<stop; ch++)
    {
        // Calculate Mean and Standard Deviation
        in_data = in->data + ch*D;
        mean=0.0f;
        std=0.0f;
        
        struct mean_std_args mean_std_args;
        mean_std_args.input = in_data;
        mean_std_args.mean = &mean;
        mean_std_args.std = &std;
        mean_std_args.var = &var;
        mean_std_args.dim = D;
        mean_std_args.epsilon = EPSILON; 

        pulp_mean_std_fp32_cl(&mean_std_args);
        
        // Generate output
        out_data = out->data + ch*D;

        gamma = coeff->data[ch];
        b = coeff->data[C + ch];
    
        gamma = gamma/std;
        
        for(int d=0; d<D; d++) {
            out_data[d] = gamma*(in_data[d] - mean) + b;
        }
    }

    return;
}


void pulp_instnorm_fp32_bw_input_grads_cl( void * InstNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_input_grads_cl, InstNorm_args);
}

void pulp_instnorm_parallelized_fp32_bw_input_grads_cl( void * InstNorm_args )
{
    struct InstNorm_args * args = (struct InstNorm_args *) InstNorm_args;
    struct blob * in = args->input;
    struct blob * out = args->output;
    struct blob * coeff = args->coeff;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (in->dim+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > in->dim ? in->dim : start+blockSize;    
    
    for (int i=start; i<stop; i++) {
        in->diff[i] = out->diff[i];
    }
}

void pulp_instnorm_fp32_bw_param_grads_cl( void * InstNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_param_grads_cl, InstNorm_args);
}

void pulp_instnorm_parallelized_fp32_bw_param_grads_cl( void * InstNorm_args )
{
    struct InstNorm_args * args = (struct InstNorm_args *) InstNorm_args;

    struct blob * in = args->input;
    struct blob * out = args->output;
    struct blob * coeff = args->coeff;

    float gamma_grad; // = 0;
    float bias_grad; // = 0;
    gamma_grad = 0.0f;
    bias_grad = 0.0f;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    float epsilon = EPSILON;

    float * in_data;
    float * out_diff;
    float * in_diff;
    float mean;
    float std; 
    float var;

    for(int c=start; c<stop; c++)
    {
        in_data = in->data + c*D;
        out_diff = out->diff + c*D;
        in_diff = in->diff + c*D;

        mean=0;
        std=0;

        struct mean_std_args mean_std_args;
        mean_std_args.input = in_data;
        mean_std_args.mean = &mean;
        mean_std_args.std = &std;
        mean_std_args.var = &var;
        mean_std_args.dim = D;
        mean_std_args.epsilon = EPSILON; 

        pulp_mean_std_fp32_cl(&mean_std_args);

        gamma_grad = 0;
        bias_grad = 0;

        for(int d=0; d<D; d++)
        {
            gamma_grad += out_diff[d]*(in_data[d] - mean);
            bias_grad += out_diff[d];
        }
        gamma_grad = gamma_grad/std;

        coeff->diff[c] = gamma_grad;
        coeff->diff[C + c] = bias_grad; 
    }
}


void pulp_instnorm_fp32_bw_cl( void * InstNorm_args )
{
    struct InstNorm_args * args = (struct InstNorm_args *) InstNorm_args;
    int skip_wg_grad = args->skip_wg_grad;
    int skip_in_grad = args->skip_in_grad;

    if (skip_wg_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_param_grads_cl, InstNorm_args);
    }

    if(skip_in_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_input_grads_cl, InstNorm_args);
    }
}