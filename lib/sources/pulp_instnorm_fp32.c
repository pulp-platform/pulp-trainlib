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
 * Authors: Giacomo Saporetti, Davide Nadalini, Luca Bompani
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

	float * running_mean = IN_args->running_mean;
    float * running_var = IN_args->running_var;
	float * running_stdev = IN_args->running_stdev;
	int freeze_running_params = IN_args->freeze_running_params;

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
        
        if (freeze_running_params == 0) {
            mean=0.0f;
            var=0.0f;
            std=0.0f;
        
            struct mean_std_args mean_std_args;
            mean_std_args.input = in_data;
            mean_std_args.mean = &mean;
            mean_std_args.std = &std;
            mean_std_args.var = &var;
            mean_std_args.dim = D;
            mean_std_args.epsilon = EPSILON; 

            pulp_mean_std_fp32_cl(&mean_std_args);

            running_mean[ch] = mean;
            running_var[ch] = var;
            running_stdev[ch] = std;
        }
        else {
            mean = running_mean[ch];
            var = running_var[ch];
            std = running_stdev[ch];
        }
        
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

	float * running_mean = args->running_mean;
    float * running_var = args->running_var;
	float * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    for (int c=start; c<stop; c++)
    {
        float * in_data  = in->data  + c*D;
        float * out_diff = out->diff + c*D;
        float * in_diff  = in->diff  + c*D;
        float mean; 
        float std;  
        float var; 
        float gamma = coeff->data[c];

        mean = running_mean[c];
        std  = running_stdev[c];
        var  = running_var[c];        

        float grad_i_sum = 0;
        float grad_i_prod = 0;
        for(int i=0; i<D; i++)
        {
            grad_i_sum  -= out_diff[i];
            grad_i_prod -= (in_data[i] - mean) * out_diff[i];
        }
                
        for(int d=0; d<D; d++)
        {
            float grad   = grad_i_sum;
            float mean_d = (in_data[d] - mean) / var;

            grad += grad_i_prod*mean_d;

            grad += D*out_diff[d];
            grad  = grad*gamma/(D*std);

            in_diff[d] = grad;
        }

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

	float * running_mean = args->running_mean;
	float * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    float gamma_grad = 0.0f;
    float bias_grad  = 0.0f;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

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

        mean = running_mean[c];
        std = running_stdev[c];

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