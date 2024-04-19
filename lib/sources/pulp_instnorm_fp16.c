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
#include "pulp_train_utils_fp16.h"
#include "pulp_instnorm_fp16.h"
#include "pulp_train_defines.h"
#include <math.h>

void pulp_instnorm_fp16_fw_cl( void * InstNorm_args_fp16 )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp16_fw_cl, InstNorm_args_fp16);
}

// Real forward function that parallelize on multicore 
void pulp_instnorm_parallelized_fp16_fw_cl( void * InstNorm_args_fp16 )
{
    struct InstNorm_args_fp16 * IN_args = (struct InstNorm_args_fp16 *) InstNorm_args_fp16;

    struct blob_fp16 * in = IN_args->input;
    struct blob_fp16 * out = IN_args->output;
    struct blob_fp16 * coeff = IN_args->coeff;

	fp16 * running_mean = IN_args->running_mean;
    fp16 * running_var = IN_args->running_var;
	fp16 * running_stdev = IN_args->running_stdev;
	int freeze_running_params = IN_args->freeze_running_params;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    fp16 * in_data = in->data;
    fp16 * out_data = out->data;
    fp16 mean; 
    fp16 std; 
    fp16 var;

    fp16 gamma = 0.0f; 
    fp16 b = 0.0f;

    for(int ch=start; ch<stop; ch++)
    {
        // Calculate Mean and Standard Deviation
        in_data = in->data + ch*D;

        if (freeze_running_params == 0) {
            mean=0.0f;
            var=0.0f;
            std=0.0f;
            
            struct mean_std_args_fp16 mean_std_args;
            mean_std_args.input = in_data;
            mean_std_args.mean = &mean;
            mean_std_args.std = &std;
            mean_std_args.var = &var;
            mean_std_args.dim = D;
            mean_std_args.epsilon = EPSILON; 

            pulp_mean_std_fp16_cl(&mean_std_args);

            running_mean[ch] = mean;
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


void pulp_instnorm_fp16_bw_input_grads_cl( void * InstNorm_args_fp16 )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp16_bw_input_grads_cl, InstNorm_args_fp16);
}

void pulp_instnorm_parallelized_fp16_bw_input_grads_cl( void * InstNorm_args_fp16 )
{
    struct InstNorm_args_fp16 * args = (struct InstNorm_args_fp16 *) InstNorm_args_fp16;
    struct blob_fp16 * in = args->input;
    struct blob_fp16 * out = args->output;
    struct blob_fp16 * coeff = args->coeff;

	fp16 * running_mean = args->running_mean;
    fp16 * running_var = args->running_var;
	fp16 * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    // Stabilize numerically
    fp16 grad_scaling = 1e6;
    fp16 grad_scaling_inv = 1 / grad_scaling;

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
        fp16 * in_data  = in->data  + c*D;
        fp16 * out_diff = out->diff + c*D;
        fp16 * in_diff  = in->diff  + c*D;
        fp16 mean; 
        fp16 std;  
        fp16 var; 
        fp16 gamma = coeff->data[c];

        mean = running_mean[c];
        std  = running_stdev[c];
        var  = running_var[c];        

        fp16 grad_i_sum = 0;
        fp16 grad_i_prod = 0;
        for(int i=0; i<D; i++)
        {
            grad_i_sum  -= out_diff[i];
            grad_i_prod -= (in_data[i] - mean) * out_diff[i];
        }
                
        for(int d=0; d<D; d++)
        {
            fp16 grad   = grad_i_sum;
            fp16 mean_d = (in_data[d] - mean) / var;

            grad += grad_i_prod*mean_d;

            grad += D*out_diff[d];
            grad  = grad*gamma/(D*std);

            in_diff[d] = grad;
        }
    }
}

void pulp_instnorm_fp16_bw_param_grads_cl( void * InstNorm_args_fp16 )
{
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp16_bw_param_grads_cl, InstNorm_args_fp16);
}

void pulp_instnorm_parallelized_fp16_bw_param_grads_cl( void * InstNorm_args_fp16 )
{
    struct InstNorm_args_fp16 * args = (struct InstNorm_args_fp16 *) InstNorm_args_fp16;

    struct blob_fp16 * in = args->input;
    struct blob_fp16 * out = args->output;
    struct blob_fp16 * coeff = args->coeff;

	fp16 * running_mean = args->running_mean;
	fp16 * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    fp16 gamma_grad = 0.0f;
    fp16 bias_grad  = 0.0f;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    fp16 * in_data;
    fp16 * out_diff;
    fp16 * in_diff;
    fp16 mean;
    fp16 std; 
    fp16 var;

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


void pulp_instnorm_fp16_bw_cl( void * InstNorm_args_fp16 )
{
    struct InstNorm_args_fp16 * args = (struct InstNorm_args_fp16 *) InstNorm_args_fp16;
    int skip_wg_grad = args->skip_wg_grad;
    int skip_in_grad = args->skip_in_grad;

    if (skip_wg_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp16_bw_param_grads_cl, InstNorm_args_fp16);
    }

    if(skip_in_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp16_bw_input_grads_cl, InstNorm_args_fp16);
    }
}