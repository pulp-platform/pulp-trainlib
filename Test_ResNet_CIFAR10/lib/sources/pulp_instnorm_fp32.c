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
 * Authors: Giacomo Saporetti
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
    float mean=0;
    float std=0;
    float var;

    float gamma=0;
    float b=0;

    for(int c=start; c<stop; c++)
    {
        // Calculate Mean and Standard Deviation
        in_data = in->data + c*D;
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
        
        // Generate output
        out_data = out->data + c*D;

        gamma = coeff->data[c];
        b = coeff->data[C + c];
    
        /*struct normalize_args normalize_args; 
        normalize_args.input = in_data;
        normalize_args.output = out_data;
        normalize_args.gamma = gamma;
        normalize_args.bias = b;
        normalize_args.mean = mean;
        normalize_args.std = std;
        normalize_args.dim = D;

        pi_cl_team_fork(NUM_CORES, pulp_normalize_fp32_cl, &normalize_args);*/
        gamma = gamma/std;
        for(int d=0; d<D; d++)
            out_data[d] = gamma*(in_data[d] - mean) + b;

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
    //struct blob * bias = args->bias;
    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = H*W;

    float epsilon = EPSILON;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;
    //int start =0;
    //int stop = C;

    for(int c=start; c<stop; c++)
    {
        float * in_data = in->data + c*D;
        float * out_diff = out->diff + c*D;
        float * in_diff = in->diff + c*D;
        float mean=0;
        float std=0; 
        float var=0;
        float gamma = coeff->data[c];

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


        for(int d=0; d<D; d++)
        {
            float grad = 0;
            for(int i=0; i<D; i++)
            {

                grad -= out_diff[i]*(1 + (in_data[i] - mean)*(in_data[d] - mean)/var);

                //grad += out_diff[i]*(1 + N*(out_data[i])*(out_data[d])/(N - 1));
            }
            grad += D*out_diff[d];
            grad = grad*gamma/(D*std);
            //grad -= out_diff[d]*N;
            //grad = -grad*gamma/(N*(std + epsilon));

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
    //struct blob * bias = args->bias;

    float gamma_grad = 0;
    float bias_grad = 0;

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
    int skip = args->skip_in_grad;
    //pulp_instnorm_fp32_bw_param_grads_cl(args);
    pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_param_grads_cl, InstNorm_args);

    if(skip == 0)
       //pulp_instnorm_fp32_bw_input_grads_cl(args);
        pi_cl_team_fork(NUM_CORES, pulp_instnorm_parallelized_fp32_bw_input_grads_cl, InstNorm_args);
}