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
 * Authors: Davide Nadalini
*/ 

#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_batchnorm_fp32.h"
#include "pulp_train_defines.h"
#include <math.h>

void pulp_batchnorm_fp32_fw_cl( void * BatchNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_fw_cl, BatchNorm_args);
}

// Real forward function that parallelize on multicore 
void pulp_batchnorm_parallelized_fp32_fw_cl( void * BatchNorm_args )
{
    struct BatchNorm_args * args = (struct BatchNorm_args *) BatchNorm_args;

    struct blob * in = args->input;
    struct blob * out = args->output;
    struct blob * coeff = args->coeff;

    int batch_size = args->batch_size;

	float * running_mean = args->running_mean;
    float * running_var = args->running_var;
	float * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int HW = H*W;
    int D = C*H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    float * in_data = in->data;
    float * out_data = out->data;
    float mean_tmp = 0; float mean = 0; 
    float std_tmp = 0;  float std = 0; 
    float var_tmp = 0;  float var = 0;

    float gamma = 0.0f; 
    float b = 0.0f;

    for (int ch=0; ch<C; ch++) {

        mean = 0;
        std  = 0;
        var  = 0;

        for(int sample=0; sample<batch_size; sample++)
        {
            // Calculate Mean and Standard Deviation
            in_data = in->data + sample*D + ch*HW;
            
            if (freeze_running_params == 0) {
                
                mean_tmp = 0.0f;
                var_tmp  = 0.0f;
                std_tmp  = 0.0f;
            
                struct mean_std_args mean_std_args;
                mean_std_args.input = in_data;
                mean_std_args.mean = &mean_tmp;
                mean_std_args.std = &std_tmp;
                mean_std_args.var = &var_tmp;
                mean_std_args.dim = HW;
                mean_std_args.epsilon = EPSILON; 

                pulp_mean_std_fp32_cl(&mean_std_args);

                mean  += mean_tmp;
                var   += var_tmp;
                std   += std_tmp;

                printf("[n=%d, ch=%d] mean = %f, mean_tmp = %f\n", sample, ch, mean, mean_tmp);
                printf("[n=%d, ch=%d] var = %f, var_tmp = %f\n", sample, ch, var, var_tmp);
                printf("[n=%d, ch=%d] std = %f, std_tmp = %f\n", sample, ch, std, std_tmp);
            }
            else {
                mean = running_mean[ch];
                var  = running_var[ch];
                std  = running_stdev[ch];
            }
            
            // Generate output
            out_data = out->data + sample*D + ch*HW;

            gamma = coeff->data[ch];
            b = coeff->data[C + ch];
        
            gamma = gamma/std;
            
            for(int d=0; d<HW; d++) {
                out_data[sample*D + ch*HW + d] = gamma*(in_data[sample*D + ch*HW + d] - mean) + b;
            }
        }
        if (freeze_running_params == 0) {
            running_mean[ch]  = mean;
            running_var[ch]   = var;
            running_stdev[ch] = std;
        }
    }

    return;
}


void pulp_batchnorm_fp32_bw_input_grads_cl( void * BatchNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_input_grads_cl, BatchNorm_args);
}

void pulp_batchnorm_parallelized_fp32_bw_input_grads_cl( void * BatchNorm_args )
{
    struct BatchNorm_args * args = (struct BatchNorm_args *) BatchNorm_args;
    struct blob * in = args->input;
    struct blob * out = args->output;
    struct blob * coeff = args->coeff;

    int batch_size = args->batch_size;

	float * running_mean = args->running_mean;
    float * running_var = args->running_var;
	float * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = C*H*W;
    int HW = H*W;

    float * gamma = coeff->data;
    float * beta = coeff->data + C;
    float * x = in->data;  

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;      
    
    for (int sample=0; sample<batch_size; sample++) {
        for (int c=start; c<stop; c++)
        {
            float * in_data  = in->data  + c*HW + sample*D;
            float * out_diff = out->diff + c*HW + sample*D;
            float * in_diff  = in->diff  + c*HW + sample*D;
            float mean; 
            float std;  
            float var; 
            float gamma = coeff->data[c];

            mean = running_mean[c];
            std  = running_stdev[c];
            var  = running_var[c];        

            for(int d=0; d<HW; d++)
            {
                float grad = 0;
                float mean_d = (in_data[d] - mean) / var;

                for(int i=0; i<HW; i++)
                {
                    grad -= out_diff[i] * (1 + (in_data[i] - mean) * mean_d);
                }
                grad += D*out_diff[d];
                grad = grad*gamma/(D*std);

                in_diff[d] = grad;
            }
        }
    }
}

void pulp_batchnorm_fp32_bw_param_grads_cl( void * BatchNorm_args )
{
    pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_param_grads_cl, BatchNorm_args);
}

void pulp_batchnorm_parallelized_fp32_bw_param_grads_cl( void * BatchNorm_args )
{
    struct BatchNorm_args * args = (struct BatchNorm_args *) BatchNorm_args;

    struct blob * in = args->input;
    struct blob * out = args->output;
    struct blob * coeff = args->coeff;

    int batch_size = args->batch_size;

	float * running_mean = args->running_mean;
	float * running_stdev = args->running_stdev;
	int freeze_running_params = args->freeze_running_params;

    float gamma_grad = 0.0f; 
    float bias_grad  = 0.0f; 

    //int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int HW = H*W;
    int D = C*H*W;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    float epsilon = EPSILON;

    float * in_data;
    float * out_diff;
    float * in_diff;
    float mean_tmp, mean;
    float std_tmp, std; 
    float var_tmp, var;

    for (int ch=start; ch<stop; ch++) {
        for(int sample=0; sample<batch_size; sample++)
        {
            in_data = in->data + sample*D + ch*HW;
            out_diff = out->diff + sample*D  + ch*HW;
            in_diff = in->diff + sample*D  + ch*HW;

            mean = running_mean[ch];
            std = running_stdev[ch];

            gamma_grad = 0;
            bias_grad = 0;

            for(int d=0; d<D; d++)
            {
                gamma_grad += out_diff[d]*(in_data[d] - mean);
                bias_grad += out_diff[d];
            }
            gamma_grad = gamma_grad/std;

            coeff->diff[ch] = gamma_grad;
            coeff->diff[C + ch] = bias_grad; 
        }
    }
}


void pulp_batchnorm_fp32_bw_cl( void * BatchNorm_args )
{
    struct BatchNorm_args * args = (struct BatchNorm_args *) BatchNorm_args;
    int skip_wg_grad = args->skip_wg_grad;
    int skip_in_grad = args->skip_in_grad;

    if (skip_wg_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_param_grads_cl, BatchNorm_args);
    }

    if(skip_in_grad == 0)
    {
        pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_input_grads_cl, BatchNorm_args);
    }
}