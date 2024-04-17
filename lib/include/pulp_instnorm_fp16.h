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


/**
 * Instance Norm layer configuration structure
 */

/**
 * @brief Structure for Instance Norm Training in FP32
 * @param input input feauture maps for the depthwise layer
 * @param output output feature maps for the depthwise layer
 * @param coeff coefficients to compute normalization, bias are included
 * @param running_mean array of running means computed during the forward step
 * @param running_var array of running variances computed during the forward step
 * @param running_stdev array of running standard deviations computed during the forward step
 * @param freeze_running_params if 1, freezes running mean and variance
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 */
struct InstNorm_args_fp16 {
	struct blob_fp16 * input;
	struct blob_fp16 * output; 
	struct blob_fp16 * coeff;
	fp16 * running_mean;
	fp16 * running_var;
	fp16 * running_stdev;
	int freeze_running_params;
	int skip_wg_grad;
	int skip_in_grad;
};

/**
 * @brief Forward function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp16_fw_cl( void * InstNorm_args_fp16 );

/**
 * @brief Function that calls both input and param gradient functions
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp16_bw_cl( void * InstNorm_args_fp16 );

/**
 * @brief Backward param gradient function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp16_bw_param_grads_cl( void * InstNorm_args_fp16 );

/**
 * @brief Backward input gradient function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp16_bw_input_grads_cl( void * InstNorm_args_fp16 );

/**
 * @brief Forward backend function parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp16_fw_cl( void * InstNorm_args_fp16 );
/**
 * @brief Backward backend function for input gradients parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp16_bw_input_grads_cl( void * InstNorm_args_fp16 );
/**
 * @brief Backward backend function for parameters gradients parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp16_bw_param_grads_cl( void * InstNorm_args_fp16 );