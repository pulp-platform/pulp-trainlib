/*
 * Copyright (C) 2021-2025 ETH Zurich and University of Bologna
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
 * Authors: Davide Nadalini, Calin Diaconu
*/


/**
 * Instance Norm layer configuration structure
 */

/**
 * @brief Structure for Instance Norm Training in FP32
 * @param input input feauture maps for the depthwise layer
 * @param output output feature maps for the depthwise layer
 * @param coeff coefficients to compute normalization, bias are included
 * @param eps epsilon value to avoid division by zero
 * @param batch_size size of the batch to be processed by the BatchNorm layer
 * @param running_mean array of running means computed during the forward step
 * @param running_var array of running variances computed during the forward step
 * @param running_stdev array of running standard deviations computed during the forward step
 * @param freeze_running_params if 1, freezes running mean and variance
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 */
struct BatchNorm_args_fp32 {
    struct blob *input;
    struct blob *output;
    struct blob *coeff;
    struct blob *bias;
    int batch_size;
    float *running_mean;
    float *running_var;
    float *running_stdev;
    int freeze_running_params;
    int skip_wg_grad;
    int skip_in_grad;

    // float *input_data;
    // float *output_data;

    // Equivalent to gamma and beta
    float *weight_data;
    float *bias_data;

    float *eps;

    int B;
    int C;
    int H;
    int W;
};


/**
 * @brief Forward function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_batch_norm_fp32_fw_cl(void *BatchNorm_args_fp32);


/**
 * @brief Forward backend function parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_batch_norm_parallelized_fp32_fw_cl(void *BatchNorm_args_fp32);
