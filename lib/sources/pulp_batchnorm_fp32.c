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

#include "pmsis.h"
#include "pulp_batchnorm_fp32.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_train_defines.h"
#include <math.h>


void pulp_batchnorm_fp32_fw_cl(void *batch_norm_args) {
    pi_cl_team_fork(NUM_CORES, pulp_batch_norm_parallelized_fp32_fw_cl, batch_norm_args);
}


// Real forward function that parallelize on multicore 
void pulp_batch_norm_parallelized_fp32_fw_cl(void *batch_norm_args) {
    /*
     * Adapted from the PyTorch implementation. Final value will be:
     * ((x - mean) / (var + eps)) * weight + bias
     *
     * where:
     *      - x -> a step_size block of the input matrix,
     *                  in order to support a normalized size parameter in the torch implementation that would be
     *                  different from the entire size of the input matrix
     *      - mean -> single float value, computed on the second dimension,
                        as the sum of all elements over the number of elements
     *      - var -> single float value, representing the standard-deviation, computed as in the PyTorch implementation
     *                  of BatchNorm: equivalent to torch.var(input, unbiased=False), or torch.var(input, correction=0)
     *                  since PyTorch 2.0, and mathematically equivalent to the sum of squares of the difference
     *                  between each element of input and the mean value, all normalized through division by the number
     *                  of elements
     *      - eps -> predetermined small constant to avoid division by 0 error
     *      - weight -> also noted with gamma in the PyTorch documentation, a matrix of the same size as input,
     *                  representing the learnable affine transform element to be multiplied to the intermediate value
     *      - bias -> also noted with beta in the PyTorch documentation, a matrix of the same size as input,
     *                  representing the learnable affine transform element to be added to the intermediate value
     *
     * For computational efficiency, using mathematical expansions and reductions, the var, initially equivalent to:
     *      sum(i from 0 to n)((input_i - mean) ** 2) / n
     * where input_i is the i-th element of the input and n is the number of elements in the input,
     * will be computed as:
     *      (sum(i from 0 to n)(input_i ** 2) - 2 * sum(i from 0 to n)(input_i) * mean) / n + (mean ** 2).
     *
     * This is parallelized such that each worker computes an equal number of consecutive blocks of the input matrix.
     */
    struct BatchNorm_args_fp32 *bn_args = (struct BatchNorm_args_fp32 *) batch_norm_args;

    float *input = bn_args->input;
    float *output = bn_args->output;

    float *weight = bn_args->weight;
    float *bias = bn_args->bias;

    float *eps = bn_args->eps;

    int B = bn_args->B;
    int C = bn_args->C;
    int H = bn_args->H;
    int W = bn_args->W;

    int current_core = pi_core_id();
    int block_size = (C + NUM_CORES - 1) / NUM_CORES;
    int start = current_core * block_size;
    int stop = start + block_size > C ? C : start + block_size;

    int elements_in_block = B * H * W;

    for (int current_c = start; current_c < stop; current_c++) {
        float temp_sum_1 = 0.0f;
        float temp_sum_2 = 0.0f;

        // OP 1: Extract necessary sums for the final computation
        for (int current_b = 0; current_b < B; current_b++) {
            for (int current_hw = 0; current_hw < H * W; current_hw++) {
                int current_idx = current_b * C * H * W + current_c * H * W + current_hw;

                temp_sum_1 += input[current_idx];
                temp_sum_2 += input[current_idx] * input[current_idx];
            }
        }

        // OP AUX: Compute mean and var
        float mean = temp_sum_1 / elements_in_block;
        float sqrt_var = sqrt(((temp_sum_2 - 2 * temp_sum_1 * mean) / elements_in_block) + (mean * mean) + eps[0]);

        // OP 2: Perform operations on the intermediate values to obtain the final result
        for (int current_b = 0; current_b < B; current_b++) {
            for (int current_hw = 0; current_hw < H * W; current_hw++) {
                int current_idx = current_b * C * H * W + current_c * H * W + current_hw;

                output[current_idx] = ((input[current_idx] - mean) / sqrt_var) * weight[current_c] + bias[current_c];
            }
        }
    }

    return;
}


void pulp_batchnorm_fp32_bw_input_grads_cl(void *BatchNorm_args_fp32) {
    pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_input_grads_cl, BatchNorm_args_fp32);
}


void pulp_batchnorm_parallelized_fp32_bw_input_grads_cl(void *BatchNorm_args_fp32) {
    struct BatchNorm_args_fp32 *args = (struct BatchNorm_args_fp32 *) BatchNorm_args_fp32;
    struct blob *in = args->input_blob;
    struct blob *out = args->output_blob;
    struct blob *coeff = args->coeff;

    int batch_size = args->batch_size;

    float *running_mean = args->running_mean;
    float *running_var = args->running_var;
    float *running_stdev = args->running_stdev;
    int freeze_running_params = args->freeze_running_params;

    int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int D = C * H * W;
    int HW = H * W;

    float *gamma = coeff->data;
    float *beta = coeff->data + C;
    float *x = in->data;

    int blockSize = (C + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > C ? C : start + blockSize;

    for (int sample = 0; sample < batch_size; sample++) {
        for (int c = start; c < stop; c++) {
            float *in_data = in->data + c * HW + sample * D;
            float *out_diff = out->diff + c * HW + sample * D;
            float *in_diff = in->diff + c * HW + sample * D;
            float mean;
            float std;
            float var;
            float gamma = coeff->data[c];

            mean = running_mean[c];
            std = running_stdev[c];
            var = running_var[c];

            for (int d = 0; d < HW; d++) {
                float grad = 0;
                float mean_d = (in_data[d] - mean) / var;

                for (int i = 0; i < HW; i++) {
                    grad -= out_diff[i] * (1 + (in_data[i] - mean) * mean_d);
                }
                grad += D * out_diff[d];
                grad = grad * gamma / (D * std);

                in_diff[d] = grad;
            }
        }
    }
}


void pulp_batchnorm_fp32_bw_param_grads_cl(void *BatchNorm_args_fp32) {
    pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_param_grads_cl, BatchNorm_args_fp32);
}


void pulp_batchnorm_parallelized_fp32_bw_param_grads_cl(void *BatchNorm_args_fp32) {
    struct BatchNorm_args_fp32 *args = (struct BatchNorm_args_fp32 *) BatchNorm_args_fp32;

    struct blob *in = args->input_blob;
    struct blob *out = args->output_blob;
    struct blob *coeff = args->coeff;

    int batch_size = args->batch_size;

    float *running_mean = args->running_mean;
    float *running_stdev = args->running_stdev;
    int freeze_running_params = args->freeze_running_params;

    float gamma_grad = 0.0f;
    float bias_grad = 0.0f;

    //int N = in->dim;
    int C = in->C;
    int H = in->H;
    int W = in->W;
    int HW = H * W;
    int D = C * H * W;

    int blockSize = (C + NUM_CORES - 1) / NUM_CORES;
    int start = pi_core_id() * blockSize;
    int stop = start + blockSize > C ? C : start + blockSize;

    float epsilon = EPSILON;

    float *in_data;
    float *out_diff;
    float *in_diff;
    float mean_tmp, mean;
    float std_tmp, std;
    float var_tmp, var;

    for (int ch = start; ch < stop; ch++) {
        for (int sample = 0; sample < batch_size; sample++) {
            in_data = in->data + sample * D + ch * HW;
            out_diff = out->diff + sample * D + ch * HW;
            in_diff = in->diff + sample * D + ch * HW;

            mean = running_mean[ch];
            std = running_stdev[ch];

            gamma_grad = 0;
            bias_grad = 0;

            for (int d = 0; d < D; d++) {
                gamma_grad += out_diff[d] * (in_data[d] - mean);
                bias_grad += out_diff[d];
            }
            gamma_grad = gamma_grad / std;

            coeff->diff[ch] = gamma_grad;
            coeff->diff[C + ch] = bias_grad;
        }
    }
}


void pulp_batchnorm_fp32_bw_cl(void *BatchNorm_args_fp32) {
    struct BatchNorm_args_fp32 *args = (struct BatchNorm_args_fp32 *) BatchNorm_args_fp32;
    int skip_wg_grad = args->skip_wg_grad;
    int skip_in_grad = args->skip_in_grad;

    if (skip_wg_grad == 0) {
        pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_param_grads_cl, BatchNorm_args_fp32);
    }

    if (skip_in_grad == 0) {
        pi_cl_team_fork(NUM_CORES, pulp_batchnorm_parallelized_fp32_bw_input_grads_cl, BatchNorm_args_fp32);
    }
}
