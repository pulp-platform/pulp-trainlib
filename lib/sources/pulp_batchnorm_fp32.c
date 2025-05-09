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


void pulp_batch_norm_fp32_fw_cl(void *batch_norm_args) {
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

    float *input = bn_args->input->data;
    float *output = bn_args->output->data;

    float *weight = bn_args->weight_data;
    float *bias = bn_args->bias_data;

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