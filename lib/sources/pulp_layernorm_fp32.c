/*
 * Copyright (C) 2024 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Calin Diaconu (calin.diaconu@studio.unibo.it)
 */

#include "math.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_layernorm_fp32.h"

// FORWARD
void pulp_layerNorm_fp32_fw_cl(void *layer_norm_args) {
    /*
     * Adapted from the PyTorch implementation. Final value will be:
     * ((x - mean) / (var + eps)) * weight + bias
     *
     * where:
     *      - x -> a step_size block of the input matrix,
     *                  in order to support a normalized size parameter in the torch implementation that would be
     *                  different from the entire size of the input matrix
     *      - mean -> single float value, computed as the sum of all elements over the number of elements
     *      - var -> single float value, representing the standard-deviation, computed as in the PyTorch implementation
     *                  of LayerNorm: equivalent to torch.var(input, unbiased=False), or torch.var(input, correction=0)
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
    struct LayerNorm_args_fp32 *ln_args = (struct LayerNorm_args_fp32 *) layer_norm_args;

    float *input = ln_args->x;
    float *weight = ln_args->weight;
    float *bias = ln_args->bias;
    float *output = ln_args->output;
    float *eps = ln_args->eps;
    int size = ln_args->size;
    int step_size = ln_args->step_size;

    int current_core = pi_core_id();
    int blockSize = ((size / step_size) + NUM_CORES - 1) / NUM_CORES;
    int start = current_core * blockSize;
    int stop = start + blockSize > size ? size : start + blockSize;

    for (int i = start; i < stop; i++) {
        float temp_sum_1 = 0.0f;
        float temp_sum_2 = 0.0f;

        // OP 1: Extract necessary sums for the final computation
        for (int j = 0; j < step_size; j++) {
            temp_sum_1 += input[i * step_size + j];
            temp_sum_2 += input[i * step_size + j] * input[i * step_size + j];
        }

        // OP AUX: Compute mean and var
        float mean = temp_sum_1 / step_size;
        float sqrt_var = sqrt(((temp_sum_2 - 2 * temp_sum_1 * mean) / step_size) + (mean * mean) + eps[0]);

        // OP 2: Perform operations on the intermediate values to obtain the final result
        for (int j = 0; j < step_size; j++) {
            output[i * step_size + j] = ((input[i * step_size + j] - mean) / sqrt_var) * weight[j] + bias[j];
        }
    }
}
