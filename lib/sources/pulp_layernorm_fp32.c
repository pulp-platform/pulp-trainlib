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
     *      - x -> input matrix
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
     */
    struct LayerNorm_args_fp32 *ln_args = (struct LayerNorm_args_fp32 *) layer_norm_args;

    float *input = ln_args->x;
    float *weight = ln_args->weight;
    float *bias = ln_args->bias;
    float *output = ln_args->output;
    float *eps = ln_args->eps;
    int size = ln_args->size;

    float temp_sum_1[NUM_CORES] = {0.0f};
    float temp_sum_2[NUM_CORES] = {0.0f};

    // OP 1: Extract necessary sums for the final computation
    struct layer_norm_op_1 ln_op1_args;

    ln_op1_args.input = input;
    ln_op1_args.sum_of_elements = temp_sum_1;
    ln_op1_args.sum_of_squared_elements = temp_sum_2;
    ln_op1_args.size = size;

    pi_cl_team_fork(NUM_CORES, pulp_layer_norm_op_1, &ln_op1_args);

    // OP AUX: Compute mean and var
    float ts_1 = 0.0f;
    float ts_2 = 0.0f;

    for (int i = 0; i < NUM_CORES; i++) {
        ts_1 += temp_sum_1[i];
        ts_2 += temp_sum_2[i];
    }

    float mean[1] = {ts_1 / size};
    float sqrt_var[1] = {sqrt(((ts_2 - 2 * ts_1 * mean[0]) / size) + (mean[0] * mean[0]) + eps[0])};

    // OP 2: Perform operations on the intermediate values to obtain the final result
    struct layer_norm_op_2 ln_op2_args;

    ln_op2_args.input = input;
    ln_op2_args.weight = weight;
    ln_op2_args.bias = bias;
    ln_op2_args.output = output;
    ln_op2_args.mean = mean;
    ln_op2_args.sqrt_var = sqrt_var;
    ln_op2_args.size = size;

    pi_cl_team_fork(NUM_CORES, pulp_layer_norm_op_2, &ln_op2_args);
}
