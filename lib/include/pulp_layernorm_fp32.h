/*
 * Copyright (C) 2024 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Calin Diaconu (calin.diaconu@studio.unibo.it)
 */

#ifndef PULP_TRAINLIB_PULP_LAYERNORM_FP32_H
#define PULP_TRAINLIB_PULP_LAYERNORM_FP32_H

#include "math.h"

/**
 * @brief Arguments for the forward pass of the LayerNorm layer.
 * @brief x: input tensor
 * @brief weight: weight tensor
 * @brief bias: bias tensor
 * @brief output: output tensor
 * @brief eps: epsilon value
 * @brief size: size of the tensors
 * @brief step_size: step size over which the normalization is performed
 */
struct LayerNorm_args_fp32 {
    float *x;
    float *weight;
    float *bias;
    float *output;
    float *eps;
    int size;
    int step_size;
};

/**
 * @brief Forward function that calls the parallelized version for the LayerNorm layer.
 * @param (void *)  (struct LayerNorm_args_fp32 void_args)
 */
void pulp_layerNorm_fp32_fw_cl(void *layer_norm_args);

#endif //PULP_TRAINLIB_PULP_LAYERNORM_FP32_H
