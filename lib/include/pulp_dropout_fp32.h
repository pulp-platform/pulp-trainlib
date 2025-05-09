/*
 * Copyright (C) 2021-2024 ETH Zurich and University of Bologna
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
 * Authors: Alberto Dequino
*/ 

#include <stdint.h>

/**
 * @brief Structure for FP32 dropout
 * @param probability the probability of the single value to be dropped
 * @param input input to apply the dropout
 * @param use_mask flag to choose wheter to do a real dropout or just apply a mask (useful for reproducing GM results)
 * @param mask vector used for masking (requires use_mask==1, and same size of input vector)
 * @param size input/mask vector size
 * @param seed initial seed value
 */
struct dropout_args_fp32{
    float probability;
    float * input;
    int use_mask;
    float * mask;
    int size;
    int seed;
};


/**
 * PULP-TrainLib's definitions
 */


/**
 * @brief FP32 Dropout function
 */
 void pulp_dropout_fp32_cl(void * dropout_args);