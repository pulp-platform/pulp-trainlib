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
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pulp_train_defines.h"

/**
 * Optimizer configuration structure
 */

/**
 * @brief Parameters for optimizer fucntions for every single layer
 * @param weights blob of the weights (with their gradient inside)
 * @param bias blob of the biases (with their gradient inside)
 * @param learning_rate the learning rate of the optimizer
 * @param use_biases flag: use bias (1) or not use bias (0).
 */
struct optim_args_fp16 {
  struct blob_fp16 * weights;
  struct blob_fp16 * biases;
  fp16 learning_rate;
  int use_biases;
};



/**
 * Optimizers
 **/

/**
 * @brief Gradient descent optimizer for a single layer. Use pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &args) to parallelize.
 * @param optim_args pointer to optim_args structure (see pulp_train_utils_fp32.h) 
 */
void pulp_gradient_descent_fp16(
    void * optim_args
);
