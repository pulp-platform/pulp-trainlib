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

/**
 * Optimizers
 **/

/**
 * @brief Gradient descent optimizer for a single layer. Use pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &args) to parallelize.
 * @param void_args pointer to optim_args structure (see pulp_train_utils_fp32.h) 
 */
void pulp_gradient_descent_fp32(
    void * void_args
);
