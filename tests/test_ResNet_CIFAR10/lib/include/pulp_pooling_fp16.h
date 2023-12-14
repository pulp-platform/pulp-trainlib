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
 * Activation functions configuration structure
 */

/**
 * @brief Structure for pooling functions
 * @param input input blob for the function
 * @param output output blob for the function
 * @param Hker vertical size of the pooling kernel
 * @param Wker horizontal size of the pooling kernel
 * @param Hstride controls the vertical stride of the kernel
 * @param Wstride controls the horizontal stride of the kernel
 */
struct pool_args_fp16 {
  struct blob_fp16 * input;
  struct blob_fp16 * output;
  int Hker;
  int Wker;
  int Hstride;
  int Wstride;
};


/**
 * Pooling functions
 **/

/**
 * @brief Forward pass function (parallelize with pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_fw_cl, &args);)
 * @param pool_args pointer to a struct pool_args structure.
*/
void pulp_avgpool_fp16_fw_cl(void * pool_args);

/**
 * @brief Backward pass function (parallelize with pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_bw_cl, &args);)
 * @param pool_args pointer to a struct pool_args structure.
*/
void pulp_avgpool_fp16_bw_cl(void * pool_args);

/**
 * @brief Forward pass function (parallelize with pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_fw_cl, &args);)
 * @param pool_args pointer to a struct pool_args structure.
*/
void pulp_maxpool_fp16_fw_cl(void * pool_args);

/**
 * @brief Backward pass function (parallelize with pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_bw_cl, &args);)
 * @param pool_args pointer to a struct pool_args structure.
*/
void pulp_maxpool_fp16_bw_cl(void * pool_args);