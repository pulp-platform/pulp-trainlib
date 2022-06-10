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
 * Pooling functions
 **/

/**
 * @brief Forward pass function.
 * @param input Input for avgpool.
 * @param output Output of avgpool.
*/
void pulp_avgpool_fp32_fw_cl(
    struct blob * input,
    struct blob * output
);

/**
 * @brief Backward pass function.
 * @param input Input for avgpool.
 * @param output Output of avgpool.
*/
void pulp_avgpool_fp32_bw_cl(
    struct blob * input,
    struct blob * output
);