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
 * Activation functions, both FW and BW
 **/

/**
 * @brief Forward pass function.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void stm32_relu_fp32_fw(
    struct blob * input,
    struct blob * output
);



/**
 * @brief Bakcward pass function.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void stm32_relu_fp32_bw(
    struct blob * input,
    struct blob * output
);



/**
 * @brief Forward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void stm32_softmax_fp32_fw(
    struct blob * input,
    struct blob * output
);

/**
 * @brief Bakcward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void stm32_softmax_fp32_bw(
    struct blob * input,
    struct blob * output
);