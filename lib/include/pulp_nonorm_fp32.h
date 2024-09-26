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
 * Nonorm training functions, grouped into FW and BW
 *
 * Authors: Alberto Dequino
*/ 


/**
 * Nonorm layer configuration structure
 */

/**
 * @brief Structure for NoNorm Training in FP32
 * @param input  input for the nonorm layer (from forward perspective)
 * @param coeff  weight vector 
 * @param bias  bias 
 * @param output  output for the nonorm layer (from forward perspective)
 */
 struct Nonorm_args {
	struct blob * input; 
	struct blob * coeff; 
	struct blob * bias; 
	struct blob * output;
};

/**
 * Nonorm layer training functions, grouped into FW and BW
*/

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param Nonorm_args structure configuring the nonorm layer.
 */
void pulp_nonorm_fp32_fw_cl( void * Nonorm_args );


