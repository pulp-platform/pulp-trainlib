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
 struct Nonorm_args_fp16 {
	struct blob_fp16 * input; 
	struct blob_fp16 * coeff; 
	struct blob_fp16 * bias; 
	struct blob_fp16 * output;
};

/**
 * Nonorm layer training functions, grouped into FW and BW
*/

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param Nonorm_args_fp16 structure configuring the nonorm layer.
 */
void pulp_nonorm_fp16_fw_cl( void * Nonorm_args );


