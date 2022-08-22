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
 * Linear layer training functions, grouped into FW and BW
*/


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function.
 * @param input  input column vector for the linear layer
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void stm32_linear_fp32_fw(
	struct blob * input, 	
	struct blob * coeff, 	
	struct blob * output,
	int opt_matmul_type	
);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 */
void stm32_linear_fp32_bw(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output,
	int skip_in_grad,
	int opt_matmul_type_wg,
	int opt_matmul_type_ig
);

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void stm32_linear_fp32_bw_param_grads(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output,
	int opt_matmul_type
);

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void stm32_linear_fp32_bw_input_grads(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output,
	int opt_matmul_type
);
