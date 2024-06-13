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
 * Fully-Connected layer configuration structure
 */

/**
 * @brief Structure for Fully-Connected Training in FP32
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager for the forward primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 */
struct Linear_args_fp16 {
	struct blob_fp16 * input; 
	struct blob_fp16 * coeff; 
	struct blob_fp16 * output;
	int skip_wg_grad;
	int skip_in_grad;
	int opt_matmul_type_fw;
	int opt_matmul_type_wg;
	int opt_matmul_type_ig;
};



/**
 * Linear layer training functions, grouped into FW and BW
*/


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input  input column vector for the linear layer
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp16_fw_cl( void * Linear_args_fp16 );


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 */
void pulp_linear_fp16_bw_cl( void * Linear_args_fp16 );

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp16_bw_param_grads_cl( void * Linear_args_fp16 );

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp16_bw_input_grads_cl( void * Linear_args_fp16 );
