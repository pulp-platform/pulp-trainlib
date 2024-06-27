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
 * @param bias  bias (Optional). Data is accessed if use_biases == 1.
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager for the forward primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 * @param use_biases flag: use bias (1) or not use bias (0).
 */
struct Linear_args {
	struct blob * input; 
	struct blob * coeff; 
	struct blob * bias; 
	struct blob * output;
	int skip_wg_grad;
	int skip_in_grad;
	int opt_matmul_type_fw;
	int opt_matmul_type_wg;
	int opt_matmul_type_ig;
	int use_biases;
};


/**
 * Linear layer training functions, grouped into FW and BW
*/


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input  input column vector for the linear layer
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp32_fw_cl( void * Linear_args );


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 */
void pulp_linear_fp32_bw_cl( void * Linear_args );

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp32_bw_param_grads_cl( void * Linear_args );

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp32_bw_input_grads_cl( void * Linear_args );



/**
 * INNER KERNELS (MATMUL-BASED)
 */

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input  input column vector for the linear layer
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer
 * @param opt_matmul_type_fw number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp32_fw_cl_kernel( void * Linear_args );

/**
 * @brief Weight gradient pass function, forked on PULP cluster.
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param coeff  weight matrix 
 * @param bias  bias (Optional)
 * @param output  categorical output for the linear layer (from forward perspective)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_linear_fp32_bw_param_grads_cl_kernel( void * Linear_args );
