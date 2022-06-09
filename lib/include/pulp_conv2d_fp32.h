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
 * Convolutional layer training functions, grouped into FW and BW
 */


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_conv2d_fp32_fw_cl(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output, 
	int Lpad,
	int Rpad,
	int Upad,
	int Dpad,
	int stride_h,
	int stride_w,
	float * i2c_buffer,
	int opt_matmul_type
);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to compute input gradients)
 * @param opt_matmul_type_wg number of the optimizer matmul to be chosen by the mm_manager for the weight gradient primitive (see mm_manager_list.txt)
 * @param opt_matmul_type_ig number of the optimizer matmul to be chosen by the mm_manager for the input gradient primitive (see mm_manager_list.txt)
 */
void pulp_conv2d_fp32_bw_cl(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output, 
	int Lpad,
	int Rpad,
	int Upad,
	int Dpad,
	int stride_h,
	int stride_w,
	float * i2c_buffer,
	float * bt_buffer,
	int skip_in_grad,
	int opt_matmul_type_wg,
	int opt_matmul_type_ig
);

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_conv2d_fp32_bw_param_grads_cl(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output, 
	int Lpad,
	int Rpad,
	int Upad,
	int Dpad,
	int stride_h,
	int stride_w,
	float * i2c_buffer,
	int opt_matmul_type
);

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param stride_w stride in input width
 * @param stride_h stride in input height
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to reshape the weights for the in grad step)
 * @param opt_matmul_type number of the optimizer matmul to be chosen by the mm_manager (see mm_manager_list.txt)
 */
void pulp_conv2d_fp32_bw_input_grads_cl(
	struct blob * input, 
	struct blob * coeff, 
	struct blob * output, 
	int Lpad,
	int Rpad,
	int Upad,
	int Dpad,
	int stride_h,
	int stride_w,
	float * i2c_buffer,
	float * bt_buffer,
	int opt_matmul_type
);
