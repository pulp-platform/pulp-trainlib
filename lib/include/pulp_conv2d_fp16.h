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
 * @param pad layer padding
 * @param i2c_buffer pointer to the im2col buffer
 */
void pulp_conv2d_fp16_fw_cl(
	struct blob_fp16 * input, 
	struct blob_fp16 * coeff, 
	struct blob_fp16 * output, 
	int pad,
	fp16 * i2c_buffer
);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param pad layer padding
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to compute input gradients)
 */
void pulp_conv2d_fp16_bw_cl(
	struct blob_fp16 * input, 
	struct blob_fp16 * coeff, 
	struct blob_fp16 * output, 
	int pad,
	fp16 * i2c_buffer,
	fp16 * bt_buffer
);

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param pad layer padding
 * @param i2c_buffer pointer to the im2col buffer
 */
void pulp_conv2d_fp16_bw_param_grads_cl(
	struct blob_fp16 * input, 
	struct blob_fp16 * coeff, 
	struct blob_fp16 * output, 
	int pad,
	fp16 * i2c_buffer
);

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input input feauture maps for the conv2d layer
 * @param coeff weight matrix 
 * @param output output feature maps for the conv2d layer 
 * @param pad layer padding
 * @param i2c_buffer pointer to the im2col buffer
 * @param bt_buffer pointer to the blocktranspose buffer (to reshape the weights for the in grad step)
 */
void pulp_conv2d_fp16_bw_input_grads_cl(
	struct blob_fp16 * input, 
	struct blob_fp16 * coeff, 
	struct blob_fp16 * output, 
	int pad,
	fp16 * i2c_buffer,
	fp16 * bt_buffer
);
