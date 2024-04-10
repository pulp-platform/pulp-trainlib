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

#include "pulp_train_defines.h"

/**
 * Depthwise layer configuration structure
 */

/**
 * @brief Structure for Depthwise Convolution Training in FP32
 * @param input input feauture maps for the depthwise layer
 * @param coeff weight matrix 
 * @param output output feature maps for the depthwise layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param HWC tells the DW Convolution if the input/output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 */
struct DepthWise_Conv_args_fp16 {
	struct blob_fp16 * input;
	struct blob_fp16 * coeff; 
	struct blob_fp16 * output; 
	int Lpad;
	int Rpad;
	int Upad;
	int Dpad;
	int skip_wg_grad;
	int skip_in_grad;
	int HWC;
};



/**
 * Depthwise layer training functions, grouped into FW and BW
 */


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input input feauture maps for the depthwise layer
 * @param coeff weight matrix 
 * @param output output feature maps for the depthwise layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param HWC tells the DW Convolution if the input tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 */
void pulp_conv_dw_fp16_fw_cl( void * DepthWise_Conv_args_fp16 );


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input input feauture maps for the depthwise layer
 * @param coeff weight matrix 
 * @param output output feature maps for the depthwise layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param skip_wg_grad skips the computation of the weight grad
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 * @param HWC tells the DW Convolution if the input/output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 */
void pulp_conv_dw_fp16_bw_cl( void * DepthWise_Conv_args_fp16 );

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input input feauture maps for the depthwise layer
 * @param coeff weight matrix 
 * @param output output feature maps for the depthwise layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param HWC tells the DW Convolution if the input tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 */
void pulp_conv_dw_fp16_bw_param_grads_cl( void * DepthWise_Conv_args_fp16 );

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input input feauture maps for the depthwise layer
 * @param coeff weight matrix 
 * @param output output feature maps for the depthwise layer
 * @param Lpad left padding
 * @param Rpad right padding
 * @param Upad upper padding
 * @param Dpad lower padding
 * @param HWC tells the DW Convolution if the output tensor is in CHW layout (HWC=0) or HWC format (HWC=1)
 */
void pulp_conv_dw_fp16_bw_input_grads_cl( void * DepthWise_Conv_args_fp16 );
