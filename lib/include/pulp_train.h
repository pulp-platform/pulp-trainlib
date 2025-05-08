/*
 * Copyright (C) 2021-2025 ETH Zurich and University of Bologna
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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
*/


/**
 * =====> GENERAL INCLUDES <=====
 */

#include "pmsis.h"
#include "pulp_train_defines.h"
#include "pulp_random.h"

// FP32 structures
#include "pulp_train_utils_fp32.h"
// FP32 primitives
#include "pulp_act_fp32.h"
#include "pulp_batchnorm_fp32.h"
#include "pulp_conv_dw_fp32.h"
#include "pulp_conv_pw_fp32.h"
#include "pulp_conv2d_fp32.h"
#include "pulp_conv_naive_fp32.h"
#include "pulp_dropout_fp32.h"
#include "pulp_im2col_fp32.h"
#include "pulp_instnorm_fp32.h"
#include "pulp_interpolation_fp32.h"
#include "pulp_linear_fp32.h"
#include "pulp_losses_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_mhsa_fp32.h"
#include "pulp_optimizers_fp32.h"
#include "pulp_pooling_fp32.h"
#include "pulp_residual_fp32.h"
#include "pulp_rnn_fp32.h"
#include "pulp_nonorm_fp32.h"
#include "pulp_transp_conv2d_fp32.h"
#include "pulp_layernorm_fp32.h"


// FP16 structures
#include "pulp_train_utils_fp16.h"
// FP16 primitives
#include "pulp_act_fp16.h"
//#include "pulp_batchnorm_fp16.h"
#include "pulp_conv_dw_fp16.h"
#include "pulp_conv_pw_fp16.h"
#include "pulp_conv2d_fp16.h"
#include "pulp_conv_naive_fp16.h"
#include "pulp_dropout_fp16.h"
#include "pulp_im2col_fp16.h"
#include "pulp_instnorm_fp16.h"
#include "pulp_interpolation_fp16.h"
#include "pulp_linear_fp16.h"
#include "pulp_losses_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_mhsa_fp16.h"
#include "pulp_optimizers_fp16.h"
#include "pulp_pooling_fp16.h"
#include "pulp_residual_fp16.h"
#include "pulp_nonorm_fp16.h"
#include "pulp_transp_conv2d_fp16.h"
#include "pulp_embedding_fp16.h"


