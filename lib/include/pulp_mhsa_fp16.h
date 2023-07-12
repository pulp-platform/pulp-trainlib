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
 * Recurrent layer training functions, grouped into FW and BW
*/

/**
 * Authors: Alberto Dequino
*/ 

#include "pulp_train_defines.h"

/**
 * Multi-Head Self Attention layer configuration structure
 */

/**
 * @brief Structure for MHSA Training in FP32
 * @param input             Input vector for the MHSA layer.
 * @param n_heads           Number of heads the attention operation is divided.
 * @param output            Output vector.
 * @param coeff_in          Weight for input projection.
 * @param coeff_out         Weight for output projection.
 * @param qkv               Query, Key and Values extracted from the input and packed into a single matrix
 * @param attention_map     Output of the MHSA module, pre-projection
 * @param temp_buffer       Support buffer used to save transposed matrices
 * @param grad              Support buffer used when calculating gradients for each computational head during MHSA backprop
 * @param head_buffer       Attention scores for every head
 * 
 */

struct Mhsa_args_fp16 {
    struct blob_fp16 * input;
    int 	n_heads; 
    int opt_matmul_type_fw;
    int opt_matmul_type_wg;
    int opt_matmul_type_ig;
    struct blob_fp16 * output;
    struct blob_fp16 * coeff_in;
    struct blob_fp16 * coeff_out;
    struct blob_fp16 * qkv;
    struct blob_fp16 * attention_map;
    fp16 * temp_buffer;
    fp16 * grad;
    struct blob_fp16 * head_buffer;
    struct blob_fp16 * softmax_buffer;
};




/**
 * MHSA layer training functions, grouped into FW and BW
 */

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param Mhsa_args_fp16 structure configuring the MHSA layer.
 */
void pulp_mhsa_fp16_fw_cl(void * Mhsa_args_fp16);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calculate both weight gradient and input gradient.
 * @param Mhsa_args_fp16 structure configuring the MHSA layer.
 */
void pulp_mhsa_fp16_bw_cl(void * Mhsa_args_fp16);
