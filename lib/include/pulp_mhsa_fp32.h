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
 *
 * Authors: Alberto Dequino, Calin Diaconu
*/ 
#include "pmsis.h"

/**
 * Multi-Head Self Attention layer configuration structure
 */

/**
 * @brief Structure for MHSA Training in FP32
 * @param input             Input vector for the MHSA layer.
 * @param input_bn          Bottlenecked input for the MHSA layer.
 * @param n_heads           Number of heads the attention operation is divided.
 * @param output            Output vector.
 * @param coeff_in_q        Weight for input projection for query.
 * @param coeff_in_k        Weight for input projection for key.
 * @param coeff_in_v        Weight for input projection for value.
 * @param bias_in_q         Bias for input projection for query.
 * @param bias_in_k         Bias for input projection for key.
 * @param bias_in_v         Bias for input projection for value.
 * @param coeff_out         Weight for output projection.
 * @param q                 Query
 * @param k                 Key
 * @param v                 Value
 * @param attention_map     Output of the MHSA module, pre-projection
 * @param temp_buffer       Support buffer used to save transposed matrices
 * @param grad              Support buffer used when calculating gradients for each computational head during MHSA backprop
 * @param head_buffer       Attention scores for every head
 * 
 */

struct Mhsa_args {
    struct blob *input;
    struct blob *input_bn;
    int n_heads;
    int opt_matmul_type_fw;
    int opt_matmul_type_wg;
    int opt_matmul_type_ig;
    struct blob *output;

    struct blob *coeff_in_q;
    struct blob *coeff_in_k;
    struct blob *coeff_in_v;

    struct blob *bias_in_q;
    struct blob *bias_in_k;
    struct blob *bias_in_v;

    struct blob *coeff_out;
    struct blob *bias_out;
    struct blob *q;
    struct blob *k;
    struct blob *v;
    struct blob *attention_map;
    float *temp_buffer;
    float *grad;
    struct blob *head_buffer;
    struct blob *softmax_buffer;
    float *global_max;
    float *partial_exp_sum;
    float *maxes;
    float *sums;
};


struct Tiled_Matmul_Mhsa_args{
    struct matMul_args * mm_args;
    struct mm_manager_args * man_args;
    float* BUFF;
    int tile_h;
    int tile_w;
    int tile_dim;
    int tile_h_p;
    int tile_w_p;
    int tile_dim_p;
    int tile_h_sm;
    int tile_w_sm;
    int tile_dim_sm;
    int tile_h_tr;
    int tile_w_tr;
    int tile_dim_tr;
    int tile_h_attv;
    int tile_w_attv;
    int tile_dim_attv;
    int tile_h_out_tr;
    int tile_w_out_tr;
    int tile_dim_out_tr;
    pi_cl_dma_cmd_t * cmd_store;
    pi_cl_dma_cmd_t * cmd_load;
};


/**
 * MHSA layer training functions, grouped into FW and BW
 */

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster. (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, pulp_mhsa_fp32_fw_cl, &args)).
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_fw_cl(void * Mhsa_args);


/**
 * @brief Forward pass function, forked on PULP cluster, using partial softmax. (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, pulp_mhsa_fp32_fw_cl_2, &args)).
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_fw_cl_2(void * Mhsa_args);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calculate both weight gradient and input gradient.
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_bw_cl(void * Mhsa_args);


// INFERENCE FUNCTIONS
/**
 * @brief Inference function for the mobilebert encoder layer, modified version of the forward function
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_mobilebert_inference_fp32_fw_cl(void* Mhsa_args);

void pulp_mhsa_mobilebert_inference_fp32_bw_cl(void* Mhsa_args);


void tiled_mhsa_fp32(void* Mhsa_args, void* tiled_matmul_mhsa_args);

void tiled_matmul_mhsa(void* matmul_args, void* tiled_matmul_mhsa_args, int projection);

void tiled_transpose_mhsa(void* transpose_args, void* Tiled_matmul_mhsa_args, int projection);