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

#include "pulp_train_defines.h"
#include "pmsis.h"
#include <bsp/bsp.h>
#include <bsp/fs/readfs.h>

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

struct Mhsa_args_fp16 {
    struct blob_fp16 *input;
    struct blob_fp16 *input_bn;
    int n_heads;
    int opt_matmul_type_fw;
    int opt_matmul_type_wg;
    int opt_matmul_type_ig;
    struct blob_fp16 *output;

    struct blob_fp16 *coeff_in_q;
    struct blob_fp16 *coeff_in_k;
    struct blob_fp16 *coeff_in_v;

    struct blob_fp16 *bias_in_q;
    struct blob_fp16 *bias_in_k;
    struct blob_fp16 *bias_in_v;

    struct blob_fp16 *coeff_out;
    struct blob_fp16 *bias_out;
    struct blob_fp16 *q;
    struct blob_fp16 *k;
    struct blob_fp16 *v;
    struct blob_fp16 *attention_map;
    fp16 *temp_buffer;
    fp16 *grad;
    struct blob_fp16 *head_buffer;
    struct blob_fp16 *softmax_buffer;
    fp16 *global_max;
    fp16 *partial_exp_sum;
    fp16 *maxes;
    fp16 *sums;
};


struct Tiled_Matmul_Mhsa_args_fp16{
    struct matMul_args_fp16 * mm_args;
    struct mm_manager_args_fp16 * man_args;
    fp16* BUFF;
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


struct Mhsa_args_fp16_db {
    struct blob_fp16 *input;
    int n_heads;
    int n_tiles;
    int opt_matmul_type_fw;
    int opt_matmul_type_wg;
    int opt_matmul_type_ig;
    struct blob_fp16 *output;
    struct blob_fp16 *coeff_in;
    struct blob_fp16 *buff1_a;
    struct blob_fp16 *buff1_b;
    struct blob_fp16 *coeff_out;
    struct blob_fp16 *qkv;
    struct blob_fp16 *buff2_a;
    struct blob_fp16 *buff2_b;
    struct blob_fp16 *attention_map;
    struct blob_fp16 *attention_map_l2;
    fp16 *temp_buffer;
    fp16 *grad;
    struct blob_fp16 *head_buffer;
    struct blob_fp16 *softmax_buffer;
    fp16 *maxes;
    fp16 *sums;
};


// Support struct
struct zero_tensor_args {
    fp16 *tensor;
    int size;
    fp16 zero_init;
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


/**
 * @brief Forward pass function, forked on PULP cluster, with double buffering strategy
 * @param Mhsa_args_fp16 structure configuring the MHSA layer.
 */
// void pulp_mhsa_fp16_fw_cl_dblbuffer(void* Mhsa_args_fp16_db);




// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calculate both weight gradient and input gradient.
 * @param Mhsa_args_fp16 structure configuring the MHSA layer.
 */
void pulp_mhsa_fp16_bw_cl(void * Mhsa_args_fp16);


// SUPPORT FUNCTIONS FOR TILED INFERENCE

void tiled_mhsa_fp16(void* Mhsa_args_fp16, void* tiled_matmul_mhsa_args_fp16);

void tiled_matmul_mhsa_fp16(void* matmul_args_fp16, void* tiled_matmul_mhsa_args_fp16, int projection);

void tiled_transpose_mhsa_fp16(void* transpose_args_fp16, void* Tiled_matmul_mhsa_args_fp16, int projection);

void tiled_mhsa_fp16_flash(void *Mhsa_args, void* Tiled_mhsa_matmul_args, fp16* BUFF_L2, pi_fs_file_t *file_p, int size);

void pulp_mhsa_mobilebert_inference_fp16_fw_cl(void *Mhsa_args);
