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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino, Calin Diaconu
*/ 

#include "pulp_train_utils_fp16.h"
#include "pulp_act_fp16.h"
#include "math.h"
#include "pulp_mhsa_fp16.h"


void pulp_sigmoid_fp16_fw_cl( void * act_args )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_fw_fp16, act_args);
}

void pulp_sigmoid_fp16_bw_cl( void * act_args )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_bw_fp16, act_args);
}

void sigmoid_core_fw_fp16( void * act_args )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    fp16 sigma = 0.0f;
    sigma = 1 + expf(-inData[i]);
    sigma = 1 / sigma;
    outData[i] = sigma;
  }
}

void sigmoid_core_bw_fp16( void * act_args )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* inDiff = args->input->diff;
  fp16* outData = args->output->data;
  fp16* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    fp16 sigma = 0.0f;
    fp16 sigma_prime = 0.0f;
    //sigma = 1 + expf(-inData[i]);
    //sigma = 1 / sigma;
    sigma = outData[i];
    sigma_prime = sigma * (1.0f - sigma);
    inDiff[i] = outDiff[i] * sigma_prime;
  }
}



void pulp_relu_fp16_fw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  pi_cl_team_fork(NUM_CORES, relu_core_fw_fp16, act_args_fp16);
}

void pulp_relu_fp16_bw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  pi_cl_team_fork(NUM_CORES, relu_core_bw_fp16, act_args_fp16);
}

void relu_core_fw_fp16( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }

}

void relu_core_bw_fp16( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* inDiff = args->input->diff;
  fp16* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}




void pulp_leakyrelu_fp16_fw_cl( void * leakyrelu_args_fp16 )
{
  struct leakyrelu_args_fp16 * args = (struct leakyrelu_args_fp16 *) leakyrelu_args_fp16;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_fw_fp16, args);
}

void pulp_leakyrelu_fp16_bw_cl( void * leakyrelu_args_fp16 )
{
  struct leakyrelu_args_fp16 * args = (struct leakyrelu_args_fp16 *) leakyrelu_args_fp16;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_bw_fp16, args);
}

void leakyrelu_core_fw_fp16( void * leakyrelu_args_fp16 )
{
  struct leakyrelu_args_fp16 * args = (struct leakyrelu_args_fp16 *) leakyrelu_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;
  fp16 neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : neg_slope*inData[i];
  }

}

void leakyrelu_core_bw_fp16( void * leakyrelu_args_fp16 )
{
  struct leakyrelu_args_fp16 * args = (struct leakyrelu_args_fp16 *) leakyrelu_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* inDiff = args->input->diff;
  fp16* outDiff = args->output->diff;
  fp16 neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : neg_slope*outDiff[i];
  }
}





void pulp_gelu_fp16_fw_cl( void* act_args_fp16)
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++){
    fp16 x = inData[i];
    fp16 halfx = (fp16) 0.5f * x;

    fp16 val = (fp16) (((x * x * x * 0.044715f) + x) * 0.7978f);
    
    fp16 val2 = val * val;
    

    fp16 a = (fp16) ((((val2 + 378.0f) * val2 + 17325.0f) * val2 + 135135.0f) * val);
    fp16 b = (fp16) (((28.0f * val2 + 3150.0f) * val2 + 62370.0f) * val2 + 135135.0f);
    val = (fp16) (a / b);

    if(val > 1)
      val = 1;
    else if(val < -1)
      val = -1;

    val = (fp16) (val * halfx + halfx);



    outData[i] = val;
  }
}


// ~~~~~~~~~~~~~~~~~~~~ SOFTMAX ~~~~~~~~~~~~~~~~~~~~
// Forward pass of the FP16 softmax
// Performs a softmax activation on each row
void pulp_softmax_fp16_fw_cl(void *act_args_fp16) {
    // Extract variables from function arguments
    struct softmax_args_fp16 *args = (struct softmax_args_fp16 *) act_args_fp16;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    fp16 *inData = args->input_data;
    fp16 *outData = args->output_data;

    fp16 *maxes = args->maxes;
    fp16 *sums = args->sums;

    for(int i = 0; i < HEIGHT; i++){
      maxes[i]=-65504.0f;
      sums[i]=0.0f;
    }

    // OP A: Compute the maximum value on each row
    struct max_args_fp16 m_args;
    m_args.input = inData;
    m_args.maxes = maxes;
    m_args.H = HEIGHT;
    m_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_row_max_fp16_cl, &m_args);

    // OP B: For each row, compute the sum of exponential of the difference between input values and the max of the row
    struct exp_sum_args_fp16 e_s_args;
    e_s_args.input = inData;
    e_s_args.output = outData;
    e_s_args.H = HEIGHT;
    e_s_args.W = WIDTH;
    e_s_args.sums = sums;
    e_s_args.maxes = maxes;

    pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp16_cl, &e_s_args);

    // OP C: Per-row division with the sum computed in the previous function
    struct row_div_args_fp16 r_d_args;
    r_d_args.input = outData;
    r_d_args.sums = sums;
    r_d_args.H = HEIGHT;
    r_d_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_row_div_fp16_cl, &r_d_args);
}

void pulp_softmax_fp16_fw_cl_tiled(void *act_args, void* Tiled_matmul_mhsa_args){
  struct softmax_args_fp16 *args = (struct softmax_args_fp16 *) act_args;
  struct Tiled_Matmul_Mhsa_args_fp16 * tiled_args = (struct Tiled_Matmul_Mhsa_args_fp16*) Tiled_matmul_mhsa_args;

  int H = args->H;
  int W = args->W;

  fp16 *maxes = args->maxes;
  fp16 *sums = args->sums;

  int tile_h = tiled_args->tile_h_sm;
  int tile_w = tiled_args->tile_w_sm;
  int tile_dim = tiled_args->tile_dim_sm;
  fp16* BUFF = tiled_args->BUFF;
  pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
  pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;

  int n_tiles_i = H / tile_h;
  int n_tiles_j = W / tile_w;

  fp16* IN_DATA = BUFF;
  fp16* OUT_DATA = BUFF + tile_dim;

  for(int i = 0; i < H; i++){
      maxes[i]=-65504.0f;
      sums[i]=0.0f;
    }

  // OP A: Compute the maximum value on each row
  struct max_args_fp16 m_args;
  m_args.input = IN_DATA;
  m_args.maxes = maxes;
  m_args.H = tile_h;
  m_args.W = tile_w;

  for(int i = 0; i < n_tiles_i; i++){
    m_args.maxes = maxes + i * tile_h;
    for(int j = 0; j < n_tiles_j; j++){
        pi_cl_dma_cmd_2d((uint32_t) (args->input_data + i * W * tile_h + j * tile_w), (uint32_t) (IN_DATA), 2 * tile_dim, 2 * W, 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
        pi_cl_dma_cmd_wait(cmd_load);
        pi_cl_team_fork(NUM_CORES, pulp_row_max_fp16_cl, &m_args);
    }
  }

  // OP B: For each row, compute the sum of exponential of the difference between input values and the max of the row
  struct exp_sum_args_fp16 e_s_args;
  e_s_args.input = IN_DATA;
  e_s_args.output = OUT_DATA;
  e_s_args.H = tile_h;
  e_s_args.W = tile_w;
  e_s_args.sums = sums;
  e_s_args.maxes = maxes;

  for(int i = 0; i < n_tiles_i; i++){
    e_s_args.maxes = maxes + i * tile_h;
    e_s_args.sums = sums + i * tile_h;
    for(int j = 0; j < n_tiles_j; j++){
      pi_cl_dma_cmd_2d((uint32_t) (args->input_data + i * W * tile_h + j * tile_w), (uint32_t) (IN_DATA), 2 * tile_dim, 2 * W, 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
      pi_cl_dma_cmd_wait(cmd_load);
      pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp16_cl, &e_s_args);
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 2 * tile_dim, 2 * W, 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
      pi_cl_dma_cmd_wait(cmd_store);
    }
  }

  // OP C: Per-row division with the sum computed in the previous function
  struct row_div_args_fp16 r_d_args;
  r_d_args.input = OUT_DATA;
  r_d_args.sums = sums;
  r_d_args.H = tile_h;
  r_d_args.W = tile_w;

  for(int i=0; i < n_tiles_i; i++){
    r_d_args.sums = sums + i * tile_h;
    for(int j=0; j < n_tiles_j; j++){
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 2 * tile_dim, 2 * W, 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
      pi_cl_dma_cmd_wait(cmd_load);
      pi_cl_team_fork(NUM_CORES, pulp_row_div_fp16_cl, &r_d_args);
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 2 * tile_dim, 2 * W, 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
      pi_cl_dma_cmd_wait(cmd_store);
    }
  }
}

// Backward pass of softmax
void pulp_softmax_fp16_bw_cl(void *act_args_fp16) {
    /*
     * The derivative of softmax is computed according to:
     * https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
     *
     * The explanation below applies to each row of the input.
     * The array that results from the softmax forward pass (named outData in the code) will be noted with
     * S(input) = [S0, S1, ..., Si, ...].
     * The partial derivative of the i-th output w.r.t. the j-th input will be noted as DjSi and stored in a DS matrix.
     *
     * For i == j, DiSi = Si * (1 - Si)
     * For i != j, DiSi = -(Si * Sj)
     *
     * If the incoming array of gradients is marked with outDiff, the d_i gradient for the i-th element of a row is
     * computed as:
     * d_i = D0Si * outDiff[0] + D1Si * outDiff[1] + ... + DjSi * outDiff[j] + ...
     *
     * Which, if expanded and then simplified, will result to:
     * d_i = Si * (outDiff[i] - (outDiff[0] * S0 + outDiff[1] * S1 + ... + outDiff[j] * Sj + ...)
     *
     * The notation in the code is:
     *      - d_i -> inDiff[row, i]
     *      - Si  -> outData[row, i]
     *      - (outDiff[0] * S0 + outDiff[1] * S1 + ... + outDiff[j] * Sj + ...) -> sum
     */
    // Extract variables from function arguments
    struct softmax_args_fp16 *args = (struct softmax_args_fp16 *) act_args_fp16;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    fp16 *inDiff = args->input_diff;
    fp16 *outData = args->output_data;
    fp16 *outDiff = args->output_diff;

    fp16 *sums = args->sums;

    // SM BW OP 1
    struct sm_bw_op_1_args_fp16 op_1_args;
    op_1_args.A = outDiff;
    op_1_args.B = outData;
    op_1_args.S = sums;
    op_1_args.H = HEIGHT;
    op_1_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_sm_bw_op_1_fp16, &op_1_args);

    // SM BW OP 2
    struct sm_bw_op_2_args_fp16 op_2_args;
    op_2_args.A = outDiff;
    op_2_args.B = outData;
    op_2_args.S = sums;
    op_2_args.output = inDiff;
    op_2_args.H = HEIGHT;
    op_2_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_sm_bw_op_2_fp16, &op_2_args);
}


void pulp_vector_softmax_fp16(fp16* out, fp16* in, fp16* buffer_n_cores, unsigned int size){ 
  struct max_args_fp16 ma;
  ma.input = in;
  ma.maxes = buffer_n_cores;
  ma.dim = size;

  pi_cl_team_fork(NUM_CORES, pulp_max_fp16_cl, &ma);

  fp16 max = ma.maxes[0];

  for(int i=1;i<NUM_CORES; i++)
    if(ma.maxes[i] > max)
      max = ma.maxes[i];
  
  struct vector_exp_sum_args_fp16 vesa;
  vesa.input = in;
  vesa.output = out;
  vesa.max = max;
  vesa.sums = buffer_n_cores;
  vesa.dim = size;
  
  pi_cl_team_fork(NUM_CORES, vector_exp_sum_fp16_cl, &vesa);

  fp16 sum = 0;

  for(int i=0; i<NUM_CORES; i++)
    sum += vesa.sums[i];

  struct div_args_fp16 da;
  da.input = out;
  da.n = sum;
  da.dim = size;

  pi_cl_team_fork(NUM_CORES, pulp_div_fp16_cl, &da); 
}


void pulp_swiglu_fp16_cl(void *swiglu_args){
  struct swiglu_args_fp16* args = (struct swiglu_args_fp16*) swiglu_args;
  fp16* in1 = args->in1;
  fp16* in2 = args->in2;
  fp16* out = args->out;
  int size = args->dim;

  const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > size ? size : start+blockSize;

  for(int i=start; i<stop; i++){
    float val = (float)in1[i];

    #ifdef FASTEXPF
    val *= (1.0f / (1.0f + fastexp_gist_fp16(-val)));
    #else
    val *= (1.0f / (1.0f + expf(-val)));
    #endif

    val *= in2[i];
    out[i] = (fp16)val;
  }
}