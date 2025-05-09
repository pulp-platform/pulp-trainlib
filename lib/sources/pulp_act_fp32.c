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

 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino, Calin Diaconu
*/

#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"
#include "math.h"
#include "pulp_mhsa_fp32.h"


void pulp_sigmoid_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_fw_fp32, act_args);
}

void pulp_sigmoid_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, sigmoid_core_bw_fp32, act_args);
}

void sigmoid_core_fw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    float sigma = 0.0f;
    sigma = 1 + expf(-inData[i]);
    sigma = 1 / sigma;
    outData[i] = sigma;
  }
}

void sigmoid_core_bw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outData = args->output->data;
  float* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i=start; i<stop; i++) {
    float sigma = 0.0f;
    float sigma_prime = 0.0f;
    //sigma = 1 + expf(-inData[i]);
    //sigma = 1 / sigma;
    sigma = outData[i];
    sigma_prime = sigma * (1.0f - sigma);
    inDiff[i] = outDiff[i] * sigma_prime;
  }
}




void pulp_relu_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, act_args);
}

void pulp_relu_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  pi_cl_team_fork(NUM_CORES, relu_core_bw_fp32, act_args);
}

void relu_core_fw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }

}

void relu_core_bw_fp32( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outDiff = args->output->diff;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}




void pulp_leakyrelu_fp32_fw_cl( void * leakyrelu_args ) 
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_fw_fp32, leakyrelu_args);
}


void pulp_leakyrelu_fp32_bw_cl( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  pi_cl_team_fork(NUM_CORES, leakyrelu_core_bw_fp32, leakyrelu_args);
}


void leakyrelu_core_fw_fp32( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;
  float neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : inData[i]*neg_slope;
  }
}


void leakyrelu_core_bw_fp32( void * leakyrelu_args )
{
  struct leakyrelu_args * args = (struct leakyrelu_args *) leakyrelu_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outDiff = args->output->diff;
  float neg_slope = args->negative_slope;

  const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > dim ? dim : start+blockSize;

  for (int i = start; i < stop; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : outDiff[i]*neg_slope;
  }
}



// ~~~~~~~~~~~~~~~~~~~~ SOFTMAX ~~~~~~~~~~~~~~~~~~~~
// Forward pass of the FP32 softmax
// Performs a softmax activation on each row
void pulp_softmax_fp32_fw_cl(void *act_args) {
    // Extract variables from function arguments
    struct softmax_args *args = (struct softmax_args *) act_args;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    float *inData = args->input_data;
    float *outData = args->output_data;

    float *maxes = args->maxes;
    float *sums = args->sums;

    for(int i = 0; i < HEIGHT; i++){
      maxes[i]=-340282346638528859811704183484516925440.0f;
      sums[i]=0.0f;
    }

    // OP A: Compute the maximum value on each row
    struct max_args m_args;
    m_args.input = inData;
    m_args.maxes = maxes;
    m_args.H = HEIGHT;
    m_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &m_args);

    // OP B: For each row, compute the sum of exponential of the difference between input values and the max of the row
    struct exp_sum_args e_s_args;
    e_s_args.input = inData;
    e_s_args.output = outData;
    e_s_args.H = HEIGHT;
    e_s_args.W = WIDTH;
    e_s_args.sums = sums;
    e_s_args.maxes = maxes;

    pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp32_cl, &e_s_args);

    // OP C: Per-row division with the sum computed in the previous function
    struct row_div_args r_d_args;
    r_d_args.input = outData;
    r_d_args.sums = sums;
    r_d_args.H = HEIGHT;
    r_d_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &r_d_args);

    // Print to terminal for debugging purposes
    #ifdef DEBUG
    if(pi_core_id()==0){
        printf("\nCurrent softmax output: %d %d\n", HEIGHT, WIDTH);
        for (int j=0; j<HEIGHT*WIDTH; j++){
            if(!(j%((int)WIDTH))) printf("\n");
            printf("%.8f ", outData[j]);
        }
    }
    printf("\n");
    #endif
}

void pulp_softmax_fp32_fw_cl_tiled(void *act_args, void* Tiled_matmul_mhsa_args){
  struct softmax_args *args = (struct softmax_args *) act_args;
  struct Tiled_Matmul_Mhsa_args * tiled_args = (struct Tiled_Matmul_Mhsa_args*) Tiled_matmul_mhsa_args;

  int H = args->H;
  int W = args->W;

  float *maxes = args->maxes;
  float *sums = args->sums;

  float *inData = args->input_data;
  float *outData = args->output_data;

  int tile_h = tiled_args->tile_h_sm;
  int tile_w = tiled_args->tile_w_sm;
  int tile_dim = tiled_args->tile_dim_sm;
  float* BUFF = tiled_args->BUFF;
  pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
  pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;

  int n_tiles_i = H / tile_h;
  int n_tiles_j = W / tile_w;

  float* IN_DATA = BUFF;
  float* OUT_DATA = BUFF + tile_dim;

  for(int i = 0; i < H; i++){
      maxes[i]=-340282346638528859811704183484516925440.0f;
      sums[i]=0.0f;
    }

  // OP A: Compute the maximum value on each row
  struct max_args m_args;
  m_args.input = IN_DATA;
  m_args.maxes = maxes;
  m_args.H = tile_h;
  m_args.W = tile_w;

  for(int i = 0; i < n_tiles_i; i++){
    m_args.maxes = maxes + i * tile_h;
    for(int j = 0; j < n_tiles_j; j++){
        pi_cl_dma_cmd_2d((uint32_t) (args->input_data + i * W * tile_h + j * tile_w), (uint32_t) (IN_DATA), 4 * tile_dim, 4 * W, 4 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
        pi_cl_dma_cmd_wait(cmd_load);
        pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &m_args);
    }
  }

  // OP B: For each row, compute the sum of exponential of the difference between input values and the max of the row
  struct exp_sum_args e_s_args;
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
      pi_cl_dma_cmd_2d((uint32_t) (args->input_data + i * W * tile_h + j * tile_w), (uint32_t) (IN_DATA), 4 * tile_dim, 4 * W, 4 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
      pi_cl_dma_cmd_wait(cmd_load);
      pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp32_cl, &e_s_args);
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 4 * tile_dim, 4 * W, 4 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
      pi_cl_dma_cmd_wait(cmd_store);
    }
  }

  // OP C: Per-row division with the sum computed in the previous function
  struct row_div_args r_d_args;
  r_d_args.input = OUT_DATA;
  r_d_args.sums = sums;
  r_d_args.H = tile_h;
  r_d_args.W = tile_w;

  for(int i=0; i < n_tiles_i; i++){
    r_d_args.sums = sums + i * tile_h;
    for(int j=0; j < n_tiles_j; j++){
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 4 * tile_dim, 4 * W, 4 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
      pi_cl_dma_cmd_wait(cmd_load);
      pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &r_d_args);
      pi_cl_dma_cmd_2d((uint32_t) (args->output_data + i * W * tile_h + j * tile_w), (uint32_t) (OUT_DATA), 4 * tile_dim, 4 * W, 4 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
      pi_cl_dma_cmd_wait(cmd_store);
    }
  }
}

// Backward pass of softmax
void pulp_softmax_fp32_bw_cl(void *act_args) {
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
    struct softmax_args *args = (struct softmax_args *) act_args;

    int HEIGHT = args->H;
    int WIDTH = args->W;

    float *inDiff = args->input_diff;
    float *outData = args->output_data;
    float *outDiff = args->output_diff;

    float *sums = args->sums;

    // SM BW OP 1
    struct sm_bw_op_1_args op_1_args;
    op_1_args.A = outDiff;
    op_1_args.B = outData;
    op_1_args.S = sums;
    op_1_args.H = HEIGHT;
    op_1_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_sm_bw_op_1, &op_1_args);

    // SM BW OP 2
    struct sm_bw_op_2_args op_2_args;
    op_2_args.A = outDiff;
    op_2_args.B = outData;
    op_2_args.S = sums;
    op_2_args.output = inDiff;
    op_2_args.H = HEIGHT;
    op_2_args.W = WIDTH;

    pi_cl_team_fork(NUM_CORES, pulp_sm_bw_op_2, &op_2_args);
}


/*
void pulp_partial_softmax_simple_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int dim = args->input->dim; // L
  int dim2 = dim * dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  float* maxes = args->maxes;
  float* sums = args->sums;

  struct max_args m_args;
  m_args.input = inData;
  m_args.maxes = maxes;
  m_args.H = dim;
  m_args.W = dim2;

  pi_cl_team_fork(NUM_CORES, pulp_row_max_fp32_cl, &m_args);

  struct shift_sum_args ss_args;
  ss_args.input = inData;
  ss_args.sums = sums;
  ss_args.output = outData;
  ss_args.dim = dim;
  ss_args.dim2 = dim2;
  ss_args.maxes = maxes;

  pi_cl_team_fork(NUM_CORES, pulp_shift_sum_fp32_cl, &ss_args);


  struct row_div_args r_d_args;
  r_d_args.input = outData;
  r_d_args.sums = sums;
  r_d_args.H = dim;
  r_d_args.W = dim2;

  pi_cl_team_fork(NUM_CORES, pulp_row_div_fp32_cl, &r_d_args);
}


void pulp_partial_softmax_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  const float eps_max = 0.03125f;
  
  float zerox = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zerox;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zerox;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = 1.0f / powf(2.0f, shift); 
        partial_exp_sum[j] = (partial_exp_sum[j]) * (1.0f / powf(2.0f, max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (1.0f / powf(2.0f, shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}


void pulp_partial_softmax_shift_fp32_fw_cl( void * act_args )
{
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  float eps_max = 0.03125f;

  float zerox = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zerox;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zerox;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = 1  >> (int)(ceilf(shift)); 
        partial_exp_sum[j] = ((int)(partial_exp_sum[j]) >> (int)ceilf(max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (1 >> (int) ceilf(shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}


void pulp_partial_softmax_approximate_fp32_fw_cl(void * act_args){
  struct softmax_args * args = (struct softmax_args *) act_args;

  int L = args->L;
  int n_heads = args->n_heads;
  float * inData = args->input->data;
  float * outData = args->output->data;

  float * partial_exp_sum = args->partial_exp_sum + pi_core_id()*L;
  float * global_max = args->global_max + pi_core_id()*L;

  float eps_max = 0.03125f;

  float zero = 0.0f;
  float minfloat = -340282346638528859811704183484516925440.0f;

  const int blockSize=(n_heads+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > n_heads ? n_heads : start+blockSize;

  //Cycle over the heads
  for(int i = start; i < stop; i++){
    //STAGE 1: Calculate the denominator
    if(i != start){
      for(int l=0; l < L; l++){
        partial_exp_sum[l] = zero;
        global_max[l] = minfloat;
      }
    }
    float* pointer = inData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float max_shift = zero;
        if(global_max[j] < (*pointer)){
          max_shift = ((*pointer) - global_max[j]) * eps_max; 
          global_max[j] = *pointer;
        }
        float shift = (global_max[j] - (*pointer)) * eps_max;
        float exp_sum = threshold(shift); 
        partial_exp_sum[j] = (partial_exp_sum[j]) * (threshold(max_shift)) + exp_sum;
        pointer++;
      }
    }
    //STAGE 2: Calculate activation value
    pointer = inData + i*L*L;
    float* out_pointer = outData + i*L*L;
    for(int j = 0; j < L; j++){
      for(int k = 0; k < L; k++){
        float shift = (global_max[j] - (*pointer)) * eps_max;
        (*out_pointer) = (threshold(shift))/partial_exp_sum[j];
        pointer++;
        out_pointer++;
      }
    }
  }
}
*/


void tanh_prll(void * args){

  struct tanh_args* args_tanh=(struct tanh_args *) args;

  const int blockSize=(args_tanh->dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > args_tanh->dim ? args_tanh->dim : start+blockSize;

  for(int i=start;i<stop;i++){
    args_tanh->output[i]=fasttanh(args_tanh->input[i]);
  }
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}

static inline float
fasttanh (float p)
{
  return -1.0f + 2.0f / (1.0f + fastexp (-2.0f * p));
}

static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { (uint32_t) ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

void pulp_vector_softmax_fp32(float* out, float* in, float* buffer_n_cores, unsigned int size){ 
  struct max_args ma;
  ma.input = in;
  ma.maxes = buffer_n_cores;
  ma.W = size;

  pi_cl_team_fork(NUM_CORES, pulp_max_fp32_cl, &ma);

  float max = ma.maxes[0];

  for(int i=1;i<NUM_CORES; i++)
    if(ma.maxes[i] > max)
      max = ma.maxes[i];
  
  struct vector_exp_sum_args vesa;
  vesa.input = in;
  vesa.output = out;
  vesa.max = max;
  vesa.sums = buffer_n_cores;
  vesa.dim = size;
  
  pi_cl_team_fork(NUM_CORES, vector_exp_sum_fp32_cl, &vesa);

  float sum = 0;

  for(int i=0; i<NUM_CORES; i++)
    sum += vesa.sums[i];

  struct div_args da;
  da.input = out;
  da.n = sum;
  da.dim = size;

  pi_cl_team_fork(NUM_CORES, pulp_div_fp32_cl, &da); 
}


void pulp_swiglu_fp32_cl(void *swiglu_args){
  struct swiglu_args* args = (struct swiglu_args*) swiglu_args;
  float* in1 = args->in1;
  float* in2 = args->in2;
  float* out = args->out;
  int size = args->dim;

  const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > size ? size : start+blockSize;

  for(int i=start; i<stop; i++){
    float val = in1[i];

    #ifdef FASTEXPF
    val *= (1.0f / (1.0f + fastexp_gist(-val)));
    #else
    val *= (1.0f / (1.0f + expf(-val)));
    #endif

    val *= in2[i];
    out[i] = val;
  }
}


void pulp_gelu_tanh_approx_fp32_fw_cl(void *void_args) {
    struct act_args *args = (struct act_args *) void_args;

    int dim = args->input->dim;
    float *inData = args->input->data;
    float *outData = args->output->data;

    const int blockSize = (dim + NUM_CORES - 1) / NUM_CORES;
    const int start = pi_core_id() * blockSize;
    const int stop = start + blockSize > dim ? dim : start + blockSize;

    for (int i = start; i < stop; i++) {
        float x = inData[i];
        float half_x = 0.5f * x;

        float val = (((x * x * x * 0.044715f) + x) * 0.7978f);

        float val_2 = val * val;

        float a = (((val_2 + 378.0f) * val_2 + 17325.0f) * val_2 + 135135.0f) * val;
        float b = ((28.0f * val_2 + 3150.0f) * val_2 + 62370.0f) * val_2 + 135135.0f;
        val = a / b;

        if (val > 1)
            val = 1;
        else if (val < -1)
            val = -1;

        val = val * half_x + half_x;

        outData[i] = val;
    }
}
