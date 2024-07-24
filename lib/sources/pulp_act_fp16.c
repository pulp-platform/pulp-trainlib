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
