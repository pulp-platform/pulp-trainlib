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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino
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


void pulp_softmax_fp16_fw_cl( void * act_args_fp16 )
{
  struct softmax_args_fp16 * args = (struct softmax_args_fp16 *) act_args_fp16;

  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  fp16* maxes = args->maxes;
  fp16* sums = args->sums;
  
  struct max_args_fp16 m_args;
  m_args.input = inData;
  m_args.maxes = maxes;
  m_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_row_max_fp16_cl, &m_args);
  
  struct exp_sum_args_fp16 e_s_args;
  e_s_args.input = inData;
  e_s_args.sums = sums;
  e_s_args.output = outData;
  e_s_args.dim = dim;
  e_s_args.maxes = maxes;
  
  pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp16_cl, &e_s_args);

  struct row_div_args_fp16 d_args;
  d_args.input = outData;
  d_args.sums = sums;
  d_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_row_div_fp16_cl, &d_args);
}

void pulp_softmax_fp16_bw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  int i = args->output->dim;
  fp16* inDiff = args->input->diff;
  fp16* outData = args->output->data;
  fp16* outDiff = args->output->diff;

  short s = 0;
  fp16 zero = (fp16) s;

  fp16 sum = zero;

  for(int j = 0; j < dim; j++){ // Cycle over the elements of the i-th head buffer
      fp16 sum = zero;
      const fp16 neg_sft_j  =  -(outData)[j]; 
      for(int z = 0; z < dim; ++z){ // Softmax involves all the elements of the i-th head buffer
          fp16 mul =  (outDiff)[z] * (outData)[z] * neg_sft_j;
          sum +=  mul; // adding to the total sum of this row.
      }
      inDiff[j] = sum;
  }

  for(int j=0; j<dim; j++){
      inDiff[j] += (outData)[j] * (outDiff)[j]; // Gradient of pre-softmax head buffer: (L x L)
  }
}

