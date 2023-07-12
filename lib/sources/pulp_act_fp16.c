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

#include "pulp_train_utils_fp16.h"
#include "pulp_act_fp16.h"
#include "math.h"

void pulp_relu_fp16_fw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  for (int i = 0; i < dim; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }
}

void pulp_relu_fp16_bw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* inDiff = args->input->diff;
  fp16* outDiff = args->output->diff;

  for (int i = 0; i < dim; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}


void pulp_softmax_fp16_fw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;

  int dim = args->input->dim;
  fp16* inData = args->input->data;
  fp16* outData = args->output->data;

  /*
  const int blockSize=(args_tanh->dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > args_tanh->dim ? args_tanh->dim : start+blockSize;
  */

  fp16 sum = 0.0;
  fp16 sum2 = 0.0;
  fp16 max = 0.0;
  fp16 maxes[NUM_CORES] = {0.0};
  fp16 sums[NUM_CORES] = {0.0};

  
  struct max_args_fp16 m_args;
  m_args.input = inData;
  m_args.maxes = maxes;
  m_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_max_fp16_cl, &m_args);

  for(int i=0; i<NUM_CORES; i++)
    if(max < maxes[i])
      max = maxes[i];
  
  struct exp_sum_args_fp16 e_s_args;
  e_s_args.input = inData;
  e_s_args.sums = sums;
  e_s_args.output = outData;
  e_s_args.dim = dim;
  e_s_args.max = max;
  
  pi_cl_team_fork(NUM_CORES, pulp_exp_sum_fp16_cl, &e_s_args);

  for(int i=0; i<NUM_CORES; i++){
    sum += sums[i];
  }

  struct div_args_fp16 d_args;
  d_args.input = outData;
  d_args.n = sum;
  d_args.dim = dim;

  pi_cl_team_fork(NUM_CORES, pulp_div_fp16_cl, &d_args);
}

void pulp_softmax_fp16_bw_cl( void * act_args_fp16 )
{
  struct act_args_fp16 * args = (struct act_args_fp16 *) act_args_fp16;
  int dim = args->input->dim;
  int i = args->output->dim;
  fp16* inDiff = args->input->diff;
  fp16* outData = args->output->data;
  fp16* outDiff = args->output->diff;
  fp16 sum = 0.0;

  for(int j = 0; j < dim; j++){ // Cycle over the elements of the i-th head buffer
      fp16 sum = 0.0;
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