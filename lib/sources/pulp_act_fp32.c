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

#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"
#include "math.h"

void pulp_relu_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;

  for (int i = 0; i < dim; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }
}

void pulp_relu_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* inDiff = args->input->diff;
  float* outDiff = args->output->diff;

  for (int i = 0; i < dim; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}


void pulp_softmax_fp32_fw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inData = args->input->data;
  float* outData = args->output->data;
  float sum = 0.0;

  for (int i = 0; i < dim; i++) {
    sum += expf(inData[i]);
  }

  for (int i = 0; i < dim; i++) {
    outData[i] = expf(inData[i])/sum;
  }
}

void pulp_softmax_fp32_bw_cl( void * act_args )
{
  struct act_args * args = (struct act_args *) act_args;
  int dim = args->input->dim;
  float* inDiff = args->input->diff;
  float* outData = args->output->data;
  float* outDiff = args->output->diff;
  float sum = 0.0;

  for (int i = 0; i < dim; i++) {
    //inDiff[i] = outDiff[i];
  }

  // Fix using: https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
  printf("[pulp_softmax_fp32_bw_cl] INVALID FORMULA, FIX!!");
}

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