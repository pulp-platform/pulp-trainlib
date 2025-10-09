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
#include "pulp_matmul_fp16.h"
#include "pulp_linear_fp16.h"

void pulp_linear_fp16_fw_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *biasData = FC_args->bias->data;
  fp16 *outData = FC_args->output->data;  
  fp16 *inputData = FC_args->input->data;

  int opt_matmul_type = FC_args->opt_matmul_type_fw;

  int use_biases_linear = FC_args->use_biases;

  struct matMul_args_fp16 matMul_args;

  matMul_args.A = coeffData;
  matMul_args.B = inputData;
  matMul_args.C = outData;
  matMul_args.bias = biasData;

  matMul_args.N = FC_args->output->dim;
  matMul_args.K = FC_args->input->dim;
  matMul_args.M = 1;
  matMul_args.trans_B = 0;
  matMul_args.USE_BIASES = use_biases_linear;

  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, pulp_linear_fp16_fw_cl_kernel, &man_args);

  #ifdef DEBUG 
    printf("\nLinear OutData: %d\n", matMul_args.N);
    for (int i=0; i<FC_args->output->dim; i++){
      printf("%4.2e ", outData[i]);
    }
    printf("\n");
  #endif
}


void pulp_linear_fp16_bw_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  int skip_wg_grad = FC_args->skip_wg_grad;
  int skip_in_grad = FC_args->skip_in_grad;

  if (skip_wg_grad == 0)
  {
    pulp_linear_fp16_bw_param_grads_cl(Linear_args_fp16);
  }
  
  if (skip_in_grad == 0) 
  {
    pulp_linear_fp16_bw_input_grads_cl(Linear_args_fp16); 
  }
}


void pulp_linear_fp16_bw_param_grads_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *inData = FC_args->input->data;
  fp16 *outData = FC_args->output->data;
  fp16 *coeffDiff = FC_args->coeff->diff;
  fp16 *outDiff = FC_args->output->diff;  
  fp16 *inDiff = FC_args->input->diff;
  fp16 *biasDiff = FC_args->bias->diff;

  int opt_matmul_type = FC_args->opt_matmul_type_wg;

  int use_biases_linear = FC_args->use_biases;

  struct matMul_args_fp16 matMul_args;

  #ifdef DEBUG
  printf("\nLinear outDiff\n");
  for(int i=0; i<FC_args->output->dim; i++)
    printf("%4.2e ", outDiff[i]);
  printf("\n");
  #endif

  matMul_args.A = outDiff;
  matMul_args.B = inData;
  matMul_args.C = coeffDiff;
  matMul_args.N = FC_args->output->dim;
  matMul_args.K = 1;
  matMul_args.M = FC_args->input->dim;
  matMul_args.trans_B = 0;  
  matMul_args.bias = biasDiff;
  matMul_args.USE_BIASES = use_biases_linear;

  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, pulp_linear_fp16_bw_param_grads_cl_kernel, &man_args);

  #ifdef DEBUG
  printf("\nLinear outDiff\n");
  for(int i=0; i<FC_args->output->dim; i++)
    printf("%4.2e ", outDiff[i]);
  printf("\n");
  #endif

  #ifdef DEBUG 
  printf("\nLinear coeffDiff ");
  for (int i=0; i<FC_args->input->dim*FC_args->output->dim; i++){
    if(!(i%(FC_args->output->dim))) printf("\n");
    printf("%4.2e (i=%d)", coeffDiff[i], i);
  }
  printf("\n");

  printf("\nLinear biasDiff\n");
  for(int i=0; i<FC_args->output->dim; i++)
    printf("%4.2e ", biasDiff[i]);
  printf("\n");
  #endif
}


void pulp_linear_fp16_bw_input_grads_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *inData = FC_args->input->data;
  fp16 *outData = FC_args->output->data;
  fp16 *coeffDiff = FC_args->coeff->diff;
  fp16 *outDiff = FC_args->output->diff;  
  fp16 *inDiff = FC_args->input->diff;

  int opt_matmul_type = FC_args->opt_matmul_type_ig;

  struct matMul_args_fp16 matMul_args;

#ifdef DEBUG
  printf("\nLinear outDiff\n");
  for(int i=0; i<FC_args->output->dim; i++)
    printf("%4.2e ", outDiff[i]);
  printf("\n");
#endif

  matMul_args.A = outDiff;
  matMul_args.B = coeffData;
  matMul_args.C = inDiff;
  matMul_args.N = 1;
  matMul_args.K = FC_args->output->dim;
  matMul_args.M = FC_args->input->dim;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_M_fp16, &matMul_args);
  #else
  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
  #endif

  #ifdef DEBUG 
  printf("\nLinear outDiff (coeffData.T * inDiff)");

    for (int i=0; i<FC_args->output->dim/*+2*/; i++){
      if(!(i%(FC_args->coeff->H))) printf("\n");
      printf("%4.2e ", outDiff[i]);
    }
    printf("\n");
  #endif
}








void pulp_linear_fp16_fw_cl_kernel( void * man_args )
{
  struct mm_manager_args_fp16 * manager_args = (struct mm_manager_args_fp16 *) man_args;
  struct matMul_args_fp16 *matMul_args = manager_args->mm_args;

  fp16 *__restrict__ inData = matMul_args->A;
  fp16 *__restrict__ coeffData = matMul_args->B;
  fp16 *__restrict__ outData = matMul_args->C;

  fp16 *__restrict__ biasData = matMul_args->bias;

  // printf("\n");
  // printf("inData: 0x%x\n", (unsigned int)inData);
  // printf("coeffData: 0x%x\n", (unsigned int)coeffData);
  // printf("outData: 0x%x\n", (unsigned int)outData);
  // printf("biasData: 0x%x\n", (unsigned int)biasData);
  // printf("\n");

  const uint32_t N = matMul_args->N;
  const uint32_t K = matMul_args->K;
  const uint32_t M = matMul_args->M;
  const uint32_t trans_B = matMul_args->trans_B;
  const uint32_t USE_BIASES = matMul_args->USE_BIASES;

  // printf("Output (before matmul): ");
  // for (int i=0; i<N; i++) {
  //   printf("[(0x%x) %f] ", (unsigned int)&outData[i], outData[i]);
  // } printf("\n");

  #ifndef OPTIMIZE
  mm_fp16(matMul_args);
  #else
  mm_manager_fp16(manager_args);
  #endif

  // printf("Output (before biases): ");
  // for (int i=0; i<N; i++) {
  //   printf("[(0x%x) %f] ", (unsigned int)&outData[i], outData[i]);
  // } printf("\n");

  // N -- Co
  const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > N ? N : start+blockSize;

  if (USE_BIASES == 1 && pi_core_id() == 0) {
    for (int i=0; i<N; i++){
    //for (uint32_t i=start; i < stop; i++) {
      // printf("outData[%d] = %f, biasData[%d] = %f\n", i, outData[i], i, biasData[i]);
      outData[i] += biasData[i];
    }
  }

  // printf("Output (after biases):  ");
  // for (int i=0; i<N; i++) {
  //   printf("[(0x%x) %f] ", (unsigned int)&outData[i], outData[i]);
  // } printf("\n");
  // printf("\nEND DEBUG\n\n");

}

void pulp_linear_fp16_bw_param_grads_cl_kernel( void * man_args )
{
  struct mm_manager_args_fp16 * manager_args = (struct mm_manager_args_fp16 *) man_args;
  struct matMul_args_fp16 *matMul_args = manager_args->mm_args;

  fp16 *__restrict__ outDiff = matMul_args->A;
  fp16 *__restrict__ inData = matMul_args->B;
  fp16 *__restrict__ coeffDiff = matMul_args->C;

  fp16 *__restrict__ biasData = matMul_args->bias;

  const uint32_t N = matMul_args->N; // A: (NxK), B: (KxM), C: (NxM)
  const uint32_t K = matMul_args->K;
  const uint32_t M = matMul_args->M;
  const uint32_t trans_B = matMul_args->trans_B;
  const uint32_t USE_BIASES = matMul_args->USE_BIASES;

  #ifndef OPTIMIZE
  mm_fp16(matMul_args);
  #else
  mm_manager_fp16(manager_args);
  #endif

  const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > N ? N : start+blockSize;

  if (USE_BIASES == 1) {
    for (uint32_t i=start; i < stop; i++) {
      biasData[i] = outDiff[i];
    }
  }
  
}
