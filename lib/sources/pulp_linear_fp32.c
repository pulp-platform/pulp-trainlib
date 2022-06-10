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
#include "pulp_matmul_fp32.h"
#include "pulp_linear_fp32.h"

void pulp_linear_fp32_fw_cl(struct blob * input, struct blob * coeff, struct blob * output)
{
  float *coeffData = coeff->data;
  float *outData = output->data;  
  float *inputData = input->data;

  struct matMul_args matMul_args;

  matMul_args.A = coeffData;
  matMul_args.B = inputData;
  matMul_args.C = outData;
  matMul_args.N = output->dim;
  matMul_args.K = input->dim;
  matMul_args.M = 1;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG 
    printf("\nLinear OutData: %d\n", matMul_args.N);
    for (int i=0; i<output->dim; i++){
      printf("%4.2e ", outData[i]);
    }
    printf("\n");
  #endif
}


void pulp_linear_fp32_bw_cl(struct blob * input, struct blob * coeff, struct blob * output) 
{
  pulp_linear_fp32_bw_param_grads_cl(input, coeff, output);
  pulp_linear_fp32_bw_input_grads_cl(input, coeff, output);
}


void pulp_linear_fp32_bw_param_grads_cl(struct blob * input, struct blob * coeff, struct blob * output) 
{
  float *coeffData = coeff->data;
  float *inData = input->data;
  float *outData = output->data;
  float *coeffDiff = coeff->diff;
  float *outDiff = output->diff;  
  float *inDiff = input->diff;

  struct matMul_args matMul_args;

#ifdef DEBUG
  printf("\nLinear outDiff\n");
  for(int i=0; i<output->dim; i++)
    printf("%4.2e ", outDiff[i]);
  printf("\n");
#endif

  matMul_args.A = output->diff;
  matMul_args.B = input->data;
  matMul_args.C = coeff->diff;
  matMul_args.N = output->dim;
  matMul_args.K = 1;
  matMul_args.M = input->dim;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_WGT_GRAD;
  man_args.matmul_type = MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG 
  printf("\nLinear coeffDiff ");
    for (int i=0; i<input->dim*output->dim; i++){
      if(!(i%(output->dim))) printf("\n");
      printf("%4.2e (i=%d)", coeffDiff[i], i);
    }
    printf("\n");
  #endif
}


void pulp_linear_fp32_bw_input_grads_cl(struct blob * input, struct blob * coeff, struct blob * output) 
{
  float *coeffData = coeff->data;
  float *inData = input->data;
  float *outData = output->data;
  float *coeffDiff = coeff->diff;
  float *outDiff = output->diff;  
  float *inDiff = input->diff;

  struct matMul_args matMul_args;

#ifdef DEBUG
  printf("\nLinear outDiff\n");
  for(int i=0; i<output->dim; i++)
    printf("%4.2e ", outDiff[i]);
  printf("\n");
#endif

  matMul_args.A = output->diff;
  matMul_args.B = coeffData;
  matMul_args.C = inDiff;
  matMul_args.N = 1;
  matMul_args.K = output->dim;
  matMul_args.M = input->dim;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_M, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG 
  printf("\nLinear outDiff (coeffData.T * inDiff)");

    for (int i=0; i<output->dim/*+2*/; i++){
      if(!(i%(coeff->H))) printf("\n");
      printf("%4.2e ", outDiff[i]);
    }
    printf("\n");
  #endif
}
