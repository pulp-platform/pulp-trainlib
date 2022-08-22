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

#include "stm32_train_utils.h"
#include "stm32_matmul.h"
#include "stm32_conv_pw.h"
#include "stm32_train_defines.h"


void stm32_conv_pw_fp32_fw(struct blob * input, struct blob * coeff, struct blob * output, int pad, int opt_matmul_type) {

  // kernel dimensions
  struct matMul_args matMul_args;

  int pW = coeff->W;
  int pH = coeff->H;
  float *coeffData = coeff->data;
  float *outData = output->data;
  float *inData = input->data;

  int W_in = input->W;
  int H_in = input->H;
  int Cin = input->C;
  int Cout = output->C;

  matMul_args.A = coeffData;
  matMul_args.B = inData;
  matMul_args.C = outData;
  matMul_args.N = Cout;
  matMul_args.M = H_in*W_in;
  matMul_args.K = pW*pH*Cin;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  mm(&matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_PW_CONV;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  mm_manager(&man_args);
  #endif

  #ifdef DEBUG
  printf("FORWARD PW LAYER \n\n");
  for (int i=0; i<Cout*output->W*output->H; i++) {
    if ((i+1)%output->W==0) {
      printf(" %f \n\n", i, outData[i]);
    }
    else
      printf(" %f \n", outData[i]);
  }
  printf("\n");
  #endif

  return;
}



void stm32_conv_pw_fp32_bw(struct blob * input, struct blob * coeff, struct blob * output, int pad, int skip_in_grad, int opt_matmul_type_wg, int opt_matmul_type_ig) 
{
  stm32_conv_pw_fp32_bw_param_grads(input, coeff, output, pad, opt_matmul_type_wg);
  if (skip_in_grad == 0)
  {
    stm32_conv_pw_fp32_bw_input_grads(input, coeff, output, pad, opt_matmul_type_ig);
  }
}



void stm32_conv_pw_fp32_bw_param_grads(struct blob * input, struct blob * coeff, struct blob * output, int pad, int opt_matmul_type) 
{
  struct matMul_args matMul_args;

  //input dimensions
  int W_in = input->W;
  int H_in = input->H;
  int C_in = input->C;
  //kernel dimensions
  int pW = coeff->W;
  int pH = coeff->H;
  //output dimensions
  int W_out = output->W;
  int H_out = output->H;
  int C_out = output->C;

  #ifdef DEBUG
  printf("OUTDIM %d %d %d ", W_in, H_in, C_in);
  #endif

  float * inData = input->data;
  float * inDiff = input->diff;

  float * coeffData = coeff->data;
  float * coeffDiff = coeff->diff;

  float * outData = output->data;
  float * outDiff = output->diff;

  // COMPUTE GRADIENT
  matMul_args.A = outDiff;
  matMul_args.B = inData;  // transpose this
  matMul_args.C = coeffDiff;
  matMul_args.N = C_out;
  matMul_args.M = C_in;
  matMul_args.K = W_out*H_out;
  matMul_args.trans_B = 1;

  #ifndef OPTIMIZE
  mm(&matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_PW_CONV;
  man_args.step_type = STEP_WGT_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  mm_manager(&man_args);
  #endif

  #ifdef DEBUG
  printf("%d %d %d %d\n\n", pW,pH,C_in,C_out);

  printf("GRADIENT PW LAYER \n\n");
  for (int i=0; i<pW*pH*C_out*C_in; i++) {
    if ((i+1)%C_out==0) {
      printf(" %f \n\n", i, coeffDiff[i]);
    }
    else
      printf(" %f \n", coeffDiff[i]);
  }
  printf("\n");
  #endif
}



void stm32_conv_pw_fp32_bw_input_grads(struct blob * input, struct blob * coeff, struct blob * output, int pad, int opt_matmul_type) 
{
  // struct for coeffDiff calculation
  struct matMul_args matMul_args;

  //input dimensions
  int W_in = input->W;
  int H_in = input->H;
  int C_in = input->C;
  //kernel dimensions
  int pW = coeff->W;
  int pH = coeff->H;
  //output dimensions
  int W_out = output->W;
  int H_out = output->H;
  int C_out = output->C;

  #ifdef DEBUG
  printf("OUTDIM %d %d %d ", W_out, H_out, C_out);
  #endif

  float * inData = input->data;
  float * inDiff = input->diff;

  float * coeffData = coeff->data;
  float * coeffDiff = coeff->diff;

  float * outData = output->data;
  float * outDiff = output->diff;

  // COMPUTE ACTIV_GRAD
  matMul_args.A = coeffData; // transp ?
  matMul_args.B = outDiff;
  matMul_args.C = inDiff;
  matMul_args.N = C_in;
  matMul_args.M = W_out*H_out;
  matMul_args.K = pW*pH*C_out;
  matMul_args.trans_B = 0;
  
  #ifndef OPTIMIZE
  mm(&matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_PW_CONV;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  mm_manager(&man_args);
  #endif

  #ifdef DEBUG
  // to PRINT outDiff orderly
  printf("ERROR PROP PW LAYER \n\n");
  for (int i=0; i<W_in*H_in*C_in; i++) {
    if ((i+1)%W_in==0) {
      printf(" %f \n", i, inDiff[i]);
      if ((i+1)%(W_in*H_in)==0)
        printf("\n");
    }
    else
      printf(" %f ", i, inDiff[i]);
  }
  #endif
}
