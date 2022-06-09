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
#include "pulp_im2col_fp32.h"
#include "pulp_conv_dw_fp32.h"
#include "pulp_train_defines.h"

void pulp_conv_dw_fp32_fw_cl(struct blob * input, struct blob * coeff, struct blob * output, int Lpad,	int Rpad,	int Upad,	int Dpad, float * i2c_buffer, int opt_matmul_type) 
{
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

  // Kernel sizes
  int pW = coeff->W;
  int pH = coeff->H;
  // helps avoiding multiple L2 acceses, when structures are stored in L2

  float *coeffData = coeff->data;
  float *outData = output->data;
  float *inData = input->data;

  int W_in = input->W;
  int H_in = input->H;
  int C_in = input->C;
  int W_out = output->W;
  int H_out = output->H;
  int C_out = output->C;

  // Set im2col args
  im2col_args.input = input;
  im2col_args.c = coeff;
  im2col_args.output = output;
  im2col_args.pBuffer = i2c_buffer;
  im2col_args.Lpad = Lpad;
  im2col_args.Rpad = Rpad;
  im2col_args.Upad = Upad;
  im2col_args.Dpad = Dpad;
  im2col_args.mod = 0;
  im2col_args.tile_start = 0;
  im2col_args.tile_h = W_in;
  im2col_args.DW = 1;

  pi_cl_team_fork(NUM_CORES, pulp_im2col_fp32, &im2col_args);

  matMul_args.A = coeffData;
  matMul_args.B = i2c_buffer;
  matMul_args.C = outData;
  matMul_args.N = C_out;
  matMul_args.K = pW*pH*C_in;
  matMul_args.M = (W_in-pW+1+Lpad+Rpad)*(H_in-pH+1+Upad+Dpad);
  matMul_args.ker_size = pW*pH;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_dw, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_dw_args = &matMul_args;
  man_args.layer_type = LAYER_DW_CONV;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG
  // to PRINT outDiff orderly
  printf("ACTIV OUTPUT DW LAYER \n\n");
  for (int i=0; i<W_out*H_out*C_out; i++) {
    if ((i+1)%W_out==0) {
      printf(" %f \n", i, outData[i]);
      if ((i+1)%(W_out*H_out)==0)
        printf("\n");
    }
    else
      printf(" %f ", i, outData[i]);
  }
  #endif

  return;
}



void pulp_conv_dw_fp32_bw_cl(struct blob * input, struct blob * coeff, struct blob * output, int Lpad,	int Rpad,	int Upad,	int Dpad, float * i2c_buffer, int skip_in_grad, int opt_matmul_type_wg, int opt_matmul_type_ig)
{
  pulp_conv_dw_fp32_bw_param_grads_cl(input, coeff, output, Lpad, Rpad, Upad, Dpad, i2c_buffer, opt_matmul_type_wg);
  if (skip_in_grad == 0)
  {
    pulp_conv_dw_fp32_bw_input_grads_cl(input, coeff, output, Lpad, Rpad, Upad, Dpad, i2c_buffer, opt_matmul_type_ig);
  }
}



void pulp_conv_dw_fp32_bw_param_grads_cl(struct blob * input, struct blob * coeff, struct blob * output, int Lpad,	int Rpad,	int Upad,	int Dpad, float * i2c_buffer, int opt_matmul_type)
{
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

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

  float * inData = input->data;
  float * inDiff = input->diff;
  float * coeffData = coeff->data;
  float * coeffDiff = coeff->diff;
  float * outDiff = output->diff;
  float * outData = output->data;

  im2col_args.input = input; 
  im2col_args.c = coeff;
  im2col_args.output = output; 
  im2col_args.pBuffer = i2c_buffer;
  im2col_args.Lpad = 0;
  im2col_args.Rpad = 0;
  im2col_args.Upad = 0;
  im2col_args.Dpad = 0;
  im2col_args.mod = 0;
  im2col_args.tile_start = 0;
  im2col_args.tile_h = W_out;
  im2col_args.DW = 1;

  pi_cl_team_fork(NUM_CORES, pulp_im2col_fp32, &im2col_args);

  // COMPUTE GRADIENT
  matMul_args.A = outDiff;
  matMul_args.B = i2c_buffer; 
  matMul_args.C = coeffDiff;
  matMul_args.N = C_in;
  matMul_args.M = pW*pH;
  matMul_args.K = (W_out)*(H_out)*C_out;
  matMul_args.ker_size = (W_out)*(H_out);

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_dw, &matMul_args);
  #else 
  struct mm_manager_args man_args;
  man_args.mm_dw_args = &matMul_args;
  man_args.layer_type = LAYER_DW_CONV;
  man_args.step_type = STEP_WGT_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG
  printf("GRADIENT DW LAYER \n\n");
  for (int i=0; i<pW*pH*C_in; i++) {
    if ((i+1)%pW==0) {
      printf(" %f \n", i, coeffDiff[i]);
      if ((i+1)%(pW*pH)==0) {
        printf("\n");
      }
    }
    else
      printf(" %f ", coeffDiff[i]);
  }
  #endif
}



void pulp_conv_dw_fp32_bw_input_grads_cl(struct blob * input, struct blob * coeff, struct blob * output, int Lpad,	int Rpad,	int Upad,	int Dpad, float * i2c_buffer, int opt_matmul_type)
{
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

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

  float * inData = input->data;
  float * inDiff = input->diff;
  float * coeffData = coeff->data;
  float * coeffDiff = coeff->diff;
  float * outDiff = output->diff;
  float * outData = output->data;

  // PREPARE im2col_buffer for ACTIV_GRAD
  im2col_args.input = input;
  im2col_args.c = coeff;
  im2col_args.output = output;
  im2col_args.pBuffer = i2c_buffer;
  im2col_args.Lpad = Lpad;
  im2col_args.Rpad = Rpad;
  im2col_args.Upad = Upad;
  im2col_args.Dpad = Dpad;
  im2col_args.mod = 1;
  im2col_args.DW = 1;
  
  //if (H_in == pH) im2col_args.pad = 2;

  pi_cl_team_fork(NUM_CORES, pulp_im2col_fp32, &im2col_args);

  #ifdef DEBUG
  printf("\nim2col buffer:\n");
  for (int idx=0; idx<W_in*H_in*C_out*(pW)*(pH); idx++) {
    if (!(idx%pW)) printf("\n");
    if (!(idx%(pW*pH))) printf("\n");
    printf("%f ", i2c_buffer[idx]);
  }
  printf("\n");

  printf("\ncoeffData:\n");
  for (int idx=0; idx<pW*pH*C_in; idx++) {
    if (!(idx%pW)) printf("\n");
    //if (!(idx&(pW*pH))) printf("\n");
    printf("%f ", coeffData[idx]);
  }
  printf("\n");
  #endif

  // COMPUTE ACTIV_GRAD
  matMul_args.A = coeffData; // to be flipped 
  matMul_args.B = i2c_buffer;
  matMul_args.C = inDiff;
  matMul_args.N = 1; 
  matMul_args.M = (W_in)*(H_in);
  matMul_args.K = pW*pH*C_in;
  matMul_args.ker_size = pW*pH;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_dw_in_grad, &matMul_args);
  #else
  struct mm_manager_args man_args;
  man_args.mm_dw_args = &matMul_args;
  man_args.layer_type = LAYER_DW_CONV;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
  #endif

  #ifdef DEBUG
  // to PRINT inDiff orderly
  printf("ERROR PROP DW LAYER \n\n");
  for (int i=0; i<(W_in)*(H_in)*C_in; i++) {
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
