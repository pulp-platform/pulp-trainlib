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

#ifndef NAIVE_DW
//#define NAIVE_DW
#endif

void pulp_conv_dw_fp32_fw_cl ( void * DepthWise_Conv_args )
{
  struct DepthWise_Conv_args * DW_args = (struct DepthWise_Conv_args *) DepthWise_Conv_args;
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

  // Kernel sizes
  int pW = DW_args->coeff->W;
  int pH = DW_args->coeff->H;

  float *coeffData = DW_args->coeff->data;
  float *outData = DW_args->output->data;
  float *inData = DW_args->input->data;

  int W_in = DW_args->input->W;
  int H_in = DW_args->input->H;
  int C_in = DW_args->input->C;
  int W_out = DW_args->output->W;
  int H_out = DW_args->output->H;
  int C_out = DW_args->output->C;

  int Lpad = DW_args->Lpad;
  int Rpad = DW_args->Rpad;
  int Upad = DW_args->Upad;
  int Dpad = DW_args->Dpad;

  float * i2c_buffer = DW_args->i2c_buffer;

  int input_layout = DW_args->HWC;
  int opt_matmul_type = DW_args->opt_matmul_type_fw;

  #ifndef NAIVE_DW
  // Set im2col args
  im2col_args.input = DW_args->input;
  im2col_args.c = DW_args->coeff;
  im2col_args.output = DW_args->output;
  im2col_args.pBuffer = DW_args->i2c_buffer;
  im2col_args.Lpad = DW_args->Lpad;
  im2col_args.Rpad = DW_args->Rpad;
  im2col_args.Upad = DW_args->Upad;
  im2col_args.Dpad = DW_args->Dpad;
  im2col_args.mod = 0;
  im2col_args.stride_h = 1;
  im2col_args.stride_w = 1;
  im2col_args.USE_DMA = 0;
  im2col_args.HWC = input_layout;

  pi_cl_team_fork(NUM_CORES, pulp_im2row_fp32, &im2col_args);

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

  #else
  
  for (int ch=0; ch<C_in; ch++) 
  {
    for (int ho=0; ho<H_out; ho++) 
    {
      for (int wo=0; wo<W_out; wo++)
      {
        float temp = 0;
        for (int hk=0; hk<pH; hk++) 
        {
          for (int wk=0; wk<pW; wk++)
          {
            temp += coeffData[wk + hk*pW + ch*pH*pW] * inData[wo+wk + (ho+hk)*W_in + ch*H_in*W_in];
          }
        }
        outData[wo + ho*W_out + ch*H_out*W_out] = temp;
      }
    }
  } 

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



void pulp_conv_dw_fp32_bw_cl( void * DepthWise_Conv_args )
{
  struct DepthWise_Conv_args * DW_args = (struct DepthWise_Conv_args *) DepthWise_Conv_args;
  int skip_in_grad = DW_args->skip_in_grad;

  pulp_conv_dw_fp32_bw_param_grads_cl(DepthWise_Conv_args); 
  if (skip_in_grad == 0)
  {
    pulp_conv_dw_fp32_bw_input_grads_cl(DepthWise_Conv_args); 
  }
}



void pulp_conv_dw_fp32_bw_param_grads_cl( void * DepthWise_Conv_args )
{
  struct DepthWise_Conv_args * DW_args = (struct DepthWise_Conv_args *) DepthWise_Conv_args;
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

  //input dimensions
  int W_in = DW_args->input->W;
  int H_in = DW_args->input->H;
  int C_in = DW_args->input->C;
  //kernel dimensions
  int pW = DW_args->coeff->W;
  int pH = DW_args->coeff->H;
  //output dimensions
  int W_out = DW_args->output->W;
  int H_out = DW_args->output->H;
  int C_out = DW_args->output->C;

  float * inData = DW_args->input->data;
  float * inDiff = DW_args->input->diff;
  float * coeffData = DW_args->coeff->data;
  float * coeffDiff = DW_args->coeff->diff;
  float * outDiff = DW_args->output->diff;
  float * outData = DW_args->output->data;

  float * i2c_buffer = DW_args->i2c_buffer;

  int input_layout = DW_args->HWC;
  int opt_matmul_type = DW_args->opt_matmul_type_wg;

  #ifndef NAIVE_DW

  im2col_args.input = DW_args->input; 
  im2col_args.c = DW_args->coeff;
  im2col_args.output = DW_args->output; 
  im2col_args.pBuffer = DW_args->i2c_buffer;
  im2col_args.Lpad = 0;
  im2col_args.Rpad = 0;
  im2col_args.Upad = 0;
  im2col_args.Dpad = 0;
  im2col_args.mod = 0;
  im2col_args.stride_h = 1;
  im2col_args.stride_w = 1;
  im2col_args.USE_DMA = 0;
  im2col_args.HWC = input_layout;

  pi_cl_team_fork(NUM_CORES, pulp_im2row_fp32, &im2col_args);

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

  #else

  for (int ch=0; ch<C_in; ch++) 
  {
    for (int hk=0; hk<pH; hk++)
    {
      for (int wk=0; wk<pW; wk++) 
      {
        float temp = 0;
        for (int ho=0; ho<H_out; ho++)
        {
          for (int wo=0; wo<W_out; wo++) 
          {
            temp += inData[wk+wo + (hk+ho)*W_in + ch*H_in*W_in] * outDiff[wo + ho*W_out + ch*H_out*W_out];
          }
        }
        coeffDiff[wk + hk*pW + ch*pH*pW] = temp;
      }
    }
  }


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



void pulp_conv_dw_fp32_bw_input_grads_cl( void * DepthWise_Conv_args )
{
  struct DepthWise_Conv_args * DW_args = (struct DepthWise_Conv_args *) DepthWise_Conv_args;
  struct matMul_DW_args matMul_args;
  struct im2col_args im2col_args;

  //input dimensions
  int W_in = DW_args->input->W;
  int H_in = DW_args->input->H;
  int C_in = DW_args->input->C;
  //kernel dimensions
  int pW = DW_args->coeff->W;
  int pH = DW_args->coeff->H;
  //output dimensions
  int W_out = DW_args->output->W;
  int H_out = DW_args->output->H;
  int C_out = DW_args->output->C;

  float * inData = DW_args->input->data;
  float * inDiff = DW_args->input->diff;
  float * coeffData = DW_args->coeff->data;
  float * coeffDiff = DW_args->coeff->diff;
  float * outDiff = DW_args->output->diff;
  float * outData = DW_args->output->data;

  float * i2c_buffer = DW_args->i2c_buffer;
  
  int output_layout = DW_args->HWC;
  int opt_matmul_type = DW_args->opt_matmul_type_ig;

  #ifndef NAIVE_DW

  // PREPARE im2col_buffer for ACTIV_GRAD
  im2col_args.input = DW_args->input;
  im2col_args.c = DW_args->coeff;
  im2col_args.output = DW_args->output;
  im2col_args.pBuffer = i2c_buffer;
  im2col_args.Lpad = 0; //Lpad;
  im2col_args.Rpad = 0; //Rpad;
  im2col_args.Upad = 0; //Upad;
  im2col_args.Dpad = 0; //Dpad;
  im2col_args.mod = 1;
  im2col_args.stride_h = 1;
  im2col_args.stride_w = 1;
  im2col_args.USE_DMA = 0;
  im2col_args.HWC = output_layout;

  pi_cl_team_fork(NUM_CORES, pulp_im2row_fp32, &im2col_args);

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

  #else

  for (int ch=0; ch<C_in; ch++) 
  {
    for (int hin=0; hin<H_in; hin++)
    {
      int ho = hin - pH + 1;
      for (int win=0; win<W_in; win++) 
      {
        int wo = win - pW + 1;
        float temp = 0;
        for (int hk=0; hk<pH; hk++)
        {
          for (int wk=0; wk<pW; wk++)
          {
            if ((wo+wk>=0) && (ho+hk>=0) && (wo+wk<W_out) && (ho+hk<H_out)) {
              temp += coeffData[(pW-1-wk) + (pH-1-hk)*pW + ch*pH*pW] * outDiff[(wo+wk) + (ho+hk)*W_out + ch*H_out*W_out]; 
            }
          }
        }
        inDiff[win + hin*W_in + ch*H_in*W_in] = temp;
      }
    }
  }

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
