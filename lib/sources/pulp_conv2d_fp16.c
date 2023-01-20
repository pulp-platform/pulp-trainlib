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
#include "pulp_im2col_fp16.h"
#include "pulp_conv2d_fp16.h"

void pulp_conv2d_fp16_fw_cl( void * Conv2D_args_fp16 )
{
    struct Conv2D_args_fp16 * C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *outData = C2D_args->output->data;
    fp16 *inData = C2D_args->input->data;

    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_in = C2D_args->input->C;
    int C_out = C2D_args->output->C;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int Lpad = C2D_args->Lpad;
    int Rpad = C2D_args->Rpad;
    int Upad = C2D_args->Upad;
    int Dpad = C2D_args->Dpad;

    fp16 * i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int USE_DMA = C2D_args->USE_DMA_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_fw;

    #ifdef DEBUG
    int in_size = C2D_args->input->dim;
    int ker_size = C2D_args->coeff->dim;
    #endif

  if (USE_IM2COL == 1) {

    // im2col on the input data
    im2col_args.input = C2D_args->input;
    im2col_args.c = C2D_args->coeff;
    im2col_args.output = C2D_args->output;
    im2col_args.pBuffer = i2c_buffer;
    im2col_args.Lpad = Lpad;
    im2col_args.Rpad = Rpad;
    im2col_args.Upad = Upad;
    im2col_args.Dpad = Dpad;
    im2col_args.mod = 0;
    im2col_args.stride_w = stride_w;
    im2col_args.stride_h = stride_h;
    im2col_args.USE_DMA = USE_DMA;
    im2col_args.HWC = HWC_layout;

    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);

    #ifdef DEBUG
    printf("\nForward input data (size: %d, address: %x):\n", in_size, inData);
    for(int index=0; index<in_size; index++) {
      printf("%f ", inData[index]);
    }
    printf("\n");

    printf("\nForward i2c buffer (size: %d, address: %x):\n", pW*pH*C_in*H_out*W_out, i2c_buffer);
    for(int index=0; index<pW*pH*C_in*H_out*W_out; index++) {
      printf("%f ", i2c_buffer[index]);
    }
    printf("\n");

    printf("\nForward kernel (size: %d, address: %x):\n", ker_size, coeffData);
    for(int index=0; index<ker_size; index++) {
      printf("%f ", coeffData[index]);
    }
    printf("\n\n");
    #endif

    matMul_args.A = coeffData;
    matMul_args.B = i2c_buffer;
    matMul_args.C = outData;
    matMul_args.N = C_out;
    matMul_args.K = pW*pH*C_in;
    matMul_args.M = (W_in-pW+stride_w+Lpad+Rpad)/stride_w*(H_in-pH+stride_h+Upad+Dpad)/stride_h;
    matMul_args.trans_B = 1;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
    #else
    struct mm_manager_args_fp16 man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = LAYER_CONV2D;
    man_args.step_type = STEP_FW;
    man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
    #endif

  }

  // Use naive kernel
  else if (USE_IM2COL == 0) {

    matMul_args.A = inData;
    matMul_args.B = coeffData;
    matMul_args.C = outData;
    matMul_args.H = H_in;
    matMul_args.W = W_in;
    matMul_args.pCin = C_in;
    matMul_args.pCout = C_out;
    matMul_args.pH = pH;
    matMul_args.pW = pW;

    pi_cl_team_fork(NUM_CORES, naive_conv2d_fw_kernel_CHW_fp16, &matMul_args);
    
  }

  // ERROR IN SELECTING IM2COL
  else {
    printf("[pulp_conv2d_fp16_fw_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
  }

    #ifdef DEBUG
    // to PRINT outData orderly
    printf("FORWARD OUTPUT CONV2D LAYER \n\n");
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
}



void pulp_conv2d_fp16_bw_cl( void * Conv2D_args_fp16 )
{
    struct Conv2D_args_fp16 * C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    int skip_in_grad = C2D_args->skip_in_grad;

    pulp_conv2d_fp16_bw_param_grads_cl(Conv2D_args_fp16); 
    if (skip_in_grad == 0)
    {
      pulp_conv2d_fp16_bw_input_grads_cl(Conv2D_args_fp16); 
    }
}



void pulp_conv2d_fp16_bw_param_grads_cl( void * Conv2D_args_fp16 )
{
    struct Conv2D_args_fp16 * C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    //input dimensions
    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int C_in = C2D_args->input->C;
    //kernel dimensions
    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    //output dimensions
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_out = C2D_args->output->C;

    fp16 * inData = C2D_args->input->data;
    fp16 * inDiff = C2D_args->input->diff;
    fp16 * coeffData = C2D_args->coeff->data;
    fp16 * coeffDiff = C2D_args->coeff->diff;
    fp16 * outDiff = C2D_args->output->diff;
    fp16 * outData = C2D_args->output->data;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int Lpad = C2D_args->Lpad;
    int Rpad = C2D_args->Rpad;
    int Upad = C2D_args->Upad;
    int Dpad = C2D_args->Dpad;

    fp16 * i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int USE_DMA = C2D_args->USE_DMA_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_wg;
    

  if (USE_IM2COL == 1) {

    im2col_args.input = C2D_args->input;
    im2col_args.c = C2D_args->coeff;
    im2col_args.output = C2D_args->output;
    im2col_args.pBuffer = i2c_buffer;
    im2col_args.Lpad = 0;
    im2col_args.Rpad = 0;
    im2col_args.Upad = 0;
    im2col_args.Dpad = 0;
    im2col_args.mod = 0;
    im2col_args.stride_w = stride_w;
    im2col_args.stride_h = stride_h;
    im2col_args.USE_DMA = USE_DMA;
    im2col_args.HWC = HWC_layout;

    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);

    matMul_args.A = outDiff;
    matMul_args.B = i2c_buffer;
    matMul_args.C = coeffDiff;
    matMul_args.N = C_out; 
    matMul_args.K = H_out*W_out; 
    matMul_args.M = pW*pH*C_in; 
    matMul_args.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
    #else
    struct mm_manager_args_fp16 man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = LAYER_CONV2D;
    man_args.step_type = STEP_WGT_GRAD;
    man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
    #endif

  }

  else if (USE_IM2COL == 0) {

    matMul_args.A = inData;
    matMul_args.B = coeffDiff;
    matMul_args.C = outDiff;
    matMul_args.H = H_in;
    matMul_args.W = W_in;
    matMul_args.pCin = C_in;
    matMul_args.pCout = C_out;
    matMul_args.pH = pH;
    matMul_args.pW = pW;

    pi_cl_team_fork(NUM_CORES, naive_conv2d_param_grad_kernel_CHW_fp16, &matMul_args);

  }

  else {
    printf("[pulp_conv2d_fp16_bw_param_grads_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
  }
    

  #ifdef DEBUG
  printf("\nBackward outDiff data (size: %d, address: %x):\n", C_out*W_out*H_out, outDiff);
  for(int index=0; index<C_out*W_out*H_out; index++) {
    if(!(index%(W_out))) printf("\n");
    if(!(index%(H_out*W_out))) printf("\n");
    printf("%f ", outDiff[index]);
  }
  printf("\n");

  printf("\nWeights (size: %d, address: %x):\n", pW*pH*C_out*C_in, coeffData);
  for(int index=0; index<pW*pH*C_out*C_in; index++) {
    if(!(index%(pW))) printf("\n");
    if(!(index%(pW*pH))) printf("\n");
    printf("%f ", coeffData[index]);
  }
  printf("\n");

  printf("\nBackward i2c buffer (size: %d, address: %x):\n", pW*pH*C_out*H_in*W_in, i2c_buffer);
  for(int index=0; index<pW*pH*C_in*H_in*W_in; index++) {
    if(!(index%(pW*pH))) printf("\n");
    if(!(index%(pW*pH*C_in))) printf("\n");
    printf("%f ", i2c_buffer[index]);
  }
  printf("\n");

  printf("\nBackward gradDiff (size: %d, address: %x):\n", pW*pH*C_out*C_in, coeffDiff);
  for(int index=0; index<pW*pH*C_out*C_in; index++) {
    if(!(index%(pW))) printf("\n");
    if(!(index%(pW*pH))) printf("\n");
    printf("%f ", inDiff[index]);
  }
  printf("\n\n");
  #endif

  #ifdef DEBUG
  printf("COEFF GRADIENT CONV2D LAYER \n\n");
  for (int i=0; i<pW*pH*C_in*C_out; i++) {
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



void pulp_conv2d_fp16_bw_input_grads_cl( void * Conv2D_args_fp16 )
{
  struct Conv2D_args_fp16 * C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
  struct matMul_args_fp16 matMul_args;
  struct im2col_args_fp16 im2col_args;

  //input dimensions
  int W_in = C2D_args->input->W;
  int H_in = C2D_args->input->H;
  int C_in = C2D_args->input->C;
  //kernel dimensions
  int pW = C2D_args->coeff->W;
  int pH = C2D_args->coeff->H;
  //output dimensions
  int W_out = C2D_args->output->W;
  int H_out = C2D_args->output->H;
  int C_out = C2D_args->output->C;

  fp16 * inData = C2D_args->input->data;
  fp16 * inDiff = C2D_args->input->diff;
  fp16 * coeffData = C2D_args->coeff->data;
  fp16 * coeffDiff = C2D_args->coeff->diff;
  fp16 * outDiff = C2D_args->output->diff;
  fp16 * outData = C2D_args->output->data;

  fp16 * i2c_buffer = C2D_args->i2c_buffer;
  fp16 * temp_bt = C2D_args->bt_buffer;

  int stride_w = C2D_args->stride_w;
  int stride_h = C2D_args->stride_h;
  int Lpad = C2D_args->Lpad;
  int Rpad = C2D_args->Rpad;
  int Upad = C2D_args->Upad;
  int Dpad = C2D_args->Dpad;

  int HWC_layout = C2D_args->HWC;
  int USE_IM2COL = C2D_args->USE_IM2COL;
  int USE_DMA = C2D_args->USE_DMA_IM2COL;
  int opt_matmul_type = C2D_args->opt_matmul_type_ig;


  if (USE_IM2COL == 1) {

    // PREPARE im2col_buffer for ACTIV_GRAD
    im2col_args.input = C2D_args->input;
    im2col_args.c = C2D_args->coeff;
    im2col_args.output = C2D_args->output;
    im2col_args.pBuffer = i2c_buffer;
    im2col_args.Lpad = 0; //pW-1;
    im2col_args.Rpad = 0; //pW-1;
    im2col_args.Upad = 0; //pH-1;
    im2col_args.Dpad = 0; //pH-1;
    im2col_args.stride_h = 1;
    im2col_args.stride_w = 1;
    im2col_args.mod = 1;
    im2col_args.USE_DMA = USE_DMA; 
    im2col_args.HWC = HWC_layout;

    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);

    // Blocktranspose weights
    struct blocktransp_args_fp16 bt_args;
    bt_args.weights = coeffData;
    bt_args.bt_weights = temp_bt;
    bt_args.Cout = C_out;
    bt_args.Cin = C_in;
    bt_args.Hk = pH;
    bt_args.Wk = pW;

    matMul_args.A = temp_bt; //coeffData;
    matMul_args.B = i2c_buffer;
    matMul_args.C = inDiff;
    matMul_args.N = C_in;
    matMul_args.K = pW*pH*C_out;
    matMul_args.M = W_in*H_in;
    matMul_args.trans_B = 1;

    pi_cl_team_fork(NUM_CORES, pulp_blocktransp_fp16, &bt_args);

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
    #else
    struct mm_manager_args_fp16 man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = LAYER_CONV2D;
    man_args.step_type = STEP_IN_GRAD;
    man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
    #endif

  }

  else if (USE_IM2COL == 0) {

    matMul_args.A = inDiff;
    matMul_args.B = coeffData;
    matMul_args.C = outDiff;
    matMul_args.H = H_in;
    matMul_args.W = W_in;
    matMul_args.pCin = C_in;
    matMul_args.pCout = C_out;
    matMul_args.pH = pH;
    matMul_args.pW = pW;

    pi_cl_team_fork(NUM_CORES, naive_conv2d_in_grad_kernel_CHW_fp16, &matMul_args);

  }

  else {
    printf("[pulp_conv2d_fp16_bw_input_grads_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
  }  

  #ifdef DEBUG
  printf("\nBackward outDiff data (size: %d, address: %x):\n", C_out*W_out*H_out, outDiff);
  for(int index=0; index<C_out*W_out*H_out; index++) {
    if(!(index%(W_out))) printf("\n");
    if(!(index%(H_out*W_out))) printf("\n");
    printf("%f ", outDiff[index]);
  }
  printf("\n");

  printf("\nWeights (size: %d, address: %x):\n", pW*pH*C_out*C_in, coeffData);
  for(int index=0; index<pW*pH*C_out*C_in; index++) {
    if(!(index%(pW))) printf("\n");
    if(!(index%(pW*pH))) printf("\n");
    printf("%f ", coeffData[index]);
  }
  printf("\n");

  printf("\nBackward i2c buffer (size: %d, address: %x):\n", pW*pH*C_out*H_in*W_in, i2c_buffer);
  for(int index=0; index<pW*pH*C_in*H_in*W_in; index++) {
    if(!(index%(pW*pH))) printf("\n");
    if(!(index%(pW*pH*C_in))) printf("\n");
    printf("%f ", i2c_buffer[index]);
  }
  printf("\n");

  printf("\nBackward inDiff (size: %d, address: %x):\n", C_in*H_in*W_in, inDiff);
  for(int index=0; index<C_in*H_in*W_in; index++) {
    if(!(index%(W_in))) printf("\n");
    if(!(index%(W_in*H_in))) printf("\n");
    printf("%f ", inDiff[index]);
  }
  printf("\n\n");
  #endif


  #ifdef DEBUG
  // to PRINT outDiff orderly
  printf("ERROR PROP CONV2D LAYER \n\n");
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
