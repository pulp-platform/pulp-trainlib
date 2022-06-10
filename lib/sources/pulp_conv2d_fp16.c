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

void pulp_conv2d_fp16_fw_cl(struct blob_fp16 * input, struct blob_fp16 * coeff, struct blob_fp16 * output, int pad, fp16 * i2c_buffer)
{
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = coeff->W;
    int pH = coeff->H;
    fp16 *coeffData = coeff->data;
    fp16 *outData = output->data;
    fp16 *inData = input->data;

    int W_in = input->W;
    int H_in = input->H;
    int W_out = output->W;
    int H_out = output->H;
    int C_in = input->C;
    int C_out = output->C;

    #ifdef DEBUG
    int in_size = input->dim;
    int ker_size = coeff->dim;
    #endif

    // im2col on the input data
    im2col_args.input = input;
    im2col_args.c = coeff;
    im2col_args.output = output;
    im2col_args.pBuffer = i2c_buffer;
    im2col_args.pad = pad;
    im2col_args.mod = 0;
    im2col_args.tile_start = 0;
    im2col_args.tile_h = pH*pW;
    im2col_args.DW = 0;

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
    matMul_args.M = (W_in-pW+1)*(H_in-pH+1);
    matMul_args.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
    #else
    struct mm_manager_args_fp16 man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = LAYER_CONV2D;
    man_args.step_type = STEP_FW;
    man_args.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
    #endif

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



void pulp_conv2d_fp16_bw_cl(struct blob_fp16 * input, struct blob_fp16 * coeff, struct blob_fp16 * output, int pad, fp16 * i2c_buffer)
{
    pulp_conv2d_fp16_bw_param_grads_cl(input, coeff, output, pad, i2c_buffer);
    pulp_conv2d_fp16_bw_input_grads_cl(input, coeff, output, pad, i2c_buffer);
}



void pulp_conv2d_fp16_bw_param_grads_cl(struct blob_fp16 * input, struct blob_fp16 * coeff, struct blob_fp16 * output, int pad, fp16 * i2c_buffer)
{
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

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

    fp16 * inData = input->data;
    fp16 * inDiff = input->diff;
    fp16 * coeffData = coeff->data;
    fp16 * coeffDiff = coeff->diff;
    fp16 * outDiff = output->diff;
    fp16 * outData = output->data;

    im2col_args.input = input;
    im2col_args.c = coeff;
    im2col_args.output = output;
    im2col_args.pBuffer = i2c_buffer;
    im2col_args.pad = 0;
    im2col_args.mod = 0;
    im2col_args.tile_start = 0;
    im2col_args.tile_h = H_out;
    im2col_args.DW = 0;

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
    man_args.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
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



void pulp_conv2d_fp16_bw_input_grads_cl(struct blob_fp16 * input, struct blob_fp16 * coeff, struct blob_fp16 * output, int pad, fp16 * i2c_buffer)
{
  struct matMul_args_fp16 matMul_args;
  struct im2col_args_fp16 im2col_args;

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

  fp16 * inData = input->data;
  fp16 * inDiff = input->diff;
  fp16 * coeffData = coeff->data;
  fp16 * coeffDiff = coeff->diff;
  fp16 * outDiff = output->diff;
  fp16 * outData = output->data;

  // PREPARE im2col_buffer for ACTIV_GRAD
  im2col_args.input = input;
  im2col_args.c = coeff;
  im2col_args.output = output;
  im2col_args.pBuffer = i2c_buffer;
  im2col_args.pad = pW-1; //1;
  im2col_args.mod = 1;
  im2col_args.DW = 0; 

  if (H_in == pH) im2col_args.pad = 2;

  pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);

  // Iterate to accumulate over adjacent receptive fields
//for (int rec_field = 0; rec_field < H_in*W_in; rec_field++)
//{
    matMul_args.A = coeffData;
    matMul_args.B = i2c_buffer;
    matMul_args.C = inDiff;
    matMul_args.N = C_in;
    matMul_args.K = pW*pH*C_out; //pW*pH;
    matMul_args.M = W_in*H_in;
    matMul_args.trans_B = 1;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
    #else
    struct mm_manager_args_fp16 man_args;
    man_args.mm_args = &matMul_args;
    man_args.layer_type = LAYER_CONV2D;
    man_args.step_type = STEP_IN_GRAD;
    man_args.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
    #endif
//}

    #ifdef DEBUG
    printf("\nBackward outDiff data (size: %d, address: %x):\n", output->dim, outDiff);
    for(int index=0; index<output->dim; index++) {
      printf("%f ", outDiff[index]);
    }
    printf("\n");

    printf("\nForward i2c buffer (size: %d, address: %x):\n", pW*pH*C_in*H_out*W_out, i2c_buffer);
    for(int index=0; index<pW*pH*C_in*H_out*W_out; index++) {
      printf("%f ", i2c_buffer[index]);
    }
    printf("\n");

    printf("\nBackward inDiff (size: %d, address: %x):\n", input->dim, inDiff);
    for(int index=0; index<input->dim; index++) {
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
