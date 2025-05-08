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
 * Authors: Davide Nadalini
*/ 

#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_im2col_fp16.h"
#include "pulp_transp_conv2d_fp16.h"
#include "pulp_conv_naive_fp16.h"

void pulp_transp_conv2d_fp16_fw_cl( void * Transp_Conv2D_args_fp16 )
{
    struct Transp_Conv2D_args_fp16 * C2D_args = (struct Transp_Conv2D_args_fp16 *) Transp_Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *biasData = C2D_args->bias->data;
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
    int inLpad = C2D_args->inLpad;
    int inRpad = C2D_args->inRpad;
    int inUpad = C2D_args->inUpad;
    int inDpad = C2D_args->inDpad;

    fp16 * i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_fw;

  /**
   * USE OPTIMIZED ALGORITHM
   */
  if (USE_IM2COL == 1) {

      /**
       * USE CHW LAYOUT
       */
      if (HWC_layout == 0) {
        printf("[pulp_transp_conv2d_fp16_fw_cl]: Unimplemented CHW Im2Col + MM version!");
      }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
        printf("[pulp_transp_conv2d_fp16_fw_cl]: Unimplemented WHC Im2Col + MM version!");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_fw_cl]: Invalid data layout format (HWC or CHW)!\n");
    }
  }

  /**
   * USE NAIVE KERNEL 
   */
  else if (USE_IM2COL == 0) {

    /**
     * USE CHW DATA LAYOUT
     */
    if (HWC_layout == 0) {
      matMul_args.A = inData;
      matMul_args.B = coeffData;
      matMul_args.C = outData;
      matMul_args.bias = biasData;
      matMul_args.USE_BIASES = USE_BIASES;
      matMul_args.H = H_in;
      matMul_args.W = W_in;
      matMul_args.pCin = C_in;
      matMul_args.pCout = C_out;
      matMul_args.pH = pH;
      matMul_args.pW = pW;
      // Stride and padding operators
      matMul_args.stride_h = stride_h;
      matMul_args.stride_w = stride_w;
      matMul_args.Lpad = inLpad;
      matMul_args.Rpad = inRpad;
      matMul_args.Upad = inUpad;
      matMul_args.Dpad = inDpad;

      #ifdef OPTIMIZE
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_fw_kernel_CHW_fp16, &matMul_args);
      #else
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_fw_kernel_CHW_fp16, &matMul_args);
      #endif
    }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
      printf("[pulp_transp_conv2d_fp16_fw_cl:] Naive kernel for HWC FW not implemented!\n");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_fw_cl:] Invalid data layout format (HWC or CHW)!\n");
    }
  }

  // ERROR IN SELECTING IM2COL
  else {
    printf("[pulp_transp_conv2d_fp16_fw_cl:] Invalid selection of the transp conv2d algorithm (im2col or not)\n");
  }
}



void pulp_transp_conv2d_fp16_bw_cl( void * Transp_Conv2D_args_fp16 )
{
    struct Transp_Conv2D_args_fp16 * C2D_args = (struct Transp_Conv2D_args_fp16 *) Transp_Conv2D_args_fp16;
    int skip_wg_grad = C2D_args->skip_wg_grad;
    int skip_in_grad = C2D_args->skip_in_grad;

    if (skip_wg_grad == 0)
    {
      pulp_transp_conv2d_fp16_bw_param_grads_cl(Transp_Conv2D_args_fp16);
    }

    if (skip_in_grad == 0)
    {
      pulp_transp_conv2d_fp16_bw_input_grads_cl(Transp_Conv2D_args_fp16); 
    }
}



void pulp_transp_conv2d_fp16_bw_param_grads_cl( void * Transp_Conv2D_args_fp16 ) {
    struct Transp_Conv2D_args_fp16 * C2D_args = (struct Transp_Conv2D_args_fp16 *) Transp_Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    fp16 *coeffDiff = C2D_args->coeff->diff;
    fp16 *biasDiff = C2D_args->bias->diff;
    fp16 *outDiff = C2D_args->output->diff;
    fp16 *inData = C2D_args->input->data;

    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_in = C2D_args->input->C;
    int C_out = C2D_args->output->C;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int inLpad = C2D_args->inLpad;
    int inRpad = C2D_args->inRpad;
    int inUpad = C2D_args->inUpad;
    int inDpad = C2D_args->inDpad;

    fp16 * i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_wg;

  /**
   * USE OPTIMIZED ALGORITHM
   */
  if (USE_IM2COL == 1) {

      /**
       * USE CHW LAYOUT
       */
      if (HWC_layout == 0) {
        printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl]: Unimplemented CHW Im2Col + MM version!");
      }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
        printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl]: Unimplemented WHC Im2Col + MM version!");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl]: Invalid data layout format (HWC or CHW)!\n");
    }
  }

  /**
   * USE NAIVE KERNEL 
   */
  else if (USE_IM2COL == 0) {

    /**
     * USE CHW DATA LAYOUT
     */
    if (HWC_layout == 0) {
      matMul_args.A = inData;
      matMul_args.B = coeffDiff;
      matMul_args.C = outDiff;
      matMul_args.bias = biasDiff;
      matMul_args.USE_BIASES = USE_BIASES;
      matMul_args.H = H_in;
      matMul_args.W = W_in;
      matMul_args.pCin = C_in;
      matMul_args.pCout = C_out;
      matMul_args.pH = pH;
      matMul_args.pW = pW;
      // Stride and padding operators
      matMul_args.stride_h = stride_h;
      matMul_args.stride_w = stride_w;
      matMul_args.Lpad = inLpad;
      matMul_args.Rpad = inRpad;
      matMul_args.Upad = inUpad;
      matMul_args.Dpad = inDpad;

      #ifdef OPTIMIZE
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_param_grad_kernel_CHW_fp16, &matMul_args);
      #else
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_param_grad_kernel_CHW_fp16, &matMul_args);
      #endif
    }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
      printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl:] Naive kernel for HWC FW not implemented!\n");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
    }
  }

  // ERROR IN SELECTING IM2COL
  else {
    printf("[pulp_transp_conv2d_fp16_bw_param_grads_cl:] Invalid selection of the transp conv2d algorithm (im2col or not)\n");
  }
}



void pulp_transp_conv2d_fp16_bw_input_grads_cl( void * Transp_Conv2D_args_fp16 ) {
    struct Transp_Conv2D_args_fp16 * C2D_args = (struct Transp_Conv2D_args_fp16 *) Transp_Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *biasData = C2D_args->bias->data;
    fp16 *outDiff = C2D_args->output->diff;
    fp16 *inDiff = C2D_args->input->diff;

    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_in = C2D_args->input->C;
    int C_out = C2D_args->output->C;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int inLpad = C2D_args->inLpad;
    int inRpad = C2D_args->inRpad;
    int inUpad = C2D_args->inUpad;
    int inDpad = C2D_args->inDpad;

    fp16 * i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_ig;

  /**
   * USE OPTIMIZED ALGORITHM
   */
  if (USE_IM2COL == 1) {

      /**
       * USE CHW LAYOUT
       */
      if (HWC_layout == 0) {
        printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl]: Unimplemented CHW Im2Col + MM version!");
      }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
        printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl]: Unimplemented WHC Im2Col + MM version!");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl]: Invalid data layout format (HWC or CHW)!\n");
    }
  }

  /**
   * USE NAIVE KERNEL 
   */
  else if (USE_IM2COL == 0) {

    /**
     * USE CHW DATA LAYOUT
     */
    if (HWC_layout == 0) {
      matMul_args.A = inDiff;
      matMul_args.B = coeffData;
      matMul_args.C = outDiff;
      //matMul_args.bias = biasData;
      matMul_args.USE_BIASES = USE_BIASES;
      matMul_args.H = H_in;
      matMul_args.W = W_in;
      matMul_args.pCin = C_in;
      matMul_args.pCout = C_out;
      matMul_args.pH = pH;
      matMul_args.pW = pW;
      // Stride and padding operators
      matMul_args.stride_h = stride_h;
      matMul_args.stride_w = stride_w;
      matMul_args.Lpad = inLpad;
      matMul_args.Rpad = inRpad;
      matMul_args.Upad = inUpad;
      matMul_args.Dpad = inDpad;

      #ifdef OPTIMIZE
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_in_grad_kernel_CHW_fp16, &matMul_args);
      #else
      pi_cl_team_fork(NUM_CORES, naive_transp_conv2d_in_grad_kernel_CHW_fp16, &matMul_args);
      #endif
    }

    /**
     * USE HWC DATA LAYOUT
     */
    else if (HWC_layout == 1) {
      printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl:] Naive kernel for HWC FW not implemented!\n");
    }
    else {
      printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
    }
  }

  // ERROR IN SELECTING IM2COL
  else {
    printf("[pulp_transp_conv2d_fp16_bw_input_grads_cl:] Invalid selection of the transp conv2d algorithm (im2col or not)\n");
  }
}


