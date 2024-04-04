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
#include "pulp_conv_naive_fp32.h"

#include "pmsis.h"

/** DEPTHWISE CONVOLUTION KERNELS **/

// Naive forward kernel for DepthWise Convolution
void dw_kernel_forward(void * kernel_DW_args) {

  struct kernel_DW_args * args = (struct kernel_DW_args *) kernel_DW_args;
  float * inData = args->input->data;
  float * coeffData = args->weights->data;
  float * outData = args->output->data;

  uint32_t C_in = args->input->C;
  uint32_t H_in = args->input->H;
  uint32_t W_in = args->input->W;
  uint32_t pH = args->weights->H;
  uint32_t pW = args->weights->W;
  uint32_t H_out = args->output->H;
  uint32_t W_out = args->output->W;

  uint32_t blockSize = (C_in+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > C_in ? C_in : start+blockSize;

  for (int ch=start; ch<stop; ch++) 
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

}



// Naive weight grad kernel for DepthWise Convolution
void dw_kernel_weight_grad(void * kernel_DW_args) {

  struct kernel_DW_args * args = (struct kernel_DW_args *) kernel_DW_args;
  float * inData = args->input->data;
  float * coeffDiff = args->weights->diff;
  float * outDiff = args->output->diff;

  uint32_t C_in = args->input->C;
  uint32_t H_in = args->input->H;
  uint32_t W_in = args->input->W;
  uint32_t pH = args->weights->H;
  uint32_t pW = args->weights->W;
  uint32_t H_out = args->output->H;
  uint32_t W_out = args->output->W;

  uint32_t blockSize = (C_in+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > C_in ? C_in : start+blockSize;

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

}



// Naive input grad kernel for DepthWise Convolution
void dw_kernel_input_grad(void * kernel_DW_args) {

  struct kernel_DW_args * args = (struct kernel_DW_args *) kernel_DW_args;
  float * inDiff = args->input->diff;
  float * coeffData = args->weights->data;
  float * outDiff = args->output->diff;

  uint32_t C_in = args->input->C;
  uint32_t H_in = args->input->H;
  uint32_t W_in = args->input->W;
  uint32_t pH = args->weights->H;
  uint32_t pW = args->weights->W;
  uint32_t H_out = args->output->H;
  uint32_t W_out = args->output->W;

  uint32_t blockSize = (C_in+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > C_in ? C_in : start+blockSize;

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

}



/** CONV2D KERNELS **/

void naive_conv2d_fw_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inData = args->A;
  float * __restrict__ coeffData = args->B;
  float * __restrict__ outData = args->C;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - pH + Upad + Dpad)/h_str + 1;
  const uint32_t W_out = (W_in - pW + Lpad + Rpad)/w_str + 1;

  const uint32_t blockSize = (C_out+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_out ? C_out : start+blockSize;  

  int padding = Lpad + Rpad + Upad + Dpad;

  if (padding == 0) {
    for (uint32_t co=start; co<stop; co++) {
      for (uint32_t ho=0; ho<H_out; ho++) {
        for (uint32_t wo=0; wo<W_out; wo++) {
          float temp = 0;
          // Receptive field
          for (uint32_t ci=0; ci<C_in; ci++) {
            for (uint32_t hk=0; hk<pH; hk++) {
              for (uint32_t wk=0; wk<pW; wk++) {
                temp += inData[w_str*wo+wk+(h_str*ho+hk)*W_in+ci*H_in*W_in] * coeffData[wk+hk*pW+ci*pH*pW+co*C_in*pH*pW];
              }
            }
          }
          outData[wo+ho*W_out+co*H_out*W_out] = temp;
          //printf("C2D_KER:   outData[%d] = %f\n", wo+ho*W_out+co*H_out*W_out, outData[wo+ho*W_out+co*H_out*W_out]);
        }
      }
    } 
  }
  else {
    
    for (uint32_t co=start; co<stop; co++) {
      for (uint32_t ho=0; ho<H_out; ho++) {
        for (uint32_t wo=0; wo<W_out; wo++) {
          float temp = 0;
          // Receptive field
          for (uint32_t ci=0; ci<C_in; ci++) {
            for (uint32_t hk=0; hk<pH; hk++) {
              for (uint32_t wk=0; wk<pW; wk++) {
                // Pad conditions
                int pad_cond_h = h_str*ho + hk - Upad;
                int pad_cond_w = w_str*wo + wk - Lpad;
                if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                    int in_idx = (w_str*wo + wk - Lpad) + (h_str*ho + hk - Upad)*W_in + ci*H_in*W_in;
                    int ker_idx = wk + hk*pW + ci*pH*pW + co*C_in*pH*pW;
                    temp += inData[in_idx] * coeffData[ker_idx];
                }
              }
            }
          }
          outData[wo+ho*W_out+co*H_out*W_out] = temp;
        }
      }
    } 

  }
}



void naive_conv2d_param_grad_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inData = args->A;
  float * __restrict__ coeffDiff = args->B;
  float * __restrict__ outDiff = args->C;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - pH + Upad + Dpad)/h_str + 1;
  const uint32_t W_out = (W_in - pW + Lpad + Rpad)/w_str + 1;

  const uint32_t blockSize = (C_out+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_out ? C_out : start+blockSize;  

  int padding = Lpad + Rpad + Upad + Dpad;

  if (padding == 0) {
    for (uint32_t co=start; co<stop; co++) {
      for (uint32_t hk=0; hk<pH; hk++) {
        for (uint32_t wk=0; wk<pW; wk++) {
          for (uint32_t ci=0; ci<C_in; ci++) {
            float temp = 0;
            for (uint32_t ho=0; ho<H_out; ho++) {
              for (uint32_t wo=0; wo<W_out; wo++) {
                temp += outDiff[wo+ho*W_out+co*H_out*W_out] * inData[w_str*wo+wk+(h_str*ho+hk)*W_in+ci*H_in*W_in];
              }
            }
            coeffDiff[wk+hk*pW+ci*pH*pW+co*pH*pW*C_in] = temp;
          }
        }
      }
    }
  }
  else {
    
    for (uint32_t co=start; co<stop; co++) {
      for (uint32_t hk=0; hk<pH; hk++) {
        for (uint32_t wk=0; wk<pW; wk++) {
          for (uint32_t ci=0; ci<C_in; ci++) {
            float temp = 0;
            for (uint32_t ho=0; ho<H_out; ho++) {
              for (uint32_t wo=0; wo<W_out; wo++) {
                // Pad conditions
                int pad_cond_h = h_str*ho + hk - Upad;
                int pad_cond_w = w_str*wo + wk - Lpad;
                if ((pad_cond_h >= 0) && (pad_cond_w >= 0) && (pad_cond_h < H_in) && (pad_cond_w < W_in)) {
                  int out_idx = wo + ho*W_out + co*H_out*W_out;
                  int in_idx = (w_str*wo + wk - Lpad) + (h_str*ho + hk - Upad)*W_in + ci*H_in*W_in;
                  temp += outDiff[out_idx] * inData[in_idx];
                }
              }
            }
            coeffDiff[wk+hk*pW+ci*pH*pW+co*pH*pW*C_in] = temp;
          }
        }
      }
    }

  }
}



void naive_conv2d_in_grad_kernel_CHW (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ inDiff = args->A;
  float * __restrict__ coeffData = args->B;
  float * __restrict__ outDiff = args->C;

  const uint32_t H_in = args->H;
  const uint32_t W_in = args->W;
  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t C_in = args->pCin;
  const uint32_t C_out = args->pCout;

  uint32_t h_str = args->stride_h;
  uint32_t w_str = args->stride_w;
  uint32_t Lpad = args->Lpad;
  uint32_t Rpad = args->Rpad;
  uint32_t Upad = args->Upad;
  uint32_t Dpad = args->Dpad;

  const uint32_t H_out = (H_in - pH + Upad + Dpad)/h_str + 1;
  const uint32_t W_out = (W_in - pW + Lpad + Rpad)/w_str + 1;
  const uint32_t pHW = pH*pW-1;

  const uint32_t blockSize = (C_in+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > C_in ? C_in : start+blockSize;  

  int padding = Lpad + Rpad + Upad + Dpad;
  int stride = h_str + w_str;

  if ((padding == 0) && (stride <= 2)) {

    for (uint32_t ci=0; ci<C_in; ci++) {
      for (uint32_t hi=0; hi<H_in; hi++) {
        for (uint32_t wi=0; wi<W_in; wi++) {
          float temp = 0;
          for (uint32_t co=0; co<C_out; co++) {
            for (uint32_t hk=0; hk<pH; hk++) {
              for (uint32_t wk=0; wk<pW; wk++) {
                // Padding conditions
                int h_padded = hi + hk - (pH-1);
                int w_padded = wi + wk - (pW-1);
                // Indices
                int ker_idx = (pHW-wk-hk*pW) + ci*pW*pH + co*pW*pH*C_in;
                int out_idx = w_padded + (h_padded)*W_out + co*H_out*W_out;

                if ((h_padded >= 0) && (w_padded >= 0) && (h_padded <= H_out - (pH-2)) && (w_padded <= W_out - (pW-2))) {
                  temp += coeffData[ker_idx] * outDiff[out_idx];
                }
              }
            }
          }
          inDiff[wi+hi*W_in+ci*H_in*W_in] = temp;
        }
      }
    }

  }
  else {

    int h_start = pH - 1 - Lpad;
    int w_start = pW - 1 - Upad;

    for (uint32_t ci=0; ci<C_in; ci++) {
      for (uint32_t hi=0; hi<H_in; hi++) {
        for (uint32_t wi=0; wi<W_in; wi++) {
          float temp = 0;
          for (uint32_t co=0; co<C_out; co++) {
            for (uint32_t hk=0; hk<pH; hk++) {
              for (uint32_t wk=0; wk<pW; wk++) {
                // Indices
                int ker_idx = (pHW-wk-hk*pW) + ci*pW*pH + co*pW*pH*C_in;
                // Border conditions
                int bord_h = (hi + hk - h_start) / h_str;
                int bord_w = (wi + wk - w_start) / w_str;
                int borders = (bord_h < H_out) && (bord_w < W_out);
                float o_grad = 0;
                float k_dat = coeffData[ker_idx];
                if (((hi+hk-h_start) % h_str == 0) && ((wi+wk-w_start) % w_str == 0) && borders == 1)
                {
                  int out_idx = bord_w + (bord_h)*W_out + co*H_out*W_out;
                  o_grad = outDiff[out_idx];
                  temp += k_dat * o_grad;
                }
              }
            }
          }
          int in_idx = (wi) + (hi)*W_in + ci*H_in*W_in;
          inDiff[in_idx] = temp;
        }
      }
    }

  }
}
