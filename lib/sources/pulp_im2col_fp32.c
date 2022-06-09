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
#include "pulp_im2col_fp32.h"

/**
 * @brief IM2COL with padding and stride
 * 
 * @param void_args 
 */
void pulp_im2col_fp32(void * void_args){

  // unpack args
  struct im2col_args * args = (struct im2col_args *)void_args;
  struct blob * input = args->input;
  struct blob * coeff = args->c;
  struct blob * output = args->output;

  float * i2c_buf = args->pBuffer;

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t DW = args->DW;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;

  // activations dimensions, w/o padding
  uint32_t Win = input->W;
  uint32_t Hin = input->H;
  uint32_t Cin = input->C;
  // kernel dimensions
  uint32_t Wk = coeff->W;
  uint32_t Hk = coeff->H;
  // input channels size
  uint32_t Wo = output->W;
  uint32_t Ho = output->H;
  uint32_t Co = output->C;

  // Set up internal variables (simpify external interface)
  Ho = Hin - Hk + 1;
  Wo = Win - Wk + 1;

  #if NUM_CORES > 1
  // Definitions for parallelism
  int blockSize, start, stop;
  if (mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
  }
  else if (mod == 1) {
    blockSize = (Co+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Co ? Co : start+blockSize;
  }
  #else
  int start, stop; 
  if (mod == 0) {
    start = 0;
    stop = Cin;    
  }
  else {
    start = 0;
    stop = Co;
  }
  #endif

  // FORWARD & WEIGHT GRAD
  if (mod==0)
  {
    uint32_t Htot, Wtot;
    if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
    else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
    if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
    else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

    int padding = Lpad + Rpad + Upad + Dpad;

    for (int ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
      for (int wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
        for (int ci=start; ci<stop; ci++) {
          for (int hk=0; hk<Hk; hk++) {
            for (int wk=0; wk<Wk; wk++) {
              // IM2COl buffer coordinates
              int kernel_idx = wk + hk*Wk + ci*Hk*Wk;
              int segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              // Input tensor coordinates
              int receptive_field_idx = wk + hk*Win + (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
              // Padding condition
              int w_pad_cond = wk + wo*Wstr;
              int h_pad_cond = hk + ho*Hstr;

              if ((padding>0)&&((h_pad_cond<Upad) || (w_pad_cond<Lpad) || (h_pad_cond>Ho+Dpad) || (w_pad_cond>Wo+Rpad))) {
                // Padding
                i2c_buf[kernel_idx+segment_idx] = 0;
                printf("(pad) i2c_buf[%d]=%f                        kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], kernel_idx, segment_idx, ho);
              }
              else {
                // Fill IM2COL buffer
                i2c_buf[kernel_idx+segment_idx] = input->data[receptive_field_idx];
                printf("(i2c) i2c_buf[%d]=%f (indata=%f)      kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], input->data[receptive_field_idx], kernel_idx, segment_idx, ho);
              }
            }
          }
        }
      }
    }
  }
  else // IN GRAD
  {
    for (int hi=0; hi<Hin; hi++) {
      for (int wi=0; wi<Win; wi++) {
        for (int co=start; co<stop; co++) {
          for (int hk=0; hk<Hk; hk++) {
            for (int wk=0; wk<Wk; wk++) {
              // IM2COl buffer coordinates
              int kernel_idx = wk + hk*Wk + co*Hk*Wk;
              int segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
              // Output grad tensor coordinates
              int ho_rf = hi - (Hk-1);
              int wo_rf = wi - (Wk-1);
              int receptive_field_idx = wk + hk*Wo + wo_rf + ho_rf*Wo + co*Ho*Wo;
              // Padding condition
              int w_pad_cond = wk + wo_rf;
              int h_pad_cond = hk + ho_rf;

              if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=Ho) || (w_pad_cond>=Wo)) {
                // Padding
                i2c_buf[kernel_idx+segment_idx] = 0;
              }
              else {
                // Fill IM2COL buffer
                i2c_buf[kernel_idx+segment_idx] = output->diff[receptive_field_idx];
              }
            }
          }
        }
      }
    }
  }
}



void pulp_blocktransp_fp32 (void * void_args)
{
  struct blocktransp_args * args = (struct blocktransp_args *)void_args;
  float * weights = args->weights;
  float * bt_weights = args->bt_weights;
  int Cin = args->Cin;
  int Cout = args->Cout;
  int Hk = args->Hk;
  int Wk = args->Wk;

  int HW = Hk*Wk;

  int blockSize = (Cout+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > Cout ? Cout : start+blockSize;

  // Block tranposition
  // for (int k=0; k<Cout; k++)
  for (int k=start; k<stop; k++)
  {
    for (int c=0; c<Cin; c++)
    {
      for (int i=0; i<Hk*Wk; i++)
      {
        // OLD 
        //temp[i+k*HW+c*Cout*HW] = weights[i+c*HW+k*Cin*HW];
        //temp[i+k*HW+c*Cout*HW] = weights[(HW-1-i)+c*HW+k*Cin*HW];

        // OTHER MATRIX
        //bt_weights[i+k*HW+c*Cout*HW] = weights[i+c*HW+k*Cin*HW];
        bt_weights[i+k*HW+c*Cout*HW] = weights[(HW-1-i)+c*HW+k*Cin*HW];
      }
    }
  } 
}

