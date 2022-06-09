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
#include "pulp_im2col_fp16.h"

/**
 *  Select if to use the legacy pointer-based IM2COL (has 
 *  bugs in the parallelization) or the simpler newer version
**/
#define NEW_IM2COL
#ifndef NEW_IM2COL
// --------------------------------- OLD (issues with parallelization) --------------------------------------
void pulp_im2col_fp16(void * void_args){

  #if NUM_CORES > 1
  #define PARALLEL
  #endif

  // unpack args
  struct im2col_args_fp16 * args = (struct im2col_args_fp16 *)void_args;
  struct blob_fp16 * input = args->input;
  struct blob_fp16 * coeff = args->c;
  struct blob_fp16 * output = args->output;

  fp16 * pBuffer = args->pBuffer;

  int pad = args->pad;
  int mod = args->mod;
  int tile_start = args->tile_start;
  int tile_h = args->tile_h;

  int DW = args->DW;

  // activations dimensions, w/o padding
  int W = input->W;
  int H = input->H;
  int Cin = input->C;
  // kernel dimensions
  int pW = coeff->W;
  int pH = coeff->H;
  // input channels size
  int Wout = output->W;
  int Hout = output->H;
  int Cout = output->C;

  // indeces for loops
  int i, j, u, t, u_pad, t_pad;

  int pW_Hmod2;
  int w_stop, h_stop;
  // im2col buffer counter
  int buffer_cont, back_buffer_size, col_size;

  // output is resized depending on the kernel
  pW_Hmod2 = pW % 2;

  col_size = pW*pH;
  int WH = W*H;
  int WHout = Wout*Hout;

  back_buffer_size = 0;
  buffer_cont = 0;

  fp16 * inData = input->data;
  fp16 * inDiff = input->diff;
  fp16 * outDiff = output->diff;

  #ifdef PARALLEL
  // Definitions for parallelism
  int blockSize, start, stop;
  if (mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
    //printf("\nCORE %d-> blockSize: %d, start: %d, stop: %d\n\n", pi_core_id(), blockSize, start, stop);
    // Initialize buffer index
    buffer_cont = start*col_size;
  }
  else if (mod == 1) {
    blockSize = (Cout+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cout ? Cout : start+blockSize;
    // Initialize buffer index
    buffer_cont = start*col_size;
  }
  #endif

  if (mod==0)
  {
  
  h_stop = H-pH+1+2*pad;
  w_stop = W-pW+1+2*pad;

  for (i=0; i<h_stop; i++)
  {
    for (j=0; j<w_stop; j++)
    {
  #ifdef PARALLEL
      for (int k=start; k<stop/*start+blockSize*/; k++)
  #else
      for (int k=0; k<Cin; k++)
  #endif
      {
        for (t=i-pad; t<i+pH-pad; t++) 
        {
          for (u=j-pad; u<j+pW-pad; u++) 
          {
                  if ((pad>0)&&((t<0)||(u<0)||(t>H-1)||(u>W-1)))
                  {
                      *(pBuffer + buffer_cont) = 0;
                      //printf("CORE %d-> k: %d, t: %d, u: %d || idx: %d, b_cont: %d  (PAD)\n", pi_core_id(), k, t, u, u+t*H+k*WH, buffer_cont);
                      buffer_cont ++;
                  }
                  else
                  {
                      *(pBuffer + buffer_cont) = *(inData + u + t*W + k*WH);
                      //printf("CORE %d-> k: %d, t: %d, u: %d || idx: %d, b_cont: %d  (VAL: %f)\n", pi_core_id(), k, t, u, u+t*H+k*WH, buffer_cont, *(inData + u + t*H + k*WH));
                      buffer_cont ++;
                  }
            }
          }
          #if NUM_CORES > 1 
          if (k==stop-1 && DW==1) buffer_cont += blockSize*(NUM_CORES-1)*col_size;
          #endif
        }
      }
    }

  }
  else // BACKWARD
  {

    h_stop = Hout-pH+1+2*pad; 
    w_stop = Wout-pW+1+2*pad;

    for (i=0; i<h_stop; i++)
    {
      for (j=0; j<w_stop; j++)
      {
          #ifdef PARALLEL
          for (int k=start; k<stop/*start+blockSize*/; k++)
          #else
                for (int k=0; k<Cout; k++) 
          #endif
          {
            for (t=i-pad; t<i+pH-pad; t++) 
            {
              for (u=j-pad; u<j+pW-pad; u++) 
              {
                  if ((pad>0) && ((t<0)||(u<0)||(t>Hout-1)||(u>Wout-1)))
                  {
                      *(pBuffer + buffer_cont) = 0;
                      //printf("CORE %d-> k: %d, t: %d, u: %d, idx: %d, b_cont: %d  (PAD)\n", pi_core_id(), k, t, u, u+t*H+k*WH, buffer_cont);
                      buffer_cont ++;
                  }
                  else
                  {
                      *(pBuffer + buffer_cont) = *(outDiff + u + t*Wout + k*WHout);
                      //printf("CORE %d-> k: %d, t: %d, u: %d, idx: %d, b_cont: %d  (VAL: %f)\n", pi_core_id(), k, t, u, u+t*H+k*WH, buffer_cont, *(outDiff + u + t*H + k*WH));
                      buffer_cont ++;
                  }
              }
            }
            #if NUM_CORES > 1 
            if (k==stop-1 && DW==1) buffer_cont += blockSize*(NUM_CORES-1)*col_size;
            #endif
          }
      }
    }
  }
}
#else
// --------- NEW ------------
void pulp_im2col_fp16(void * void_args){

  // unpack args
  struct im2col_args_fp16 * args = (struct im2col_args_fp16 *)void_args;
  struct blob_fp16 * input = args->input;
  struct blob_fp16 * coeff = args->c;
  struct blob_fp16 * output = args->output;

  fp16 * i2c_buf = args->pBuffer;

  int pad = args->pad;
  int mod = args->mod;
  int DW = args->DW;

  // activations dimensions, w/o padding
  int Win = input->W;
  int Hin = input->H;
  int Cin = input->C;
  // kernel dimensions
  int Wk = coeff->W;
  int Hk = coeff->H;
  // input channels size
  int Wo = output->W;
  int Ho = output->H;
  int Co = output->C;

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
    for (int ho=0; ho<Ho+2*pad; ho++) {
      for (int wo=0; wo<Wo+2*pad; wo++) {
        for (int ci=start; ci<stop; ci++) {
          for (int hk=0; hk<Hk; hk++) {
            for (int wk=0; wk<Wk; wk++) {
              // IM2COl buffer coordinates
              int kernel_idx = wk + hk*Wk + ci*Hk*Wk;
              int segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wo+2*pad);
              // Input tensor coordinates
              int receptive_field_idx = wk + hk*Win + (wo-pad) + (ho-pad)*Win + ci*Hin*Win;
              // Padding condition
              int w_pad_cond = wk + wo;
              int h_pad_cond = hk + ho;

              if ((pad>0)&&((h_pad_cond<pad) || (w_pad_cond<pad) || (h_pad_cond>Ho+pad) || (w_pad_cond>Wo+pad))) {
                // Padding
                i2c_buf[kernel_idx+segment_idx] = 0;
              }
              else {
                // Fill IM2COL buffer
                i2c_buf[kernel_idx+segment_idx] = input->data[receptive_field_idx];
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

              if ((pad>0) && ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=Ho) || (w_pad_cond>=Wo))) {
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
#endif







void pulp_blocktransp_fp16 (void * void_args)
{
  struct blocktransp_args_fp16 * args = (struct blocktransp_args_fp16 *)void_args;
  fp16 * weights = args->weights;
  fp16 * bt_weights = args->bt_weights;
  int Cin = args->Cin;
  int Cout = args->Cout;
  int Hk = args->Hk;
  int Wk = args->Wk;

  int HW = Hk*Wk;

  // Temporary matrix
  //fp16 temp[Cin*Cout*HW];

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

  // OLD NOT WORKING
  // #if NUM_CORES > 1
  // pi_cl_team_barrier();
  // #endif

  // // Store back
  // // for (int k=0; k<Cout; k++)
  // for (int k=start; k<stop; k++)
  // {
  //   for (int c=0; c<Cin; c++)
  //   {
  //     for (int i=0; i<HW; i++)
  //     {
  //       weights[i+k*HW+c*Cout*HW] = temp[i+k*HW+c*Cout*HW];     
  //     }
  //   }
  // }
}