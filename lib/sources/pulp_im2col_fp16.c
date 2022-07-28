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
    pad = pW-1;

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

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t DW = args->DW;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;
  // Flag to activate the DMA version of the IM2COL
  uint8_t USE_DMA = args->USE_DMA;

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

  /**
   * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
   */
  if (USE_DMA == 0) {
    // FORWARD & WEIGHT GRAD
    if (mod==0)
    {
      uint32_t Htot, Wtot;
      if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp16: 259] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
      else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
      if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp16: 261] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
      else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

      int padding = Lpad + Rpad + Upad + Dpad;

      for (int ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
        for (int wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
          for (int ci=start; ci<stop; ci++) {
            // IM2COL buffer coordinates
            int kernel_idx = ci*Hk*Wk;
            int segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
            // Input tensor coordinates
            int receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
            for (int hk=0; hk<Hk; hk++) {
              for (int wk=0; wk<(Wk & 0xfffffffe); wk+=2) {
                // IM2COl buffer coordinate update
                int i2c_inner_idx = wk + hk*Wk;
                // Input tensor coordinate update
                int in_inner_idx = wk + hk*Win;
                // Padding condition
                int w_pad_cond_0 = wk + wo*Wstr;      int pad_w0_l = (w_pad_cond_0<Lpad);   int pad_w0_r = (w_pad_cond_0>Wo+Rpad);
                int w_pad_cond_1 = wk+1 + wo*Wstr;    int pad_w1_l = (w_pad_cond_1<Lpad);   int pad_w1_r = (w_pad_cond_1>Wo+Rpad);
                int h_pad_cond = hk + ho*Hstr;
                // Vector of the final data
                v2f16 im2col_fill = (v2f16) {0, 0};

                im2col_fill = *((v2f16 *) &input->data[receptive_field_idx+in_inner_idx]);

                if ((padding>0)) {      // FIXME!!
                  // Fill the padding vector with correct bits
                  if (((h_pad_cond<Upad) || (h_pad_cond>Ho+Dpad)))  {im2col_fill = (v2f16) {0, 0};}
                  else if (pad_w0_l)                                {im2col_fill[0] = 0;}
                  else if (pad_w1_l)                                {im2col_fill[1] = 0;}
                  else if (pad_w0_r)                                {im2col_fill[0] = 0;}
                  else if (pad_w1_r)                                {im2col_fill[1] = 0;}
                }
                // Fill IM2COL buffer
                v2f16 *I2C = (v2f16 *) &i2c_buf[kernel_idx+segment_idx+i2c_inner_idx];
                *I2C = im2col_fill;
              }
              if (Wk & 0x00000001) {
                // IM2COl buffer coordinate update
                int i2c_inner_idx = (Wk-1) + hk*Wk;
                // Input tensor coordinate update
                int in_inner_idx = (Wk-1) + hk*Win;
                // Padding condition
                int w_pad_cond = (Wk-1) + wo*Wstr;
                int h_pad_cond = hk + ho*Hstr;

                if ((padding>0)&&((h_pad_cond<Upad) || (w_pad_cond<Lpad) || (h_pad_cond>Ho+Dpad) || (w_pad_cond>Wo+Rpad))) {
                  // Padding
                  i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                  //printf("(pad) i2c_buf[%d]=%f                        kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], kernel_idx, segment_idx, ho);
                }
                else {
                  // Fill IM2COL buffer
                  i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                  //printf("(i2c) i2c_buf[%d]=%f (indata=%f)      kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], input->data[receptive_field_idx], kernel_idx, segment_idx, ho);
                }                
              }
            }
          }
        }
      }
    }
    else // IN GRAD
    {
      // Set up variables for the in grad propagation
      Ho = (Hin-Hk+Upad+Dpad+Hstr);
      Wo = (Win-Wk+Rpad+Lpad+Wstr);
      
      for (int hi=0; hi<Hin; hi++) {
        for (int wi=0; wi<Win; wi++) {
          for (int co=start; co<stop; co++) {
            // IM2COL buffer coordinates
            int kernel_idx = co*Hk*Wk;
            int segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
            // Output grad tensor coordinates
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);
            int receptive_field_idx = wo_rf + ho_rf*Wo + co*Ho*Wo;
            for (int hk=0; hk<Hk; hk++) {
              for (int wk=0; wk<(Wk & 0xfffffffe); wk+=2) {
                // IM2COl buffer coordinates
                int i2c_inner_idx = wk +hk*Wk;
                // Output grad tensor coordinates
                int out_inner_idx = wk + hk*Wo;
                // Padding condition
                int w_pad_cond_0 = wk + wo_rf;    int pad_w0_l = w_pad_cond_0<0;    int pad_w0_r = w_pad_cond_0>=Wo;
                int w_pad_cond_1 = wk+1 + wo_rf;  int pad_w1_l = w_pad_cond_1<0;    int pad_w1_r = w_pad_cond_1>=Wo;
                int h_pad_cond = hk + ho_rf;      int pad_h_u = h_pad_cond<0;       int pad_h_d = h_pad_cond>=Ho;
                // Vector for the final data
                v2f16 im2col_fill = (v2f16) {0, 0};

                im2col_fill = *((v2f16 *) &output->diff[receptive_field_idx+out_inner_idx]);

                if (pad_w0_l || pad_w0_r || pad_w1_l || pad_w1_r || pad_h_u || pad_h_d) {
                  if (pad_h_u || pad_h_d)         {im2col_fill = (v2f16) {0, 0};}
                  else if (pad_w0_l)              {im2col_fill[0] = 0;}
                  else if (pad_w1_l)              {im2col_fill[1] = 0;}
                  else if (pad_w0_r)              {im2col_fill[0] = 0;}
                  else if (pad_w1_r)              {im2col_fill[1] = 0;}
                }
                // Fill IM2COL buffer
                v2f16 *I2C = (v2f16 *) &i2c_buf[kernel_idx+segment_idx+i2c_inner_idx];
                *I2C = im2col_fill;
              }
              if (Wk & 0x00000001) {
                // IM2COl buffer coordinates
                int i2c_inner_idx = (Wk-1) + hk*Wk;
                // Output grad tensor coordinates
                int out_inner_idx = (Wk-1) + hk*Wo;
                // Padding condition
                int w_pad_cond = (Wk-1) + wo_rf;
                int h_pad_cond = hk + ho_rf;

                if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=Ho) || (w_pad_cond>=Wo)) {
                  // Padding
                  i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                }
                else {
                  // Fill IM2COL buffer
                  i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
              }
            }
          }
        }
      }
    }
  }
  }

  /**
   * IM2COL FROM L2 DATA TO L1 IM2COL_BUFFER
   */
  else if (USE_DMA == 1) {
    // FORWARD & WEIGHT GRAD
    if (mod==0)
    {
      uint32_t Htot, Wtot;
      if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp16: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
      else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
      if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp16: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
      else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

      int padding = Lpad + Rpad + Upad + Dpad;

      if (padding == 0) {
        Htot = Hin;
        Wtot = Win;
        for (int ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (int wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            for (int ci=start; ci<stop; ci++) {
              // IM2COl buffer coordinates
              int segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              int kernel_idx = ci*Hk*Wk;
              // Input tensor coordinates
              int receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;

              // DMA Copy structures
              pi_cl_dma_copy_2d_t dma_i2cfw;

              // Load first data into L1A
              dma_i2cfw.dir = PI_CL_DMA_DIR_EXT2LOC;
              dma_i2cfw.merge = 0;
              dma_i2cfw.stride = 2*Win;
              dma_i2cfw.length = 2*Wk;
              dma_i2cfw.size = 2*Hk*Wk;
              dma_i2cfw.id = pi_core_id();
              dma_i2cfw.ext = (uint32_t) (input->data + receptive_field_idx);
              dma_i2cfw.loc = (uint32_t) &i2c_buf[segment_idx+kernel_idx];
              pi_cl_dma_memcpy_2d(&dma_i2cfw);    

              pi_cl_dma_wait(&dma_i2cfw);      
            }
          }
        }
      }
      else {
        for (int ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (int wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            printf("\nho=%d, wo=%d\n", ho, wo);
            // Initialize padding conditions and variables
            int pad_l = wo*Wstr - Lpad;  
            int pad_r = wo*Wstr + Rpad;
            int pad_u = ho*Hstr - Upad;
            int pad_d = ho*Hstr + Dpad;
            int row_size = Wk;                // Transfer lenght (length of a row)
            int transfer_size = Hk*Wk;        // Transfer size (depends on the padding)
            int in_shift_idx = 0;             // Index to shift input reading
            printf("transfer_size=%d\n", transfer_size);
            // Check if conditions for padding are met and assign zeros
            if (pad_l < 0)  {pad_l = -pad_l;  row_size -= pad_l;  transfer_size -= pad_l*Hk;  in_shift_idx += pad_l;}       
            else {pad_l = 0;}
            if (pad_r > Wo) {pad_r = pad_r-Wo;  row_size -= pad_r;  transfer_size -= pad_r*Hk;}   
            else {pad_r = 0;}
            if (pad_u < 0)  {pad_u = -pad_u;  transfer_size -= pad_u*Wk;  in_shift_idx += Win*pad_u;  if(pad_l>0 || pad_r>0) {transfer_size++;}}       
            else {pad_u = 0;}
            if (pad_d > Ho) {if(pad_u==0) {pad_d = pad_d-Ho;  transfer_size -= pad_d*Wk; if(pad_l>0 || pad_r>0) {transfer_size++;}}}   
            else {pad_d = 0;}
            printf("pad_l=%d, pad_r=%d, pad_u=%d, pad_d=%d\n", pad_l, pad_r, pad_u, pad_d);
            printf("row_size=%d, transfer_size=%d\n", row_size, transfer_size);
            // Check if to perform padding operation
            int padding_int = pad_l + pad_r + pad_u + pad_d;
            printf("padding_int=%d, in_shift_idx=%d\n", padding_int, in_shift_idx);

            for (int ci=start; ci<stop; ci++) {
              // IM2COL buffer coordinates
              int kernel_idx = ci*Hk*Wk;
              int segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              // Input tensor coordinates
              int receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;

              if (padding_int == 0) {
                // DMA Copy structures
                pi_cl_dma_copy_2d_t dma_i2cfw_pad;

                // Load first data into L1A
                dma_i2cfw_pad.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_i2cfw_pad.merge = 0;
                dma_i2cfw_pad.stride = 2*Win;
                dma_i2cfw_pad.length = 2*Wk;
                dma_i2cfw_pad.size = 2*Hk*Wk;
                dma_i2cfw_pad.id = pi_core_id();
                dma_i2cfw_pad.ext = (uint32_t) (input->data + receptive_field_idx);
                dma_i2cfw_pad.loc = (uint32_t) &i2c_buf[segment_idx+kernel_idx];
                pi_cl_dma_memcpy_2d(&dma_i2cfw_pad);    

                pi_cl_dma_wait(&dma_i2cfw_pad);                      
              }
              else {
                // DMA Copy structures
                pi_cl_dma_copy_2d_t dma_i2cfw_pad;
                fp16 load_buffer[transfer_size];
                fp16 pad_buffer[Wk*Hk];

                // Load first data into L1A
                dma_i2cfw_pad.dir = PI_CL_DMA_DIR_EXT2LOC;
                dma_i2cfw_pad.merge = 0;
                dma_i2cfw_pad.stride = 2*Win;
                dma_i2cfw_pad.length = 2*row_size;
                dma_i2cfw_pad.size = 2*transfer_size;
                dma_i2cfw_pad.id = pi_core_id();
                dma_i2cfw_pad.ext = (uint32_t) (input->data + receptive_field_idx + in_shift_idx);
                dma_i2cfw_pad.loc = (uint32_t) load_buffer; 
                pi_cl_dma_memcpy_2d(&dma_i2cfw_pad);    

                // Initialize pad_buffer
                for (int i=0; i<Wk*Hk; i++) pad_buffer[i]=0;

                pi_cl_dma_wait(&dma_i2cfw_pad);    

                // Fill the pad_buffer
                for (int i=0; i<transfer_size; i++) {
                  for (int j=0; j<row_size; j++) {
                    int pad_buffer_idx = pad_l + j + (pad_u+i)*Wk;
                    pad_buffer[pad_buffer_idx] = load_buffer[j+i*row_size];
                  }
                } 

                // Fill im2col
                for (int i=0; i<Wk*Hk; i++)   i2c_buf[segment_idx+kernel_idx+i] = pad_buffer[i];
              }
            }
          }
        }
      }
    }    
    else // IN GRAD
    {
      printf("\n[pulp_im2col_fp16: 320] IM2COL for In Grad with DMA not implemented!\n");
      return;

      // Set up variables for the in grad propagation
      Ho = (Hin-Hk+Upad+Dpad+Hstr);
      Wo = (Win-Wk+Rpad+Lpad+Wstr);
      
      for (int hi=0; hi<Hin; hi++) {
        for (int wi=0; wi<Win; wi++) {
          for (int co=start; co<stop; co++) {
            // IM2COL buffer coordinates
            int kernel_idx = co*Hk*Wk;
            int segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
            // Output grad tensor coordinates
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);
            int receptive_field_idx = wo_rf + ho_rf*Wo + co*Ho*Wo;
            // Padding conditions
            int row_length = Wk; int col_length = Hk; int tr_residual = 0;
            int pad_l = Wk + wo_rf;  int pad_r = wo_rf;
            int pad_u = Hk + ho_rf;  int pad_d = ho_rf;
            if (pad_l<0)    {if(pad_r<Wo) row_length -= (Wk-1)-wi;}
            if (pad_r>=Wo)  {if(pad_l>0)  row_length -= Wo+(Wk-1)-wi;}
            if (pad_u<0)    {if(pad_u<Ho) row_length -= (Hk-1)-hi;}
            if (pad_d>=Ho)  {if(pad_u>0)  row_length -= Wo+(Wk-1)-wi;}

            int transfer_lenght = row_length*col_length + tr_residual;
            fp16 load_buffer[transfer_lenght];
            fp16 pad_buffer[Hk*Wk];

            printf("pad_l=%d, pad_r=%d, pad_u=%d, pad_d=%d\n", pad_l, pad_r, pad_u, pad_d);
            printf("");

            // // DMA Copy structures
            // pi_cl_dma_copy_2d_t dma_i2cbw;

            // // Load first data into L1A
            // dma_i2cbw.dir = PI_CL_DMA_DIR_EXT2LOC;
            // dma_i2cbw.merge = 0;
            // dma_i2cbw.stride = 2*Wo;
            // dma_i2cbw.length = 2*row_length; //4*Wk;
            // dma_i2cbw.size = 2*transfer_lenght; //4*Hk*Wk;
            // dma_i2cbw.id = pi_core_id();
            // dma_i2cbw.ext = (uint32_t) (output->diff + receptive_field_idx);
            // dma_i2cbw.loc = (uint32_t) load_buffer; //&i2c_buf[segment_idx+kernel_idx];
            // pi_cl_dma_memcpy_2d(&dma_i2cbw);    

            // // Initialize buffer
            // for (int i=0; i<Hk*Wk; i++) pad_buffer[i] = 0;

            // pi_cl_dma_wait(&dma_i2cbw);      

            // // Fill the pad_buffer
            // for (int i=0; i<transfer_lenght; i++) {
            //   for (int j=0; j<row_length; j++) {
            //     int pad_buffer_idx = j + i*Wk; //pad_l + j + (pad_u+i)*Wk;
            //     pad_buffer[pad_buffer_idx] = load_buffer[j+i*row_length];
            //   }
            // } 

            // // Fill im2col
            // for (int i=0; i<Wk*Hk; i++)   i2c_buf[segment_idx+kernel_idx+i] = pad_buffer[i];



            // for (int hk=0; hk<Hk; hk++) {
            //   for (int wk=0; wk<Wk; wk++) {
            //     // IM2COl buffer coordinates
            //     int int_ker_idx = wk + hk*Wk;
            //     // Output grad tensor coordinates
            //     int int_rec_field_idx = wk + hk*Wo;
            //     // Padding condition
            //     int w_pad_cond = wk + wo_rf;
            //     int h_pad_cond = hk + ho_rf;

            //     if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=Ho) || (w_pad_cond>=Wo)) {
            //       // Padding
            //       i2c_buf[kernel_idx+int_ker_idx+segment_idx] = 0;
            //     }
            //     else {
            //       // Fill IM2COL buffer
            //       i2c_buf[kernel_idx+int_ker_idx+segment_idx] = output->diff[receptive_field_idx+int_rec_field_idx];
            //     }
            //   }
            // }

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