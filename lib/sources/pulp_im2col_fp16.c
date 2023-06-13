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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino
*/ 

#include "pulp_train_utils_fp16.h"
#include "pulp_im2col_fp16.h"

/**
 * @brief IM2ROW with padding and stride
 * 
 * @param im2col_args 
 */
void pulp_im2row_fp16(void * im2col_args_fp16){

  // unpack args
  struct im2col_args_fp16 * args = (struct im2col_args_fp16 *)im2col_args_fp16;
  struct blob_fp16 * input = args->input;
  struct blob_fp16 * coeff = args->c;
  struct blob_fp16 * output = args->output;

  fp16 * i2c_buf = args->pBuffer;

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;
  // Flag to activate the DMA version of the IM2COL
  uint8_t USE_DMA = args->USE_DMA;
  uint8_t HWC = args->HWC;

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

  // Set up im2col variables for padding and stride
  uint32_t Htot=0, Wtot=0;
  Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
  Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

  #if NUM_CORES > 1
  // Definitions for parallelism
  uint32_t blockSize=0, start=0, stop=0;
  if (HWC == 0 && mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
  }
  else if (HWC == 0 && mod == 1) {
    blockSize = (Co+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Co ? Co : start+blockSize;
  }
  else if (HWC == 1 && mod == 0) {
    blockSize = (Htot+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Htot ? Htot : start+blockSize;
  }
  else if (HWC == 1 && mod == 1) {
    blockSize = (Hin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Hin ? Hin : start+blockSize;
  }
  #else
  uint32_t start=0, stop=0; 
  if (HWC == 0 && mod == 0) {
    start = 0;
    stop = Cin;    
  }
  else if (HWC == 0 && mod == 1) {
    start = 0;
    stop = Co;
  }
  else if (HWC == 1 && mod == 0) {
    start = 0;
    stop = Htot;
  }
  else if (HWC == 1 && mod == 1) {
    start = 0;
    stop = Hin;
  }
  #endif

  /**
   * USE CHW FORMAT (ADJACENT ELEMENTS ARE ROW ELEMENTS OF THE INPUT OR OUTPUT MATRIX)
   */
  if (HWC == 0) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp16] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp16] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;

        if (padding == 0) {
          for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Hk*Wk;
                uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr) + (ho*Hstr)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<(Wk & 0xfffffffe); wk+=2) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;
                    // Vector of the final data
                    v2f16 im2col_fill = (v2f16) {0, 0};
                    im2col_fill = *((v2f16 *) &input->data[receptive_field_idx+in_inner_idx]);

                    // Fill IM2COL buffer
                    v2f16 *I2C = (v2f16 *) &i2c_buf[kernel_idx+segment_idx+i2c_inner_idx];
                    *I2C = im2col_fill;
                  }
                }
                if (Wk & 0x00000001) {
                  for (uint32_t hk=0; hk<Hk; hk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = (Wk-1) + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = (Wk-1) + hk*Win;

                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];             
                  }
                }
              }
            }
          }        
        }

        else {
          for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Hk*Wk;
                uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<(Wk & 0xfffffffe); wk+=2) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;
                    // Padding condition
                    uint32_t w_pad_cond_0 = wk + wo*Wstr;      uint32_t pad_w0_l = (w_pad_cond_0<Lpad);   uint32_t pad_w0_r = (w_pad_cond_0>Wo+Rpad);
                    uint32_t w_pad_cond_1 = wk+1 + wo*Wstr;    uint32_t pad_w1_l = (w_pad_cond_1<Lpad);   uint32_t pad_w1_r = (w_pad_cond_1>Wo+Rpad);
                    uint32_t h_pad_cond = hk + ho*Hstr;
                    // Vector of the final data
                    v2f16 im2col_fill = (v2f16) {0, 0};

                    im2col_fill = *((v2f16 *) &input->data[receptive_field_idx+in_inner_idx]);

                    // OLD PADDING (SLOW!!)
                    if (padding>0) {      // FIXME!!
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
                }
                if (Wk & 0x00000001) {
                  for (uint32_t hk=0; hk<Hk; hk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = (Wk-1) + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = (Wk-1) + hk*Win;
                    // Padding condition
                    uint32_t w_pad_cond = (Wk-1) + wo*Wstr;
                    uint32_t h_pad_cond = hk + ho*Hstr;

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

      }
      else // IN GRAD
      {
      //   // Set up variables for the in grad propagation
      //   Ho = (Hin-Hk+Upad+Dpad+Hstr);
      //   Wo = (Win-Wk+Rpad+Lpad+Wstr);
        
      //   for (uint32_t hi=0; hi<Hin; hi++) {
      //     for (uint32_t wi=0; wi<Win; wi++) {
      //       for (uint32_t co=start; co<stop; co++) {
      //         // IM2COL buffer coordinates
      //         uint32_t kernel_idx = co*Hk*Wk;
      //         uint32_t segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
      //         // Output grad tensor coordinates
      //         int ho_rf = hi - (Hk-1);
      //         int wo_rf = wi - (Wk-1);
      //         int receptive_field_idx = wo_rf + ho_rf*Wo + co*Ho*Wo;
      //         for (uint32_t hk=0; hk<Hk; hk++) {
      //           for (uint32_t wk=0; wk<(Wk & 0xfffffffe); wk+=2) {
      //             // IM2COl buffer coordinates
      //             uint32_t i2c_inner_idx = wk +hk*Wk;
      //             // Output grad tensor coordinates
      //             uint32_t out_inner_idx = wk + hk*Wo;
      //             // Padding condition
      //             int w_pad_cond_0 = wk + wo_rf;    int pad_w0_l = w_pad_cond_0<0;    int pad_w0_r = w_pad_cond_0>=(int)Wo;
      //             int w_pad_cond_1 = wk+1 + wo_rf;  int pad_w1_l = w_pad_cond_1<0;    int pad_w1_r = w_pad_cond_1>=(int)Wo;
      //             int h_pad_cond = hk + ho_rf;      int pad_h_u = h_pad_cond<0;       int pad_h_d = h_pad_cond>=(int)Ho;
      //             // Vector for the final data
      //             v2f16 im2col_fill = (v2f16) {0, 0};

      //             im2col_fill = *((v2f16 *) &output->diff[receptive_field_idx+out_inner_idx]);

      //             if (pad_w0_l || pad_w0_r || pad_w1_l || pad_w1_r || pad_h_u || pad_h_d) {
      //               if (pad_h_u || pad_h_d)         {im2col_fill = (v2f16) {0, 0};}
      //               else if (pad_w0_l)              {im2col_fill[0] = 0;}
      //               else if (pad_w1_l)              {im2col_fill[1] = 0;}
      //               else if (pad_w0_r)              {im2col_fill[0] = 0;}
      //               else if (pad_w1_r)              {im2col_fill[1] = 0;}
      //             }
      //             // Fill IM2COL buffer
      //             v2f16 *I2C = (v2f16 *) &i2c_buf[kernel_idx+segment_idx+i2c_inner_idx];
      //             *I2C = im2col_fill;
      //           }
      //         }
      //         if (Wk & 0x00000001) {
      //           for (uint32_t hk=0; hk<Hk; hk++) {
      //             // IM2COl buffer coordinates
      //             uint32_t i2c_inner_idx = (Wk-1) + hk*Wk;
      //             // Output grad tensor coordinates
      //             uint32_t out_inner_idx = (Wk-1) + hk*Wo;
      //             // Padding condition
      //             int w_pad_cond = (Wk-1) + wo_rf;
      //             int h_pad_cond = hk + ho_rf;

      //             if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Ho) || (w_pad_cond>=(int)Wo)) {
      //               // Padding
      //               i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
      //             }
      //             else {
      //               // Fill IM2COL buffer
      //               i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
      //           }
      //         }
      //       }
      //     }
      //   }
      // }

        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        
        for (uint32_t hi=0; hi<Hin; hi++) {
          for (uint32_t wi=0; wi<Win; wi++) {
            for (uint32_t co=start; co<stop; co++) {
              // IM2COL buffer coordinates
              uint32_t kernel_idx = co*Hk*Wk;
              uint32_t segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
              // Output grad tensor coordinates
              int ho_rf = hi - (Hk-1);
              int wo_rf = wi - (Wk-1);
              int receptive_field_idx = wo_rf + ho_rf*Wox + co*Hox*Wox;

              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  // IM2COl buffer coordinates
                  uint32_t i2c_inner_idx = wk + hk*Wk;
                  // Output grad tensor coordinates
                  uint32_t out_inner_idx = wk + hk*Wox;
                  // Padding condition
                  int w_pad_cond = wk + wo_rf;
                  int h_pad_cond = hk + ho_rf;

                  if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
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

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32: 414] Invalid USE_DMA parameter (not 0 or 1)\n");
    }
  }

  /**
   * USE HWC FORMAT (ADJACENT ELEMENTS ARE CHANNEL ELEMENTS IN THE INPUT OR OUTPUT MATRIX)
   */
  else if (HWC == 1) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;

        if (padding == 0) {

          for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
              // Im2Col indices
              uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              // Input activation indices
              uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  for (uint32_t ci=0; ci<(Cin & 0xfffffffe); ci+=2) {
                    // Im2Col indices
                    uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
                    // Input activation indices                    
                    uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
                    // Im2Col data
                    v2f16 im2col_fill = (v2f16) {0, 0};
                    im2col_fill = *((v2f16*) &input->data[input_idx+act_idx]);
                    // Fill Im2Col buffer
                    v2f16* I2C = (v2f16*) &i2c_buf[segment_idx+i2c_inner_idx];
                    *I2C = im2col_fill;
                  }
                }
              }
              if (Cin & 0x00000001) {
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // Im2Col indices
                    uint32_t i2c_inner_idx = (Cin-1) + wk*Cin + hk*Cin*Wk;
                    // Input activation indices                    
                    uint32_t act_idx = (Cin-1) + wk*Cin + hk*Cin*Win;
                    // Fill im2col buffer
                    i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];                    
                  }
                }
              }
            }
          }

        }
        else {

          printf("\n[pulp_im2col_fp16.c:] Padding not implemented for HWC im2col without DMA!\n");

          // for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
          //   for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
          //     // Im2Col indices
          //     uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
          //     // Input activation indices
          //     uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
          //     for (uint32_t hk=0; hk<Hk; hk++) {
          //       for (uint32_t wk=0; wk<Wk; wk++) {
          //         for (uint32_t ci=0; ci<(Cin & 0xfffffffe); ci+=2) {
          //           // Im2Col indices
          //           uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
          //           // Input activation indices                    
          //           uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
          //           // Im2Col data
          //           v2f16 im2col_fill = (v2f16) {0, 0};
          //           im2col_fill = *((v2f16*) &input->data[input_idx+act_idx]);
          //           // Fill Im2Col buffer
          //           v2f16* I2C = (v2f16*) &i2c_buf[segment_idx+i2c_inner_idx];
          //           *I2C = im2col_fill;
          //         }
          //         if (Cin & 0x00000001) {
          //           // Im2Col indices
          //           uint32_t i2c_inner_idx = (Cin-1) + wk*Cin + hk*Cin*Wk;
          //           // Input activation indices                    
          //           uint32_t act_idx = (Cin-1) + wk*Cin + hk*Cin*Win;
          //           // Fill im2col buffer
          //           i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];                    
          //         }
          //       }
          //     }
          //   }
          // }

        }
      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;

        for (uint32_t hi=start/*0*/; hi<stop/*Hin*/; hi++) {
          for (uint32_t wi=0; wi<Win; wi++) {
            // Padding variables
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);

            for (uint32_t hk=0; hk<Hk; hk++) {
              for (uint32_t wk=0; wk<Wk; wk++) {
                // Padding conditions
                int w_pad_cond = wk + wo_rf;
                int h_pad_cond = hk + ho_rf;    

                // Set padding loop
                if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                  for (uint32_t co=0; co<Co; co++) {
                    // IM2COL buffer coordinates
                    uint32_t segment_idx = wi*Co*Hk*Wk + hi*Co*Hk*Wk*Win;
                    uint32_t kernel_idx = wk*Co + hk*Co*Wk;
                    uint32_t i2c_inner_idx = co;  

                    // Fill with zeroes  
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0.0f;             
                  }
                }
                else {
                  // Non-padded iteration
                  for (uint32_t co=0; co<Co; co++) {
                    // OutDiff coordinates
                    int receptive_field_idx = (wo_rf+wk)*Co + (ho_rf+hk)*Co*Wox;
                    uint32_t out_inner_idx = co;

                    // IM2COL buffer coordinates
                    uint32_t segment_idx = wi*Co*Hk*Wk + hi*Co*Hk*Wk*Win;
                    uint32_t kernel_idx = wk*Co + hk*Co*Wk;
                    uint32_t i2c_inner_idx = co;

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

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32: 414] Invalid USE_DMA parameter (not 0 or 1)\n");
    }
    
  }

  // ERROR SIGNAL
  else {
    printf("[pulp_im2col_fp32:] Invalid HWC parameter (not 0 or 1)\n");
  }  

}







/**
 * @brief IM2COL with padding and stride
 * 
 * @param im2col_args 
 */
void pulp_im2col_fp16(void * im2col_args_fp16){

  // unpack args
  struct im2col_args_fp16 * args = (struct im2col_args_fp16 *)im2col_args_fp16;
  struct blob_fp16 * input = args->input;
  struct blob_fp16 * coeff = args->c;
  struct blob_fp16 * output = args->output;

  fp16 * i2c_buf = args->pBuffer;

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;
  // Flag to activate the DMA version of the IM2COL
  uint8_t USE_DMA = args->USE_DMA;
  uint8_t HWC = args->HWC;

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

  // Set up im2col variables for padding and stride
  uint32_t Htot=0, Wtot=0;
  Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
  Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

  #if NUM_CORES > 1
  // Definitions for parallelism
  uint32_t blockSize=0, start=0, stop=0;
  if (HWC == 0 && mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
  }
  else if (HWC == 0 && mod == 1) {
    blockSize = (Co+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Co ? Co : start+blockSize;
  }
  else if (HWC == 1 && mod == 0) {
    blockSize = (Htot+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Htot ? Htot : start+blockSize;
  }
  else if (HWC == 1 && mod == 1) {
    blockSize = (Hin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Hin ? Hin : start+blockSize;
  }
  #else
  uint32_t start=0, stop=0; 
  if (HWC == 0 && mod == 0) {
    start = 0;
    stop = Cin;    
  }
  else if (HWC == 0 && mod == 1) {
    start = 0;
    stop = Co;
  }
  else if (HWC == 1 && mod == 0) {
    start = 0;
    stop = Htot;
  }
  else if (HWC == 1 && mod == 1) {
    start = 0;
    stop = Hin;
  }
  #endif

  /**
   * USE CHW FORMAT (ADJACENT ELEMENTS ARE ROW ELEMENTS OF THE INPUT OR OUTPUT MATRIX)
   */
  if (HWC == 0) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;

        if (padding == 0) {
          for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Htot*Wtot*Hk*Wk;
                uint32_t segment_idx = wo + ho*Wtot;
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk*Htot*Wtot + hk*Htot*Wtot*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;

                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                  }
                }
              }
            }
          }          
        }

        else {
          for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Htot*Wtot*Hk*Wk;
                uint32_t segment_idx = wo + ho*Wtot;
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk*Htot*Wtot + hk*Htot*Wtot*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;
                    // Padding condition
                    uint32_t w_pad_cond = wk + wo*Wstr;
                    uint32_t h_pad_cond = hk + ho*Hstr;

                    if ((padding>0)&&((h_pad_cond<Upad) || (w_pad_cond<Lpad) || (h_pad_cond>Ho+(Hk)-Dpad) || (w_pad_cond>Wo+(Wk)-Rpad))) {
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

      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        
        for (uint32_t hi=0; hi<Hin; hi++) {
          for (uint32_t wi=0; wi<Win; wi++) {
            for (uint32_t co=start; co<stop; co++) {
              // IM2COL buffer coordinates
              uint32_t kernel_idx = co*Hin*Win*Hk*Wk;
              uint32_t segment_idx = wi + hi*Win;
              // Output grad tensor coordinates
              int ho_rf = hi - (Hk-1);
              int wo_rf = wi - (Wk-1);
              int receptive_field_idx = wo_rf + ho_rf*Wox + co*Hox*Wox;

              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  // IM2COl buffer coordinates
                  uint32_t i2c_inner_idx = wk*Hin*Win + hk*Hin*Win*Wk;
                  // Output grad tensor coordinates
                  uint32_t out_inner_idx = wk + hk*Wox;
                  // Padding condition
                  int w_pad_cond = wk + wo_rf;
                  int h_pad_cond = hk + ho_rf;

                  if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
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
    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32] DMA not implemented!\n");
    }
  }

  /**
   * USE HWC FORMAT (ADJACENT ELEMENTS ARE CHANNEL ELEMENTS IN THE INPUT OR OUTPUT MATRIX)
   */
  else if (HWC == 1) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;

        if (padding == 0) {

          for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
              // Im2Col indices
              uint32_t segment_idx = wo + ho*Wtot;
              // Input activation indices
              uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  for (uint32_t ci=0; ci<Cin; ci++) {
                    // Im2Col indices
                    uint32_t i2c_inner_idx = ci*Htot*Wtot + wk*Htot*Wtot*Cin + hk*Htot*Wtot*Cin*Wk;
                    // Input activation indices                    
                    uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
                    // Fill im2col buffer
                    i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
                  }
                }
              }
            }
          }

        }
        else {

          printf("\n[pulp_im2col_fp32.c:] Padding not implemented for HWC im2col without DMA!\n");

          // for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
          //   for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
          //     // Im2Col indices
          //     uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
          //     // Input activation indices
          //     uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
          //     for (uint32_t hk=0; hk<Hk; hk++) {
          //       for (uint32_t wk=0; wk<Wk; wk++) {
          //         for (uint32_t ci=0; ci<Cin; ci++) {
          //           // Im2Col indices
          //           uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
          //           // Input activation indices                    
          //           uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
          //           // Fill im2col buffer
          //           i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
          //         }
          //       }
          //     }
          //   }
          // }

        }
      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;

        for (uint32_t hi=start/*0*/; hi<stop/*Hin*/; hi++) {
          for (uint32_t wi=0; wi<Win; wi++) {
            // Padding variables
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);

            for (uint32_t hk=0; hk<Hk; hk++) {
              for (uint32_t wk=0; wk<Wk; wk++) {
                // Padding conditions
                int w_pad_cond = wk + wo_rf;
                int h_pad_cond = hk + ho_rf;  

                // Set padding loop
                if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                  for (uint32_t co=0; co<Co; co++) {
                    // IM2COL buffer coordinates
                    uint32_t segment_idx = wi + hi*Win;
                    uint32_t kernel_idx = wk*Co*Hin*Win + hk*Co*Hin*Win*Wk;
                    uint32_t i2c_inner_idx = co*Hin*Win;  

                    // Fill with zeroes  
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0.0f;             
                  }
                }
                else {
                  // Non-padded iteration
                  for (uint32_t co=0; co<Co; co++) {
                    // OutDiff coordinates
                    int receptive_field_idx = (wo_rf+wk)*Co + (ho_rf+hk)*Co*Wox;
                    uint32_t out_inner_idx = co;

                    // IM2COL buffer coordinates
                    uint32_t segment_idx = wi + hi*Win;
                    uint32_t kernel_idx = wk*Co*Hin*Win + hk*Co*Hin*Win*Wk;
                    uint32_t i2c_inner_idx = co*Hin*Win;

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

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32] DMA not implemented!\n");
    }
    
  }

  // ERROR SIGNAL
  else {
    printf("[pulp_im2col_fp32:] Invalid HWC parameter (not 0 or 1)\n");
  }
}







#ifndef OPTIMIZE_BT
#define OPTIMIZE_BT
#endif

void pulp_blocktransp_fp16 (void * blocktransp_args_fp16)
{
  struct blocktransp_args_fp16 * args = (struct blocktransp_args_fp16 *)blocktransp_args_fp16;
  fp16 * weights = args->weights;
  fp16 * bt_weights = args->bt_weights;
  uint32_t Cin = args->Cin;
  uint32_t Cout = args->Cout;
  uint32_t Hk = args->Hk;
  uint32_t Wk = args->Wk;
  uint8_t HWC_layout = args->HWC;

  uint32_t HW = Hk*Wk;

  uint32_t blockSize = (Cout+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > Cout ? Cout : start+blockSize;

  // USE CHW LAYOUT
  if (HWC_layout == 0) {
    #ifdef OPTIMIZE_BT
    // Block transposition
    for (uint32_t k=start; k<stop; k++) {
      for (uint32_t c=0; c<Cin; c++) {
        for (uint32_t i=0; i<(HW & 0xfffffffe); i+=2) {
          v2f16 wgt_elems = (v2f16) {0, 0};
          wgt_elems = *((v2f16 *) &weights[(HW-1-i-1)+c*HW+k*Cin*HW]);
          wgt_elems = (v2f16)(__builtin_shuffle(wgt_elems, (v2s){1,0}));
          v2f16 *BUF = (v2f16 *) &bt_weights[i+k*HW+c*Cout*HW];
          *BUF = wgt_elems;
        }
        if (HW & 0x00000001) {
          bt_weights[(HW-1)+k*HW+c*Cout*HW] = weights[c*HW+k*Cin*HW];
        }
      }
    } 
    #else 
    for (uint32_t k=start; k<stop; k++)
    {
      for (uint32_t c=0; c<Cin; c++)
      {
        for (uint32_t i=0; i<Hk*Wk; i++)
        {
          // OTHER MATRIX
          //bt_weights[i+k*HW+c*Cout*HW] = weights[i+c*HW+k*Cin*HW];
          bt_weights[i+k*HW+c*Cout*HW] = weights[(HW-1-i)+c*HW+k*Cin*HW];
        }
      }
    } 
    #endif
  }

  // USE HWC LAYOUT
  else if (HWC_layout == 1) {
    #ifdef OPTIMIZE_BT
    for (uint32_t co=0; co<Cout; co++) {
      for (uint32_t hk=0; hk<Hk; hk++) {
        for (uint32_t wk=0; wk<Wk; wk++) {
          for (uint32_t ci=0; ci<Cin; ci++) {
            bt_weights[ci*Hk*Wk*Cout + wk*Cout + hk*Wk*Cout + co] = weights[ci + (Wk-1-wk)*Cin + (Hk-1-hk)*Wk*Cin + co*Wk*Hk*Cin];
          }
        }
      }
    }
    #else
    for (uint32_t co=0; co<Cout; co++) {
      for (uint32_t hk=0; hk<Hk; hk++) {
        for (uint32_t wk=0; wk<Wk; wk++) {
          for (uint32_t ci=0; ci<Cin; ci++) {
            // OPTIMIZE ME!!
            bt_weights[ci*Hk*Wk*Cout + wk*Cout + hk*Wk*Cout + co] = weights[ci + (Wk-1-wk)*Cin + (Hk-1-hk)*Wk*Cin + co*Wk*Hk*Cin];
          }
        }
      }
    }
    #endif
  }

  // LAYOUT ERROR
  else {
    printf("[pulp_blocktransp_fp16.c] Invalid data layout (not 0 or 1)!!\n");
  }
}
