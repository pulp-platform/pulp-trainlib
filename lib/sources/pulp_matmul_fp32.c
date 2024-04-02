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

#include "pmsis.h"


/**
 * NAIVE VERSIONS
 */

void mm(void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const uint32_t N = args->N;
  const uint32_t M = args->M;
  const uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    if (K == 1) 
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          C[i*M+j] = A[i*K] * B[j];
          #ifdef DEBUG
          //printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K, j, C[i*M+j], A[i], B[j]);
          #endif
        }
      }
    }
    else if (K > 0)
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          float temp = 0;
          for (uint32_t k = 0; k < K; k++) 
          {
                temp += A[i*K+k] * B[j+k*M];
                #ifdef DEBUG
                //printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
                #endif
          } 
          C[i*M+j] = temp;
        } 
      } 
    }
  }

  // =====> B IS TRANSPOSED <=====  
  else 
  {
    if (K == 1) 
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          C[i*M+j] = A[i*K] * B[j*K];
          #ifdef DEBUG
          //printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i, j*K, C[i*M+j], A[i*K], B[j*K]);
          #endif
        } 
      } 
    }
    else if (K > 0)
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          float temp = 0;
          for (uint32_t k = 0; k < K; k++) 
          {
              temp += A[i*K+k] * B[k+j*K];
              #ifdef DEBUG
              //printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
          } 
          C[i*M+j] = temp;
        } 
      } 
    }
  }
}

// Naive version with add on the output matrix
void mm_add(void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const uint32_t N = args->N;
  const uint32_t M = args->M;
  const uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    if (K == 1) 
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          C[i*M+j] += A[i*K] * B[j];
          #ifdef DEBUG
          printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K, j, C[i*M+j], A[i], B[j]);
          #endif
        }
      }
    }
    else if (K > 0)
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          float temp = 0;
          for (uint32_t k = 0; k < K; k++) 
          {
                temp += A[i*K+k] * B[j+k*M];
                #ifdef DEBUG
                printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
                #endif
          } 
          C[i*M+j] += temp;
        } 
      } 
    }
  }

  // =====> B IS TRANSPOSED <=====  
  else 
  {
    if (K == 1) 
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          C[i*M+j] += A[i*K] * B[j*K];
          #ifdef DEBUG
          printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i, j*K, C[i*M+j], A[i*K], B[j*K]);
          #endif
        } 
      } 
    }
    else if (K > 0)
    {
      for (uint32_t i=start; i < stop; i++) 
      {
        for (uint32_t j = 0; j < M; j++) 
        {
          float temp = 0;
          for (uint32_t k = 0; k < K; k++) 
          {
              temp += A[i*K+k] * B[k+j*K];
              #ifdef DEBUG
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
          } 
          C[i*M+j] += temp;
        } 
      } 
    }
  }
}


// Naive matmul with parallelism on M
void mm_M(void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const uint32_t N = args->N;
  const uint32_t M = args->M;
  const uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  const uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > M ? M: start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (uint32_t i = 0; i < N; i++) 
    {
      for (uint32_t j=start; j < stop; j++) 
      {
        float temp = 0;
        for (uint32_t k = 0; k < K; k++) 
        {
              temp += A[i*K+k] * B[j+k*M];
              #ifdef DEBUG
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
              #endif
        } 
        C[i*M+j] = temp;
      } 
    } 
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (uint32_t i = 0; i < N; i++) 
    {
      for (uint32_t j=start; j < stop; j++) 
      {
        float temp = 0;
        for (uint32_t k = 0; k < K; k++) 
        {
              temp += A[i*K+k] * B[j*K+k];
              #ifdef DEBUG              
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
        } 
        C[i*M+j] = temp;
      } 
    } 
  }
}



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



void mm_conv2d_in_grad (void * matMul_args) 
{

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const uint32_t N = args->N;
  const uint32_t K = args->K;
  const uint32_t M = args->M;

  const uint32_t pW = args->pW;
  const uint32_t pH = args->pH;
  const uint32_t pCin = args->pCin;
  const uint32_t pCout = args->pCout;

  const uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > M ? M : start+blockSize;

  // ALGORITHM
  // For each receptive field of the output on the weights
  for (uint32_t rec_field = 0; rec_field < M; rec_field++) {
    // For each channel of the output
    for (uint32_t Ci = 0; Ci < pCin; Ci++) {
      // Multiply each receptive field for the corresponding
      // set of channels and accumulate on the input channel by channel
      float temp = 0;
      printf("\ntemp = 0\n");
      for (uint32_t Co = 0; Co < pCout; Co++) {  
        for (uint32_t elem = 0; elem < pW*pH; elem++) {
          temp += A[pW*pH*pCin*Co+pW*pH*Ci+elem] * B[pH*pW*pCout*rec_field+pW*pH*Co+elem];
          //#ifdef DEBUG
          #if 1
          printf("coeffdata[%d]=%f, i2c_buffer[%d]=%f, temp=%f\n",
                  pW*pH*pCin*Co+pW*pH*Ci+elem, A[pW*pH*pCin*Co+pW*pH*Ci+elem], 
                  pH*pW*pCout*rec_field+pW*pH*Co+elem, B[pH*pW*pCout*rec_field+pW*pH*Co+elem],
                  temp);                  
          #endif
        }
      }
      C[M*Ci+rec_field] = temp;
      #ifdef DEBUG
      printf("C[%d]=%f\n", M*Ci+rec_field, C[M*Ci+rec_field]);
      #endif
    }
  }
}



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

    for (uint32_t ci=0; ci<C_in; ci++) {
      for (uint32_t hi=0; hi<H_in; hi++) {
        for (uint32_t wi=0; wi<W_in; wi++) {
          float temp = 0;
          for (uint32_t co=0; co<C_out; co++) {
            for (uint32_t hk=0; hk<pH; hk++) {
              for (uint32_t wk=0; wk<pW; wk++) {
                // Padding conditions
                int h_padded = hi + hk - (pH-1) + Upad;
                int w_padded = wi + wk - (pW-1) + Lpad;
                // Indices
                int ker_idx = (pHW-wk-hk*pW) + ci*pW*pH + co*pW*pH*C_in;
                int out_idx = w_padded + (h_padded)*W_out + co*H_out*W_out;

                if ((h_padded >= 0) && (w_padded >= 0) && (h_padded < H_out - (pH-2-Dpad)) && (w_padded < W_out - (pW-2-Rpad))) {
                  temp += coeffData[ker_idx] * outDiff[out_idx];
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







/**
 * OPTIMIZED VERSIONS
 */

// Naive mm with unrolling of 2
void mm_u2 (void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (uint32_t i=start; i < stop; i++) 
    {
      for (uint32_t j = 0; j < M; j++) 
      {
        float temp = 0;
        for (uint32_t k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[j+k*M];
              temp += A[i*K+k+1] * B[j+(k+1)*M];
        } 
        C[i*M+j] = temp;
      } 
    } 
    // Leftover on K
    if (K & 0x00000001)
    {
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<M; j++) 
        {
          C[i*M+j] += A[i*K+(K-1)] * B[j+(K-1)*M];
        }
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (uint32_t i=start; i < stop; i++) 
    {
      for (uint32_t j = 0; j < M; j++) 
      {
        float temp = 0;
        for (uint32_t k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[k+j*K];
              temp += A[i*K+k+1] * B[k+1+j*K];              
        } 
        C[i*M+j] = temp;
        //temp = 0;
      } 
    } 
    // Leftover on K 
    if (K & 0x00000001)
    {
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<M; j++) 
        {
          C[i*M+j] += A[i*K+(K-1)] * B[(K-1)+j*K];
        }
      }
    }
  }
}



void mm_unroll_1x2 (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 2) { mm(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
        }
      }
      // Leftover on M
      if (M & 0x00000001) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          float temp = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp += A[i*K+k] * B[k*M+M-1];
          }
          C[i*M+M-1] = temp;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
        }
      }
      // Leftover on M
      if (M & 0x00000001) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          float temp = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp += A[i*K+k] * B[k+(M-1)*K];
          }
          C[i*M+M-1] = temp;
        }
      }    
    }
  }
}



void mm_unroll_1x4 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 4) { mm_unroll_1x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1];
            temp2     += Ash * B[idx+2];
            temp3     += Ash * B[idx+3];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
        }
      }
      // Leftover on M
      if (M & 0x00000003) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k*M+j];
            }
          C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K];
            temp2     += Ash * B[idx+2*K];
            temp3     += Ash * B[idx+3*K];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
        }
      }
      // Leftover on M
      if (M & 0x00000003) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k+j*K];
            }
          C[i*M+j] = temp;
          }
        }
      }  
    }
  }
}



void mm_unroll_1x8 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 8) { mm_unroll_1x4(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffff8); j=j+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1];
            temp2     += Ash * B[idx+2];
            temp3     += Ash * B[idx+3];
            temp4     += Ash * B[idx+4];
            temp5     += Ash * B[idx+5];
            temp6     += Ash * B[idx+6];
            temp7     += Ash * B[idx+7];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
          C[i*M+j+4]  = temp4;
          C[i*M+j+5]  = temp5;
          C[i*M+j+6]  = temp6;
          C[i*M+j+7]  = temp7;
        }
      }
      // Leftover on M
      if (M & 0x00000007) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          for (uint32_t j=(M-(M & 0x00000007)); j<M; j++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k*M+j];
            }
          C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i++) 
      {
        for (uint32_t j=0; j<(M & 0xfffffff8); j=j+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K];
            temp2     += Ash * B[idx+2*K];
            temp3     += Ash * B[idx+3*K];
            temp4     += Ash * B[idx+4*K];
            temp5     += Ash * B[idx+5*K];
            temp6     += Ash * B[idx+6*K];
            temp7     += Ash * B[idx+7*K];
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
          C[i*M+j+4]  = temp4;
          C[i*M+j+5]  = temp5;
          C[i*M+j+6]  = temp6;
          C[i*M+j+7]  = temp7;
        }
      }
      // Leftover on M
      if (M & 0x00000007) 
      {
        for (uint32_t i=start; i<stop; i++) 
        {
          for (uint32_t j=(M-(M & 0x00000007)); j<M; j++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k+j*K];
            }
          C[i*M+j] = temp;
          }
        }
      }    
    }
  }
}



void mm_unroll_2x1 (void * matMul_args)
{

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;
  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 2) { mm(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (uint32_t i=start; i<stop; i=i+2)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k*M+j];
            temp0     += A[idx]   * Bsh;
            temp1     += A[idx+K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t jj=j_start; jj<j_stop; jj++)
        {
          for (uint32_t ii=N-N_left; ii<N; ii++)
          {
            float temp = 0;
            for (uint32_t kk=0; kk<K; kk++)
            {
              temp += A[ii*K+kk] * B[kk*M+jj];
            }
            C[ii*M+jj] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t i=start; i<stop; i=i+2)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k+j*K];
            temp0     += A[idx]   * Bsh;
            temp1     += A[idx+K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t jj=j_start; jj<j_stop; jj++)
        {
          for (uint32_t ii=N-N_left; ii<N; ii++)
          {
            float temp = 0;
            for (uint32_t kk=0; kk<K; kk++)
            {
              temp += A[ii*K+kk] * B[kk+jj*K];
            }
            C[ii*M+jj] = temp;
          }
        }
      }
    }
  }
}



void mm_unroll_4x1 (void * matMul_args)
{

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;
  uint32_t N_par = N & 0xfffffffc;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 4) { mm_unroll_2x1(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (uint32_t i=start; i<stop; i=i+4)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k*M+j];
            temp0     += A[idx]     * Bsh;
            temp1     += A[idx+K]   * Bsh;
            temp2     += A[idx+2*K] * Bsh;
            temp3     += A[idx+3*K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k*M+j];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t i=start; i<stop; i=i+4)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k+j*K];
            temp0 += A[idx]     * Bsh;
            temp1 += A[idx+K]   * Bsh;
            temp2 += A[idx+2*K] * Bsh;
            temp3 += A[idx+3*K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k+j*K];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }
  }
}



void mm_unroll_8x1 (void * matMul_args)
{

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;
  uint32_t N_par = N & 0xfffffff8;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 8) { mm_unroll_4x1(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (uint32_t i=start; i<stop; i=i+8)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k*M+j];
            temp0     += A[idx]     * Bsh;
            temp1     += A[idx+K]   * Bsh;
            temp2     += A[idx+2*K] * Bsh;
            temp3     += A[idx+3*K] * Bsh;
            temp4     += A[idx+4*K] * Bsh;
            temp5     += A[idx+5*K] * Bsh;
            temp6     += A[idx+6*K] * Bsh;
            temp7     += A[idx+7*K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
          C[(i+4)*M+j]  = temp4;
          C[(i+5)*M+j]  = temp5;
          C[(i+6)*M+j]  = temp6;
          C[(i+7)*M+j]  = temp7;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k*M+j];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t i=start; i<stop; i=i+8)
      {
        for (uint32_t j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            float Bsh = B[k+j*K];
            temp0     += A[idx]     * Bsh;
            temp1     += A[idx+K]   * Bsh;
            temp2     += A[idx+2*K] * Bsh;
            temp3     += A[idx+3*K] * Bsh;
            temp4     += A[idx+4*K] * Bsh;
            temp5     += A[idx+5*K] * Bsh;
            temp6     += A[idx+6*K] * Bsh;
            temp7     += A[idx+7*K] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
          C[(i+4)*M+j]  = temp4;
          C[(i+5)*M+j]  = temp5;
          C[(i+6)*M+j]  = temp6;
          C[(i+7)*M+j]  = temp7;
        }
      }
      // Leftover on N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp += A[i*K+k] * B[k+j*K];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }
  }
}



void mm_unroll_2x2 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 2) { mm_unroll_1x8(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+2) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+1];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp2     += Ash * Ba;
            temp3     += Ash * Bb;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[(i+1)*M+j]    = temp2;
          C[(i+1)*M+j+1]  = temp3;
        }
        // Leftover in M
        if (M & 0x00000001) 
        {
          for (uint32_t ii=i; ii<i+2; ii++) 
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[ii*K+k] * B[k*M+(M-1)];
            }
            C[ii*M+M-1] = left_temp;
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j+k*M];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+2) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+K];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp2     += Ash * Ba;
            temp3     += Ash * Bb;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[(i+1)*M+j]    = temp2;
          C[(i+1)*M+j+1]  = temp3;
        }
        // Leftover in M
        if (M & 0x00000001) 
        {
          for (uint32_t ii=i; ii<i+2; ii++) 
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[ii*K+k] * B[k+(M-1)*K];
            }
            C[ii*M+M-1] = left_temp;
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }    
    }
  }
}



void mm_unroll_2x4 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;
  float temp4 = 0;
  float temp5 = 0;
  float temp6 = 0;
  float temp7 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((N_par/NUM_CORES) < 2) { mm_unroll_1x8(args); }
  else if (M < 4)                 { mm_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+2) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+1];
            float Bc  = B[idx+2];
            float Bd  = B[idx+3];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            temp2     += Ash * Bc;
            temp3     += Ash * Bd;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            temp6     += Ash * Bc;
            temp7     += Ash * Bd;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[i*M+j+2]      = temp2;
          C[i*M+j+3]      = temp3;
          C[(i+1)*M+j]    = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+1)*M+j+2]  = temp6;
          C[(i+1)*M+j+3]  = temp7;
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (uint32_t ii=i; ii<i+2; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k*M+j];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j+k*M];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+2) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+K];
            float Bc  = B[idx+2*K];
            float Bd  = B[idx+3*K];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            temp2     += Ash * Bc;
            temp3     += Ash * Bd;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            temp6     += Ash * Bc;
            temp7     += Ash * Bd;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[i*M+j+2]      = temp2;
          C[i*M+j+3]      = temp3;
          C[(i+1)*M+j]    = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+1)*M+j+2]  = temp6;
          C[(i+1)*M+j+3]  = temp7;
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (uint32_t ii=i; ii<i+2; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k+j*K];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (uint32_t k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }
    }
  }
}



void mm_unroll_4x2 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffc;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;
  float temp4 = 0;
  float temp5 = 0;
  float temp6 = 0;
  float temp7 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((N_par/NUM_CORES) < 4) { mm_unroll_1x8(args); }
  else if (M < 2)                 { mm_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+4) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+1];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp2     += Ash * Ba;
            temp3     += Ash * Bb;
            // Third A row
            Ash       = A[(i+2)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            // Fourth A row
            Ash       = A[(i+3)*K+k];
            temp6     += Ash * Ba;
            temp7     += Ash * Bb;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[(i+1)*M+j]    = temp2;
          C[(i+1)*M+j+1]  = temp3;
          C[(i+2)*M+j]    = temp4;
          C[(i+2)*M+j+1]  = temp5;
          C[(i+3)*M+j]    = temp6;
          C[(i+3)*M+j+1]  = temp7;
        }
        // Leftover in M
        if (M & 0x00000001) 
        {
          for (uint32_t ii=i; ii<i+4; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000001)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k*M+j];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=N-N_left; i<N; i++)
          {
            float temp_left = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp_left += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = temp_left;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+4) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+K];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp2     += Ash * Ba;
            temp3     += Ash * Bb;
            // Third A row
            Ash       = A[(i+2)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            // Fourth A row
            Ash       = A[(i+3)*K+k];
            temp6     += Ash * Ba;
            temp7     += Ash * Bb;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[(i+1)*M+j]    = temp2;
          C[(i+1)*M+j+1]  = temp3;
          C[(i+2)*M+j]    = temp4;
          C[(i+2)*M+j+1]  = temp5;
          C[(i+3)*M+j]    = temp6;
          C[(i+3)*M+j+1]  = temp7;
        }
        // Leftover in M
        if (M & 0x00000001) 
        {
          for (uint32_t ii=i; ii<i+4; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000001)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k+j*K];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t j=j_start; j<j_stop; j++)
        {
          for (uint32_t i=N-N_left; i<N; i++)
          {  
            float temp_left = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp_left += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = temp_left;
          }
        }
      }
    }
  }
}



void mm_unroll_4x4 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffc;
  uint32_t N_left = N - N_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;  float temp8   = 0;
  float temp1 = 0;  float temp9   = 0;
  float temp2 = 0;  float temp10  = 0;
  float temp3 = 0;  float temp11  = 0;
  float temp4 = 0;  float temp12  = 0;
  float temp5 = 0;  float temp13  = 0;
  float temp6 = 0;  float temp14  = 0;
  float temp7 = 0;  float temp15  = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((N_par/NUM_CORES) < 2) { mm_unroll_1x8(args); }
  else if ((N_par/NUM_CORES) < 4) { mm_unroll_2x4(args); }
  else if (M < 4)                 { mm_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED
    if (transp == 0)
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+4) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;  temp8   = 0;
          temp1 = 0;  temp9   = 0;
          temp2 = 0;  temp10  = 0;
          temp3 = 0;  temp11  = 0;
          temp4 = 0;  temp12  = 0;
          temp5 = 0;  temp13  = 0;
          temp6 = 0;  temp14  = 0;
          temp7 = 0;  temp15  = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k*M+j;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+1];
            float Bc  = B[idx+2];
            float Bd  = B[idx+3];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            temp2     += Ash * Bc;
            temp3     += Ash * Bd;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            temp6     += Ash * Bc;
            temp7     += Ash * Bd;
            // Third A row
            Ash       = A[(i+2)*K+k];
            temp8     += Ash * Ba;
            temp9     += Ash * Bb;
            temp10    += Ash * Bc;
            temp11    += Ash * Bd;
            // Fourth A row
            Ash       = A[(i+3)*K+k];
            temp12    += Ash * Ba;
            temp13    += Ash * Bb;
            temp14    += Ash * Bc;
            temp15    += Ash * Bd;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[i*M+j+2]      = temp2;
          C[i*M+j+3]      = temp3;
          C[(i+1)*M+j]    = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+1)*M+j+2]  = temp6;
          C[(i+1)*M+j+3]  = temp7;
          C[(i+2)*M+j]    = temp8;
          C[(i+2)*M+j+1]  = temp9;
          C[(i+2)*M+j+2]  = temp10;
          C[(i+2)*M+j+3]  = temp11;
          C[(i+3)*M+j]    = temp12;
          C[(i+3)*M+j+1]  = temp13;
          C[(i+3)*M+j+2]  = temp14;
          C[(i+3)*M+j+3]  = temp15;
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (uint32_t ii=i; ii<i+4; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k*M+j];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t i=N-N_left; i<N; i++)
        {
          for (uint32_t j=j_start; j<j_stop; j++)
          {
            float temp_left = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp_left += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = temp_left;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      // Unrolled core
      for (uint32_t i=start; i<stop; i=i+4) 
      {
        for (uint32_t j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;  temp8   = 0;
          temp1 = 0;  temp9   = 0;
          temp2 = 0;  temp10  = 0;
          temp3 = 0;  temp11  = 0;
          temp4 = 0;  temp12  = 0;
          temp5 = 0;  temp13  = 0;
          temp6 = 0;  temp14  = 0;
          temp7 = 0;  temp15  = 0;

          for (uint32_t k=0; k<K; k++) 
          {
            uint32_t idx   = k+j*K;
            // First A row
            float Ash = A[i*K+k];
            float Ba  = B[idx];
            float Bb  = B[idx+K];
            float Bc  = B[idx+2*K];
            float Bd  = B[idx+3*K];
            temp0     += Ash * Ba;
            temp1     += Ash * Bb;
            temp2     += Ash * Bc;
            temp3     += Ash * Bd;
            // Second A row
            Ash       = A[(i+1)*K+k];
            temp4     += Ash * Ba;
            temp5     += Ash * Bb;
            temp6     += Ash * Bc;
            temp7     += Ash * Bd;
            // Third A row
            Ash       = A[(i+2)*K+k];
            temp8     += Ash * Ba;
            temp9     += Ash * Bb;
            temp10    += Ash * Bc;
            temp11    += Ash * Bd;
            // Fourth A row
            Ash       = A[(i+3)*K+k];
            temp12    += Ash * Ba;
            temp13    += Ash * Bb;
            temp14    += Ash * Bc;
            temp15    += Ash * Bd;
          }
          C[i*M+j]        = temp0;
          C[i*M+j+1]      = temp1;
          C[i*M+j+2]      = temp2;
          C[i*M+j+3]      = temp3;
          C[(i+1)*M+j]    = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+1)*M+j+2]  = temp6;
          C[(i+1)*M+j+3]  = temp7;
          C[(i+2)*M+j]    = temp8;
          C[(i+2)*M+j+1]  = temp9;
          C[(i+2)*M+j+2]  = temp10;
          C[(i+2)*M+j+3]  = temp11;
          C[(i+3)*M+j]    = temp12;
          C[(i+3)*M+j+1]  = temp13;
          C[(i+3)*M+j+2]  = temp14;
          C[(i+3)*M+j+3]  = temp15;
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (uint32_t ii=i; ii<i+4; ii++) 
          {
            for (uint32_t j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[k+j*K];
              }
              C[ii*M+j] = left_temp;
            }
          }
        }
      }

      // Leftover in N (parallel on M)
      if (N_left > 0)
      {
        uint32_t j_block = (M+NUM_CORES-1) / NUM_CORES;
        uint32_t j_start = core_id*j_block;
        uint32_t j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (uint32_t i=N-N_left; i<N; i++)
        {
          for (uint32_t j=j_start; j<j_stop; j++)
          {
            float temp_left = 0;
            for (uint32_t k=0; k<K; k++)
            {
              temp_left += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = temp_left;
          }
        }
      }
    }
  }
}











// Naive mm with unrolling of 2, parallelizes on M
void mm_M_u2 (void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > M ? M : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (uint32_t i = 0; i < N; i++) 
    {
      for (uint32_t j = start; j < stop; j++) 
      {
        float temp = 0;
        for (uint32_t k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[j+k*M];
              temp += A[i*K+k+1] * B[j+(k+1)*M];
              #ifdef DEBUG
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
              #endif
        } //k
        C[i*M+j] = temp;
      } //j
    } //i

    // Leftover on K
    if (K & 0x00000001)
    for (uint32_t i=0; i<N; i++) {
      for (uint32_t j=start; j<stop; j++) {
        C[i*M+j] += A[i*K+(K-1)] * B[j+(K-1)*M];
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (uint32_t i = 0; i < N; i++) 
    {
      float temp = 0;
      for (uint32_t j = start; j < stop; j++) 
      {
        for (uint32_t k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[k+j*K];
              temp += A[i*K+k+1] * B[k+1+j*K];
              #ifdef DEBUG              
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
        } //k
        C[i*M+j] = temp;
      } //j
    } //i

    // Leftover on K 
    if (K & 0x00000001)
    for (uint32_t i=0; i<N; i++) {
      for (uint32_t j=start; j<stop; j++) {
        C[i*M+j] += A[i*K+(K-1)] * B[(K-1)+j*K];
      }
    }
  }

}



void mm_M_unroll_2x1 (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 2) { mm_M(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffffe); i=i+2)
        {
          float temp0 = 0;
          float temp1 = 0;
          
          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          float temp = 0;
          for (uint32_t k=0; k<K; k++)
          { 
            temp += A[(N-1)*K+k] * B[j+k*M];
          }
          C[(N-1)*M+j] = temp;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffffe); i=i+2)
        {
          float temp0 = 0;
          float temp1 = 0;
          
          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          float temp = 0;
          for (uint32_t k=0; k<K; k++)
          { 
            temp += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp;
        }
      }
    }
  }
}



void mm_M_unroll_4x1 (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffffc;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 4) { mm_M_unroll_2x1(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffffc); i=i+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          
          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            uint32_t idx2 = (i+2)*K+k;
            uint32_t idx3 = (i+3)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
            temp2 += A[idx2] * Bsh;
            temp3 += A[idx3] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          for (uint32_t i=(N-N_left); i<N; i++) 
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            { 
              temp += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffffc); i=i+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            uint32_t idx2 = (i+2)*K+k;
            uint32_t idx3 = (i+3)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
            temp2 += A[idx2] * Bsh;
            temp3 += A[idx3] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            { 
              temp += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_8x1 (void * matMul_args) 
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t N_par = N & 0xfffffff8;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 8) { mm_M_unroll_4x1(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffff8); i=i+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;
          
          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            uint32_t idx2 = (i+2)*K+k;
            uint32_t idx3 = (i+3)*K+k;
            uint32_t idx4 = (i+4)*K+k;
            uint32_t idx5 = (i+5)*K+k;
            uint32_t idx6 = (i+6)*K+k;
            uint32_t idx7 = (i+7)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
            temp2 += A[idx2] * Bsh;
            temp3 += A[idx3] * Bsh;
            temp4 += A[idx4] * Bsh;
            temp5 += A[idx5] * Bsh;
            temp6 += A[idx6] * Bsh;
            temp7 += A[idx7] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
          C[(i+4)*M+j]  = temp4;
          C[(i+5)*M+j]  = temp5;
          C[(i+6)*M+j]  = temp6;
          C[(i+7)*M+j]  = temp7;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          for (uint32_t i=(N-N_left); i<N; i++) 
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            { 
              temp += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j++)
      {
        for (uint32_t i=0; i<(N & 0xfffffff8); i=i+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            uint32_t idx0 = i*K+k;
            uint32_t idx1 = (i+1)*K+k;
            uint32_t idx2 = (i+2)*K+k;
            uint32_t idx3 = (i+3)*K+k;
            uint32_t idx4 = (i+4)*K+k;
            uint32_t idx5 = (i+5)*K+k;
            uint32_t idx6 = (i+6)*K+k;
            uint32_t idx7 = (i+7)*K+k;
            temp0 += A[idx0] * Bsh;
            temp1 += A[idx1] * Bsh;
            temp2 += A[idx2] * Bsh;
            temp3 += A[idx3] * Bsh;
            temp4 += A[idx4] * Bsh;
            temp5 += A[idx5] * Bsh;
            temp6 += A[idx6] * Bsh;
            temp7 += A[idx7] * Bsh;
          }
          C[i*M+j]      = temp0;
          C[(i+1)*M+j]  = temp1;
          C[(i+2)*M+j]  = temp2;
          C[(i+3)*M+j]  = temp3;
          C[(i+4)*M+j]  = temp4;
          C[(i+5)*M+j]  = temp5;
          C[(i+6)*M+j]  = temp6;
          C[(i+7)*M+j]  = temp7;
        }
      }
      // Leftover on N
      if (N_left > 0)
      {
        for (uint32_t j=start; j<stop; j++) 
        {
          for (uint32_t i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (uint32_t k=0; k<K; k++)
            { 
              temp += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_1x2 (void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffe;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 2) { mm_M(args); }
  else
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j+k*M;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;   

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++) 
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj+kk*M];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j*K+k;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj*K+kk];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_1x4 (void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffc;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 4) { mm_M_unroll_1x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (uint32_t j=start; j<stop; j=j+4)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j+k*M;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1];
            temp2     += Ash * B[idx+2];
            temp3     += Ash * B[idx+3]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj+kk*M];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j*K+k;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj*K+kk];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_1x8 (void * matMul_args) {

  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;

  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffff8;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 8) { mm_M_unroll_1x4(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (uint32_t j=start; j<stop; j=j+8)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j+k*M;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+1];
            temp2     += Ash * B[idx+2];
            temp3     += Ash * B[idx+3];
            temp4     += Ash * B[idx+4];
            temp5     += Ash * B[idx+5];
            temp6     += Ash * B[idx+6];
            temp7     += Ash * B[idx+7]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
          C[i*M+j+4]  = temp4;
          C[i*M+j+5]  = temp5;
          C[i*M+j+6]  = temp6;
          C[i*M+j+7]  = temp7;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj+kk*M];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+8)
      {
        for (uint32_t i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = j*K+k;
            float Ash = A[i*K+k];
            temp0     += Ash * B[idx];
            temp1     += Ash * B[idx+K];
            temp2     += Ash * B[idx+2*K];
            temp3     += Ash * B[idx+3*K];
            temp4     += Ash * B[idx+4*K];
            temp5     += Ash * B[idx+5*K];
            temp6     += Ash * B[idx+6*K];
            temp7     += Ash * B[idx+7*K]; 
          }
          C[i*M+j]    = temp0;
          C[i*M+j+1]  = temp1;
          C[i*M+j+2]  = temp2;
          C[i*M+j+3]  = temp3;
          C[i*M+j+4]  = temp4;
          C[i*M+j+5]  = temp5;
          C[i*M+j+6]  = temp6;
          C[i*M+j+7]  = temp7;
        }
      }
      // Leftover in M (parallel in N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (uint32_t ii=i_start; ii<i_stop; ii++)
        {
          for (uint32_t jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (uint32_t kk=0; kk<K; kk++)
            {
              left_temp += A[ii*K+kk] * B[jj*K+kk];
            }
            C[ii*M+jj] = left_temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_2x2 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffe;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 2) { mm_M_unroll_8x1(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<(N & 0xfffffffe); i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B column
            float Bsh = B[j+k*M];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            // Second B column
            Bsh       = B[j+1+k*M];
            temp2     += Aa * Bsh;
            temp3     += Ab * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[i*M+j+1]      = temp2;
          C[(i+1)*M+j+1]  = temp3;
        }
        // Leftover on N
        if (N & 0x00000001)
        {
          for (uint32_t jj=j; jj<j+2; jj++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[(N-1)*K+k] * B[jj+k*M];
            }
            C[(N-1)*M+jj] = left_temp;
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;

          for (uint32_t k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)+k*M];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B column
            float Bsh = B[j*K+k];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            // Second B column
            Bsh       = B[(j+1)*K+k];
            temp2     += Aa * Bsh;
            temp3     += Ab * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[i*M+j+1]      = temp2;
          C[(i+1)*M+j+1]  = temp3;
        }
        // Leftover on N
        if (N & 0x00000001)
        {
          for (uint32_t jj=j; jj<j+2; jj++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[(N-1)*K+k] * B[jj*K+k];
            }
            C[(N-1)*M+jj] = left_temp;
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;

          for (uint32_t k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)*K+k];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }
  }
}



void mm_M_unroll_4x2 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffe;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t N_par = N & 0xfffffffc;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;
  float temp4 = 0;
  float temp5 = 0;
  float temp6 = 0;
  float temp7 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 2) { mm_M_unroll_8x1(args); }
  else if (N < 4)                 { mm_M_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N_par; i=i+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            uint32_t idx   = i*K+k;
            temp0 += A[idx]     * Bsh;
            temp1 += A[idx+K]   * Bsh;
            temp2 += A[idx+2*K] * Bsh;
            temp3 += A[idx+3*K] * Bsh;

            Bsh = B[j+1+k*M];
            temp4 += A[idx]     * Bsh;
            temp5 += A[idx+K]   * Bsh;
            temp6 += A[idx+2*K] * Bsh;
            temp7 += A[idx+3*K] * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[(i+2)*M+j]    = temp2;
          C[(i+3)*M+j]    = temp3;
          C[i*M+j+1]      = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+2)*M+j+1]  = temp6;
          C[(i+3)*M+j+1]  = temp7;
        }
        // Leftover on N
        if (N & 0x00000003)
        {
          for (uint32_t jj=j; jj<j+2; jj++)
          {
            for (uint32_t i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[i*K+k] * B[jj+k*M];
              }
              C[i*M+jj] = left_temp;
            }
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;
          for (uint32_t k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)+k*M];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+2)
      {
        for (uint32_t i=0; i<N_par; i=i+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            uint32_t idx   = i*K+k;
            temp0 += A[idx]     * Bsh;
            temp1 += A[idx+K]   * Bsh;
            temp2 += A[idx+2*K] * Bsh;
            temp3 += A[idx+3*K] * Bsh;

            Bsh = B[(j+1)*K+k];
            temp4 += A[idx]     * Bsh;
            temp5 += A[idx+K]   * Bsh;
            temp6 += A[idx+2*K] * Bsh;
            temp7 += A[idx+3*K] * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[(i+2)*M+j]    = temp2;
          C[(i+3)*M+j]    = temp3;
          C[i*M+j+1]      = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+2)*M+j+1]  = temp6;
          C[(i+3)*M+j+1]  = temp7;
        }
        // Leftover on N
        if (N & 0x00000003)
        {
          for (uint32_t jj=j; jj<j+2; jj++)
          {
            for (uint32_t i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[i*K+k] * B[jj*K+k];
              }
              C[i*M+jj] = left_temp;
            }
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;
          for (uint32_t k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)*K+k];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }
  }
}



void mm_M_unroll_2x4 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffc;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t N_par = N & 0xfffffffe;
  uint32_t N_left = N - N_par;

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;
  float temp1 = 0;
  float temp2 = 0;
  float temp3 = 0;
  float temp4 = 0;
  float temp5 = 0;
  float temp6 = 0;
  float temp7 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 4) { mm_M_unroll_8x1(args); }
  else if (N < 2)                 { mm_M_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      for (uint32_t j=start; j<stop; j=j+4)
      {
        for (uint32_t i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B column
            float Bsh = B[j+k*M];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            // Second B column
            Bsh       = B[j+1+k*M];
            temp2     += Aa * Bsh;
            temp3     += Ab * Bsh;
            // Third B column
            Bsh       = B[j+2+k*M];
            temp4     += Aa * Bsh;
            temp5     += Ab * Bsh;
            // Fourth B column
            Bsh       = B[j+3+k*M];
            temp6     += Aa * Bsh;
            temp7     += Ab * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[i*M+j+1]      = temp2;
          C[(i+1)*M+j+1]  = temp3;
          C[i*M+j+2]      = temp4;
          C[(i+1)*M+j+2]  = temp5;
          C[i*M+j+3]      = temp6;
          C[(i+1)*M+j+3]  = temp7;
        }
        // Leftover on N
        if (N & 0x00000001)
        {
          for (uint32_t jj=j; jj<j+4; jj++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[(N-1)*K+k] * B[jj+k*M];
            }
            C[(N-1)*M+jj] = left_temp;
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          for (uint32_t j=M-M_left; j<M; j++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = left_temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+4)
      {
        for (uint32_t i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B row
            float Bsh = B[j*K+k];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            // Second B row
            Bsh       = B[(j+1)*K+k];
            temp2     += Aa * Bsh;
            temp3     += Ab * Bsh;
            // Second B row
            Bsh       = B[(j+2)*K+k];
            temp4     += Aa * Bsh;
            temp5     += Ab * Bsh;
            // Second B row
            Bsh       = B[(j+3)*K+k];
            temp6     += Aa * Bsh;
            temp7     += Ab * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[i*M+j+1]      = temp2;
          C[(i+1)*M+j+1]  = temp3;
          C[i*M+j+2]      = temp4;
          C[(i+1)*M+j+2]  = temp5;
          C[i*M+j+3]      = temp6;
          C[(i+1)*M+j+3]  = temp7;
        }
        // Leftover on N
        if (N & 0x00000001)
        {
          for (uint32_t jj=j; jj<j+4; jj++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[(N-1)*K+k] * B[jj*K+k];
            }
            C[(N-1)*M+jj] = left_temp;
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t i=i_start; i<i_stop; i++)
        {
          for (uint32_t j=M-M_left; j<M; j++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = left_temp;
          }
        }
      }
    }
  }
}



void mm_M_unroll_4x4 (void * matMul_args)
{
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  uint32_t N = args->N;
  uint32_t M = args->M;
  uint32_t K = args->K;
  uint32_t transp = args->trans_B;

  uint32_t M_par = M & 0xfffffffc;
  uint32_t M_left = M - M_par;
  uint32_t core_id = pi_core_id();

  uint32_t blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  uint32_t start = core_id*blockSize;
  uint32_t stop = start+blockSize > M_par ? M_par : start+blockSize;

  // Global accumulators
  float temp0 = 0;  float temp8  = 0;
  float temp1 = 0;  float temp9  = 0;
  float temp2 = 0;  float temp10 = 0;
  float temp3 = 0;  float temp11 = 0;
  float temp4 = 0;  float temp12 = 0;
  float temp5 = 0;  float temp13 = 0;
  float temp6 = 0;  float temp14 = 0;
  float temp7 = 0;  float temp15 = 0;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 2) { mm_M_unroll_8x1(args); }
  else if ((M_par/NUM_CORES) < 4) { mm_M_unroll_4x2(args); }
  else if (N < 4)                 { mm_M_unroll_2x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      for (uint32_t j=start; j<stop; j=j+4)
      {
        for (uint32_t i=0; i<(N & 0xfffffffc); i=i+4)
        {
          temp0 = 0;  temp8  = 0;
          temp1 = 0;  temp9  = 0;
          temp2 = 0;  temp10 = 0;
          temp3 = 0;  temp11 = 0;
          temp4 = 0;  temp12 = 0;
          temp5 = 0;  temp13 = 0;
          temp6 = 0;  temp14 = 0;
          temp7 = 0;  temp15 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B colums
            float Bsh = B[j+k*M];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            float Ac  = A[idx+2*K];
            float Ad  = A[idx+3*K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            temp2     += Ac * Bsh;
            temp3     += Ad * Bsh;
            // Second B column
            Bsh       = B[j+1+k*M];
            temp4     += Aa * Bsh;
            temp5     += Ab * Bsh;
            temp6     += Ac * Bsh;
            temp7     += Ad * Bsh;
            // third B column
            Bsh       = B[j+2+k*M];
            temp8     += Aa * Bsh;
            temp9     += Ab * Bsh;
            temp10    += Ac * Bsh;
            temp11    += Ad * Bsh;
            // Fourth B column
            Bsh       = B[j+3+k*M];
            temp12    += Aa * Bsh;
            temp13    += Ab * Bsh;
            temp14    += Ac * Bsh;
            temp15    += Ad * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[(i+2)*M+j]    = temp2;
          C[(i+3)*M+j]    = temp3;
          C[i*M+j+1]      = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+2)*M+j+1]  = temp6;
          C[(i+3)*M+j+1]  = temp7;
          C[i*M+j+2]      = temp8;
          C[(i+1)*M+j+2]  = temp9;
          C[(i+2)*M+j+2]  = temp10;
          C[(i+3)*M+j+2]  = temp11;
          C[i*M+j+3]      = temp12;
          C[(i+1)*M+j+3]  = temp13;
          C[(i+2)*M+j+3]  = temp14;
          C[(i+3)*M+j+3]  = temp15;
        }
        // Leftover on N
        if (N & 0x00000003)
        {
          for (uint32_t jj=j; jj<j+4; jj++)
          {
            for (uint32_t i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[i*K+k] * B[jj+k*M];
              }
              C[i*M+jj] = left_temp;
            }
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t j=(M-M_left); j<M; j++)
        {
          for (uint32_t i=i_start; i<i_stop; i++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[i*K+k] * B[j+k*M];
            }
            C[i*M+j] = left_temp;
          }
        }
      }
    }

    // =====> B IS TRANSPOSED <=====
    else 
    {
      for (uint32_t j=start; j<stop; j=j+4)
      {
        for (uint32_t i=0; i<(N & 0xfffffffc); i=i+4)
        {
          temp0 = 0;  temp8  = 0;
          temp1 = 0;  temp9  = 0;
          temp2 = 0;  temp10 = 0;
          temp3 = 0;  temp11 = 0;
          temp4 = 0;  temp12 = 0;
          temp5 = 0;  temp13 = 0;
          temp6 = 0;  temp14 = 0;
          temp7 = 0;  temp15 = 0;

          for (uint32_t k=0; k<K; k++)
          {
            uint32_t idx   = i*K+k;
            // First B row
            float Bsh = B[j*K+k];
            float Aa  = A[idx];
            float Ab  = A[idx+K];
            float Ac  = A[idx+2*K];
            float Ad  = A[idx+3*K];
            temp0     += Aa * Bsh;
            temp1     += Ab * Bsh;
            temp2     += Ac * Bsh;
            temp3     += Ad * Bsh;
            // Second B row
            Bsh       = B[(j+1)*K+k];
            temp4     += Aa * Bsh;
            temp5     += Ab * Bsh;
            temp6     += Ac * Bsh;
            temp7     += Ad * Bsh;
            // Third B row
            Bsh       = B[(j+2)*K+k];
            temp8     += Aa * Bsh;
            temp9     += Ab * Bsh;
            temp10    += Ac * Bsh;
            temp11    += Ad * Bsh;
            // Fourth B row
            Bsh       = B[(j+3)*K+k];
            temp12    += Aa * Bsh;
            temp13    += Ab * Bsh;
            temp14    += Ac * Bsh;
            temp15    += Ad * Bsh;
          }
          C[i*M+j]        = temp0;
          C[(i+1)*M+j]    = temp1;
          C[(i+2)*M+j]    = temp2;
          C[(i+3)*M+j]    = temp3;
          C[i*M+j+1]      = temp4;
          C[(i+1)*M+j+1]  = temp5;
          C[(i+2)*M+j+1]  = temp6;
          C[(i+3)*M+j+1]  = temp7;
          C[i*M+j+2]      = temp8;
          C[(i+1)*M+j+2]  = temp9;
          C[(i+2)*M+j+2]  = temp10;
          C[(i+3)*M+j+2]  = temp11;
          C[i*M+j+3]      = temp12;
          C[(i+1)*M+j+3]  = temp13;
          C[(i+2)*M+j+3]  = temp14;
          C[(i+3)*M+j+3]  = temp15;
        }
        // Leftover on N
        if (N & 0x00000003)
        {
          for (uint32_t jj=j; jj<j+4; jj++)
          {
            for (uint32_t i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (uint32_t k=0; k<K; k++)
              {
                left_temp += A[i*K+k] * B[jj*K+k];
              }
              C[i*M+jj] = left_temp;
            }
          }
        }
      }

      // Leftover on M (parallel on N)
      if (M_left > 0)
      {
        uint32_t i_block = (N+NUM_CORES-1) / NUM_CORES;
        uint32_t i_start = core_id*i_block;
        uint32_t i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (uint32_t j=M-M_left; j<M; j++)
        {
          for (uint32_t i=i_start; i<i_stop; i++)
          {
            float left_temp = 0;
            for (uint32_t k=0; k<K; k++)
            {
              left_temp += A[i*K+k] * B[j*K+k];
            }
            C[i*M+j] = left_temp;
          }
        }
      }
    }
  }
}
