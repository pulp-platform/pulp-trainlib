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

void mm(void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const int N = args->N;
  const int M = args->M;
  const int K = args->K;

  int transp = args->trans_B;

  const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > N ? N : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    if (K == 1) 
    {
      for (int i=start; i < stop; i++) 
      {
        for (int j = 0; j < M; j++) 
        {
          C[i*M+j] = A[i*K] * B[j];
          #ifdef DEBUG
          printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
          #endif
        }
      }
    }
    else if (K > 0)
    {
      for (int i=start; i < stop; i++) 
      {
        for (int j = 0; j < M; j++) 
        {
          float temp = 0;
          for (int k = 0; k < K; k++) 
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
      for (int i=start; i < stop; i++) 
      {
        for (int j = 0; j < M; j++) 
        {
          C[i*M+j] = A[i*K] * B[j*K];
          #ifdef DEBUG
          printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
          #endif
        } 
      } 
    }
    else if (K > 0)
    {
      for (int i=start; i < stop; i++) 
      {
        for (int j = 0; j < M; j++) 
        {
          float temp = 0;
          for (int k = 0; k < K; k++) 
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
void mm_M(void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const int N = args->N;
  const int M = args->M;
  const int K = args->K;

  int transp = args->trans_B;

  const int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > M ? M: start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (int i = 0; i < N; i++) 
    {
      for (int j=start; j < stop; j++) 
      {
        float temp = 0;
        for (int k = 0; k < K; k++) 
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

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (int i = 0; i < N; i++) 
    {
      for (int j=start; j < stop; j++) 
      {
        float temp = 0;
        for (int k = 0; k < K; k++) 
        {
              temp += A[i*K+k] * B[j*K+k];
              #ifdef DEBUG              
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
        } 
        C[i*M+j] += temp;
      } 
    } 
  }
}



// Matmul for depthwise convolutions
void mm_dw(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  #ifdef DEBUG
  int num_MAC = 0;
  #endif

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  #ifdef DEBUG
  float a = 0;
  float b = 0;
  int idx_a = 0;
  int idx_b = 0;
  #endif

  for (int j = 0; j < M; j++) 
  {
    for (int k=start; k < stop; k++) 
    {
      #ifdef DEBUG
        printf("\nCORE %d: start=%d, stop=%d\n", pi_core_id(), start, stop);
      #endif
      float temp = 0; 
      for (int t = 0; t < ker_dim; t++) 
      {
        #ifdef DEBUG
          // variables needed for debugging, remove to measure performances
          idx_a = /*i*K+*/(k*ker_dim+t);
          a = A[idx_a];
          idx_b = j*(N*ker_dim)+(k*ker_dim+t);
          b = B[idx_b];
          temp += a * b;
          num_MAC++;
          printf("idx_a=%d, a=%f, idx_b=%d, b=%f, temp=%f\n", idx_a, a, idx_b, b, temp);
        #else
          temp += A[k*ker_dim+t] * B[j*(N*ker_dim)+(k*ker_dim+t)];
        #endif
      }
      C[j+k*M] = temp;
      #ifdef DEBUG
        printf("C[%d] = %f\n", j+k*M, temp);
      #endif
    } 
  }

  #ifdef DEBUG
  printf("\n\n=====> MM_DW MAC: %d <=====\n\n", num_MAC);
  #endif
}



void mm_dw_in_grad(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

  #ifdef DEBUG
  int num_MAC = 0;
  #endif

  for (int i=0; i<N; i++) 
  {
    for (int j=0; j<M; j++) 
    {
      for (int k=start; k<stop; k++)
      {
        float temp = 0;
        for (int u=0; u<ker_dim; u++) 
        {
          // In-order weights (A matrix)
          // temp += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
          
          // Flipped weights (A matrix inverted channel-by-channel)
          temp += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+j*K];
          
          #ifdef DEBUG
          num_MAC++;
          #endif
        }
        C[j+k*M] = temp;
      }
    }
  }

  #ifdef DEBUG
  printf("\n\n=====> MM_DW_IN_GRAD MAC: %d <=====\n\n", num_MAC);
  #endif
}



void mm_conv2d_in_grad (void * void_args) 
{

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  const int H = args->H;
  const int W = args->W;

  const int pW = args->pW;
  const int pH = args->pH;
  const int pCin = args->pCin;
  const int pCout = args->pCout;

  //const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  //const int start = pi_core_id()*blockSize;
  //const int stop = start+blockSize > N ? N : start+blockSize;

  // Addressing variables
  

  for (int k=0; k<pCin; k++) 
  {
    for (int v=0; v<pCout; v++)
    {
      int col_index = 0;
      for (int i=0; i<H; i++)
      {
        for (int j=0; j<W; j++)
        {
          float temp = 0;
          for (int t=0; t<pW; t++)
          {
            for (int u=0; u<pH; u++)
            {
              // in_grad = ker * out_grad
              // Non-flipped weights
              temp += A[t*pW + u + v*pW*pH*pCin + k*pH*pW] * B[t*pW + u + col_index*pW*pH*pCout + v*H*W];

              // Flipped weights
              //temp += A[(pW*pH - t*pW + u - 1) + v*pW*pH*pCin + k*pH*pW] * B[t*pW + u + col_index*pW*pH*pCout + v*H*W];
            }
          }
          C[i*W+j+k*H*W] = temp;
          col_index++;
        }
        col_index++;
      }
    }
  }
}













/**
 * OPTIMIZED VERSIONS
 */

// Naive mm with unrolling of 2
void mm_u2 (void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (int i=start; i < stop; i++) 
    {
      for (int j = 0; j < M; j++) 
      {
        float temp = 0;
        for (int k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[j+k*M];
              temp += A[i*K+k+1] * B[j+(k+1)*M];
        } 
        C[i*M+j] += temp;
      } 
    } 
    // Leftover on K
    if (K & 0x00000001)
    {
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<M; j++) 
        {
          C[i*M+j] += A[i*K+(K-1)] * B[j+(K-1)*M];
        }
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (int i=start; i < stop; i++) 
    {
      for (int j = 0; j < M; j++) 
      {
        float temp = 0;
        for (int k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[k+j*K];
              temp += A[i*K+k+1] * B[k+1+j*K];              
        } 
        C[i*M+j] += temp;
        temp = 0;
      } 
    } 
    // Leftover on K 
    if (K & 0x00000001)
    {
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<M; j++) 
        {
          C[i*M+j] += A[i*K+(K-1)] * B[(K-1)+j*K];
        }
      }
    }
  }
}



void mm_unroll_1x2 (void * void_args) 
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 2) { mm(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      // Unrolled core
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
        for (int i=start; i<stop; i++) 
        {
          float temp = 0;
          for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
        for (int i=start; i<stop; i++) 
        {
          float temp = 0;
          for (int k=0; k<K; k++)
          {
            temp += A[i*K+k] * B[k+(M-1)*K];
          }
          C[i*M+M-1] = temp;
        }
      }    
    }
  }
}



void mm_unroll_1x4 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 4) { mm_unroll_1x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
        for (int i=start; i<stop; i++) 
        {
          for (int j=(M-(M & 0x00000003)); j<M; j++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
        for (int i=start; i<stop; i++) 
        {
          for (int j=(M-(M & 0x00000003)); j<M; j++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_unroll_1x8 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if (M < 8) { mm_unroll_1x4(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0)
    {
      // Unrolled core
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffff8); j=j+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
        for (int i=start; i<stop; i++) 
        {
          for (int j=(M-(M & 0x00000007)); j<M; j++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i++) 
      {
        for (int j=0; j<(M & 0xfffffff8); j=j+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
        for (int i=start; i<stop; i++) 
        {
          for (int j=(M-(M & 0x00000007)); j<M; j++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_unroll_2x1 (void * void_args)
{

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;
  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 2) { mm(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (int i=start; i<stop; i=i+2)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int jj=j_start; jj<j_stop; jj++)
        {
          for (int ii=N-N_left; ii<N; ii++)
          {
            float temp = 0;
            for (int kk=0; kk<K; kk++)
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
      for (int i=start; i<stop; i=i+2)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int jj=j_start; jj<j_stop; jj++)
        {
          for (int ii=N-N_left; ii<N; ii++)
          {
            float temp = 0;
            for (int kk=0; kk<K; kk++)
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



void mm_unroll_4x1 (void * void_args)
{

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;
  int N_par = N & 0xfffffffc;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 4) { mm_unroll_2x1(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (int i=start; i<stop; i=i+4)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+4)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_unroll_8x1 (void * void_args)
{

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;
  int N_par = N & 0xfffffff8;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if ((N_par/NUM_CORES) < 8) { mm_unroll_4x1(args); }
  else
  {  
    // =====> B NOT TRANSPOSED <=====
    if (transp==0) 
    {
      for (int i=start; i<stop; i=i+8)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+8)
      {
        for (int j=0; j<M; j++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;       

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_unroll_2x2 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

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
      for (int i=start; i<stop; i=i+2) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
          for (int ii=i; ii<i+2; ii++) 
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+2) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
          for (int ii=i; ii<i+2; ii++) 
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (int k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }    
    }
  }
}



void mm_unroll_2x4 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

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
      for (int i=start; i<stop; i=i+2) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
          for (int ii=i; ii<i+2; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+2) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
          for (int ii=i; ii<i+2; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          float temp_left = 0;
          for (int k=0; k<K; k++)
          {
            temp_left += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp_left;
        }
      }
    }
  }
}



void mm_unroll_4x2 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffc;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

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
      for (int i=start; i<stop; i=i+4) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
          for (int ii=i; ii<i+4; ii++) 
          {
            for (int j=(M-(M & 0x00000001)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=N-N_left; i<N; i++)
          {
            float temp_left = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+4) 
      {
        for (int j=0; j<(M & 0xfffffffe); j=j+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
          for (int ii=i; ii<i+4; ii++) 
          {
            for (int j=(M-(M & 0x00000001)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int j=j_start; j<j_stop; j++)
        {
          for (int i=N-N_left; i<N; i++)
          {  
            float temp_left = 0;
            for (int k=0; k<K; k++)
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



void mm_unroll_4x4 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffc;
  int N_left = N - N_par;
  int core_id = pi_core_id();

  int blockSize = (N_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > N_par ? N_par : start+blockSize;

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
      for (int i=start; i<stop; i=i+4) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;  temp8   = 0;
          temp1 = 0;  temp9   = 0;
          temp2 = 0;  temp10  = 0;
          temp3 = 0;  temp11  = 0;
          temp4 = 0;  temp12  = 0;
          temp5 = 0;  temp13  = 0;
          temp6 = 0;  temp14  = 0;
          temp7 = 0;  temp15  = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k*M+j;
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
          for (int ii=i; ii<i+4; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int i=N-N_left; i<N; i++)
        {
          for (int j=j_start; j<j_stop; j++)
          {
            float temp_left = 0;
            for (int k=0; k<K; k++)
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
      for (int i=start; i<stop; i=i+4) 
      {
        for (int j=0; j<(M & 0xfffffffc); j=j+4)
        {
          temp0 = 0;  temp8   = 0;
          temp1 = 0;  temp9   = 0;
          temp2 = 0;  temp10  = 0;
          temp3 = 0;  temp11  = 0;
          temp4 = 0;  temp12  = 0;
          temp5 = 0;  temp13  = 0;
          temp6 = 0;  temp14  = 0;
          temp7 = 0;  temp15  = 0;

          for (int k=0; k<K; k++) 
          {
            int idx   = k+j*K;
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
          for (int ii=i; ii<i+4; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int j_block = (M+NUM_CORES-1) / NUM_CORES;
        int j_start = core_id*j_block;
        int j_stop = j_start+j_block > M ? M : j_start+j_block;

        for (int i=N-N_left; i<N; i++)
        {
          for (int j=j_start; j<j_stop; j++)
          {
            float temp_left = 0;
            for (int k=0; k<K; k++)
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
void mm_M_u2 (void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;

  int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > M ? M : start+blockSize;

  // =====> B NOT TRANSPOSED <=====
  if (transp==0)
  {
    for (int i = 0; i < N; i++) 
    {
      for (int j = start; j < stop; j++) 
      {
        float temp = 0;
        for (int k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[j+k*M];
              temp += A[i*K+k+1] * B[j+(k+1)*M];
              #ifdef DEBUG
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f", i*M+j, i*K+k, j+k*M, C[i*M+j], A[i*K+k], B[j+k*M]);
              #endif
        } //k
        C[i*M+j] += temp;
      } //j
    } //i

    // Leftover on K
    if (K & 0x00000001)
    for (int i=0; i<N; i++) {
      for (int j=start; j<stop; j++) {
        C[i*M+j] += A[i*K+(K-1)] * B[j+(K-1)*M];
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    for (int i = 0; i < N; i++) 
    {
      float temp = 0;
      for (int j = start; j < stop; j++) 
      {
        for (int k = 0; k < (K & 0xfffffffe); k=k+2) 
        {
              temp += A[i*K+k]   * B[k+j*K];
              temp += A[i*K+k+1] * B[k+1+j*K];
              #ifdef DEBUG              
              printf("C[%i] += A[%i] * B[%i] -> %f = %f * %f\n", i*M+j, i*K+k, k+j*K, C[i*M+j], A[i*K+k], B[k+j*K]);
              #endif
        } //k
        C[i*M+j] += temp;
      } //j
    } //i

    // Leftover on K 
    if (K & 0x00000001)
    for (int i=0; i<N; i++) {
      for (int j=start; j<stop; j++) {
        C[i*M+j] += A[i*K+(K-1)] * B[(K-1)+j*K];
      }
    }
  }

}



void mm_M_unroll_2x1 (void * void_args) 
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;

  int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 2) { mm_M(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffffe); i=i+2)
        {
          float temp0 = 0;
          float temp1 = 0;
          
          for (int k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          float temp = 0;
          for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffffe); i=i+2)
        {
          float temp0 = 0;
          float temp1 = 0;
          
          for (int k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          float temp = 0;
          for (int k=0; k<K; k++)
          { 
            temp += A[(N-1)*K+k] * B[j*K+k];
          }
          C[(N-1)*M+j] = temp;
        }
      }
    }
  }
}



void mm_M_unroll_4x1 (void * void_args) 
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffffc;
  int N_left = N - N_par;

  int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 4) { mm_M_unroll_2x1(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffffc); i=i+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          
          for (int k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
            int idx2 = (i+2)*K+k;
            int idx3 = (i+3)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          for (int i=(N-N_left); i<N; i++) 
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffffc); i=i+4)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
            int idx2 = (i+2)*K+k;
            int idx3 = (i+3)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_M_unroll_8x1 (void * void_args) 
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int N_par = N & 0xfffffff8;
  int N_left = N - N_par;

  int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > M ? M : start+blockSize;

   // Check if sizes are smaller than the unrolling, and take countermeasures
  if      (N < 8) { mm_M_unroll_4x1(args); }
  else
  { 
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffff8); i=i+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;
          
          for (int k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
            int idx2 = (i+2)*K+k;
            int idx3 = (i+3)*K+k;
            int idx4 = (i+4)*K+k;
            int idx5 = (i+5)*K+k;
            int idx6 = (i+6)*K+k;
            int idx7 = (i+7)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          for (int i=(N-N_left); i<N; i++) 
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j++)
      {
        for (int i=0; i<(N & 0xfffffff8); i=i+8)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            int idx0 = i*K+k;
            int idx1 = (i+1)*K+k;
            int idx2 = (i+2)*K+k;
            int idx3 = (i+3)*K+k;
            int idx4 = (i+4)*K+k;
            int idx5 = (i+5)*K+k;
            int idx6 = (i+6)*K+k;
            int idx7 = (i+7)*K+k;
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
        for (int j=start; j<stop; j++) 
        {
          for (int i=(N-N_left); i<N; i++)
          {
            float temp = 0;
            for (int k=0; k<K; k++)
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



void mm_M_unroll_1x2 (void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;

  int M_par = M & 0xfffffffe;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 2) { mm_M(args); }
  else
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j+k*M;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;   

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++) 
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j*K+k;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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



void mm_M_unroll_1x4 (void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;

  int M_par = M & 0xfffffffc;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 4) { mm_M_unroll_1x2(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (int j=start; j<stop; j=j+4)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j+k*M;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j*K+k;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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



void mm_M_unroll_1x8 (void * void_args) {

  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int transp = args->trans_B;

  int M_par = M & 0xfffffff8;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par: start+blockSize;

  // Check if sizes are smaller than the unrolling, and take countermeasures
  if      ((M_par/NUM_CORES) < 8) { mm_M_unroll_1x4(args); }
  else 
  {
    // =====> B NOT TRANSPOSED <=====
    if (transp==0)
    {
      for (int j=start; j<stop; j=j+8)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j+k*M;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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
      for (int j=start; j<stop; j=j+8)
      {
        for (int i=0; i<N; i++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;
          float temp4 = 0;
          float temp5 = 0;
          float temp6 = 0;
          float temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = j*K+k;
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;      

        for (int ii=i_start; ii<i_stop; ii++)
        {
          for (int jj=M-M_left; jj<M; jj++)
          {
            float left_temp = 0;

            for (int kk=0; kk<K; kk++)
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



void mm_M_unroll_2x2 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int M_par = M & 0xfffffffe;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par : start+blockSize;

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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<(N & 0xfffffffe); i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+2; jj++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;

          for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+2; jj++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;

          for (int k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)*K+k];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }
  }
}



void mm_M_unroll_4x2 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int M_par = M & 0xfffffffe;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int N_par = N & 0xfffffffc;
  int N_left = N - N_par;

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par : start+blockSize;

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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N_par; i=i+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++)
          {
            float Bsh = B[j+k*M];
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+2; jj++)
          {
            for (int i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;
          for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j=j+2)
      {
        for (int i=0; i<N_par; i=i+4)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++)
          {
            float Bsh = B[j*K+k];
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+2; jj++)
          {
            for (int i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          float left_temp = 0;
          for (int k=0; k<K; k++)
          {
            left_temp += A[i*K+k] * B[(M-1)*K+k];
          }
          C[i*M+(M-1)] = left_temp;
        }
      }
    }
  }
}



void mm_M_unroll_2x4 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int M_par = M & 0xfffffffc;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par : start+blockSize;

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
      for (int j=start; j<stop; j=j+4)
      {
        for (int i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+4; jj++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          for (int j=M-M_left; j<M; j++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j=j+4)
      {
        for (int i=0; i<N_par; i=i+2)
        {
          temp0 = 0;
          temp1 = 0;
          temp2 = 0;
          temp3 = 0;
          temp4 = 0;
          temp5 = 0;
          temp6 = 0;
          temp7 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+4; jj++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int i=i_start; i<i_stop; i++)
        {
          for (int j=M-M_left; j<M; j++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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



void mm_M_unroll_4x4 (void * void_args)
{
  struct matMul_args* args = (struct matMul_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int transp = args->trans_B;

  int M_par = M & 0xfffffffc;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
  int start = core_id*blockSize;
  int stop = start+blockSize > M_par ? M_par : start+blockSize;

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
      for (int j=start; j<stop; j=j+4)
      {
        for (int i=0; i<(N & 0xfffffffc); i=i+4)
        {
          temp0 = 0;  temp8  = 0;
          temp1 = 0;  temp9  = 0;
          temp2 = 0;  temp10 = 0;
          temp3 = 0;  temp11 = 0;
          temp4 = 0;  temp12 = 0;
          temp5 = 0;  temp13 = 0;
          temp6 = 0;  temp14 = 0;
          temp7 = 0;  temp15 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+4; jj++)
          {
            for (int i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int j=(M-M_left); j<M; j++)
        {
          for (int i=i_start; i<i_stop; i++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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
      for (int j=start; j<stop; j=j+4)
      {
        for (int i=0; i<(N & 0xfffffffc); i=i+4)
        {
          temp0 = 0;  temp8  = 0;
          temp1 = 0;  temp9  = 0;
          temp2 = 0;  temp10 = 0;
          temp3 = 0;  temp11 = 0;
          temp4 = 0;  temp12 = 0;
          temp5 = 0;  temp13 = 0;
          temp6 = 0;  temp14 = 0;
          temp7 = 0;  temp15 = 0;

          for (int k=0; k<K; k++)
          {
            int idx   = i*K+k;
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
          for (int jj=j; jj<j+4; jj++)
          {
            for (int i=(N-(N & 0x00000003)); i<N; i++)
            {
              float left_temp = 0;
              for (int k=0; k<K; k++)
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
        int i_block = (N+NUM_CORES-1) / NUM_CORES;
        int i_start = core_id*i_block;
        int i_stop = i_start+i_block > N ? N : i_start+i_block;

        for (int j=M-M_left; j<M; j++)
        {
          for (int i=i_start; i<i_stop; i++)
          {
            float left_temp = 0;
            for (int k=0; k<K; k++)
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

















// Matmul for depthwise convolutions
void mm_dw_u2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  int k = 0;

  if (ker_dim < 2) { mm_dw(args); }
  else
  {
    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

    float a = 0;
    float b = 0;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < 1; i++) {  
      for (int j = 0; j < M; j++) {
        for (int k=start; k < stop; k++) {
          float temp = 0;
          for (int t = 0; t < (ker_dim & 0xfffffffe); t=t+2) {
              temp += A[i*K+(k*ker_dim+t)] * B[j*(N*ker_dim)+(k*ker_dim+t)];
              temp += A[i*K+(k*ker_dim+t+1)] * B[j*(N*ker_dim)+(k*ker_dim+t+1)];            
          }
          // Leftover on ker_dim
          if (ker_dim & 1) {
            temp += A[i*K+(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
          }
          C[i*M+j+k*M] = temp;
        } //k
      } //j
    } //i
  }
}



// Matmul for depthwise convolutions
void mm_dw_u3(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  int k = 0;

  if (ker_dim < 3) { mm_dw(args); }
  else
  {
    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

    float a = 0;
    float b = 0;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < 1; i++) {  
      for (int j = 0; j < M; j++) {
        for (int k=start; k < stop; k++) {
          float temp = 0;
          for (int t = 0; t < (ker_dim & 0xfffffffd); t=t+3) {
              temp += A[i*K+(k*ker_dim+t)] * B[j*(N*ker_dim)+(k*ker_dim+t)];
              temp += A[i*K+(k*ker_dim+t+1)] * B[j*(N*ker_dim)+(k*ker_dim+t+1)];
              temp += A[i*K+(k*ker_dim+t+2)] * B[j*(N*ker_dim)+(k*ker_dim+t+2)];
          }
          // Leftover on ker_dim
          if (ker_dim % 3) {
            if (ker_dim & 0x00000001) {
              temp += A[i*K+(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            }
            else {
              temp += A[i*K+(k*ker_dim+ker_dim-2)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-2)];
              temp += A[i*K+(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            }
          }
          C[i*M+j+k*M] = temp;
        } //k
      } //j
    } //i
  }
}



// Matmul for depthwise convolutions
void mm_dw_unroll_1x2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int ker_dim = args->ker_size;

  int k = 0;

  if (M < 2) { mm_dw(args); }
  else 
  {
    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

      for (int j = 0; j < (M & 0xfffffffe); j+=2) 
      {
        for (int k=start; k < stop; k++) 
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int t = 0; t < ker_dim; t++) 
          {
            float Ash  = A[k*ker_dim+t];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t)];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
        }
        // Leftover in M
        if (M % 2) 
        {
          for (int k=start; k < stop; k++) 
          {
            float temp = 0;

            for (int t = 0; t < ker_dim; t++) 
            {
              temp += A[(k*ker_dim+t)] * B[(M-1)*(N*ker_dim)+(k*ker_dim+t)];
            }
            C[(M-1)+k*M]    = temp;
          }        
        } 
      } 
    } 
}



// Matmul for depthwise convolutions
void mm_dw_unroll_1x4(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int ker_dim = args->ker_size;

  int k = 0;

  if (M < 4) { mm_dw_unroll_1x2(args); }
  else
  {
    int M_left = M % 4;

    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

      for (int j = 0; j < (M & 0xfffffffc); j+=4) 
      {
        for (int k=start; k < stop; k++) 
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int t = 0; t < ker_dim; t++) 
          {
            float Ash  = A[(k*ker_dim+t)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t)];
            temp2     += Ash * B[(j+2)*(N*ker_dim)+(k*ker_dim+t)];
            temp3     += Ash * B[(j+3)*(N*ker_dim)+(k*ker_dim+t)];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
          C[j+2+k*M]  = temp2;
          C[j+3+k*M]  = temp3;
        }
        // Leftover in M
        if (M % 4) 
        {
          for (int j = M-M_left; j < M; j++) 
          {
            for (int k=start; k < stop; k++) 
            {
              float temp = 0; 
              for (int t = 0; t < ker_dim; t++) 
              {
                temp += A[(k*ker_dim+t)] * B[j*(N*ker_dim)+(k*ker_dim+t)];
              }
              C[j+k*M] = temp;
            }        
          }
        } 
      } 
    }
}



// Matmul for depthwise convolutions
void mm_dw_unroll_1x2_u2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int ker_dim = args->ker_size;

  int k = 0;

  if      (M < 2)         { mm_dw(args); }
  else if (ker_dim < 2)   { mm_dw_unroll_1x2(args); }
  else 
  {
    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

      for (int j = 0; j < (M & 0xfffffffe); j+=2) 
      {
        for (int k=start; k < stop; k++) 
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int t = 0; t < (ker_dim & 0xfffffffe); t+=2) 
          {
            float Ash  = A[(k*ker_dim+t)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t)];
            Ash        = A[(k*ker_dim+t+1)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t+1)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t+1)];
          }
          if (ker_dim & 1) {
            temp0     += A[(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            temp1     += A[(k*ker_dim+ker_dim-1)] * B[(j+1)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
        }
        // Leftover in M
        if (M % 2) 
        {
          for (int k=start; k < stop; k++) 
          {
            float temp = 0;

            for (int t = 0; t < (ker_dim & 0xfffffffe); t+=2) 
            {
              temp += A[(k*ker_dim+t)] * B[(M-1)*(N*ker_dim)+(k*ker_dim+t)];
              temp += A[(k*ker_dim+t+1)] * B[(M-1)*(N*ker_dim)+(k*ker_dim+t+1)];
            }
            if (ker_dim & 1) {
              temp += A[(k*ker_dim+ker_dim-1)] * B[(M-1)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            }
            C[(M-1)+k*M]    = temp;
          }        
        } 
      } 
    } 
}




// Matmul for depthwise convolutions
void mm_dw_unroll_1x4_u2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;

  int ker_dim = args->ker_size;

  int k = 0;

  if      (M < 4)         { mm_dw_unroll_1x2(args); }
  else if (ker_dim < 2)   { mm_dw_unroll_1x4(args); }
  else
  {
    int M_left = M % 4;

    int blockSize = (N+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > N ? N : start+blockSize;

      for (int j = 0; j < (M & 0xfffffffc); j+=4) 
      {
        for (int k=start; k < stop; k++) 
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int t = 0; t < (ker_dim & 0xfffffffe); t+=2) 
          {
            float Ash  = A[(k*ker_dim+t)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t)];
            temp2     += Ash * B[(j+2)*(N*ker_dim)+(k*ker_dim+t)];
            temp3     += Ash * B[(j+3)*(N*ker_dim)+(k*ker_dim+t)];
            Ash  = A[(k*ker_dim+t+1)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+t+1)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+t+1)];
            temp2     += Ash * B[(j+2)*(N*ker_dim)+(k*ker_dim+t+1)];
            temp3     += Ash * B[(j+3)*(N*ker_dim)+(k*ker_dim+t+1)];
          }
          if (ker_dim & 1) {
            float Ash  = A[(k*ker_dim+ker_dim-1)];
            temp0     += Ash * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            temp1     += Ash * B[(j+1)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            temp2     += Ash * B[(j+2)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
            temp3     += Ash * B[(j+3)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];           
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
          C[j+2+k*M]  = temp2;
          C[j+3+k*M]  = temp3;
        }
        // Leftover in M
        if (M % 4) 
        {
          for (int j = M-M_left; j < M; j++) 
          {
            for (int k=start; k < stop; k++) 
            {
              float temp = 0; 
              for (int t = 0; t < (ker_dim & 0xfffffffe); t+=2) 
              {
                temp += A[(k*ker_dim+t)] * B[j*(N*ker_dim)+(k*ker_dim+t)];
                temp += A[(k*ker_dim+t+1)] * B[j*(N*ker_dim)+(k*ker_dim+t+1)];
              }
              if (ker_dim & 1) {
                temp += A[(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
              }
              C[j+k*M] = temp;
            }        
          }
        } 
      } 
    }
}



// Matmul for dw input grad with unrolling of 2
void mm_dw_in_grad_u2(void * void_args)
{

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if (ker_dim < 2)  { mm_dw_in_grad(args); }
  else
  {
    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j<M; j++) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp = 0;
          for (int u=0; u < (ker_dim & 0xfffffffe); u=u+2) 
          {
            temp += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
            temp += A[u+1+k*ker_dim] * B[u+1+k*ker_dim+j*K];
          }
          if (ker_dim & 1) {
            temp += A[ker_dim-1+k*ker_dim] * B[ker_dim-1+k*ker_dim+j*K];
          }
          C[j+k*M] = temp;
        }
      }
    }
  }
}



// Matmul for dw input grad with unrolling of 2
void mm_dw_in_grad_u3(void * void_args)
{

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if (ker_dim < 3)  { mm_dw_in_grad(args); }
  else
  {
    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j<M; j++) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp = 0;
          for (int u=0; u < (ker_dim & 0xfffffffd); u=u+3) 
          {
            temp += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
            temp += A[u+1+k*ker_dim] * B[u+1+k*ker_dim+j*K];
            temp += A[u+2+k*ker_dim] * B[u+2+k*ker_dim+j*K];
          }
          if (ker_dim % 3) {
            if (ker_dim & 0x00000001) {
              temp += A[ker_dim-1+k*ker_dim] * B[ker_dim-1+k*ker_dim+j*K];
            }
            else {
              temp += A[ker_dim-2+k*ker_dim] * B[ker_dim-2+k*ker_dim+j*K];
              temp += A[ker_dim-1+k*ker_dim] * B[ker_dim-1+k*ker_dim+j*K];
            }
          }
          C[j+k*M] = temp;
        }
      }
    }
  }
}



void mm_dw_in_grad_unroll_1x2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if (M < 2) { mm_dw_in_grad(args); }
  else 
  {
    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j< (M & 0xfffffffe); j+=2) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int u=0; u<ker_dim; u++) 
          {
            // In-order weights (A matrix)
            // float Ash = A[u+k*ker_dim];
            // temp0    += Ash * B[u+k*ker_dim+j*K];
            // temp1    += Ash * B[u+k*ker_dim+(j+1)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-u+k*ker_dim];
            temp0     += Ash * B[u+k*ker_dim+j*K];
            temp1     += Ash * B[u+k*ker_dim+(j+1)*K];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
        }
      }
      // Leftover in M
      if (M % 1) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp = 0;
          for (int u=0; u<ker_dim; u++) 
          {
            // In-order weights (A matrix)
            // temp += A[u+k*ker_dim] * B[u+k*ker_dim+(M-1)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel
            temp += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+(M-1)*K];
          }
          C[(M-1)+k*M]    = temp;
        }        
      }
    }
  }
}



void mm_dw_in_grad_unroll_1x4(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if (M < 4) { mm_dw_in_grad_unroll_1x2(args); }
  else 
  {
    int M_left = M % 4;

    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j< (M & 0xfffffffc); j+=4) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int u=0; u<ker_dim; u++) 
          {
            // In-order weights (A matrix)
            // float Ash = A[u+k*ker_dim];
            // temp0    += Ash * B[u+k*ker_dim+j*K];
            // temp1    += Ash * B[u+k*ker_dim+(j+1)*K];
            // temp2    += Ash * B[u+k*ker_dim+(j+2)*K];
            // temp3    += Ash * B[u+k*ker_dim+(j+3)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-u+k*ker_dim];
            temp0     += Ash * B[u+k*ker_dim+j*K];
            temp1     += Ash * B[u+k*ker_dim+(j+1)*K];
            temp2     += Ash * B[u+k*ker_dim+(j+2)*K];
            temp3     += Ash * B[u+k*ker_dim+(j+3)*K];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
          C[j+2+k*M]  = temp2;
          C[j+3+k*M]  = temp3;
        }
      }
      // Leftover in M
      if (M % 4) 
      {
        for (int j=M-M_left; j< M; j++) 
        {
          for (int k=start; k<stop; k++)
          {
            float temp = 0;
            for (int u=0; u<ker_dim; u++) 
            {
              // In-order weights (A matrix)
              // temp0    += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
              
              // Flipped weights (A matrix inverted channel-by-channel)
              temp     += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+j*K];
            }
            C[j+k*M]    = temp;
          }
        }        
      }
    }
  }
}



void mm_dw_in_grad_unroll_1x2_u2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if      (M < 2)         { mm_dw_in_grad(args); }
  else if (ker_dim < 2)   { mm_dw_in_grad_unroll_1x2(args); }
  else 
  {
    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j< (M & 0xfffffffe); j+=2) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp0 = 0;
          float temp1 = 0;

          for (int u=0; u<(ker_dim & 0xfffffffe); u+=2) 
          {
            // In-order weights (A matrix)
            // float Ash = A[u+k*ker_dim];
            // temp0    += Ash * B[u+k*ker_dim+j*K];
            // temp1    += Ash * B[u+k*ker_dim+(j+1)*K];
            // Ash       = A[u+1+k*ker_dim];
            // temp0    += Ash * B[u+1+k*ker_dim+j*K];
            // temp1    += Ash * B[u+1+k*ker_dim+(j+1)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-u+k*ker_dim];
            temp0     += Ash * B[u+k*ker_dim+j*K];
            temp1     += Ash * B[u+k*ker_dim+(j+1)*K];
            Ash        = A[(ker_dim-1)-(u+1)+k*ker_dim];
            temp0     += Ash * B[u+1+k*ker_dim+j*K];
            temp1     += Ash * B[u+1+k*ker_dim+(j+1)*K];
          }
          if (ker_dim & 1) {
            // In-order weights (A matrix)
            // float Ash = A[ker_dim-1+k*ker_dim];
            // temp0    += Ash * B[ker_dim-1+k*ker_dim+j*K];
            // temp1    += Ash * B[ker_dim-1+k*ker_dim+(j+1)*K];

            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-(ker_dim-1)+k*ker_dim];
            temp0     += Ash * B[ker_dim-1+k*ker_dim+j*K];
            temp1     += Ash * B[ker_dim-1+k*ker_dim+(j+1)*K];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
        }
      }
      // Leftover in M
      if (M % 1) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp = 0;
          for (int u=0; u<(ker_dim & 0xfffffffe); u+=2) 
          {
            // In-order weights (A matrix)
            // temp += A[u+k*ker_dim] * B[u+k*ker_dim+(M-1)*K];
            // temp += A[u+1+k*ker_dim] * B[u+1+k*ker_dim+(M-1)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel
            temp += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+(M-1)*K];
            temp += A[(ker_dim-1)-u+1+k*ker_dim] * B[u+1+k*ker_dim+(M-1)*K];
          }
          if (ker_dim & 1) {
            temp += A[(ker_dim-1)-(ker_dim-1)+k*ker_dim] * B[(ker_dim-1)+k*ker_dim+(M-1)*K];
          }
          C[(M-1)+k*M]    = temp;
        }        
      }
    }
  }
}



void mm_dw_in_grad_unroll_1x4_u2(void * void_args) {

  struct matMul_DW_args* args = (struct matMul_DW_args *)void_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  if      (M < 4)         { mm_dw_in_grad_unroll_1x2(args); }
  else if (ker_dim < 2)   { mm_dw_in_grad_unroll_1x4(args); }
  else 
  {
    int M_left = M % 4;

    int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

    for (int i=0; i<N; i++) 
    {
      for (int j=0; j< (M & 0xfffffffc); j+=4) 
      {
        for (int k=start; k<stop; k++)
        {
          float temp0 = 0;
          float temp1 = 0;
          float temp2 = 0;
          float temp3 = 0;

          for (int u=0; u<(ker_dim & 0xfffffffe); u+=2) 
          {
            // In-order weights (A matrix)
            // float Ash = A[u+k*ker_dim];
            // temp0    += Ash * B[u+k*ker_dim+j*K];
            // temp1    += Ash * B[u+k*ker_dim+(j+1)*K];
            // temp2    += Ash * B[u+k*ker_dim+(j+2)*K];
            // temp3    += Ash * B[u+k*ker_dim+(j+3)*K];
            // Ash       = A[u+1+k*ker_dim];
            // temp0    += Ash * B[u+1+k*ker_dim+j*K];
            // temp1    += Ash * B[u+1+k*ker_dim+(j+1)*K];
            // temp2    += Ash * B[u+1+k*ker_dim+(j+2)*K];
            // temp3    += Ash * B[u+1+k*ker_dim+(j+3)*K];
            
            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-u+k*ker_dim];
            temp0     += Ash * B[u+k*ker_dim+j*K];
            temp1     += Ash * B[u+k*ker_dim+(j+1)*K];
            temp2     += Ash * B[u+k*ker_dim+(j+2)*K];
            temp3     += Ash * B[u+k*ker_dim+(j+3)*K];
            Ash        = A[(ker_dim-1)-(u+1)+k*ker_dim];
            temp0     += Ash * B[u+1+k*ker_dim+j*K];
            temp1     += Ash * B[u+1+k*ker_dim+(j+1)*K];
            temp2     += Ash * B[u+1+k*ker_dim+(j+2)*K];
            temp3     += Ash * B[u+1+k*ker_dim+(j+3)*K];
          }
          if (ker_dim & 1) {
            // In-order weights (A matrix)
            // float Ash = A[(ker_dim-1)+k*ker_dim];
            // temp0    += Ash * B[(ker_dim-1)+k*ker_dim+j*K];
            // temp1    += Ash * B[(ker_dim-1)+k*ker_dim+(j+1)*K];
            // temp2    += Ash * B[(ker_dim-1)+k*ker_dim+(j+2)*K];
            // temp3    += Ash * B[(ker_dim-1)+k*ker_dim+(j+3)*K];

            // Flipped weights (A matrix inverted channel-by-channel)
            float Ash  = A[(ker_dim-1)-(ker_dim-1)+k*ker_dim];
            temp0     += Ash * B[(ker_dim-1)+k*ker_dim+j*K];
            temp1     += Ash * B[(ker_dim-1)+k*ker_dim+(j+1)*K];
            temp2     += Ash * B[(ker_dim-1)+k*ker_dim+(j+2)*K];
            temp3     += Ash * B[(ker_dim-1)+k*ker_dim+(j+3)*K];
          }
          C[j+k*M]    = temp0;
          C[j+1+k*M]  = temp1;
          C[j+2+k*M]  = temp2;
          C[j+3+k*M]  = temp3;
        }
      }
      // Leftover in M
      if (M % 4) 
      {
        for (int j=M-M_left; j< M; j++) 
        {
          for (int k=start; k<stop; k++)
          {
            float temp = 0;
            for (int u=0; u<(ker_dim & 0xfffffffe); u+=2) 
            {
              // In-order weights (A matrix)
              // temp0    += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
              // temp0    += A[u+1+k*ker_dim] * B[u+1+k*ker_dim+j*K];
              
              // Flipped weights (A matrix inverted channel-by-channel)
              temp     += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+j*K];
              temp     += A[(ker_dim-1)-(u+1)+k*ker_dim] * B[u+1+k*ker_dim+j*K];
            }
            if (ker_dim & 1) {
              // In-order weights (A matrix)
              // temp0    += A[(ker_dim-1)+k*ker_dim] * B[(ker_dim-1)+k*ker_dim+j*K];

              // Flipped weights (A matrix inverted channel-by-channel)
              temp     += A[(ker_dim-1)-(ker_dim-1)+k*ker_dim] * B[(ker_dim-1)+k*ker_dim+j*K];                            
            }
            C[j+k*M]    = temp;
          }
        }        
      }
    }
  }
}
