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

#include "pmsis.h"


void mm_fp16(void * void_args) {

  struct matMul_args_fp16* args = (struct matMul_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

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
          fp16 temp = 0;
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
          fp16 temp = 0;
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
void mm_M_fp16(void * void_args) {

  struct matMul_args_fp16* args = (struct matMul_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

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
        fp16 temp = 0;
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
        fp16 temp = 0;
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
void mm_dw_fp16(void * void_args) {

  struct matMul_DW_args_fp16* args = (struct matMul_DW_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

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
  fp16 a = 0;
  fp16 b = 0;
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
      fp16 temp = 0; 
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



void mm_dw_in_grad_fp16(void * void_args) {

  struct matMul_DW_args_fp16* args = (struct matMul_DW_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

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
        fp16 temp = 0;
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


















/**
 * Optimized versions
 */

void __attribute__((noinline)) mm_fp16_SIMD_2x4 (void * void_args) 
{

  struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
  fp16 * __restrict__ A = args->A; 
  fp16 * __restrict__ B = args->B; 
  fp16 * __restrict__ C = args->C; 
  int N = args->N;  
  int M = args->M; 
  int K = args->K;  
  int transp = args->trans_B;

  int indexA, indexB;
  v2f16 Av;
  v2f16 Bv0, Bv1;
  v2f16 *Cv;


  // =====> B NOT TRANSPOSED <=====
  if (transp == 0) 
  {
    #if NUM_CORES > 1
      const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
      const int start = pi_core_id()*blockSize;
      const int stop = start+blockSize < N? start+blockSize: N;

      for (int i = start; i < stop; i++) {

    #else
      const int start = 0;
      const int stop = N;

      for (int i = start; i < stop; i++) {
    #endif
      
      for (int j = 0; j < (M & 0xfffffffe); j+=2) {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j;
            
        for (int k = 0; k < (K & 0xfffffffe) ; k+=2) {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Bv0 = *((v2f16 *) &B[indexB/*k*M+j*/]);
          Bv1 = *((v2f16 *) &B[indexB+M/*k*M+j+M*/]);
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2*M;
          }
          // Leftover on K
          if (K & 1) {
            Av  = (v2f16) {A[indexA], A[indexA]};
            Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
            temp += Av * Bv0;
          } 
          Cv = (v2f16 *)&C[i*M+j];
          *Cv = temp;
          }
      }
      // Leftover on M
      if(M & 1) {
        for (int i = start; i < stop; i++) {
          fp16 val = 0;
          for (int k = 0; k < K; k++) {
            val += A[i*K+k]*B[k*M+(M-1)];
          }
          C[i*M+(M-1)] = val;
        }
      }

  }

  
  // =====> B IS TRANSPOSED <=====
  else 
  {
    #if NUM_CORES > 1
      const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
      const int start = pi_core_id()*blockSize;
      const int stop = start+blockSize < N? start+blockSize: N;

      for (int i = start; i < stop; i++) {

    #else
      const int start = 0;
      const int stop = N;

      for (int i = start; i < stop; i++) {
    #endif
      
      for (int j = 0; j < (M & 0xfffffffe); j+=2) {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j*M;

        for (int k = 0; k < (K & 0xfffffffe) ; k+=2) {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);

          // Bv0 = *((v2f16 *) &B[indexB]);
          // Bv1 = *((v2f16 *) &B[indexB+M]);
          // temp += (v2f16)(Av * Bv0);
          // temp += (v2f16)(Av * Bv1);

          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          Bv1 = (v2f16) {B[indexB+1], B[indexB+M+1]};
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2; 
        }
        // Leftover on K
        if (K & 1) {
          Av  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          #ifdef DEBUG
            printf("------ MM K left (indexA=%d, indexB=%d) => Av = {0x%x, 0x%x}, Bv0 = {0x%x, 0x%x}\n", indexA, indexB, 
              ((unsigned int) Av)&(0xffff0000)>>16, ((unsigned int) Av)&(0x0000ffff), 
              ((unsigned int) Bv0)&(0xffff0000)>>16, ((unsigned int) Bv0)&(0x0000ffff));
          #endif
          temp += (v2f16) (Av * Bv0);
          #ifdef DEBUG
            printf("------ MM K left => temp = {0x%x, 0x%x}\n", 
              ((unsigned int) temp)&(0xffff0000)>>16, ((unsigned int) temp)&(0x0000ffff));
          #endif
        } 
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;
      }
    }

    // Leftover on N
    if(M & 1) {
      for (int i = start; i < stop; i++) {
        fp16 val = 0;
        for (int k = 0; k < K; k++) {
          val += A[i*K+k]*B[(M-1)+k];
        }
      C[i*M+(M-1)] = val;
      }
    }
  }

}



void __attribute__((noinline)) mm_fp16_SIMD_4x8 (void * void_args) 
{

  struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
  fp16 * __restrict__ A = args->A; 
  fp16 * __restrict__ B = args->B; 
  fp16 * __restrict__ C = args->C; 
  int N = args->N; 
  int M = args->M; 
  int K = args->K;  
  int transp = args->trans_B;

  int indexA, indexB;
  v2f16 Av0, Av1;
  v2f16 Bv0, Bv1, Bv2, Bv3;
  v2f16 *Cv;


  // =====> B NOT TRANSPOSED <=====
  if (transp == 0) 
  {
    #if NUM_CORES > 1
      const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
      const int start = pi_core_id()*blockSize;
      const int stop = start+blockSize < N? start+blockSize: N;

      for (int i = start; i < stop; i++) {

    #else
      const int start = 0;
      const int stop = N;

      for (int i = start; i < stop; i++) {
    #endif
      
      for (int j = 0; j < (M & 0xfffffffc); j+=4) {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j;
            
        for (int k = 0; k < (K & 0xfffffffc) ; k+=4) {
          Av0  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Av1  = *((v2f16 *) &A[indexA+2/*i*K+k+2*/]);
          Bv0 = *((v2f16 *) &B[indexB/*k*M+j*/]);
          Bv1 = *((v2f16 *) &B[indexB+M/*k*M+j+M*/]);
          Bv2 = *((v2f16 *) &B[indexB+2*M/*k*M+j+2*M*/]);
          Bv3 = *((v2f16 *) &B[indexB+3*M/*k*M+j+3*M*/]);
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;

          indexA += 4;
          indexB += 4*M;
          }
          // Leftover on K
          if (K & 0x00000002) {
            Av0  = *((v2f16*) &A[indexA]);
            Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
            Bv1 = *((v2f16 *)&B[indexB+M/*k*M+j+M*/]);
            temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
            temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;

            indexA += 2;
            indexB += 2*M;
          } 
          if (K & 0x00000001) {
            Av0  = (v2f16) {A[indexA], A[indexA]};
            Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
            temp += Av0 * Bv0;            
          }
          Cv = (v2f16 *)&C[i*M+j];
          *Cv = temp;
          }
      }
      // Leftover on M
      if(M & 1) {
        for (int i = start; i < stop; i++) {
          fp16 val = 0;
          for (int k = 0; k < K; k++) {
            val += A[i*K+k]*B[k*M+(M-1)];
          }
          C[i*M+(M-1)] = val;
        }
      }

  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    #if NUM_CORES > 1
      const int blockSize = (N+NUM_CORES-1) / NUM_CORES;
      const int start = pi_core_id()*blockSize;
      const int stop = start+blockSize < N? start+blockSize: N;

      for (int i = start; i < stop; i++) {

    #else
      const int start = 0;
      const int stop = N;

      for (int i = start; i < stop; i++) {
    #endif
      
      for (int j = 0; j < (M & 0xfffffffc); j+=4) {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j*M;

        for (int k = 0; k < (K & 0xfffffffc) ; k+=4) {
          Av0  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Av1  = *((v2f16 *) &A[indexA+2/*i*K+k*/]);

          // Bv0 = *((v2f16 *) &B[indexB]);
          // Bv1 = *((v2f16 *) &B[indexB+M]);
          // Bv2 = *((v2f16 *) &B[indexB+2]);
          // Bv3 = *((v2f16 *) &B[indexB+M+2]);
          // temp += (v2f16)(Av0 * Bv0);
          // temp += (v2f16)(Av0 * Bv1);
          // temp += (v2f16)(Av1 * Bv2);
          // temp += (v2f16)(Av1 * Bv3);

          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          Bv1 = (v2f16) {B[indexB+1], B[indexB+M+1]};
          Bv2 = (v2f16) {B[indexB+2], B[indexB+M+2]};
          Bv3 = (v2f16) {B[indexB+3], B[indexB+M+3]};
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;

          indexA += 4; //4;
          indexB += 4; //2;
        }
        // Leftover on K
        if (K & 1) {
          Av0  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          temp += Av0 * Bv0;
        } 
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;
      }
    }

    // Leftover on N
    if(M & 1) {
      for (int i = start; i < stop; i++) {
        fp16 val = 0;
        for (int k = 0; k < K; k++) {
          val += A[i*K+k]*B[(M-1)+k];
        }
      C[i*M+(M-1)] = val;
      }
    }
  }

}



















void __attribute__((noinline)) mm_M_fp16_SIMD_2x4 (void * void_args)
{

  struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
  fp16 * __restrict__ A = args->A; 
  fp16 * __restrict__ B = args->B; 
  fp16 * __restrict__ C = args->C; 
  int N = args->N;  
  int M = args->M; 
  int K = args->K;  
  int transp = args->trans_B;

  int indexA, indexB;
  v2f16 Av;
  v2f16 Bv0, Bv1;
  v2f16 *Cv;

  int M_par = M & 0xfffffffe;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  // =====> B NOT TRANSPOSED <=====
  if (transp == 0) 
  {
    #if NUM_CORES > 1
    const int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
    const int start = core_id*blockSize;
    const int stop = start+blockSize < M_par? start+blockSize: M_par;

    for (int j = start; j < stop; j+=2)
    #else 
    const int start = 0;
    const int stop = M_par;

    for (int j = start; j < stop; j+=2)
    #endif
    {
      for (int i = 0; i < N; i++)
      {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j;

        for (int k = 0; k < (K & 0xfffffffe) ; k+=2) 
        {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Bv0 = *((v2f16 *) &B[indexB/*k*M+j*/]);
          Bv1 = *((v2f16 *) &B[indexB+M/*k*M+j+M*/]);
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2*M;
        }
        // Leftover in K
        if (K & 1)
        {
          Av  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
          temp += Av * Bv0;          
        }
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;        
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      for (int i = start; i < stop; i++) 
      {
        fp16 val = 0;
        for (int k = 0; k < K; k++) 
        {
          val += A[i*K+k]*B[k*M+(M-1)];
        }
        C[i*M+(M-1)] = val;
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    #if NUM_CORES > 1
    const int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
    const int start = core_id*blockSize;
    const int stop = start+blockSize < M_par? start+blockSize: M_par;

    for (int j = start; j < stop; j+=2)
    #else 
    const int start = 0;
    const int stop = M_par;

    for (int j = start; j < stop; j+=2)
    #endif
    {
      for (int i = 0; i < N; i++)
      {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j*M;

        for (int k = 0; k < (K & 0xfffffffe) ; k+=2) 
        {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);

          // Bv0 = *((v2f16 *) &B[indexB]);
          // Bv1 = *((v2f16 *) &B[indexB+M]);
          // temp += (v2f16)(Av * Bv0);
          // temp += (v2f16)(Av * Bv1);

          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          Bv1 = (v2f16) {B[indexB+1], B[indexB+M+1]};
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2;
        }
        // Leftover in K
        if (K & 1)
        {
          Av  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          temp += (v2f16) (Av * Bv0);        
        }
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;        
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      for (int i = start; i < stop; i++) 
      {
        fp16 val = 0;
        for (int k = 0; k < K; k++) 
        {
          val += A[i*K+k]*B[(M-1)+k];
        }
      C[i*M+(M-1)] = val;
      }
    }
  }

}



void __attribute__((noinline)) mm_M_fp16_SIMD_4x8 (void * void_args)
{

  struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
  fp16 * __restrict__ A = args->A; 
  fp16 * __restrict__ B = args->B; 
  fp16 * __restrict__ C = args->C; 
  int N = args->N;  
  int M = args->M; 
  int K = args->K;  
  int transp = args->trans_B;

  int indexA, indexB;
  v2f16 Av0, Av1;
  v2f16 Bv0, Bv1, Bv2, Bv3;
  v2f16 *Cv;

  int M_par = M & 0xfffffffc;
  int M_left = M - M_par;
  int core_id = pi_core_id();

  // =====> B NOT TRANSPOSED <=====
  if (transp == 0) 
  {
    #if NUM_CORES > 1
    const int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
    const int start = core_id*blockSize;
    const int stop = start+blockSize < M_par? start+blockSize: M_par;

    for (int j = start; j < stop; j+=4)
    #else 
    const int start = 0;
    const int stop = M_par;

    for (int j = start; j < stop; j+=4)
    #endif
    {
      for (int i = 0; i < N; i++)
      {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j;

        for (int k = 0; k < (K & 0xfffffffc) ; k+=4) 
        {
          Av0  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Av1  = *((v2f16 *) &A[indexA+2/*i*K+k+2*/]);
          Bv0 = *((v2f16 *) &B[indexB/*k*M+j*/]);
          Bv1 = *((v2f16 *) &B[indexB+M/*k*M+j+M*/]);
          Bv2 = *((v2f16 *) &B[indexB+2*M/*k*M+j+2*M*/]);
          Bv3 = *((v2f16 *) &B[indexB+3*M/*k*M+j+3*M*/]);
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;

          indexA += 4;
          indexB += 4*M;
        }
        // Leftover in K
        if (K & 0x00000002) 
        {
          Av0  = *((v2f16*) &A[indexA]);
          Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
          Bv1 = *((v2f16 *)&B[indexB+M/*k*M+j+M*/]);
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2*M;
        } 
        if (K & 0x00000001) 
        {
          Av0  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = *((v2f16 *)&B[indexB/*k*M+j*/]);
          temp += Av0 * Bv0;            
        }
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;     
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      for (int i = start; i < stop; i++) {
        fp16 val = 0;
        for (int k = 0; k < K; k++) {
          val += A[i*K+k]*B[k*M+(M-1)];
        }
        C[i*M+(M-1)] = val;
      }
    }
  }

  // =====> B IS TRANSPOSED <=====
  else 
  {
    #if NUM_CORES > 1
    const int blockSize = (M_par+NUM_CORES-1) / NUM_CORES;
    const int start = core_id*blockSize;
    const int stop = start+blockSize < M_par? start+blockSize: M_par;

    for (int j = start; j < stop; j+=4)
    #else 
    const int start = 0;
    const int stop = M_par;

    for (int j = start; j < stop; j+=4)
    #endif
    {
      for (int i = 0; i < N; i++)
      {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j*M;

        for (int k = 0; k < (K & 0xfffffffc) ; k+=4) 
        {
          Av0  = *((v2f16 *) &A[indexA/*i*K+k*/]);
          Av1  = *((v2f16 *) &A[indexA+2/*i*K+k*/]);

          // Bv0 = *((v2f16 *) &B[indexB]);
          // Bv1 = *((v2f16 *) &B[indexB+M]);
          // Bv2 = *((v2f16 *) &B[indexB+2]);
          // Bv3 = *((v2f16 *) &B[indexB+M+2]);
          // temp += (v2f16)(Av0 * Bv0);
          // temp += (v2f16)(Av0 * Bv1);
          // temp += (v2f16)(Av1 * Bv2);
          // temp += (v2f16)(Av1 * Bv3);

          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          Bv1 = (v2f16) {B[indexB+1], B[indexB+M+1]};
          Bv2 = (v2f16) {B[indexB+2], B[indexB+M+2]};
          Bv3 = (v2f16) {B[indexB+3], B[indexB+M+3]};
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
          temp += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;

          indexA += 4; 
          indexB += 4; 
        }
        // Leftover in K
        if (K & 0x00000002) 
        {
          Av0  = *((v2f16*) &A[indexA]);
          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          Bv1 = (v2f16) {B[indexB+1], B[indexB+M+1]};
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
          temp += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;

          indexA += 2;
          indexB += 2*M;
        } 
        if (K & 0x00000001) 
        {
          Av0  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+M]};
          temp += Av0 * Bv0;           
        }
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;        
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      for (int i = start; i < stop; i++) 
      {
        fp16 val = 0;
        for (int k = 0; k < K; k++) 
        {
          val += A[i*K+k]*B[(M-1)+k];
        }
      C[i*M+(M-1)] = val;
      }
    }
  }

}






















// Matmul for depthwise convolutions
void __attribute__((noinline)) mm_dw_fp16_SIMD_1x2_u2(void * void_args) {

  struct matMul_DW_args_fp16* args = (struct matMul_DW_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  int blockSize = (N+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > N ? N : start+blockSize;

  for (int j = 0; j < (M & 0xfffffffe); j+=2) 
  {
    for (int k=start; k < stop; k++) 
    {
      v2f16 temp0 = (v2f16) {0, 0};
      v2f16 temp1 = (v2f16) {0, 0}; 
      fp16 res0 = 0;
      fp16 res1 = 0;
      for (int t = 0; t < ker_dim; t+=2) 
      {
          int Aidx     = k*ker_dim+t;
          int Bidx     = j*(N*ker_dim)+(k*ker_dim+t);
          v2f16 Avsh   = *((v2f16*) &A[Aidx]);
          v2f16 Bv     = *((v2f16*) &B[Bidx]);
          // Compute first couple of operands
          temp0        = Avsh * Bv;
          // Sum them
          res0        += (float) (((unsigned int) temp0) & 0x0000ffff) + (((unsigned int) temp0) & 0xffff0000 >> 16);
          
          Bidx         = (j+1)*(N*ker_dim)+(k*ker_dim+t);
          Bv           = *((v2f16*) &B[Bidx]);
          // Compute first couple of operands
          temp1        = Avsh * Bv;
          res1        += (float) (((unsigned int) temp1) & 0x0000ffff) + (((unsigned int) temp1) & 0xffff0000 >> 16);          
      }
      if (ker_dim & 1) 
      {
        res0     += A[(k*ker_dim+ker_dim-1)] * B[j*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
        res1     += A[(k*ker_dim+ker_dim-1)] * B[(j+1)*(N*ker_dim)+(k*ker_dim+ker_dim-1)];
      }
      C[j+k*M]   = res0;
      C[j+1+k*M] = res1;
    } 
    // Leftover in M
    if (M % 2) 
    {
      for (int k=start; k < stop; k++) 
      {
        fp16 temp = 0;

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



void __attribute__((noinline)) mm_dw_in_grad_fp16_SIMD_1x2_u2(void * void_args) {

  struct matMul_DW_args_fp16* args = (struct matMul_DW_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

  int N = args->N;
  int M = args->M;
  int K = args->K;
  int ker_dim = args->ker_size;

  int blockSize = ((K/ker_dim)+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > (K/ker_dim) ? (K/ker_dim) : start+blockSize;

  for (int i=0; i<N; i++) 
  {
    for (int j=0; j<M; j++) 
    {
      for (int k=start; k<stop; k++)
      {
        fp16 temp = 0;
        for (int u=0; u<ker_dim; u++) 
        {
          // In-order weights (A matrix)
          // temp += A[u+k*ker_dim] * B[u+k*ker_dim+j*K];
          
          // Flipped weights (A matrix inverted channel-by-channel)
          temp += A[(ker_dim-1)-u+k*ker_dim] * B[u+k*ker_dim+j*K];          
        }
        C[j+k*M] = temp;
      }
    }
  }

}
