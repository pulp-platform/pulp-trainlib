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
          C[i*M+j] = temp;
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
        C[i*M+j] = temp;
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
        C[i*M+j] = temp;
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



void mm_conv2d_in_grad_fp16 (void * void_args) 
{

  struct matMul_args_fp16* args = (struct matMul_args_fp16 *)void_args;
  fp16 * __restrict__ A = args->A;
  fp16 * __restrict__ B = args->B;
  fp16 * __restrict__ C = args->C;

  const int N = args->N;
  const int K = args->K;
  const int M = args->M;

  const int pW = args->pW;
  const int pH = args->pH;
  const int pCin = args->pCin;
  const int pCout = args->pCout;

  const int blockSize = (M+NUM_CORES-1) / NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start+blockSize > M ? M : start+blockSize;

  // ALGORITHM
  // For each receptive field of the output on the weights
  for (int rec_field = 0; rec_field < M; rec_field++) {
    // For each channel of the output
    for (int Ci = 0; Ci < pCin; Ci++) {
      // Multiply each receptive field for the corresponding
      // set of channels and accumulate on the input channel by channel
      fp16 temp = 0;
      for (int Co = 0; Co < pCout; Co++) {  
        for (int elem = 0; elem < pW*pH; elem++) {
          temp += A[pW*pH*pCin*Co+pW*pH*Ci+elem] * B[pH*pW*pCout*rec_field+pW*pH*Co+elem];
          #ifdef DEBUG
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

  // Optimized looping variables
  int M_loop = (M & 0xfffffffe);
  int K_loop = (K & 0xfffffffe);

  if      (M < 2) mm_fp16(args);
  else if (K < 2) mm_fp16(args);
  else {
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
      
      for (int j = 0; j < M_loop; j+=2) {
        v2f16 temp = (v2f16) {0, 0};
        indexA = i*K;
        indexB = j;
            
        for (int k = 0; k < K_loop; k+=2) {
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
            Av  = (v2f16) {A[i*K+(K-1)], A[i*K+(K-1)]};
            Bv0 = *((v2f16 *)&B[(K-1)*M+j]);
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
      
      for (int j = 0; j < M_loop; j+=2) {
        // Global accumulator
        v2f16 temp = (v2f16) {0, 0};
        // Dot product accumulators
        v2f16 tmp0 = (v2f16) {0, 0};
        v2f16 tmp1 = (v2f16) {0, 0};
        // K leftover accumulator
        v2f16 vtmp = (v2f16) {0, 0};
        // Scalar accumulators for final result
        fp16 a = 0;
        fp16 b = 0;
        // Indices
        indexA = i*K;
        indexB = j*K; //j*M;

        for (int k = 0; k < K_loop; k+=2) {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);

          Bv0 = *((v2f16 *) &B[indexB]);
          Bv1 = *((v2f16 *) &B[indexB+K]);
          tmp0 += (v2f16)(Av * Bv0);
          tmp1 += (v2f16)(Av * Bv1);

          indexA += 2;
          indexB += 2; 
        }
        // Leftover on K
        if (K & 1) {
          Av  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+K]};
          #ifdef DEBUG
            printf("------ MM K left (indexA=%d, indexB=%d) => Av = {0x%x, 0x%x}, Bv0 = {0x%x, 0x%x}\n", indexA, indexB, 
              ((unsigned int) Av)&(0xffff0000)>>16, ((unsigned int) Av)&(0x0000ffff), 
              ((unsigned int) Bv0)&(0xffff0000)>>16, ((unsigned int) Bv0)&(0x0000ffff));
          #endif
          vtmp += (v2f16) (Av * Bv0);
          a += vtmp[0];
          b += vtmp[1];
          #ifdef DEBUG
            printf("------ MM K left => temp = {0x%x, 0x%x}\n", ((unsigned int) vtmp[0]), ((unsigned int) vtmp[1]));
          #endif
        } 
        // Complete dot product
        a += tmp0[0] + tmp0[1];
        b += tmp1[0] + tmp1[1];
        temp = (v2f16) {a, b};
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;
      }
    }

    // Leftover on N
    if(M & 1) {
      for (int i = start; i < stop; i++) {
        fp16 val = 0;
        for (int k = 0; k < K; k++) {
          val += A[i*K+k]*B[(M-1)*K+k];
        }
      C[i*M+(M-1)] = val;
      }
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

  int indexA0, indexA1; 
  int indexB;
  v2f16 Av0, Av1;
  v2f16 Bv0, Bv1, Bv2, Bv3;
  v2f16 *Cv;

  // Looping variables for leftovers
  int N_loop = N & 0xfffffffe;
  int N_left = N - N_loop;
  int core_id = pi_core_id();

  // Integrity barrier for oversized unrolling
  int N_bound = (N_loop)/NUM_CORES;
  int M_bound = (M-(M&0x00000003));
  int K_bound = (K-(K&0x00000001));

  if      (M_bound < 4) mm_fp16_SIMD_2x4(args);
  else if (K_bound < 2) mm_fp16_SIMD_2x4(args);
  else if (N_bound < 2) mm_fp16_SIMD_2x4(args);
  else {
    // =====> B NOT TRANSPOSED <=====
    if (transp == 0) 
    {
    #if NUM_CORES > 1
      const int blockSize = (N_loop+NUM_CORES-1) / NUM_CORES;
      const int start = core_id*blockSize;
      const int stop = start+blockSize < N_loop? start+blockSize: N_loop;

      for (int i = start; i < stop; i+=2) {

    #else
      const int start = 0;
      const int stop = N_loop;

      for (int i = start; i < stop; i+=2) {
    #endif
        for (int j = 0; j < (M & 0xfffffffc); j+=4) 
        {
          v2f16 temp0 = (v2f16) {0, 0};
          v2f16 temp1 = (v2f16) {0, 0};
          v2f16 temp2 = (v2f16) {0, 0};
          v2f16 temp3 = (v2f16) {0, 0};

          for (int k = 0; k < (K & 0xfffffffe); k+=2) 
          {
            // A vectors
            Av0 = *(v2f16 *) &A[i*K+k];
            Av1 = *(v2f16 *) &A[(i+1)*K+k];
            // B vectors
            Bv0 = *(v2f16 *) &B[k*M+j];
            Bv1 = *(v2f16 *) &B[(k+1)*M+j];
            Bv2 = *(v2f16 *) &B[k*M+j+2];
            Bv3 = *(v2f16 *) &B[(k+1)*M+j+2];

            // Ci,j, Ci,j+1
            temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
            temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
            // Ci,j+2, Ci,j+3
            temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv2;
            temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv3;
            // Ci+1,j, Ci+1,j+1
            temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv0;
            temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv1;
            // Ci+1,j+2, Ci+1,j+3
            temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
            temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;          
          }
          // Leftover on K
          if (K & 1)
          {
            Av0  = (v2f16) {A[i*K+(K-1)], A[i*K+(K-1)]};
            Av1  = (v2f16) {A[(i+1)*K+(K-1)], A[(i+1)*K+(K-1)]};
            Bv0 = *((v2f16 *)&B[(K-1)*M+j]);
            Bv1 = *((v2f16 *)&B[(K-1)*M+j+2]);
            temp0 += Av0 * Bv0;
            temp1 += Av0 * Bv1;
            temp2 += Av1 * Bv0;
            temp3 += Av1 * Bv1;
          }
          Cv = (v2f16 *)&C[i*M+j];
          *Cv = temp0;

          Cv = (v2f16 *)&C[i*M+j+2];
          *Cv = temp1;

          Cv = (v2f16 *)&C[(i+1)*M+j];
          *Cv = temp2;

          Cv = (v2f16 *)&C[(i+1)*M+j+2];
          *Cv = temp3;          
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (int ii=i; ii<i+2; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              fp16 left_temp = 0;
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
          fp16 temp_left = 0;
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
    #if NUM_CORES > 1
      const int blockSize = (N_loop+NUM_CORES-1) / NUM_CORES;
      const int start = pi_core_id()*blockSize;
      const int stop = start+blockSize < N_loop? start+blockSize: N_loop;

      for (int i = start; i < stop; i+=2) {

    #else
      const int start = 0;
      const int stop = N_loop;

      for (int i = start; i < stop; i+=2) {
    #endif
        for (int j = 0; j < (M & 0xfffffffc); j+=4) 
        {
          // Global accumulator
          v2f16 temp = (v2f16) {0, 0};
          // Dot product accumulators
          v2f16 tmp0 = (v2f16) {0, 0};
          v2f16 tmp1 = (v2f16) {0, 0};
          v2f16 tmp2 = (v2f16) {0, 0};
          v2f16 tmp3 = (v2f16) {0, 0};
          v2f16 tmp4 = (v2f16) {0, 0};
          v2f16 tmp5 = (v2f16) {0, 0};
          v2f16 tmp6 = (v2f16) {0, 0};
          v2f16 tmp7 = (v2f16) {0, 0};
          // Scalar accumulators
          fp16 a = 0;
          fp16 b = 0;

          for (int k = 0; k < (K & 0xfffffffe); k+=2) 
          {
            // A vectors
            Av0 = *(v2f16 *) &A[i*K+k];
            Av1 = *(v2f16 *) &A[(i+1)*K+k];
            // B vectors (transposed matrix)
            Bv0 = *(v2f16 *) &B[j*K+k];
            Bv1 = *(v2f16 *) &B[(j+1)*K+k];
            Bv2 = *(v2f16 *) &B[(j+2)*K+k];
            Bv3 = *(v2f16 *) &B[(j+3)*K+k];

            // Products in Ci,j and successive with Av0
            tmp0 += Av0 * Bv0;
            tmp1 += Av0 * Bv1;
            tmp2 += Av0 * Bv2;
            tmp3 += Av0 * Bv3;
            // Products in Ci+1,j and successive with Av1
            tmp4 += Av1 * Bv0;
            tmp5 += Av1 * Bv1;
            tmp6 += Av1 * Bv2;
            tmp7 += Av1 * Bv3;
          }
          // Leftover on K
          if (K & 1) {
            // A elements
            fp16 A0 = A[i*K+(K-1)];
            fp16 A1 = A[(i+1)*K+(K-1)];
            // B elements (transposed matrix)
            fp16 B0 = B[j*K+(K-1)];
            fp16 B1 = B[(j+1)*K+(K-1)];
            fp16 B2 = B[(j+2)*K+(K-1)];
            fp16 B3 = B[(j+3)*K+(K-1)];

            // Products in Ci,j and successive with Av0
            tmp0[0] += A0 * B0;
            tmp1[0] += A0 * B1;
            tmp2[0] += A0 * B2;
            tmp3[0] += A0 * B3;
            // Products in Ci+1,j and successive with Av1
            tmp4[0] += A1 * B0;
            tmp5[0] += A1 * B1;
            tmp6[0] += A1 * B2;
            tmp7[0] += A1 * B3;
          }
          // Accumulate to compute dot product and store
          // Row 1
          a = tmp0[0] + tmp0[1];
          b = tmp1[0] + tmp1[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[i*M+j];
          *Cv = temp;

          a = tmp2[0] + tmp2[1];
          b = tmp3[0] + tmp3[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[i*M+j+2];
          *Cv = temp;

          // Row 2
          a = tmp4[0] + tmp4[1];
          b = tmp5[0] + tmp5[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[(i+1)*M+j];
          *Cv = temp;

          a = tmp6[0] + tmp6[1];
          b = tmp7[0] + tmp7[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[(i+1)*M+j+2];
          *Cv = temp;
        }
        // Leftover in M
        if (M & 0x00000003) 
        {
          for (int ii=i; ii<i+2; ii++) 
          {
            for (int j=(M-(M & 0x00000003)); j<M; j++)
            {
              fp16 left_temp = 0;
              for (int k=0; k<K; k++)
              {
                left_temp += A[ii*K+k] * B[j*K+k];
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
        //for (int j=0; j<M; j++)
        {
          fp16 temp_left = 0;
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

  if      (M_par < 2) mm_fp16(args);
  else if (K < 2)     mm_fp16(args);
  else {
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
          // Leftover on K
          if (K & 1) {
            Av  = (v2f16) {A[i*K+(K-1)], A[i*K+(K-1)]};
            Bv0 = *((v2f16 *)&B[(K-1)*M+j]);
            temp += Av * Bv0;
          } 
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;        
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      int i_block = (N+NUM_CORES-1) / NUM_CORES;
      int i_start = core_id*i_block;
      int i_stop = i_start+i_block > N ? N : i_start+i_block;

      for (int i = i_start; i < i_stop; i++) 
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
        // Global accumulator
        v2f16 temp = (v2f16) {0, 0};
        // Dot product accumulators
        v2f16 tmp0 = (v2f16) {0, 0};
        v2f16 tmp1 = (v2f16) {0, 0};
        // K leftover accumulator
        v2f16 vtmp = (v2f16) {0, 0};
        // Scalar accumulators for final result
        fp16 a = 0;
        fp16 b = 0;
        // Indices
        indexA = i*K;
        indexB = j*K; //j*M;

        for (int k = 0; k < (K & 0xfffffffe) ; k+=2) 
        {
          Av  = *((v2f16 *) &A[indexA/*i*K+k*/]);

          Bv0 = *((v2f16 *) &B[indexB]);
          Bv1 = *((v2f16 *) &B[indexB+K]);
          tmp0 += (v2f16)(Av * Bv0);
          tmp1 += (v2f16)(Av * Bv1);

          indexA += 2;
          indexB += 2; 
        }
        // Leftover in K
        if (K & 1) {
          Av  = (v2f16) {A[indexA], A[indexA]};
          Bv0 = (v2f16) {B[indexB], B[indexB+K]};
          #ifdef DEBUG
            printf("------ MM K left (indexA=%d, indexB=%d) => Av = {0x%x, 0x%x}, Bv0 = {0x%x, 0x%x}\n", indexA, indexB, 
              ((unsigned int) Av)&(0xffff0000)>>16, ((unsigned int) Av)&(0x0000ffff), 
              ((unsigned int) Bv0)&(0xffff0000)>>16, ((unsigned int) Bv0)&(0x0000ffff));
          #endif
          vtmp += (v2f16) (Av * Bv0);
          a += vtmp[0];
          b += vtmp[1];
          #ifdef DEBUG
            printf("------ MM K left => temp = {0x%x, 0x%x}\n", ((unsigned int) vtmp[0]), ((unsigned int) vtmp[1]));
          #endif
        } 
        // Complete dot product
        a += tmp0[0] + tmp0[1];
        b += tmp1[0] + tmp1[1];
        temp = (v2f16) {a, b};
        Cv = (v2f16 *)&C[i*M+j];
        *Cv = temp;      
      }
    }
    // Leftover on M
    if(M_left > 0) 
    {
      int i_block = (N+NUM_CORES-1) / NUM_CORES;
      int i_start = core_id*i_block;
      int i_stop = i_start+i_block > N ? N : i_start+i_block;
      
      for (int i = i_start; i < i_stop; i++) 
      {
        fp16 val = 0;
        for (int k = 0; k < K; k++) 
        {
          val += A[i*K+k]*B[(M-1)*K+k];
        }
      C[i*M+(M-1)] = val;
      }
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

  int N_par = N & 0xfffffffe;
  int N_left = N - N_par;

  // Integrity barrier for oversized unrolling
  int N_bound = (N-(N&0x00000001));
  int M_bound = (M_par)/NUM_CORES;
  int K_bound = (K-(K&0x00000001));

  if      (M_bound < 4) mm_M_fp16_SIMD_2x4(args);
  else if (K_bound < 2) mm_M_fp16_SIMD_2x4(args);
  else if (N_bound < 2) mm_M_fp16_SIMD_2x4(args);
  else {
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
        for (int i = 0; i < N_par; i+=2)
        {
          v2f16 temp0 = (v2f16) {0, 0};
          v2f16 temp1 = (v2f16) {0, 0};
          v2f16 temp2 = (v2f16) {0, 0};
          v2f16 temp3 = (v2f16) {0, 0};

          for (int k = 0; k < (K & 0xfffffffe) ; k+=2) 
          {
            // A vectors
            Av0 = *(v2f16 *) &A[i*K+k];
            Av1 = *(v2f16 *) &A[(i+1)*K+k];
            // B vectors
            Bv0 = *(v2f16 *) &B[k*M+j];
            Bv1 = *(v2f16 *) &B[(k+1)*M+j];
            Bv2 = *(v2f16 *) &B[k*M+j+2];
            Bv3 = *(v2f16 *) &B[(k+1)*M+j+2];

            // Ci,j, Ci,j+1
            temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv0;
            temp0 += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv1;
            // Ci,j+2, Ci,j+3
            temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s){0,0})) * Bv2;
            temp1 += (v2f16)(__builtin_shuffle(Av0, (v2s){1,1})) * Bv3;
            // Ci+1,j, Ci+1,j+1
            temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv0;
            temp2 += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv1;
            // Ci+1,j+2, Ci+1,j+3
            temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s){0,0})) * Bv2;
            temp3 += (v2f16)(__builtin_shuffle(Av1, (v2s){1,1})) * Bv3;   
          }
          // Leftover on K
          if (K & 1)
          {
            Av0  = (v2f16) {A[i*K+(K-1)], A[i*K+(K-1)]};
            Av1  = (v2f16) {A[(i+1)*K+(K-1)], A[(i+1)*K+(K-1)]};
            Bv0 = *((v2f16 *)&B[(K-1)*M+j]);
            Bv1 = *((v2f16 *)&B[(K-1)*M+j+2]);
            temp0 += Av0 * Bv0;
            temp1 += Av0 * Bv1;
            temp2 += Av1 * Bv0;
            temp3 += Av1 * Bv1;
          }
          Cv = (v2f16 *)&C[i*M+j];
          *Cv = temp0;

          Cv = (v2f16 *)&C[i*M+j+2];
          *Cv = temp1;

          Cv = (v2f16 *)&C[(i+1)*M+j];
          *Cv = temp2;

          Cv = (v2f16 *)&C[(i+1)*M+j+2];
          *Cv = temp3;    
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
        for (int i = 0; i < (N & 0xfffffffe); i+=2)
        {
          // Global accumulator
          v2f16 temp = (v2f16) {0, 0};
          // Dot product accumulators
          v2f16 tmp0 = (v2f16) {0, 0};
          v2f16 tmp1 = (v2f16) {0, 0};
          v2f16 tmp2 = (v2f16) {0, 0};
          v2f16 tmp3 = (v2f16) {0, 0};
          v2f16 tmp4 = (v2f16) {0, 0};
          v2f16 tmp5 = (v2f16) {0, 0};
          v2f16 tmp6 = (v2f16) {0, 0};
          v2f16 tmp7 = (v2f16) {0, 0};
          // Scalar accumulators
          fp16 a = 0;
          fp16 b = 0;

          for (int k = 0; k < (K & 0xfffffffe) ; k+=2) 
          {
            // A vectors
            Av0 = *(v2f16 *) &A[i*K+k];
            Av1 = *(v2f16 *) &A[(i+1)*K+k];
            // B vectors (transposed matrix)
            Bv0 = *(v2f16 *) &B[j*K+k];
            Bv1 = *(v2f16 *) &B[(j+1)*K+k];
            Bv2 = *(v2f16 *) &B[(j+2)*K+k];
            Bv3 = *(v2f16 *) &B[(j+3)*K+k];

            // Products in Ci,j and successive with Av0
            tmp0 += Av0 * Bv0;
            tmp1 += Av0 * Bv1;
            tmp2 += Av0 * Bv2;
            tmp3 += Av0 * Bv3;
            // Products in Ci+1,j and successive with Av1
            tmp4 += Av1 * Bv0;
            tmp5 += Av1 * Bv1;
            tmp6 += Av1 * Bv2;
            tmp7 += Av1 * Bv3;
          }
          if (K & 1) {
            // A elements
            fp16 A0 = A[i*K+(K-1)];
            fp16 A1 = A[(i+1)*K+(K-1)];
            // B elements (transposed matrix)
            fp16 B0 = B[j*K+(K-1)];
            fp16 B1 = B[(j+1)*K+(K-1)];
            fp16 B2 = B[(j+2)*K+(K-1)];
            fp16 B3 = B[(j+3)*K+(K-1)];

            // Products in Ci,j and successive with Av0
            tmp0[0] += A0 * B0;
            tmp1[0] += A0 * B1;
            tmp2[0] += A0 * B2;
            tmp3[0] += A0 * B3;
            // Products in Ci+1,j and successive with Av1
            tmp4[0] += A1 * B0;
            tmp5[0] += A1 * B1;
            tmp6[0] += A1 * B2;
            tmp7[0] += A1 * B3;
          }
          // Accumulate to compute dot product and store
          // Row 1
          a = tmp0[0] + tmp0[1];
          b = tmp1[0] + tmp1[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[i*M+j];
          *Cv = temp;

          a = tmp2[0] + tmp2[1];
          b = tmp3[0] + tmp3[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[i*M+j+2];
          *Cv = temp;

          // Row 2
          a = tmp4[0] + tmp4[1];
          b = tmp5[0] + tmp5[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[(i+1)*M+j];
          *Cv = temp;

          a = tmp6[0] + tmp6[1];
          b = tmp7[0] + tmp7[1];
          temp = (v2f16) {a, b};
          Cv = (v2f16*)&C[(i+1)*M+j+2];
          *Cv = temp;     
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
