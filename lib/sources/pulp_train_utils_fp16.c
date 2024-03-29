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

#include "pmsis.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"
#include <math.h>


int verify_tensor_fp16(fp16 * tensor_out, fp16 * tensor_ref, int size, fp16 tolerance){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > tolerance ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned short int*) &tensor_ref[i], tensor_out[i], *(unsigned short int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



void transpose_fp16(void * void_args) 
{
    struct transp_args_fp16 args = *((struct transp_args_fp16 *)void_args);
    fp16 * matrix = args.matrix;
    fp16 * transp_matrix = args.transp_matrix;
    int N = args.N;
    int M = args.M;

    // Parallelize on N or M depending on the wides available dimension
    if (N > M) 
    {
        int blockSize = (N+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > N ? N : start+blockSize;

        for (int i=start; i<stop; i++)
        {
            for (int j=0; j<M; j++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }
    else 
    {
        int blockSize = (M+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > M ? M : start+blockSize;

        for (int j=start; j<stop; j++)
        {
            for (int i=0; i<N; i++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }
}



void copy_fp16 (void * void_args)
{
  struct copy_args_fp16 args = *((struct copy_args_fp16 *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size  : start+blockSize;
 
  for(int i=start; i<stop; i++)
    args.to[i] = args.from[i];

}



void set_to_value_fp16 (void * void_args)
{
  struct set_to_value_args_fp16 args = *((struct set_to_value_args_fp16 *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for(int i=start; i<stop; i++)
    args.to[i] = args.value;
}


//#define DEBUG
void vect_sum_fp16 (void * vect_sum_args)
{
  struct vect_sum_args_fp16 * args = (struct vect_sum_args_fp16*) vect_sum_args;
  fp16 * op_1 = args->op_1;
  fp16 * op_2 = args->op_2;
  fp16 * dest = args->dest;
  int size = args->size;
  int size_left = size & 0x00000001;

// SIMD implementation
  v2f16 TEMP;
  v2f16 OP1;
  v2f16 OP2;
  v2f16 * DEST = (v2f16 *) dest;

  int blockSize = (size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > size  ? size-1 : start+blockSize;

  if (start & 0x0001) start++;
  if (0x1 != (stop & 0x1)) stop--;
  for (int i=start; i<stop; i+=2) 
  {
    OP1 = *(v2f16 *) &op_1[i];
    OP2 = *(v2f16 *) &op_2[i];
    TEMP = OP1 + OP2;
    DEST = (v2f16 *)&dest[i];
    *DEST = TEMP;
  }
  if (size_left)
    if(pi_core_id()== NUM_CORES-1)
    {
     int idx = size-1;
     dest[idx] = op_1[idx] + op_2[idx];
    }
 
}




void cast_fp32_tensor_to_fp16 (void * cast_32t16_args) 
{
  struct cast_32t16_args args = *((struct cast_32t16_args *)cast_32t16_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for (int i=start; i<stop; i++) {
    args.destination[i] = (fp16) args.source[i];
  }
}




void HWC_to_CHW_fp16 (void * layout_args) 
{
    struct layout_args_fp16 * args = (struct layout_args_fp16 *) layout_args;
    fp16 * data = args->tensor->data;
    fp16 * grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    fp16 * buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args_fp16 tr_args;
    struct copy_args_fp16 cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        tr_args.matrix = data;
        tr_args.transp_matrix = buff;
        tr_args.N = H*W;
        tr_args.M = C;
        pi_cl_team_fork(NUM_CORES, transpose_fp16, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);
    }

    if (transpose_grad == 1) {
        // Transpose grad
        tr_args.matrix = grad;
        tr_args.transp_matrix = buff;
        tr_args.N = H*W;
        tr_args.M = C;
        pi_cl_team_fork(NUM_CORES, transpose_fp16, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);    
    }
}




void CHW_to_HWC_fp16 (void * layout_args) 
{
    struct layout_args_fp16 * args = (struct layout_args_fp16 *) layout_args;
    fp16 * data = args->tensor->data;
    fp16 * grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    fp16 * buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args_fp16 tr_args;
    struct copy_args_fp16 cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        tr_args.matrix = data;
        tr_args.transp_matrix = buff;
        tr_args.N = C;
        tr_args.M = H*W;
        pi_cl_team_fork(NUM_CORES, transpose_fp16, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);
    }

    if (transpose_grad == 1) {
        // Transpose grad
        tr_args.matrix = grad;
        tr_args.transp_matrix = buff;
        tr_args.N = C;
        tr_args.M = H*W;
        pi_cl_team_fork(NUM_CORES, transpose_fp16, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy_fp16, &cpy_args);    
    }
}


void pad_tensor_fp16 (void * pad_args_fp16) 
{
    struct pad_args_fp16 * args = (struct pad_args_fp16*) pad_args_fp16;
    fp16 * source = args->source;
    fp16 * dest = args->dest;
    int C = args->C;
    int H = args->H;
    int W = args->W;
    int L_PAD = args->T_LPAD;
    int R_PAD = args->T_RPAD;
    int U_PAD = args->T_UPAD;
    int D_PAD = args->T_DPAD;
    int HWC = args->HWC_lay;
    
    int H_out = H + U_PAD + D_PAD;
    int W_out = W + L_PAD + R_PAD;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    if (HWC == 0) 
    {
        for (int ch=0; ch<C; ch++) 
        {
            for (int ht=0; ht<H_out; ht++) 
            {
                for (int wt=0; wt<W_out; wt++) 
                {
                    // Compute matrix idx
                    int in_t_idx = (wt-L_PAD) + (ht-U_PAD)*W + ch*H*W;
                    int out_t_idx = wt + ht*W_out + ch*H_out*W_out;
                    // Padding conditions
                    int zero_cond = (wt < L_PAD || wt > W) || (ht < U_PAD || ht > H);
                    if (zero_cond == 1) { dest[out_t_idx] = 0; }
                    else 
                    {
                        dest[out_t_idx] = source[in_t_idx];
                    }
                }
            }
        }
    }
    else if (HWC == 1)
    {
        for (int ht=0; ht<H_out; ht++) 
        {
            for (int wt=0; wt<W_out; wt++)
            {
                for (int ch=0; ch<C; ch++) 
                {
                    // Compute matrix idx
                    int in_t_idx = ch + (wt-L_PAD)*C + (ht-U_PAD)*C*W;
                    int out_t_idx = ch + wt*C + ht*C*W_out;
                    // Padding conditions
                    int zero_cond = (wt < L_PAD || wt > W) || (ht < U_PAD || ht > H);
                    if (zero_cond == 1) { dest[out_t_idx] = 0; }
                    else 
                    {
                        dest[out_t_idx] = source[in_t_idx];
                    }                    
                }
            }
        }
    }
    else 
    {
        printf("[pad_tensor_fp16] HWC layout not implemented!!");
    }
}


void pulp_max_fp16_cl(void * void_args){
    struct max_args_fp16* args = (struct max_args_fp16 *) void_args;

    fp16* input = args->input;
    fp16 max = args->maxes[pi_core_id()];
    int dim = args->dim;

    const int blockSize=(args->dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    for(int i=start; i<stop; i++)
        if(max < input[i])
            max = input[i];

    args->maxes[pi_core_id()] = max;
}

void pulp_row_max_fp16_cl(void * void_args){
    struct max_args_fp16* args = (struct max_args_fp16 *) void_args;

    fp16* input = args->input;
    int dim = args->dim; // L
    fp16* max = args->maxes;
    
    const int blockSize=(dim + NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    input = input + start * dim;

    for(int i=start; i<stop; i++){
        max[i] = *input;
        input++;
        for(int j=1; j<dim; j++){
            if(max[i] < *input)
                max[i] = *input;
            input++;    
        }    
    }
}


float fastexp_gist_fp16(float x) {
    x = GIST_A * x + GIST_B;

    if (x < GIST_C || x > GIST_D)
        x = (x < GIST_C) ? 0.0f : GIST_D;

    uint32_t n = (uint32_t) (x);
    return *(float*) &n;
}

float q_rsqrt_fp16(float number)
{
  long i;
  float x2, y;
  const float threehalfs = 1.5f;

  x2 = number * 0.5f;
  y  = number;
  i  = * ( long * ) &y;                       // evil floating point bit level hacking
  i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
  y  = * ( float * ) &i;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration

  return y;
}

void pulp_exp_sum_fp16_cl(void* void_args){
    struct exp_sum_args_fp16* args = (struct exp_sum_args_fp16 *) void_args;

    fp16* input = args->input;
    fp16* output = args->output;
    fp16* sums = args->sums;
    int dim = args->dim;
    fp16* maxes = args->maxes;
    

    #ifdef DEBUG
    if(pi_core_id()==0){
        int L = dim;
        printf("\nCurrent input - max in softmax: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%((int)L))) printf("\n");
            printf("%.8f ", (input[j] - maxes[j]));
        }
    }
    printf("\n");
    #endif


    const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    input += start * dim;
    output += start * dim;

    for(int i=start; i<stop; i++){
        sums[i] = 0;
        for(int j=0; j<dim; j++){
            fp16 o = (fp16)(fastexp_gist_fp16((float)(*input - maxes[i])));
            //fp16 o = (fp16)expf((float)(*input - maxes[i]));
            *output = o;
            sums[i] += o;
            input++;
            output++;    
        }   
    }
}

void pulp_row_div_fp16_cl(void* void_args){
    struct row_div_args_fp16* args = (struct row_div_args_fp16 *) void_args;

    fp16* input = args->input;
    fp16* sums = args->sums;
    int dim = args->dim;

    const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    int row = 0;

    for(int i=start; i<stop; i++){
        row = i * dim;
        for(int j=0; j<dim; j++){
            input[row + j] = input[row + j]/sums[i];    
        }   
    }
}

void pulp_div_fp16_cl(void* void_args){
    struct div_args_fp16* args = (struct div_args_fp16 *) void_args;

    fp16* input = args->input;
    fp16 n = args->n;
    int dim = args->dim;

    const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    for(int i=start; i<stop; i++){
        input[i] = input[i]/n;
    }
}

void pulp_scalar_mul_fp16_cl(void* void_args){
    struct scalar_mul_args_fp16* args = (struct scalar_mul_args_fp16 *) void_args;

    fp16* input = args->input;
    fp16 scalar = args->scalar;
    int dim = args->dim;

    const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    for(int i=start; i<stop; i++){
        input[i] = (fp16) (input[i]*scalar);
    }
}



/**
 * Choose the user-selected matmul for the chosen layer.
 */
void mm_manager_fp16 (void * void_args) 
{
    struct mm_manager_args_fp16* args = (struct mm_manager_args_fp16 *) void_args;
    
    struct matMul_args_fp16 *matMul_args = args->mm_args;    
    struct matMul_DW_args_fp16 *matMul_DW_args = args->mm_dw_args;
    int layer_type = args->layer_type;
    int step_type = args->step_type;
    int matmul_type = args->matmul_type;

    #ifdef DEBUG
    printf("Running layer %d, step %d, matmul %d\n", layer_type, step_type, matmul_type);
    #endif

// =====> CONV2D
    if (layer_type == LAYER_CONV2D) 
    {
        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection

    }

// =====> POINTWISE CONVOLUTION
    else if (layer_type == LAYER_PW_CONV)
    {

        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
        
    }

// =====> LINEAR LAYER
    else if (layer_type == LAYER_LINEAR)
    {
        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M_fp16((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_fp16_SIMD_4x8((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 4)     { mm_M_fp16_SIMD_2x4((void *) matMul_args);}
            else if (matmul_type == 5)     { mm_M_fp16_SIMD_4x8((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
        
    }

// =====> WRONG LAYER SELECTION
    else
    {
        printf("\nWrong layer_type selection!!\n");
    }

}




// FP16 dot product in floating point
inline fp16 vfdotp(v2f16 a, v2f16 b) {
  fp16 result;
  //asm ("vfdotp.ah %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return 0; //result;
}

// Packs 2 FP16 into v2f16
inline v2f16 vfpack(fp16 a, fp16 b) {
  v2f16 result = (v2f16) {0,0};
  //asm ("pv.pack.ah %0, %1, %2" : "=f" (result): "f" (a), "f" (b) : );
  return result;
}


void pulp_mean_std_fp16_cl(void * mean_std_args)
{
    struct mean_std_args_fp16 * args = (struct mean_std_args_fp16 *) mean_std_args;

    fp16 * data = args->input;
    int D = args->dim;
    fp16 D_inverse = (1/(fp16)D);
    fp16 * mean = args->mean;
    fp16 * std = args->std;
    fp16 * var = args->var;
    fp16 epsilon = args->epsilon;

    fp16 m=0;
    fp16 v=0;
    fp16 s=0;

    int var_was_infinite = 0;

     for(int d=0; d<D; d++)
        {
            fp16 t = data[d];
            m += t;
            v += t*t;
        }

        m = m*D_inverse;
        v = v*D_inverse;

        // Test for infinite variance   
        if (*(__int16_t *)&v == 0x7c00)
        {
            var_was_infinite=1;
            v = 0;
            for(int d=0; d<D; d++)
            {
                fp16 t = data[d];
                fp16 temp = t - m;
                v += temp*temp*D_inverse;
            }
        }

        if(!var_was_infinite)   v -= m*m;
        v = v + epsilon;
        if ((v)<0) v=epsilon;
        *mean = m;
        *var = v;
        *std = (fp16)sqrtf(v);
}