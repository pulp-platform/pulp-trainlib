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

#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"


int verify_tensor(float * tensor_out, float * tensor_ref, int size, float tolerance){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > tolerance ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



void transpose(void * void_args) 
{
    struct transp_args args = *((struct transp_args *)void_args);
    float * matrix = args.matrix;
    float * transp_matrix = args.transp_matrix;
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



void copy (void * void_args)
{
  struct copy_args args = *((struct copy_args *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for(int i=start; i<stop; i++)
    args.to[i] = args.from[i];
}



void set_to_value (void * void_args)
{
  struct set_to_value_args args = *((struct set_to_value_args *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for(int i=start; i<stop; i++)
    args.to[i] = args.value;  
}




void cast_fp16_tensor_to_fp32 (void * cast_16t32_args) 
{
  struct cast_16t32_args args = *((struct cast_16t32_args *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for (int=start; i<stop; i++) {
    args.destination[i] = (float) args.source[i];
  }
}





/**
 * Choose the correct matmul for the chosen layer.
 */
void mm_manager (void * void_args)
{
    struct mm_manager_args* args = (struct mm_manager_args *) void_args;
    
    struct matMul_args *matMul_args = args->mm_args;    
    struct matMul_DW_args *matMul_DW_args = args->mm_dw_args;
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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

// =====> DEPTHWISE CONVOLUTION
    else if (layer_type == LAYER_DW_CONV) 
    {

        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_u2((void *) matMul_DW_args);}
            else if (matmul_type == 2)      { mm_dw_u3((void *) matMul_DW_args);}
            else if (matmul_type == 3)      { mm_dw_unroll_1x2((void *) matMul_DW_args);}
            else if (matmul_type == 4)      { mm_dw_unroll_1x4((void *) matMul_DW_args);}
            else if (matmul_type == 5)      { mm_dw_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_unroll_1x4_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_u2((void *) matMul_DW_args);}
            else if (matmul_type == 2)      { mm_dw_u3((void *) matMul_DW_args);}
            else if (matmul_type == 3)      { mm_dw_unroll_1x2((void *) matMul_DW_args);}
            else if (matmul_type == 4)      { mm_dw_unroll_1x4((void *) matMul_DW_args);}
            else if (matmul_type == 5)      { mm_dw_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_unroll_1x4_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw_in_grad((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_in_grad_u2((void *) matMul_DW_args); }
            else if (matmul_type == 2)      { mm_dw_in_grad_u3((void *) matMul_DW_args); }
            else if (matmul_type == 3)      { mm_dw_in_grad_unroll_1x2((void *) matMul_DW_args); }
            else if (matmul_type == 4)      { mm_dw_in_grad_unroll_1x4((void *) matMul_DW_args); }
            else if (matmul_type == 5)      { mm_dw_in_grad_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_in_grad_unroll_1x4_u2((void *) matMul_DW_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
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
