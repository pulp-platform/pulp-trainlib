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
#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"


void copy_fp16 (void * void_args)
{
  struct copy_args_fp16 args = *((struct copy_args_fp16 *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

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
            if      (matmul_type == 0)      { mm_fp16((void *) matMul_args); }
            //else if (matmul_type == 1)      { }
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
            if      (matmul_type == 0)      { mm_dw_fp16((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_fp16_SIMD_1x2_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw_fp16((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_fp16_SIMD_1x2_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw_in_grad_fp16((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_in_grad_fp16_SIMD_1x2_u2((void *) matMul_DW_args); }
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