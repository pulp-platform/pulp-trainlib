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


void cast_fp32_tensor_to_fp16 (void * cast_32t16_args) 
{
  struct cast_16t32_args args = *((struct cast_16t32_args *)cast_32t16_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for (int i=start; i<stop; i++) {
    args.destination[i] = (fp16) args.source[i];
  }
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