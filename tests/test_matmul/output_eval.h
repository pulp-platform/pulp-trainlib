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

#include "pmsis.h"
#include "net.h"

/*
 *  OUTPUT EVALUATION LIBRARY
*/

// Additive quantity to avoid diff error when tensors are perfectly equal
#define DIFF_OFFS 1e-10

// Elementwise checker
#ifdef FLOAT32
int check_tensor(float * tensor_out, float * tensor_ref, int size){
#endif

#ifdef FLOAT16
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size){
#endif

#ifdef BFLOAT16
int check_tensor(bf16 * tensor_out, bf16 * tensor_ref, int size){
#endif

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");

            #ifdef FLOAT32
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            #endif

            #ifdef FLOAT16
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned short int*) &tensor_ref[i], tensor_out[i], *(unsigned short int*) &tensor_out[i]);
            #endif            
            error_flag = 1;
        }
    }
    return error_flag;
}


// Checksum generator
#ifdef FLOAT32
void compare_tensors(float *A, float *B, int length){

  float mean_err_rel = 0.0f;
  float diff = 0.0f;
  float den = 0.000001f;
#endif

#ifdef FLOAT16
void compare_tensors(fp16 *A, fp16 *B, int length){

  fp16 mean_err_rel = 0.0f;
  fp16 diff = 0.0f;
  fp16 den = 0.000001f;
#endif

#ifdef BFLOAT16
void compare_tensors(bf16 *A, bf16 *B, int length){

  bf16 mean_err_rel = 0.0f;
  bf16 diff = 0.0f;
  bf16 den = 0.000001f;
#endif

  for(int i=0; i<length; i++){
     //printf("A %f B %f \n", A[i], B[i]);
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-(B[i]+DIFF_OFFS);
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        else den = -A[i]; // missing A = 0
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-(B[i]+DIFF_OFFS);
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       else den = -A[i];
       mean_err_rel = mean_err_rel + (diff / den)/length;
     }
  }
  // printf("ERR REL AVERAGE: %f\n", mean_err_rel);
  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n");
  else printf(">>>TENSOR NOT MATCHING!\n");

}





