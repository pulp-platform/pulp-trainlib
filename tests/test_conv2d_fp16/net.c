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

#include "pulp_train.h"

#include "input-image.h"
#include "conv2d-output.h"
#include "conv2d-grads.h"
#include "init-defines.h"

#include "step-check.h"
#include "stats.h"

#include "net.h"

// DATA DEFINITION

// CONV2D
PI_L1 struct blob_fp16 layer1_in, layer1_wgt, layer1_out;
// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

#ifdef FORWARD
#define IM2COL_SIZE (Tker_H_l1*Tker_W_l1*Tin_C_l1*(Tin_H_l1-Tker_H_l1+1)*(Tin_W_l1-Tker_W_l1+1))
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
PI_L1 fp16 l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef BACKWARD_ERROR   // PASS TO BE FIXED
//#define IM2COL_SIZE (Tout_H_l1*Tout_W_l1*Tout_C_l1*Tin_H_l1*Tin_W_l1*Tin_C_l1)
#define IM2COL_SIZE (Tin_H_l1*Tin_W_l1*Tin_C_l1*Tout_C_l1*Tker_W_l1*Tker_H_l1)
PI_L1 fp16 l1_in_diff[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef BACKWARD_GRAD
//#define IM2COL_SIZE (Tker_W_l1*Tker_H_l1*Tout_W_l1*Tout_H_l1*Tout_C_l1)
#define IM2COL_SIZE (Tker_W_l1*Tker_H_l1*Tout_W_l1*Tout_H_l1*Tin_C_l1)
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
PI_L1 fp16 l1_ker_diff[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif



#ifdef FORWARD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in[i] = INPUT[i];
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker[i] = WEIGHTS[i]; //weight_init;
  for (int i=0; i<IM2COL_SIZE; i++)     im2col_buffer[i] = 0.0f;
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out[i] =  0.0f;
}

static inline void connect_blobs(){

  // Copy golden model's data into L1 tensor
  struct copy_args_fp16 cpy;
  cpy.from = INPUT;
  cpy.to = l1_in;
  cpy.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.data = l1_in; //INPUT;
  layer1_in.dim = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  layer1_in.W = Tin_W_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.C = Tin_C_l1;

  layer1_out.data = l1_out; 
  layer1_out.dim = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  layer1_out.W = Tout_W_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.C = Tout_C_l1;

  layer1_wgt.data = l1_ker;
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;
}

static inline void compute_memory_occupation(){
  //printf("\n----------L1----------\n");
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  //printf("Input_tensor: %d bytes\n", Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16));
  L1_memocc_bytes += IM2COL_SIZE*sizeof(fp16);
  //printf("Im2Col: %d bytes\n", IM2COL_SIZE*sizeof(fp16));
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16);
  //printf("Weights: %d bytes\n", Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);
  //printf("Output: %d bytes\n", Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += INPUT_SIZE*sizeof(fp16);
  //printf("Input_image: %d bytes\n", INPUT_SIZE*sizeof(fp16));
  //printf("----------------------");

  L2_memocc_bytes += G_IN_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_WGT_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
}

#ifdef DEBUG
static inline void print_data() {
  printf("\n>>> DEBUG FORWARD DATA <<<\n");
  printf("l1_in data (size: %d):\n", Tin_H_l1*Tin_W_l1*Tin_C_l1);
  for(int index=0; index<Tin_H_l1*Tin_W_l1*Tin_C_l1; index++) {
    if(!(index%Tin_H_l1)) printf("\n");
    printf("%f ", l1_in[index]);
  }
  printf("\n\nl1_ker (size: %d):\n", Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1);
  for(int index=0; index<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; index++) {
    if(!(index%Tker_H_l1)) printf("\n");
    printf("%f ", l1_ker[index]);   
  }
  printf("\n\nl1_out (size: %d):\n", Tout_H_l1*Tout_W_l1*Tout_C_l1);
  for(int index=0; index<Tout_H_l1*Tout_W_l1*Tout_C_l1; index++) {
    if(!(index%Tout_H_l1)) printf("\n");
    printf("%f ", l1_out[index]);
  }
  printf("\n\n");
}
#endif
#endif


#ifdef BACKWARD_GRAD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in[i] = INPUT[i]; 
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker_diff[i] = 0.0f;
  for (int i=0; i<IM2COL_SIZE; i++)                                            im2col_buffer[i] = 0.0f; 
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out_diff[i] = OUTPUT_GRAD[i]; 
}

static inline void connect_blobs(){

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.data = l1_in; //INPUT;
  layer1_in.dim = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  layer1_in.W = Tin_W_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.C = Tin_C_l1;

  layer1_out.diff = l1_out_diff; //OUTPUT_GRAD; 
  layer1_out.dim = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  layer1_out.W = Tout_W_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.C = Tout_C_l1;

  layer1_wgt.diff = l1_ker_diff;
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;

}

static inline void compute_memory_occupation(){
  //printf("\n----------L1----------\n");
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  //printf("Input: %d bytes\n", Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16));
  L1_memocc_bytes += IM2COL_SIZE*sizeof(fp16);
  //printf("Im2Col: %d bytes\n", IM2COL_SIZE*sizeof(fp16));
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16);
  //printf("Weights: %d bytes\n", Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);
  //printf("Output: %d bytes\n", Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  //printf("Out_gradient: %d bytes\n", G_OUTPUT_SIZE*sizeof(fp16));
  //printf("----------------------\n");

  L2_memocc_bytes += G_IN_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_WGT_SIZE*sizeof(fp16);
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  L2_memocc_bytes += INPUT_SIZE*sizeof(fp16);
}

#endif


#ifdef BACKWARD_ERROR
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in_diff[i] = 0.0f;
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker[i] = WEIGHTS[i]; //weight_init;
  for (int i=0; i<IM2COL_SIZE; i++)                                            im2col_buffer[i] = 0.0f; 
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out_diff[i] = OUTPUT_GRAD[i]; //0.0f;
}

static inline void connect_blobs(){

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.diff = l1_in_diff;
  layer1_in.dim = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  layer1_in.W = Tin_W_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.C = Tin_C_l1;

  layer1_out.diff = l1_out_diff; //OUTPUT_GRAD; 
  layer1_out.dim = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  layer1_out.W = Tout_W_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.C = Tout_C_l1;

  layer1_wgt.data = l1_ker;
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;

}

static inline void compute_memory_occupation(){
  L1_memocc_bytes += IM2COL_SIZE*sizeof(fp16);
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16);
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);
  L1_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);

  L2_memocc_bytes += G_IN_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_WGT_SIZE*sizeof(fp16);
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  L2_memocc_bytes += INPUT_SIZE*sizeof(fp16);
}
#endif


static inline void forward(){

  /**  FORWARD convPW #1   **/
  #ifdef FORWARD
  pulp_conv2d_fp16_fw_cl(&layer1_in, &layer1_wgt, &layer1_out, Tpad_l1, im2col_buffer);
  #endif
}

static inline void compare_tensors(fp16 *A, fp16 *B, int length){

  fp16 mean_err_rel = 0.0f;
  fp16 diff = 0.0f;
  fp16 den = 0.000001f;

  for(int i=0; i<length; i++){
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-B[i];
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        else den = -A[i]; // missing A = 0
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-B[i];
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       else den = -A[i];
       mean_err_rel = mean_err_rel + (diff / den)/length;
     }
  }
  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n");
  else printf(">>>TENSOR NOT MATCHING!\n");

}


// Elementwise checker
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}


static inline void train(){

  #ifdef PROF_FWD
  printf("\nForward stats\n");
  START_STATS();
  #endif

  #ifdef FORWARD
  pulp_conv2d_fp16_fw_cl(&layer1_in, &layer1_wgt, &layer1_out, Tpad_l1, im2col_buffer);
  #endif

  #ifdef PROF_FWD
  STOP_STATS();
  #endif


  #ifdef PROF_BKWD
  printf("\nBackward stats\n");
  START_STATS();
  #endif

  #ifdef BACKWARD_GRAD
  pulp_conv2d_fp16_bw_param_grads_cl(&layer1_in, &layer1_wgt, &layer1_out, Tpad_l1, im2col_buffer);
  #endif

  #ifdef BACKWARD_ERROR
  pulp_conv2d_fp16_bw_input_grads_cl(&layer1_in, &layer1_wgt, &layer1_out, Tpad_l1, im2col_buffer);
  #endif

  #ifdef PROF_BKWD
  STOP_STATS();
  #endif


  #ifdef FORWARD
  printf("FORWARD CHECK: \n");
  compare_tensors(l1_out, OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  check_tensor(l1_out, OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer1_in, &layer1_wgt, &layer1_out);
  for (int index=0; index<Tout_H_l1*Tout_W_l1*Tout_C_l1; index++) {
    if (!(index%Tout_H_l1)) printf("\n");
    printf("%f ", l1_out[index]);
  }
  #endif

  #ifdef BACKWARD_GRAD
  printf("WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1);
  check_tensor(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x, BUFF:%x\n", &layer1_in, &layer1_wgt, &layer1_out, im2col_buffer);
  for (int index=0; index<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; index++) {
   if (!(index%Tker_H_l1)) printf("\n");
   printf("%f ", l1_ker_diff[index]);
  }
  #endif

  #ifdef BACKWARD_ERROR
  printf("INPUTS GRADIENT CHECK: \n");
  compare_tensors(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  check_tensor(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x, BUFF:%x\n", &layer1_in, &layer1_wgt, &layer1_out, im2col_buffer);
  for (int index=0; index<Tin_H_l1*Tin_W_l1*Tin_C_l1; index++) {
    if (!(index%Tin_H_l1)) printf("\n");
    printf("%f ", l1_in_diff[index]);
  }
  #endif
}



// Most important function: it connects each passage to step the net and perform training
void net_step()
{
  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  #endif

  #ifdef MEMOCC_COMP
  compute_memory_occupation();
  printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
  printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
  #endif

  tensor_init();

  connect_blobs();

  train();

  #if defined(DEBUG) && defined(FORWARD) 
  print_data();
  #endif

  return;
}
