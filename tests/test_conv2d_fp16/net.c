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
PI_L1 fp16 zero_init = 0.0f;
PI_L1 struct Conv2D_args_fp16 C2D_args;
PI_L1 struct blob_fp16 layer1_in, layer1_wgt, layer1_bias, layer1_out;
// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

#ifdef FORWARD
#if (IM2COL == 1)
#define IM2COL_SIZE (Tker_H_l1*Tker_W_l1*Tin_C_l1*((Tin_H_l1-Tker_H_l1+PAD_U+PAD_D+STRIDE_H)/STRIDE_H)*((Tin_W_l1-Tker_W_l1+PAD_L+PAD_R+STRIDE_W)/STRIDE_W))
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
#else 
#define IM2COL_SIZE 1
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
#endif
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
#if (USE_BIAS == 1)
PI_L1 fp16 l1_bias[Tout_C_l1];
#endif
PI_L1 fp16 l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
PI_L1 fp16 bt_buffer[1];
#endif

#ifdef BACKWARD_ERROR   
//#define IM2COL_SIZE (Tker_H_l1*Tker_W_l1*Tout_C_l1*Tin_H_l1*Tin_W_l1)
#define IM2COL_SIZE (Tin_H_l1*Tin_W_l1*Tin_C_l1*Tout_C_l1*Tker_W_l1*Tker_H_l1)
PI_L1 fp16 l1_in_diff[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
PI_L1 fp16 bt_buffer[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef BACKWARD_GRAD
//#define IM2COL_SIZE (Tker_W_l1*Tker_H_l1*Tout_W_l1*Tout_H_l1*Tout_C_l1)
#define IM2COL_SIZE (Tker_W_l1*Tker_H_l1*Tout_W_l1*Tout_H_l1*Tin_C_l1)
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 im2col_buffer[IM2COL_SIZE];
PI_L1 fp16 l1_ker_diff[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
#if (USE_BIAS == 1)
PI_L1 fp16 l1_bias_diff[Tout_C_l1];
#endif
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
PI_L1 fp16 bt_buffer[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif



#ifdef FORWARD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in[i] = INPUT[i];
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker[i] = WEIGHTS[i]; //weight_init;
  #if (USE_BIAS == 1)
  for (int i=0; i<Tout_C_l1; i++)                                              l1_bias[i] = BIASES[i]; //bias_init;
  #endif
  for (int i=0; i<IM2COL_SIZE; i++)                                            im2col_buffer[i] = zero_init;
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into L1 tensor
  //struct copy_args cpy;
  //cpy.from = INPUT;
  //cpy.to = l1_in;
  //cpy.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  //pi_cl_team_fork(NUM_CORES, copy, (void*)&cpy);

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

  #if (USE_BIAS == 1)
  layer1_bias.data = l1_bias;
  layer1_bias.dim = Tout_C_l1;
  #endif

  C2D_args.input = &layer1_in;
  C2D_args.coeff = &layer1_wgt;
  C2D_args.bias = &layer1_bias;
  C2D_args.output = &layer1_out;
  C2D_args.Lpad = PAD_L;
  C2D_args.Rpad = PAD_R;
  C2D_args.Upad = PAD_U;
  C2D_args.Dpad = PAD_D;
  C2D_args.stride_h = STRIDE_H;
  C2D_args.stride_w = STRIDE_W;
  C2D_args.i2c_buffer = im2col_buffer;
  C2D_args.bt_buffer = bt_buffer;
  C2D_args.skip_wg_grad = 0;
  C2D_args.skip_in_grad = 0;
  C2D_args.HWC = HWC_LAYOUT;
  C2D_args.opt_matmul_type_fw = MATMUL_TYPE;
  C2D_args.opt_matmul_type_wg = MATMUL_TYPE;
  C2D_args.opt_matmul_type_ig = MATMUL_TYPE;
  C2D_args.USE_IM2COL = IM2COL;
  C2D_args.USE_DMA_IM2COL = DMA;
  C2D_args.USE_BIASES = USE_BIAS;
}

static inline void compute_memory_occupation(){
  //printf("\n----------L1----------\n");
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  //printf("Input_tensor: %d bytes\n", Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16));
  L1_memocc_bytes += IM2COL_SIZE*sizeof(fp16);
  //printf("Im2Col: %d bytes\n", IM2COL_SIZE*sizeof(fp16));
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16);
  //printf("Weights: %d bytes\n", Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16));
  #if (USE_BIAS == 1)
  L1_memocc_bytes += Tout_C_l1 * sizeof(fp16);
  //printf("Biases: %d bytes\n", Tout_C_l1*sizeof(fp16));
  #endif
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);
  //printf("Output: %d bytes\n", Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += INPUT_SIZE*sizeof(fp16);
  //printf("Input_image: %d bytes\n", INPUT_SIZE*sizeof(fp16));
  //printf("----------------------");

  L2_memocc_bytes += G_IN_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_WGT_SIZE*sizeof(fp16);
  #if (USE_BIAS == 1)
  L2_memocc_bytes += G_BIAS_SIZE * sizeof(fp16);
  #endif
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
  #if (USE_BIAS == 1)
  printf("\n\nl1_bias (size: %d):\n", Tout_C_l1);
  for(int index=0; index<Tout_C_l1; index++) {
    printf("%f ", l1_bias[index]);
  }
  #endif
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
  for (int i=0; i<Tout_C_l1*Tker_H_l1*Tker_W_l1*Tin_C_l1; i++)                 l1_ker_diff[i] = zero_init;
  #if (USE_BIAS == 1)
  for (int i=0; i<Tout_C_l1; i++)                                              l1_bias_diff[i] = zero_init;
  #endif
  for (int i=0; i<IM2COL_SIZE; i++)                                            im2col_buffer[i] = zero_init; 
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

  #if (USE_BIAS == 1)
  layer1_bias.diff = l1_bias_diff;
  layer1_bias.dim = Tout_C_l1;
  C2D_args.bias = &layer1_bias;
  #endif

  C2D_args.input = &layer1_in;
  C2D_args.coeff = &layer1_wgt;
  C2D_args.output = &layer1_out;
  C2D_args.Lpad = PAD_L;
  C2D_args.Rpad = PAD_R;
  C2D_args.Upad = PAD_U;
  C2D_args.Dpad = PAD_D;
  C2D_args.stride_h = STRIDE_H;
  C2D_args.stride_w = STRIDE_W;
  C2D_args.i2c_buffer = im2col_buffer;
  C2D_args.bt_buffer = bt_buffer;
  C2D_args.skip_wg_grad = 0;
  C2D_args.skip_in_grad = 0;
  C2D_args.HWC = HWC_LAYOUT;
  C2D_args.opt_matmul_type_fw = MATMUL_TYPE;
  C2D_args.opt_matmul_type_wg = MATMUL_TYPE;
  C2D_args.opt_matmul_type_ig = MATMUL_TYPE;
  C2D_args.USE_IM2COL = IM2COL;
  C2D_args.USE_DMA_IM2COL = DMA;
  C2D_args.USE_BIASES = USE_BIAS;
}

static inline void compute_memory_occupation(){
  //printf("\n----------L1----------\n");
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  //printf("Input: %d bytes\n", Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16));
  L1_memocc_bytes += IM2COL_SIZE*sizeof(fp16);
  //printf("Im2Col: %d bytes\n", IM2COL_SIZE*sizeof(fp16));
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16);
  //printf("Weights: %d bytes\n", Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1*sizeof(fp16));
  #if (USE_BIAS == 1)
  L1_memocc_bytes += Tout_C_l1 * sizeof(fp16);
  //printf("Biases: %d bytes\n", Tout_C_l1 * sizeof(fp16));
  #endif
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);
  //printf("Output: %d bytes\n", Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16));
  L1_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  //printf("Out_gradient: %d bytes\n", G_OUTPUT_SIZE*sizeof(fp16));
  //printf("----------------------\n");

  L2_memocc_bytes += G_IN_SIZE*sizeof(fp16);
  L2_memocc_bytes += G_WGT_SIZE*sizeof(fp16);
  #if (USE_BIAS == 1)
  L2_memocc_bytes += G_BIAS_SIZE * sizeof(fp16);
  #endif
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  L2_memocc_bytes += INPUT_SIZE*sizeof(fp16);
}

#endif


#ifdef BACKWARD_ERROR
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in_diff[i] = zero_init;
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker[i] = WEIGHTS[i]; //weight_init;
  for (int i=0; i<IM2COL_SIZE; i++)                                            im2col_buffer[i] = zero_init; 
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out_diff[i] = OUTPUT_GRAD[i]; //0.0f;
}

static inline void connect_blobs(){

  // // Copy golden model's data into L1 tensor
  // struct copy_args cpy;
  // cpy.from = OUTPUT_GRAD;
  // cpy.to = l1_out_diff;
  // cpy.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  // pi_cl_team_fork(NUM_CORES, copy, (void*)&cpy);

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

  C2D_args.input = &layer1_in;
  C2D_args.coeff = &layer1_wgt;
  C2D_args.output = &layer1_out;
  C2D_args.Lpad = PAD_L;
  C2D_args.Rpad = PAD_R;
  C2D_args.Upad = PAD_U;
  C2D_args.Dpad = PAD_D;
  C2D_args.stride_h = STRIDE_H;
  C2D_args.stride_w = STRIDE_W;
  C2D_args.i2c_buffer = im2col_buffer;
  C2D_args.bt_buffer = bt_buffer;
  C2D_args.skip_wg_grad = 0;
  C2D_args.skip_in_grad = 0;
  C2D_args.HWC = HWC_LAYOUT;
  C2D_args.opt_matmul_type_fw = MATMUL_TYPE;
  C2D_args.opt_matmul_type_wg = MATMUL_TYPE;
  C2D_args.opt_matmul_type_ig = MATMUL_TYPE;
  C2D_args.USE_IM2COL = IM2COL;
  C2D_args.USE_DMA_IM2COL = DMA;
  C2D_args.USE_BIASES = USE_BIAS;
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
  pulp_conv2d_fp16_fw_cl(&C2D_args);
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
                tensor_ref[i], *(uint16_t*) &tensor_ref[i], tensor_out[i], *(uint16_t*) &tensor_out[i]);
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
  pulp_conv2d_fp16_fw_cl(&C2D_args);
  #endif

  #ifdef PROF_FWD
  STOP_STATS();
  #endif


  #ifdef PROF_BKWD
  printf("\nBackward stats\n");
  START_STATS();
  #endif

  #ifdef BACKWARD_GRAD
  pulp_conv2d_fp16_bw_param_grads_cl(&C2D_args);
  #endif

  #ifdef BACKWARD_ERROR
  pulp_conv2d_fp16_bw_input_grads_cl(&C2D_args);
  #endif

  #ifdef PROF_BKWD
  STOP_STATS();
  #endif


  #ifdef FORWARD
  printf("FORWARD CHECK: \n");
  compare_tensors(l1_out, OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  check_tensor(l1_out, OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  // TEST
  printf("\nOUT SIZES: [%d, %d, %d]\n", Tout_C_l1, Tout_H_l1, Tout_W_l1);
  //printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer1_in, &layer1_wgt, &layer1_out);
  printf("\nOUT_ELEMENTS: %d\n", Tout_H_l1*Tout_W_l1*Tout_C_l1);
  for (int index=0; index<Tout_H_l1*Tout_W_l1*Tout_C_l1; index++) {
    #if HWC_LAYOUT == 0
    if (!(index%Tout_W_l1)) printf("\n");
    #else
    if (!(index%Tout_C_l1)) printf("\n");
    #endif
    printf("%f ", l1_out[index]);
  }
  printf("\n");
  #endif

  #ifdef BACKWARD_GRAD
  printf("WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1);
  check_tensor(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1);
  // TEST
  printf("\nOUT SIZES: [%d, %d, %d]\n", Tout_C_l1, Tout_H_l1, Tout_W_l1);

  #if (USE_BIAS ==1)
  printf("BIASES GRADIENT CHECK: \n");
  compare_tensors(l1_bias_diff, BIAS_GRAD, Tout_C_l1);
  check_tensor(l1_bias_diff, BIAS_GRAD, Tout_C_l1);
  printf("\nBIAS SIZES: [%d]\n", Tout_C_l1);
  #endif

  //printf("\nADDR\nIN: %x, WGT: %x, OUT: %x, BUFF:%x\n", &layer1_in, &layer1_wgt, &layer1_out, im2col_buffer);
  for (int index=0; index<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; index++) {
    #if HWC_LAYOUT == 0 
    if (!(index%Tker_W_l1)) printf("\n");
    #else
    if (!(index%Tin_C_l1*Tker_H_l1*Tker_W_l1)) printf("\n");
    #endif
    printf("%f ", l1_ker_diff[index]);
  }
  printf("\n");
  #endif

  #ifdef BACKWARD_ERROR
  printf("INPUTS GRADIENT CHECK: \n");
  compare_tensors(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  check_tensor(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x, BUFF:%x\n", &layer1_in, &layer1_wgt, &layer1_out, im2col_buffer);
  for (int index=0; index<Tin_H_l1*Tin_W_l1*Tin_C_l1; index++) {
    #if HWC_LAYOUT == 0
    if (!(index%Tin_W_l1)) printf("\n");
    #else
    if (!(index%Tin_C_l1)) printf("\n");
    #endif
    printf("%f ", l1_in_diff[index]);
  }
  printf("\n");
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
