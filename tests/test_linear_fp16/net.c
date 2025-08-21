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

#include "linear-data.h"
#include "stats.h"

#include "net.h"

// DATA DEFINITION

// LINEAR
PI_L1 struct Linear_args_fp16 FC_args;
PI_L1 struct blob_fp16 layer0_in, layer0_wgt, layer0_out, layer0_bias;
// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

PI_L1 fp16 zero_init = 0.0f;

#ifdef FORWARD
PI_L1 fp16 l0_in[Tin_l0];
PI_L1 fp16 l0_ker[Tker_l0];
PI_L1 fp16 l0_out[Tout_l0]; 
PI_L1 fp16 l0_bias[Tout_l0]; 
#endif

#ifdef BACKWARD_ERROR
PI_L1 fp16 l0_in_diff [Tin_l0];
PI_L1 fp16 l0_ker[Tker_l0];
PI_L1 fp16 l0_out_diff [Tout_l0];
#endif

#ifdef BACKWARD_GRAD
PI_L1 fp16 l0_in[Tin_l0];
PI_L1 fp16 l0_ker_diff[Tker_l0];
PI_L1 fp16 l0_out_diff [Tout_l0];
PI_L1 fp16 l0_bias_diff[Tout_l0]; 
#endif



#ifdef FORWARD
static inline void tensor_init() 
{
  for (int i=0; i<Tin_l0; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tker_l0; i++)       l0_ker[i] = L0_WEIGHTS_params[i];
  for (int i=0; i<Tout_l0; i++)       l0_out[i] = zero_init; 
  for (int i=0; i<Tout_l0; i++)       l0_bias[i] = L0_BIAS_params[i]; 
}

static inline void connect_blobs() 
{
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_l0;

  layer0_wgt.data = l0_ker;
  layer0_wgt.dim = Tker_l0;

  layer0_out.data = l0_out;
  layer0_out.dim = Tout_l0;

  layer0_bias.data = l0_bias;
  layer0_bias.dim = Tout_l0;

  FC_args.input = &layer0_in;
  FC_args.coeff = &layer0_wgt;
  FC_args.output = &layer0_out;
  FC_args.bias = &layer0_bias;
  FC_args.skip_wg_grad = 0;
  FC_args.skip_in_grad = 0;
  FC_args.opt_matmul_type_fw = MATMUL_TYPE;
  FC_args.opt_matmul_type_wg = MATMUL_TYPE;
  FC_args.opt_matmul_type_ig = MATMUL_TYPE;
  FC_args.use_biases = USE_BIASES_LINEAR;
}

static inline void compute_memory_occupation(){
  // Input
  L1_memocc_bytes += Tin_l0*sizeof(fp16);
  // Kernel
  L1_memocc_bytes += Tker_l0*sizeof(fp16); 
  // Output
  L1_memocc_bytes += Tout_l0*sizeof(fp16);
  // Bias
  L1_memocc_bytes += Tout_l0*sizeof(float);

  // Input data
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
  // Weights
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Output gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Weight gradient
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Input gradient
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
  // Bias gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
}
#endif


#ifdef BACKWARD_ERROR
static inline void tensor_init() 
{
  for (int i=0; i<Tin_l0; i++)        l0_in_diff[i] = zero_init;
  for (int i=0; i<Tker_l0; i++)       l0_ker[i] = L0_WEIGHTS_params[i];
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = L0_OUT_GRAD[i]; 
}

static inline void connect_blobs() 
{
  layer0_in.diff = l0_in_diff;
  layer0_in.dim = Tin_l0;

  layer0_wgt.data = l0_ker;
  layer0_wgt.dim = Tker_l0;

  layer0_out.diff = l0_out_diff;
  layer0_out.dim = Tout_l0;  

  FC_args.input = &layer0_in;
  FC_args.coeff = &layer0_wgt;
  FC_args.output = &layer0_out;
  FC_args.skip_wg_grad = 0;
  FC_args.skip_in_grad = 0;
  FC_args.opt_matmul_type_fw = MATMUL_TYPE;
  FC_args.opt_matmul_type_wg = MATMUL_TYPE;
  FC_args.opt_matmul_type_ig = MATMUL_TYPE;
  FC_args.use_biases = USE_BIASES_LINEAR;
}

static inline void compute_memory_occupation(){
  // Input grad
  L1_memocc_bytes += Tin_l0*sizeof(fp16);
  // Kernel
  L1_memocc_bytes += Tker_l0*sizeof(fp16); 
  // Output grad
  L1_memocc_bytes += Tout_l0*sizeof(fp16);

  // Input data
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
  // Weights
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Output gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Weight gradient
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Input gradient
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
}
#endif


#ifdef BACKWARD_GRAD
static inline void tensor_init() 
{
  for (int i=0; i<Tin_l0; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tker_l0; i++)       l0_ker_diff[i] = zero_init;
  for (int i=0; i<Tout_l0; i++)       l0_bias_diff[i] = zero_init; 
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = L0_OUT_GRAD[i];   
}

static inline void connect_blobs() 
{
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_l0;

  layer0_wgt.diff = l0_ker_diff;
  layer0_wgt.dim = Tker_l0;

  layer0_out.diff = l0_out_diff;
  layer0_out.dim = Tout_l0;  

  layer0_bias.diff = l0_bias_diff;
  layer0_bias.dim = Tout_l0;

  FC_args.input = &layer0_in;
  FC_args.coeff = &layer0_wgt;
  FC_args.output = &layer0_out;
  FC_args.bias = &layer0_bias;
  FC_args.skip_wg_grad = 0;
  FC_args.skip_in_grad = 0;
  FC_args.opt_matmul_type_fw = MATMUL_TYPE;
  FC_args.opt_matmul_type_wg = MATMUL_TYPE;
  FC_args.opt_matmul_type_ig = MATMUL_TYPE;
  FC_args.use_biases = USE_BIASES_LINEAR;
}

static inline void compute_memory_occupation(){
  // Input
  L1_memocc_bytes += Tin_l0*sizeof(fp16);
  // Kernel grad
  L1_memocc_bytes += Tker_l0*sizeof(fp16); 
  // Output grad
  L1_memocc_bytes += Tout_l0*sizeof(fp16);
  // Bias grad
  L1_memocc_bytes += Tout_l0*sizeof(float);

  // Input data
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
  // Weights
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Output gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(fp16);
  // Weight gradient
  L2_memocc_bytes += L0_WEIGHTS*sizeof(fp16);
  // Input gradient
  L2_memocc_bytes += L0_IN_CH*sizeof(fp16);
  // Bias gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
}
#endif



static inline void net_forward(){
  /**  FORWARD FC #1   **/
  #ifdef FORWARD
  pulp_linear_fp16_fw_cl(&FC_args);
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
  pulp_linear_fp16_fw_cl(&FC_args);
  #endif

  #ifdef PROF_FWD
  STOP_STATS();
  #endif

  #ifdef PROF_BCKWD
  printf("\nBackward stats\n");
  START_STATS();
  #endif

  #ifdef BACKWARD_ERROR
  pulp_linear_fp16_bw_input_grads_cl(&FC_args);
  #endif

  #ifdef BACKWARD_GRAD
  pulp_linear_fp16_bw_param_grads_cl(&FC_args);
  #endif

  #ifdef PROF_BCKWD
  STOP_STATS();
  #endif



  #ifdef FORWARD
  printf("FORWARD CHECK: \n");
  //printf("\nLinear forward in: \n");
  //for (int i=0; i<Tin_l0; i++) {
  //  printf("%f ", l0_in[i]);
  //} printf("\n");
  //printf("FC weights: \n");
  //for (int i=0; i<Tin_l0*Tout_l0; i++) {
  //  if ((i%Tin_l0==0) && (i!=0)) printf("\n");
  //  printf("%f ", l0_ker[i]);
  //} printf("\n");
  //printf("FC biases: \n");
  //for (int i=0; i<Tout_l0; i++) {
  //  printf("%f ", l0_bias[i]);
  //} printf("\n");
  //printf("Linear forward out: \n");
  //for (int i=0; i<Tout_l0; i++) {
  //  printf("%f ", l0_out[i]);
  //} printf("\n\n");
  compare_tensors(l0_out, L0_OUT_FW, Tout_l0);
  check_tensor(l0_out, L0_OUT_FW, Tout_l0);
  #endif

  #ifdef BACKWARD_ERROR
  printf("INPUTS GRADIENT CHECK: \n");
  compare_tensors(l0_in_diff, L0_IN_GRAD, Tin_l0);
  check_tensor(l0_in_diff, L0_IN_GRAD, Tin_l0);
  #endif

  #ifdef BACKWARD_GRAD
  printf("WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l0_ker_diff, L0_WEIGHT_GRAD, Tker_l0);
  check_tensor(l0_ker_diff, L0_WEIGHT_GRAD, Tker_l0);

  printf("BIASES GRADIENT CHECK: \n");
  compare_tensors(l0_bias_diff, L0_BIAS_GRAD, Tout_l0);
  check_tensor(l0_bias_diff, L0_BIAS_GRAD, Tout_l0);
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

  return;
}
