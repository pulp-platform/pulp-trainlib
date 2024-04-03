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
#include "dw-output.h"
#include "dw-grads.h"
#include "pw-output.h"
#include "pw-grads.h"
#include "init-defines.h"

#include "step-check.h"
#include "stats.h"

#include "net.h"

// DATA DEFINITION
PI_L1 fp16 zero_init = 0.0f;
PI_L1 fp16 value_init = 0.1f;

// SEPARABLE CONV
PI_L1 struct DepthWise_Conv_args_fp16 DW_args;
PI_L1 struct blob_fp16 layer1_in, layer1_wgt, layer1_out;

// // POINTWISE CONV
PI_L1 struct PointWise_Conv_args_fp16 PW_args;
PI_L1 struct blob_fp16 layer2_in, layer2_wgt, layer2_out;

// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

#ifdef DW_FORWARD
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1];
PI_L1 fp16 l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef DW_BACKWARD_ERROR
PI_L1 fp16 l1_in_diff[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1];
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef DW_BACKWARD_GRAD
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
PI_L1 fp16 l1_ker_diff[Tker_H_l1*Tker_W_l1*Tin_C_l1];
PI_L1 fp16 l1_out_diff[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#ifdef PW_FORWARD
PI_L1 fp16 l2_in[Tin_H_l2*Tin_W_l2*Tin_C_l2];
PI_L1 fp16 l2_ker[Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2];
PI_L1 fp16 l2_out[Tout_H_l2*Tout_W_l2*Tout_C_l2];
#endif

#ifdef PW_BACKWARD_ERROR
PI_L1 fp16 l2_in_diff[Tin_H_l2*Tin_W_l2*Tin_C_l2];
PI_L1 fp16 l2_ker[Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2];
PI_L1 fp16 l2_out_diff[Tout_H_l2*Tout_W_l2*Tout_C_l2];
PI_L1 fp16 tr_buff[Tin_C_l2*Tout_C_l2];
#endif

#ifdef PW_BACKWARD_GRAD
PI_L1 fp16 l2_in[Tin_H_l2*Tin_W_l2*Tin_C_l2];
PI_L1 fp16 l2_ker_diff[Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2];
PI_L1 fp16 l2_out_diff[Tout_H_l2*Tout_W_l2*Tout_C_l2];
PI_L1 fp16 tr_buff[Tin_C_l2*Tin_H_l2*Tin_W_l2+Tout_C_l2*Tout_H_l2*Tout_W_l2];
#endif



#ifdef DW_FORWARD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in[i] = OUTPUT[i]; //0.4f;
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1; i++)                           l1_ker[i] = DW_WEIGHTS[i]; //weight_init;
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = OUTPUT;
  cpy.to = l1_in;
  cpy.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.data = l1_in; //OUTPUT;
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
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;

  DW_args.input = &layer1_in;
  DW_args.coeff = &layer1_wgt;
  DW_args.output = &layer1_out;
  DW_args.Lpad = LPAD;
  DW_args.Rpad = RPAD;
  DW_args.Upad = UPAD;
  DW_args.Dpad = DPAD;
  DW_args.skip_wg_grad = 0;
  DW_args.skip_in_grad = 0;
  DW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Kernel
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Output
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);

  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
}
#endif


#ifdef DW_BACKWARD_GRAD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in[i] = OUTPUT[i]; //0.4f;
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1; i++)                           l1_ker_diff[i] = zero_init;
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out_diff[i] =  zero_init;
}

static inline void connect_blobs(){

  // // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = OUTPUT;
  cpy.to = l1_in;
  cpy.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  cpy.from = OUTPUT_GRAD;
  cpy.to = l1_out_diff;
  cpy.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.data = l1_in; //OUTPUT;
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
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;

  DW_args.input = &layer1_in;
  DW_args.coeff = &layer1_wgt;
  DW_args.output = &layer1_out;
  DW_args.Lpad = LPAD;
  DW_args.Rpad = RPAD;
  DW_args.Upad = UPAD;
  DW_args.Dpad = DPAD;
  DW_args.skip_wg_grad = 0;
  DW_args.skip_in_grad = 0;
  DW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Kernel grad
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Output grad
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);

  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
}
#endif


#ifdef DW_BACKWARD_ERROR
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             l1_in_diff[i] = zero_init;
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1; i++)                           l1_ker[i] = DW_WEIGHTS[i]; 
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          l1_out_diff[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = OUTPUT_GRAD;
  cpy.to = l1_out_diff;
  cpy.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

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
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;

  DW_args.input = &layer1_in;
  DW_args.coeff = &layer1_wgt;
  DW_args.output = &layer1_out;
  DW_args.Lpad = LPAD;
  DW_args.Rpad = RPAD;
  DW_args.Upad = UPAD;
  DW_args.Dpad = DPAD;
  DW_args.skip_wg_grad = 0;
  DW_args.skip_in_grad = 0;
  DW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input grad
  L1_memocc_bytes += Tin_H_l1*Tin_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Kernel
  L1_memocc_bytes += Tker_H_l1*Tker_W_l1*Tin_C_l1*sizeof(fp16);
  // DW Output grad
  L1_memocc_bytes += Tout_H_l1*Tout_W_l1*Tout_C_l1*sizeof(fp16);

  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
}
#endif

// ................................ POINTWISE

#ifdef PW_FORWARD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l2*Tin_W_l2*Tin_C_l2; i++)                             l2_in[i] = zero_init;
  for (int i=0; i<Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2; i++)                 l2_ker[i] = PW_WEIGHTS[i]; // 0.1f;
  for (int i=0; i<Tout_H_l2*Tout_W_l2*Tout_C_l2; i++)                          l2_out[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = DW_OUTPUT;
  cpy.to = l2_in;
  cpy.size = Tin_H_l2*Tin_W_l2*Tin_C_l2;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);  

  layer2_in.data = l2_in; //DW_OUTPUT;
  layer2_in.dim = Tin_H_l2*Tin_W_l2*Tin_C_l2;
  layer2_in.W = Tin_W_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.C = Tin_C_l2;

  layer2_out.data = l2_out;
  layer2_out.dim = Tout_H_l2*Tout_W_l2*Tout_C_l2;
  layer2_out.W = Tout_W_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.C = Tout_C_l2;

  layer2_wgt.data = l2_ker;
  layer2_wgt.dim = Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.C = Tout_C_l2; //Tin_C_l2;

  PW_args.input = &layer2_in;
  PW_args.coeff = &layer2_wgt;
  PW_args.output = &layer2_out;
  PW_args.skip_wg_grad = 0;
  PW_args.skip_in_grad = 0;
  PW_args.opt_matmul_type_fw = MATMUL_TYPE;
  PW_args.opt_matmul_type_wg = MATMUL_TYPE;
  PW_args.opt_matmul_type_ig = MATMUL_TYPE;
  PW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input
  L1_memocc_bytes += Tin_H_l2*Tin_W_l2*Tin_C_l2*sizeof(fp16);
  // PW Kernel
  L1_memocc_bytes += Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2*sizeof(fp16);
  // PW Output
  L1_memocc_bytes += Tout_H_l2*Tout_W_l2*Tout_C_l2*sizeof(fp16);

  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
}
#endif


#ifdef PW_BACKWARD_GRAD
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l2*Tin_W_l2*Tin_C_l2; i++)                             l2_in[i] = zero_init;
  for (int i=0; i<Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2; i++)                 l2_ker_diff[i] = zero_init;
  for (int i=0; i<Tout_H_l2*Tout_W_l2*Tout_C_l2; i++)                          l2_out_diff[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = DW_OUTPUT;
  cpy.to = l2_in;
  cpy.size = Tin_H_l2*Tin_W_l2*Tin_C_l2;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  cpy.from = PW_OUTPUT_GRAD;
  cpy.to = l2_out_diff;
  cpy.size = Tout_H_l2*Tout_W_l2*Tout_C_l2;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy); 

  layer2_in.data = l2_in; //DW_OUTPUT;
  layer2_in.dim = Tin_H_l2*Tin_W_l2*Tin_C_l2;
  layer2_in.W = Tin_W_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.C = Tin_C_l2;

  layer2_out.diff = l2_out_diff; //PW_OUTPUT_GRAD; 
  layer2_out.dim = Tout_H_l2*Tout_W_l2*Tout_C_l2;
  layer2_out.W = Tout_W_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.C = Tout_C_l2;

  layer2_wgt.diff = l2_ker_diff;
  layer2_wgt.dim = Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.C = Tin_C_l2;

  PW_args.input = &layer2_in;
  PW_args.coeff = &layer2_wgt;
  PW_args.output = &layer2_out;
  PW_args.skip_wg_grad = 0;
  PW_args.skip_in_grad = 0;
  PW_args.opt_matmul_type_fw = MATMUL_TYPE;
  PW_args.opt_matmul_type_wg = MATMUL_TYPE;
  PW_args.opt_matmul_type_ig = MATMUL_TYPE;
  PW_args.transpose_buffer = tr_buff;
  PW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input
  L1_memocc_bytes += Tin_H_l2*Tin_W_l2*Tin_C_l2*sizeof(fp16);
  // PW Kernel grad
  L1_memocc_bytes += Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2*sizeof(fp16);
  // PW Output grad
  L1_memocc_bytes += Tout_H_l2*Tout_W_l2*Tout_C_l2*sizeof(fp16);
  // Transposition buffer
  L1_memocc_bytes += (Tin_C_l2*Tin_H_l2*Tin_W_l2+Tout_C_l2*Tout_H_l2*Tout_W_l2)*sizeof(fp16);

  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
}
#endif


#ifdef PW_BACKWARD_ERROR
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l2*Tin_W_l2*Tin_C_l2; i++)                             l2_in_diff[i] = zero_init;
  for (int i=0; i<Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2; i++)                 l2_ker[i] = PW_WEIGHTS[i]; // 0.1f;
  for (int i=0; i<Tout_H_l2*Tout_W_l2*Tout_C_l2; i++)                          l2_out_diff[i] =  zero_init;
}

static inline void connect_blobs(){

  // Copy golden model's data into the correct tensor
  struct copy_args_fp16 cpy;
  cpy.from = PW_OUTPUT_GRAD;
  cpy.to = l2_out_diff;
  cpy.size = Tout_H_l2*Tout_W_l2*Tout_C_l2;
  pi_cl_team_fork(NUM_CORES, copy_fp16, (void*)&cpy);

  layer2_in.diff = l2_in_diff;
  layer2_in.dim = Tin_H_l2*Tin_W_l2*Tin_C_l2;
  layer2_in.W = Tin_W_l2;
  layer2_in.H = Tin_H_l2;
  layer2_in.C = Tin_C_l2;

  layer2_out.diff = l2_out_diff; //PW_OUTPUT_GRAD; 
  layer2_out.dim = Tout_H_l2*Tout_W_l2*Tout_C_l2;
  layer2_out.W = Tout_W_l2;
  layer2_out.H = Tout_H_l2;
  layer2_out.C = Tout_C_l2;

  layer2_wgt.data = l2_ker;
  layer2_wgt.dim = Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2;
  layer2_wgt.W = Tker_W_l2;
  layer2_wgt.H = Tker_H_l2;
  layer2_wgt.C = Tin_C_l2;

  PW_args.input = &layer2_in;
  PW_args.coeff = &layer2_wgt;
  PW_args.output = &layer2_out;
  PW_args.skip_wg_grad = 0;
  PW_args.skip_in_grad = 0;
  PW_args.opt_matmul_type_fw = MATMUL_TYPE;
  PW_args.opt_matmul_type_wg = MATMUL_TYPE;
  PW_args.opt_matmul_type_ig = MATMUL_TYPE;
  PW_args.transpose_buffer = tr_buff;
  PW_args.HWC = HWC_LAYOUT;
}

static inline void compute_memory_occupation() {
  // Input grad
  L1_memocc_bytes += Tin_H_l2*Tin_W_l2*Tin_C_l2*sizeof(fp16);
  // PW Kernel
  L1_memocc_bytes += Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2*sizeof(fp16);
  // PW Output grad
  L1_memocc_bytes += Tout_H_l2*Tout_W_l2*Tout_C_l2*sizeof(fp16);
  // Transposition buffer
  L1_memocc_bytes += Tin_C_l2*Tout_C_l2*sizeof(fp16);

  // PW Output grad
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // DW Output compare
  L2_memocc_bytes += OUTPUT_SIZE*sizeof(fp16);
  // DW Output grad
  L2_memocc_bytes += G_OUTPUT_SIZE*sizeof(fp16);
  // DW Input grad
  L2_memocc_bytes += IN_SIZE*sizeof(fp16);
  // DW Weight grad
  L2_memocc_bytes += WGT_SIZE*sizeof(fp16);
  // PW Input grad
  L2_memocc_bytes += PW_IN_SIZE*sizeof(fp16);
  // PW Weight grad
  L2_memocc_bytes += PW_WGT_SIZE*sizeof(fp16);
  // PW Output
  L2_memocc_bytes += PW_OUTPUT_SIZE*sizeof(fp16);
  // PW Input (DW Output)
  L2_memocc_bytes += DW_OUTPUT_SIZE*sizeof(fp16);
}
#endif


static inline void forward(){

  /**  FORWARD convPW #1   **/
  #ifdef DW_FORWARD
  pulp_conv_dw_fp16_fw_cl(&DW_args);
  #endif

  #ifdef PW_FORWARD
  pulp_conv_pw_fp16_fw_cl(&PW_args);
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
                tensor_ref[i], *(unsigned short int*) &tensor_ref[i], tensor_out[i], *(unsigned short int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



static inline void train(){

  #ifdef PROF_DW_FWD
  printf("\nForward stats\n");
  START_STATS();
  #endif

  #ifdef DW_FORWARD
  pulp_conv_dw_fp16_fw_cl(&DW_args);
  #endif

  #ifdef PROF_DW_FWD
  STOP_STATS();
  #endif

  #ifdef PROF_PW_FWD
  printf("\nForward stats\n");
  START_STATS();
  #endif

  #ifdef PW_FORWARD
  pulp_conv_pw_fp16_fw_cl(&PW_args);
  #endif

  #ifdef PROF_PW_FWD
  STOP_STATS();
  #endif

  #ifdef PROF_PW_BKWD
  printf("\nBackward PW stats\n");
  START_STATS();
  #endif

  #ifdef PW_BACKWARD_GRAD
  pulp_conv_pw_fp16_bw_param_grads_cl(&PW_args);
  #endif

  #ifdef PW_BACKWARD_ERROR
  pulp_conv_pw_fp16_bw_input_grads_cl(&PW_args);
  #endif

  #ifdef PROF_PW_BKWD
  STOP_STATS();
  #endif

  #ifdef PROF_DW_BKWD
  printf("\nBackward DW stats\n");
  START_STATS();
  #endif

  #ifdef DW_BACKWARD_GRAD
  pulp_conv_dw_fp16_bw_param_grads_cl(&DW_args);
  #endif

  #ifdef DW_BACKWARD_ERROR
  pulp_conv_dw_fp16_bw_input_grads_cl(&DW_args);
  #endif


  #ifdef PROF_DW_BKWD
  STOP_STATS();
  #endif


  #ifdef CHECK_PRINT

  #ifdef DW_FORWARD
  printf("DW FORWARD CHECK: \n");
  compare_tensors(l1_out, DW_OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  check_tensor(l1_out, DW_OUTPUT, Tout_H_l1*Tout_W_l1*Tout_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer1_in, &layer1_wgt, &layer1_out);
  for (int index=0; index<Tout_H_l1*Tout_W_l1*Tout_C_l1; index++) {
    if (!(index%Tout_H_l1)) printf("\n");
    printf("%f ", l1_out[index]);
  }
  printf("\n");
  #endif

  #ifdef DW_BACKWARD_GRAD
  printf("DW WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1);
  check_tensor(l1_ker_diff, WEIGHT_GRAD, Tker_H_l1*Tker_W_l1*Tin_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer1_in, &layer1_wgt, &layer1_out);
  printf("\n\nWEIGHT GRAD:");
  for (int index=0; index<WGT_SIZE; index++) {
   if (!(index%Tker_H_l1)) printf("\n");
   printf("%f ", l1_ker_diff[index]);
  }
  printf("\n");
  #endif

  #ifdef DW_BACKWARD_ERROR
  printf("DW INPUTS GRADIENT CHECK: \n");
  compare_tensors(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  check_tensor(l1_in_diff, INPUT_GRAD, Tin_H_l1*Tin_W_l1*Tin_C_l1);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer1_in, &layer1_wgt, &layer1_out);
  for (int index=0; index<Tin_H_l1*Tin_W_l1*Tin_C_l1; index++) {
    if (!(index%Tin_H_l1)) printf("\n");
    printf("%f ", l1_in_diff[index]);
  }
  printf("\n");
  #endif

  #ifdef PW_FORWARD
  printf("PW FORWARD CHECK: \n");
  compare_tensors(l2_out, PW_OUTPUT, Tout_H_l2*Tout_W_l2*Tout_C_l2);
  check_tensor(l2_out, PW_OUTPUT, Tout_H_l2*Tout_W_l2*Tout_C_l2);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer2_in, &layer2_wgt, &layer2_out);
  for (int index=0; index<Tout_H_l2*Tout_W_l2*Tout_C_l2; index++) {
    #if HWC_LAYOUT == 0
    if (!(index%Tout_W_l2)) printf("\n");
    #elif HWC_LAYOUT == 1
    if (!(index%Tout_C_l2)) printf("\n");
    #endif
    printf("%f ", l2_out[index]);
  }
  printf("\n");
  #endif

  #ifdef PW_BACKWARD_GRAD
  printf("PW WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l2_ker_diff, PW_WEIGHT_GRAD, Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2);
  check_tensor(l2_ker_diff, PW_WEIGHT_GRAD, Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer2_in, &layer2_wgt, &layer2_out);
  for (int index=0; index<Tker_H_l2*Tker_W_l2*Tin_C_l2*Tout_C_l2; index++) {
    #if HWC_LAYOUT == 0
    if (!(index%Tin_C_l2)) printf("\n");
    #elif HWC_LAYOUT == 1
    if (!(index%Tout_C_l2)) printf("\n");
    #endif
    printf("%f ", l2_ker_diff[index]);
  }
  printf("\n");
  #endif

  #ifdef PW_BACKWARD_ERROR
  printf("PW INPUTS GRADIENT CHECK: \n");
  compare_tensors(l2_in_diff, PW_INPUT_GRAD, Tin_H_l2*Tin_W_l2*Tin_C_l2);
  check_tensor(l2_in_diff, PW_INPUT_GRAD, Tin_H_l2*Tin_W_l2*Tin_C_l2);
  // TEST
  printf("\nADDR\nIN: %x, WGT: %x, OUT: %x\n", &layer2_in, &layer2_wgt, &layer2_out);
  for (int index=0; index<Tin_H_l2*Tin_W_l2*Tin_C_l2; index++) {
    #if HWC_LAYOUT == 0
    if (!(index%Tin_W_l2)) printf("\n");
    #elif HWC_LAYOUT == 1
    if (!(index%Tin_C_l2)) printf("\n");
    #endif
    printf("%f ", l2_in_diff[index]);
  }
  printf("\n");
  #endif

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
