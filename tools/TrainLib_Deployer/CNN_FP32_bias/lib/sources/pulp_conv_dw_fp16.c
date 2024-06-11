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
#include "pulp_im2col_fp16.h"
#include "pulp_conv_dw_fp16.h"
#include "pulp_conv_naive_fp16.h"
#include "pulp_train_defines.h"

void pulp_conv_dw_fp16_fw_cl ( void * DepthWise_Conv_args_fp16 )
{
  struct DepthWise_Conv_args_fp16 * DW_args = (struct DepthWise_Conv_args_fp16 *) DepthWise_Conv_args_fp16;

  struct kernel_DW_args_fp16 ker_args;
  ker_args.input = DW_args->input;
  ker_args.weights = DW_args->coeff;
  ker_args.output = DW_args->output;

  pi_cl_team_fork(NUM_CORES, dw_kernel_forward_fp16, &ker_args);

  return;
}



void pulp_conv_dw_fp16_bw_cl( void * DepthWise_Conv_args_fp16 )
{
  struct DepthWise_Conv_args_fp16 * DW_args = (struct DepthWise_Conv_args_fp16 *) DepthWise_Conv_args_fp16;
  int skip_wg_grad = DW_args->skip_wg_grad;
  int skip_in_grad = DW_args->skip_in_grad;

  if (skip_wg_grad == 0)
  {
    pulp_conv_dw_fp16_bw_param_grads_cl(DepthWise_Conv_args_fp16); 
  }
  if (skip_in_grad == 0)
  {
    pulp_conv_dw_fp16_bw_input_grads_cl(DepthWise_Conv_args_fp16); 
  }
}



void pulp_conv_dw_fp16_bw_param_grads_cl( void * DepthWise_Conv_args_fp16 )
{
  struct DepthWise_Conv_args_fp16 * DW_args = (struct DepthWise_Conv_args_fp16 *) DepthWise_Conv_args_fp16;

  struct kernel_DW_args_fp16 ker_args;
  ker_args.input = DW_args->input;
  ker_args.weights = DW_args->coeff;
  ker_args.output = DW_args->output;

  pi_cl_team_fork(NUM_CORES, dw_kernel_weight_grad_fp16, &ker_args);

}



void pulp_conv_dw_fp16_bw_input_grads_cl( void * DepthWise_Conv_args_fp16 )
{
  struct DepthWise_Conv_args_fp16 * DW_args = (struct DepthWise_Conv_args_fp16 *) DepthWise_Conv_args_fp16;

  struct kernel_DW_args_fp16 ker_args;
  ker_args.input = DW_args->input;
  ker_args.weights = DW_args->coeff;
  ker_args.output = DW_args->output;

  pi_cl_team_fork(NUM_CORES, dw_kernel_input_grad_fp16, &ker_args);

}
