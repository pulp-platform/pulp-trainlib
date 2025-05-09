/*
 * Copyright (C) 2021-2025 ETH Zurich and University of Bologna
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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
*/

#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_im2col_fp16.h"
#include "pulp_conv2d_fp16.h"
#include "pulp_conv_naive_fp16.h"

void pulp_conv2d_fp16_fw_cl(void *Conv2D_args_fp16) {
    struct Conv2D_args_fp16 *C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *biasData = C2D_args->bias->data;
    fp16 *outData = C2D_args->output->data;
    fp16 *inData = C2D_args->input->data;

    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_in = C2D_args->input->C;
    int C_out = C2D_args->output->C;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int Lpad = C2D_args->Lpad;
    int Rpad = C2D_args->Rpad;
    int Upad = C2D_args->Upad;
    int Dpad = C2D_args->Dpad;

    fp16 *i2c_buffer = C2D_args->i2c_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int USE_DMA = C2D_args->USE_DMA_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_fw;

    /**
     * USE OPTIMIZED ALGORITHM
     */
    if (USE_IM2COL == 1) {

        /**
         * USE CHW LAYOUT
         */
        if (HWC_layout == 0) {
            // im2col on the input data
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = Lpad;
            im2col_args.Rpad = Rpad;
            im2col_args.Upad = Upad;
            im2col_args.Dpad = Dpad;
            im2col_args.mod = 0;
            im2col_args.stride_w = stride_w;
            im2col_args.stride_h = stride_h;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);

            // Perform matmul
            matMul_args.A = coeffData;
            matMul_args.B = i2c_buffer;
            matMul_args.C = outData;
            matMul_args.N = C_out;
            matMul_args.K = pW * pH * C_in;
            matMul_args.M =
                    (W_in - pW + stride_w + Lpad + Rpad) / stride_w * (H_in - pH + stride_h + Upad + Dpad) / stride_h;
            matMul_args.trans_B = 1;
            matMul_args.HWC = HWC_layout;
            matMul_args.bias = biasData;
            matMul_args.USE_BIASES = USE_BIASES;

            matMul_args.H = H_in;
            matMul_args.W = W_in;
            matMul_args.pCin = C_in;
            matMul_args.pCout = C_out;
            matMul_args.pH = H_out;
            matMul_args.pW = W_out;

            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_FW;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;

            pi_cl_team_fork(NUM_CORES, im2col_conv2d_fw_kernel_fp16, &man_args);
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            // im2col on the input data
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = Lpad;
            im2col_args.Rpad = Rpad;
            im2col_args.Upad = Upad;
            im2col_args.Dpad = Dpad;
            im2col_args.mod = 0;
            im2col_args.stride_w = stride_w;
            im2col_args.stride_h = stride_h;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);

            matMul_args.A = i2c_buffer;
            matMul_args.B = coeffData;
            matMul_args.C = outData;
            matMul_args.N =
                    (W_in - pW + stride_w + Lpad + Rpad) / stride_w * (H_in - pH + stride_h + Upad + Dpad) / stride_h;
            matMul_args.K = pW * pH * C_in;
            matMul_args.M = C_out;
            matMul_args.trans_B = 1;
            matMul_args.HWC = HWC_layout;
            matMul_args.bias = biasData;
            matMul_args.USE_BIASES = USE_BIASES;

            matMul_args.H = H_in;
            matMul_args.W = W_in;
            matMul_args.pCin = C_in;
            matMul_args.pCout = C_out;
            matMul_args.pH = H_out;
            matMul_args.pW = W_out;

            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_FW;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;

            pi_cl_team_fork(NUM_CORES, im2col_conv2d_fw_kernel_fp16, &man_args);
        } else {
            printf("[pulp_conv2d_fp16_fw_cl:] Invalid data layout format (HWC or CHW)!\n");
        }
    }

        /**
         * USE NAIVE KERNEL
         */
    else if (USE_IM2COL == 0) {

        /**
         * USE CHW DATA LAYOUT
         */
        if (HWC_layout == 0) {
            matMul_args.A = inData;
            matMul_args.B = coeffData;
            matMul_args.C = outData;
            matMul_args.bias = biasData;
            matMul_args.USE_BIASES = USE_BIASES;
            matMul_args.H = H_in;
            matMul_args.W = W_in;
            matMul_args.pCin = C_in;
            matMul_args.pCout = C_out;
            matMul_args.pH = pH;
            matMul_args.pW = pW;
            // Stride and padding operators
            matMul_args.stride_h = stride_h;
            matMul_args.stride_w = stride_w;
            matMul_args.Lpad = Lpad;
            matMul_args.Rpad = Rpad;
            matMul_args.Upad = Upad;
            matMul_args.Dpad = Dpad;

            #ifdef OPTIMIZE
            int padding = Lpad + Rpad + Upad + Dpad;
            int stride = stride_h + stride_w;
            if (pH == 3 && pW == 3 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_fw_kernel_CHW_k3x3_s2_p1_fp16, &matMul_args);
            else if (pH == 5 && pW == 5 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_fw_kernel_CHW_k5x5_s2_p1_fp16, &matMul_args);
            else
            #endif
            pi_cl_team_fork(NUM_CORES, naive_conv2d_fw_kernel_CHW_fp16, &matMul_args);
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            printf("[pulp_conv2d_fp16_fw_cl:] Naive kernel for HWC FW Conv2D not implemented!\n");
        } else {
            printf("[pulp_conv2d_fp16_fw_cl:] Invalid data layout format (HWC or CHW)!\n");
        }
    }

        // ERROR IN SELECTING IM2COL
    else {
        printf("[pulp_conv2d_fp16_fw_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
    }
}


void pulp_conv2d_fp16_bw_cl(void *Conv2D_args_fp16) {
    struct Conv2D_args_fp16 *C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    int skip_wg_grad = C2D_args->skip_wg_grad;
    int skip_in_grad = C2D_args->skip_in_grad;

    if (skip_wg_grad == 0) {
        pulp_conv2d_fp16_bw_param_grads_cl(Conv2D_args_fp16);
    }

    if (skip_in_grad == 0) {
        pulp_conv2d_fp16_bw_input_grads_cl(Conv2D_args_fp16);
    }
}


void pulp_conv2d_fp16_bw_param_grads_cl(void *Conv2D_args_fp16) {
    struct Conv2D_args_fp16 *C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    //input dimensions
    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int C_in = C2D_args->input->C;
    //kernel dimensions
    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    //bias dimensions
    int bias_dim = C2D_args->output->C;
    //output dimensions
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_out = C2D_args->output->C;

    fp16 *inData = C2D_args->input->data;
    fp16 *inDiff = C2D_args->input->diff;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *coeffDiff = C2D_args->coeff->diff;
    fp16 *biasData = C2D_args->bias->data;
    fp16 *biasDiff = C2D_args->bias->diff;
    fp16 *outDiff = C2D_args->output->diff;
    fp16 *outData = C2D_args->output->data;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int Lpad = C2D_args->Lpad;
    int Rpad = C2D_args->Rpad;
    int Upad = C2D_args->Upad;
    int Dpad = C2D_args->Dpad;

    fp16 *i2c_buffer = C2D_args->i2c_buffer;
    // Transposition buffer for HWC Conv2D
    fp16 *tr_buffer = C2D_args->bt_buffer;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int USE_DMA = C2D_args->USE_DMA_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_wg;

    /**
     * USE OPTIMIZED ALGORITHM
     */
    if (USE_IM2COL == 1) {

        /**
         * USE CHW LAYOUT
         */
        if (HWC_layout == 0) {
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = 0; //Lpad;
            im2col_args.Rpad = 0; //Rpad;
            im2col_args.Upad = 0; //Upad;
            im2col_args.Dpad = 0; //Dpad;
            im2col_args.mod = 0;
            im2col_args.stride_w = stride_w;
            im2col_args.stride_h = stride_h;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);

            matMul_args.A = outDiff;
            matMul_args.B = i2c_buffer;
            matMul_args.C = coeffDiff;
            matMul_args.N = C_out;
            matMul_args.K = H_out * W_out;
            matMul_args.M = pW * pH * C_in;
            matMul_args.trans_B = 0;

            matMul_args.HWC = HWC_layout;
            matMul_args.bias = biasDiff;
            matMul_args.USE_BIASES = USE_BIASES;

            matMul_args.pH = H_out;
            matMul_args.pW = W_out;

            matMul_args.bias_dim = bias_dim;

            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_WGT_GRAD;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;

            pi_cl_team_fork(NUM_CORES, im2col_conv2d_param_grad_kernel_fp16, &man_args);
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = Lpad;
            im2col_args.Rpad = Rpad;
            im2col_args.Upad = Upad;
            im2col_args.Dpad = Dpad;
            im2col_args.mod = 0;
            im2col_args.stride_w = stride_w;
            im2col_args.stride_h = stride_h;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);

            // int dim[] = {C_out, H_out * W_out};
            // int tr_axes[] = {1, 0};

            struct transp_args_fp16 tr_args;

            tr_args.in_matrix = outDiff;
            tr_args.out_matrix = tr_buffer;
            tr_args.N = H_out * W_out;
            tr_args.M = C_out;
            // tr_args.dim = dim;
            // tr_args.transposed_axes = tr_axes;
            // tr_args.n_dim = 2;

            pi_cl_team_fork(NUM_CORES, transpose_matrix_fp16, &tr_args);

            matMul_args.A = tr_buffer; // outDiff;
            matMul_args.B = i2c_buffer;
            matMul_args.C = coeffDiff;
            matMul_args.N = C_out;
            matMul_args.K = H_out * W_out;
            matMul_args.M = pW * pH * C_in;
            matMul_args.trans_B = 1;

            matMul_args.HWC = HWC_layout;
            matMul_args.bias = biasDiff;
            matMul_args.USE_BIASES = USE_BIASES;

            matMul_args.pH = H_out;
            matMul_args.pW = W_out;

            matMul_args.bias_dim = bias_dim;

            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_WGT_GRAD;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;

            pi_cl_team_fork(NUM_CORES, im2col_conv2d_param_grad_kernel_fp16, &man_args);
        } else {
            printf("[pulp_conv2d_fp16_bw_param_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
        }

    }

        /**
         * USE NAIVE KERNEL
         */
    else if (USE_IM2COL == 0) {

        /**
         * USE CHW DATA LAYOUT
         */
        if (HWC_layout == 0) {
            matMul_args.A = inData;
            matMul_args.B = coeffDiff;
            matMul_args.C = outDiff;
            matMul_args.H = H_in;
            matMul_args.W = W_in;
            matMul_args.pCin = C_in;
            matMul_args.pCout = C_out;
            matMul_args.pH = pH;
            matMul_args.pW = pW;
            // Stride and padding operators
            matMul_args.stride_h = stride_h;
            matMul_args.stride_w = stride_w;
            matMul_args.Lpad = Lpad;
            matMul_args.Rpad = Rpad;
            matMul_args.Upad = Upad;
            matMul_args.Dpad = Dpad;

            // Handle bias
            matMul_args.bias = biasDiff;
            matMul_args.USE_BIASES = USE_BIASES;

#ifdef OPTIMIZE
            int padding = Lpad + Rpad + Upad + Dpad;
            int stride = stride_h + stride_w;
            if (pH == 3 && pW == 3 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_param_grad_kernel_CHW_k3x3_s2_p1_fp16, &matMul_args);
            else if (pH == 5 && pW == 5 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_param_grad_kernel_CHW_k5x5_s2_p1_fp16, &matMul_args);
            else
#endif
            pi_cl_team_fork(NUM_CORES, naive_conv2d_param_grad_kernel_CHW_fp16, &matMul_args);
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            printf("[pulp_conv2d_fp16_bw_param_grads_cl:] Naive kernel for HWC FW Conv2D not implemented!\n");
        } else {
            printf("[pulp_conv2d_fp16_bw_param_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
        }

    } else {
        printf("[pulp_conv2d_fp16_bw_param_grads_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
    }
}


void pulp_conv2d_fp16_bw_input_grads_cl(void *Conv2D_args_fp16) {
    struct Conv2D_args_fp16 *C2D_args = (struct Conv2D_args_fp16 *) Conv2D_args_fp16;
    struct matMul_args_fp16 matMul_args;
    struct im2col_args_fp16 im2col_args;

    //input dimensions
    int W_in = C2D_args->input->W;
    int H_in = C2D_args->input->H;
    int C_in = C2D_args->input->C;
    //kernel dimensions
    int pW = C2D_args->coeff->W;
    int pH = C2D_args->coeff->H;
    //output dimensions
    int W_out = C2D_args->output->W;
    int H_out = C2D_args->output->H;
    int C_out = C2D_args->output->C;

    fp16 *inData = C2D_args->input->data;
    fp16 *inDiff = C2D_args->input->diff;
    fp16 *coeffData = C2D_args->coeff->data;
    fp16 *coeffDiff = C2D_args->coeff->diff;
    fp16 *outDiff = C2D_args->output->diff;
    fp16 *outData = C2D_args->output->data;

    fp16 *i2c_buffer = C2D_args->i2c_buffer;
    fp16 *temp_bt = C2D_args->bt_buffer;

    int stride_w = C2D_args->stride_w;
    int stride_h = C2D_args->stride_h;
    int Lpad = C2D_args->Lpad;
    int Rpad = C2D_args->Rpad;
    int Upad = C2D_args->Upad;
    int Dpad = C2D_args->Dpad;

    int HWC_layout = C2D_args->HWC;
    int USE_BIASES = C2D_args->USE_BIASES;
    int USE_IM2COL = C2D_args->USE_IM2COL;
    int USE_DMA = C2D_args->USE_DMA_IM2COL;
    int opt_matmul_type = C2D_args->opt_matmul_type_ig;

    /**
     * USE OPTIMIZED ALGORITHM
     */
    if (USE_IM2COL == 1) {

        /**
         * USE CHW LAYOUT
         */
        if (HWC_layout == 0) {
            // PREPARE im2col_buffer for ACTIV_GRAD
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = 0; //pW-1;
            im2col_args.Rpad = 0; //pW-1;
            im2col_args.Upad = 0; //pH-1;
            im2col_args.Dpad = 0; //pH-1;
            im2col_args.stride_h = 1;
            im2col_args.stride_w = 1;
            im2col_args.mod = 1;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);

            // Blocktranspose weights
            struct blocktransp_args_fp16 bt_args;
            bt_args.weights = coeffData;
            bt_args.bt_weights = temp_bt;
            bt_args.Cout = C_out;
            bt_args.Cin = C_in;
            bt_args.Hk = pH;
            bt_args.Wk = pW;
            bt_args.HWC = HWC_layout;

            matMul_args.A = temp_bt; //coeffData;
            matMul_args.B = i2c_buffer;
            matMul_args.C = inDiff;
            matMul_args.N = C_in;
            matMul_args.K = pW * pH * C_out;
            matMul_args.M = W_in * H_in;
            matMul_args.trans_B = 1;

            pi_cl_team_fork(NUM_CORES, pulp_blocktransp_fp16, &bt_args);

#ifndef OPTIMIZE
            pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
#else
            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_IN_GRAD;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
            pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
#endif
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            // PREPARE im2col_buffer for ACTIV_GRAD
            im2col_args.input = C2D_args->input;
            im2col_args.c = C2D_args->coeff;
            im2col_args.output = C2D_args->output;
            im2col_args.pBuffer = i2c_buffer;
            im2col_args.Lpad = 0; //pW-1;
            im2col_args.Rpad = 0; //pW-1;
            im2col_args.Upad = 0; //pH-1;
            im2col_args.Dpad = 0; //pH-1;
            im2col_args.stride_h = 1;
            im2col_args.stride_w = 1;
            im2col_args.mod = 1;
            im2col_args.USE_DMA = USE_DMA;
            im2col_args.HWC = HWC_layout;

            pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);

            // Blocktranspose weights
            struct blocktransp_args_fp16 bt_args;
            bt_args.weights = coeffData;
            bt_args.bt_weights = temp_bt;
            bt_args.Cout = C_out;
            bt_args.Cin = C_in;
            bt_args.Hk = pH;
            bt_args.Wk = pW;
            bt_args.HWC = HWC_layout;

            matMul_args.A = i2c_buffer;
            matMul_args.B = temp_bt; //coeffData;
            matMul_args.C = inDiff;
            matMul_args.N = W_in * H_in;
            matMul_args.K = pW * pH * C_out;
            matMul_args.M = C_in;
            matMul_args.trans_B = 1;

            pi_cl_team_fork(NUM_CORES, pulp_blocktransp_fp16, &bt_args);

#ifndef OPTIMIZE
            pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
#else
            struct mm_manager_args_fp16 man_args;
            man_args.mm_args = &matMul_args;
            man_args.layer_type = LAYER_CONV2D;
            man_args.step_type = STEP_IN_GRAD;
            man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
            pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
#endif
        } else {
            printf("[pulp_conv2d_fp16_bw_input_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
        }

    }

        /**
         * USE NAIVE KERNEL
         */
    else if (USE_IM2COL == 0) {

        /**
         * USE CHW DATA LAYOUT
         */
        if (HWC_layout == 0) {
            matMul_args.A = inDiff;
            matMul_args.B = coeffData;
            matMul_args.C = outDiff;
            matMul_args.H = H_in;
            matMul_args.W = W_in;
            matMul_args.pCin = C_in;
            matMul_args.pCout = C_out;
            matMul_args.pH = pH;
            matMul_args.pW = pW;
            // Stride and padding operators
            matMul_args.stride_h = stride_h;
            matMul_args.stride_w = stride_w;
            matMul_args.Lpad = Lpad;
            matMul_args.Rpad = Rpad;
            matMul_args.Upad = Upad;
            matMul_args.Dpad = Dpad;

            // Handle bias
            matMul_args.USE_BIASES = USE_BIASES;

#ifdef OPTIMIZE
            int padding = Lpad + Rpad + Upad + Dpad;
            int stride = stride_h + stride_w;
            if (pH == 3 && pW == 3 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_in_grad_kernel_CHW_k3x3_s2_p1_fp16, &matMul_args);
            else if (pH == 5 && pW == 5 && padding == 4 && stride == 4)
            pi_cl_team_fork(NUM_CORES, naive_conv2d_in_grad_kernel_CHW_k5x5_s2_p1_fp16, &matMul_args);
            else
#endif
            pi_cl_team_fork(NUM_CORES, naive_conv2d_in_grad_kernel_CHW_fp16, &matMul_args);
        }

            /**
             * USE HWC DATA LAYOUT
             */
        else if (HWC_layout == 1) {
            printf("[pulp_conv2d_fp16_bw_input_grads_cl:] Naive kernel for HWC IG Conv2D not implemented!\n");
        } else {
            printf("[pulp_conv2d_fp16_bw_input_grads_cl:] Invalid data layout format (HWC or CHW)!\n");
        }

    } else {
        printf("[pulp_conv2d_fp16_bw_input_grads_cl:117] Invalid selection of the conv2d algorithm (im2col or not)\n");
    }
}


void im2col_conv2d_fw_kernel_fp16(void *void_args) {
    struct mm_manager_args_fp16 *man_args = (struct mm_manager_args_fp16 *) void_args;
    struct matMul_args_fp16 *args = man_args->mm_args;

    fp16 *__restrict__ inData = args->A;
    fp16 *__restrict__ coeffData = args->B;
    fp16 *__restrict__ outData = args->C;

    fp16 *__restrict__ biasData = args->bias;
    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t pW = args->pW;
    const uint32_t pH = args->pH;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->pCout;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;
    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    // const uint32_t H_out = (H_in - pH + Upad + Dpad) / h_str + 1;
    // const uint32_t W_out = (W_in - pW + Lpad + Rpad) / w_str + 1;

    const uint32_t H_out = pH;
    const uint32_t W_out = pW;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    const uint32_t HWC = args->HWC;

    int padding = Lpad + Rpad + Upad + Dpad;

    // Perform simple matrix multiplication
#ifndef OPTIMIZE
    mm_fp16(args);
#else
    mm_manager_fp16(man_args);
#endif


    // Handle biases
    if (USE_BIASES == 1) {
        for (uint32_t co = start; co < stop; co++) {
            for (uint32_t ho = 0; ho < H_out; ho++) {
                for (uint32_t wo = 0; wo < W_out; wo++) {
                    if (HWC == 0) {
                        // CHW layout
                        outData[wo + ho * W_out + co * H_out * W_out] += biasData[co];
                    } else if (HWC == 1) {
                        // HWC layout
                        outData[co + wo * C_out + ho * W_out * C_out] += biasData[co];
                    }
                }
            }
        }
    }

    if (HWC != 0 && HWC != 1) {
        // Unsupported layout
        printf("[im2col_conv2d_fw_kernel_fp16:] Invalid selection of the HWC layout (1 for HWC, 0 for CHW). Actual value: %d. Biases not used, even if provided!\n",
               HWC);
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[im2col_conv2d_fw_kernel_fp16:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}


void im2col_conv2d_param_grad_kernel_fp16(void *void_args) {
    struct mm_manager_args_fp16 *man_args = (struct mm_manager_args_fp16 *) void_args;
    struct matMul_args_fp16 *args = man_args->mm_args;

    fp16 *__restrict__ inData = args->A;
    fp16 *__restrict__ coeffDiff = args->B;
    fp16 *__restrict__ outDiff = args->C;

    fp16 *__restrict__ biasDiff = args->bias;
    const uint32_t USE_BIASES = args->USE_BIASES;

    const uint32_t H_in = args->H;
    const uint32_t W_in = args->W;
    const uint32_t pW = args->pW;
    const uint32_t pH = args->pH;
    const uint32_t C_in = args->pCin;
    const uint32_t C_out = args->N;

    uint32_t h_str = args->stride_h;
    uint32_t w_str = args->stride_w;
    uint32_t Lpad = args->Lpad;
    uint32_t Rpad = args->Rpad;
    uint32_t Upad = args->Upad;
    uint32_t Dpad = args->Dpad;

    const uint32_t H_out = (H_in - pH + Upad + Dpad) / h_str + 1;
    const uint32_t W_out = (W_in - pW + Lpad + Rpad) / w_str + 1;

    const uint32_t blockSize = (C_out + NUM_CORES - 1) / NUM_CORES;
    const uint32_t start = pi_core_id() * blockSize;
    const uint32_t stop = start + blockSize > C_out ? C_out : start + blockSize;

    const uint32_t HWC = args->HWC;

    int padding = Lpad + Rpad + Upad + Dpad;

    // Perform simple matrix multiplication
#ifndef OPTIMIZE
    mm_fp16(args);
#else
    mm_manager_fp16(man_args);
#endif

    // Handle biases
    if (USE_BIASES == 1) {
        for (uint32_t co = start; co < stop; co++) {
            float temp = 0;
            for (uint32_t ho = 0; ho < pH; ho++) {
                for (uint32_t wo = 0; wo < pW; wo++) {
                    temp += inData[wo + ho * pW + co * pH * pW];
                }
            }
            biasDiff[co] = temp;
        }
    }

    if (HWC != 0 && HWC != 1) {
        // Unsupported layout
        printf("[im2col_conv2d_param_grad_kernel_fp16:] Invalid selection of the HWC layout (1 for HWC, 0 for CHW). Actual value: %d. Biases not used, even if provided!\n",
               HWC);
    }

    if (USE_BIASES != 0 && USE_BIASES != 1) {
        printf("[im2col_conv2d_param_grad_kernel_fp16:] Invalid selection of the bias option (1 or 0 - use biases or not). Actual value: %d. Biases not used, even if provided!\n",
               USE_BIASES);
    }
}
