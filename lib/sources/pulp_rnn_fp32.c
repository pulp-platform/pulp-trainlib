/*
 * Copyright (C) 2023 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Francesco Conoscenti (francesco.conoscenti@studio.unibo.it), Alberto Dequino (alberto.dequino@unibo.it), Calin Diaconu (calin.diaconu2@unibo.it)
 */


#include "pulp_rnn_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"


//FORWARD
void pulp_rnn_fp32_fw_cl(void *Rnn_args) {
    struct Rnn_args *rnn_args = (struct Rnn_args *) Rnn_args;
    float *coeffDataWx = rnn_args->coeff_x->data; // Input Weights
    float *coeffDataWs = rnn_args->coeff_s->data; // State Weights
    float *outData = rnn_args->output->data;
    float *inputData = rnn_args->input->data;
    float *stateData = rnn_args->state->data;

    int N = rnn_args->input->H; // Input/Output Sequence length
    int K = rnn_args->input->W; // Input Sequence element length
    int M = rnn_args->output->W; // Output Sequence element length

    //matmul setup 1
    struct matMul_args matMul_args1;
    matMul_args1.A = inputData;
    matMul_args1.B = coeffDataWx;
    matMul_args1.C = outData;
    matMul_args1.N = N;
    matMul_args1.K = K;
    matMul_args1.M = M;
    matMul_args1.trans_B = 0;

#ifdef DEBUG
    printf("\ninputData: %d %d\n", N, K);
    for (int j=0; j<K; j++){
        printf("%4.2e ", matMul_args1.A[j]);
    }
    printf("\n");

    printf("\nWx: %d %d\n", K, M);
    for (int j=0; j<K*M; j++){
        if(!(j%(M))) printf("\n");
        printf("%4.2e ",  matMul_args1.B[j]);
    }
    printf("\n");
#endif

    pi_cl_team_fork(NUM_CORES, mm, &matMul_args1);


#ifdef DEBUG
    printf("\noutData: %d %d\n", N, M);
    for (int j=0; j<N*M; j++){
        printf("%4.2e ", matMul_args1.C[j]);
    }
    printf("\n");
#endif


    //matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = stateData;
    matMul_args2.B = coeffDataWs;
    matMul_args2.C = outData;
    matMul_args2.N = N;
    matMul_args2.K = M;
    matMul_args2.M = M;
    matMul_args2.trans_B = 0;


#ifdef DEBUG
    printf("\nprev_state: %d %d\n", N, M);
    for (int j=0; j<M; j++){
        printf("%4.2e ", matMul_args2.A[j]);
    }
    printf("\n");

    printf("\nWs: %d %d\n", M, M);
    for (int j=0; j<M*M; j++){
        if(!(j%(M))) printf("\n");
        printf("%4.2e ",  matMul_args2.B[j]);
    }
    printf("\n");
#endif


    pi_cl_team_fork(NUM_CORES, mm_add, &matMul_args2);


#ifdef DEBUG
    printf("\noutData: %d %d\n", N, M);
    for (int j=0; j<N*M; j++){
        printf("%4.2e ", matMul_args2.C[j]);
    }
    printf("\n");
#endif

    struct tanh_args tanh_arg;
    tanh_arg.input = matMul_args2.C;
    tanh_arg.dim = N * M;
    tanh_arg.output = matMul_args2.C;

    pi_cl_team_fork(NUM_CORES, tanh_prll, &tanh_arg);


#ifdef DEBUG
    printf("\nLinear OutData: %d %d\n", N, M);
    for (int j=0; j<N*M; j++){
        printf("%4.2e ", matMul_args2.C[j]);
    }
#endif

    /*
    if(i != ((rnn_args->input->H)-1)){
        for(int j=0; j<M; j++)
            stateData[(i+1)*M+j]=matMul_args2.C[j];
    }
    else{
        for(int j=0; j<M; j++)
            stateData[j]=matMul_args2.C[j];
    }
    */
}


//BACKWARD
void pulp_rnn_fp32_bw_cl(void *Rnn_args) {
    struct Rnn_args *rnn_args = (struct Rnn_args *) Rnn_args;

    float *coeffDataWx = rnn_args->coeff_x->data;
    float *coeffDataWs = rnn_args->coeff_s->data;
    float *inData = rnn_args->input->data;
    float *temp = rnn_args->temp_buffer; // Temporary buffer to save transposed matrices
    float *outData = rnn_args->output->data;
    float *coeffDiffWx = rnn_args->coeff_x->diff;
    float *coeffDiffWs = rnn_args->coeff_s->diff;

    float *outDiff = rnn_args->output->diff;

    float *inDiff = rnn_args->input->diff;
    float *hiddState = rnn_args->state->data;
    float *hiddStateDiff = rnn_args->state->diff;
    float *grad = rnn_args->grad_buffer; // Buffer that saves output gradients for tanh

    int total_dim = rnn_args->output->dim;
    int N = rnn_args->input->H; // Input sequence length
    int K = rnn_args->input->W; // Input sequence element size
    int M = rnn_args->output->W; // Output sequence element size



#ifdef DEBUG
    printf("\nHIDDEN STATES\n");
    for(int i=0; i<(total_dim); i++){
    if(!(i%M)) printf("\n");
      printf("%4.2e  ",hiddState[i]);
    }
    printf("\n");

    printf("\noutData\n");
    for(int i=0; i<total_dim; i++){
    if(!(i%M)) printf("\n");
      printf("%4.2e  ",outData[i]);
    }
    printf("\n");

    printf("\nLinear outDiff\n");
    for(int i=0; i<total_dim; i++){
    if(!(i%M)) printf("\n");
      printf("%4.2e  ",outDiff[i]);
    }
    printf("\n");
#endif

    for (int i = 0; i < total_dim; i++)
        grad[i] = (1 - (outData[i] * outData[i])) * outDiff[i]; // Taylor series expansion for derivative(tanh)


    // Calculate gradient for Input Weights

    // Transpose Input
    int dims[] = {N, K};
    int t_axes[] = {1, 0};

    struct transp_args transp_args1;

    transp_args1.in_matrix = inData;
    transp_args1.out_matrix = temp;
    transp_args1.dim = dims;
    transp_args1.transposed_axes = t_axes;
    transp_args1.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);


    // matmul setup 1
    struct matMul_args matMul_args1;
    matMul_args1.A = temp;
    matMul_args1.B = grad;
    matMul_args1.C = coeffDiffWx;
    matMul_args1.N = K;
    matMul_args1.K = N;
    matMul_args1.M = M;
    matMul_args1.trans_B = 0;


#ifdef DEBUG
    printf("\ngrad \n");
    for (int i=0; i<total_dim; i++){
        if(!(i%M)) printf("\n");
        printf("%4.2e  ", grad[i]);
    }
    printf("\n");

    printf("\nTransposed input sequence\n");
    for (int i=0; i<N*K; i++){
        if(!(i%N)) printf("\n");
        printf("%4.2e  ", matMul_args1.A[i]);
    }
    printf("\n");
#endif


    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args1);


#ifdef DEBUG
    printf("\nLinear coeffDiffWx");
    for (int i=0; i<K*M; i++){
      if(!(i%M)) printf("\n");
      printf("%4.2e (i=%d)", matMul_args1.C[i], i);
    }
    printf("\n");
#endif


    // Calculate gradient for State Weights
    // Transpose State
    int dims[] = {N, M};
    int t_axes[] = {1, 0};

    struct transp_args transp_args2;

    transp_args2.matrix = hiddState;
    transp_args2.transp_matrix = temp;
    transp_args2.dim = dims;
    transp_args2.transposed_axes = t_axes;
    transp_args2.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);


    // matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = temp;
    matMul_args2.B = grad;
    matMul_args2.C = coeffDiffWs;
    matMul_args2.N = M;
    matMul_args2.K = N;
    matMul_args2.M = M;
    matMul_args2.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args2);


#ifdef DEBUG
    printf("\nLinear coeffDiffWs");
    for (int i=0; i<M*M; i++){
      if(!(i%M)) printf("\n");
      printf("%4.2e (i=%d)", matMul_args2.C[i], i);
    }
    printf("\n");
#endif


#ifdef DEBUG
    printf("\ngrad \n");
    for (int i=0; i<total_dim; i++){
        if(!(i%M)) printf("\n");
        printf("%4.2e  ", matMul_args2.B[i]);
    }
    printf("\n");

    printf("\nTransposed hidden state\n");
    for (int i=0; i<total_dim; i++){
        if(!(i%N)) printf("\n");
        printf("%4.2e  ",matMul_args2.A[i]);
    }
    printf("\n");

#endif


    // Calculate the Gradient of the Input
    // Transpose Input Weights
    dims = {K, M};

    struct transp_args transp_args3;

    transp_args3.in_matrix = coeffDataWx;
    transp_args3.out_matrix = temp;
    transp_args3.dim = dims;
    transp_args3.transposed_axes = t_axes;
    transp_args3.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args3);


    // matmul setup 3
    struct matMul_args matMul_args3;
    matMul_args3.A = grad;
    matMul_args3.B = temp;
    matMul_args3.C = inDiff;
    matMul_args3.N = N;
    matMul_args3.K = M;
    matMul_args3.M = K;
    matMul_args3.trans_B = 0;


    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x1, &matMul_args3);


#ifdef DEBUG
    printf("\noutputDiff\n");
    for (int i=0; i<total_dim; i++){
        if(!(i%M)) printf("\n");
        printf("%4.2e  ", matMul_args1.A[i]);
    }
    printf("\n");

    printf("\nTransposed Wx\n");
    for (int i=0; i<K*M; i++){
        if(!(i%K)) printf("\n");
        printf("%4.2e  ", matMul_args1.B[i]);
    }
    printf("\n");

    printf("\nInput Gradients\n");
    for (int i=0; i<N*K; i++){
        if(!(i%K)) printf("\n");
        printf("%4.2e  ", matMul_args1.C[i]);
    }
    printf("\n");
#endif
}
  



