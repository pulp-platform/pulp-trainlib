/*
 * Copyright (C) 2023 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alberto Dequino (alberto.dequino@unibo.it)
 */


#include "pulp_mhsa_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"


//FORWARD
void pulp_mhsa_fp32_fw_cl(void* Mhsa_args){
    struct Mhsa_args *mhsa_args = (struct Mhsa_args *) Mhsa_args;
    float *coeffDataWin = mhsa_args->coeff_in->data; // Input Projection Weights
    float *coeffDataWout = mhsa_args->coeff_out->data; // Output Projection Weights
    float *attention_map = mhsa_args->attention_map->data; // Buffer saving the MHSA map before projection
    float *outData = mhsa_args->output->data;  
    float *inputData = mhsa_args->input->data;
    float *temp = mhsa_args->temp_buffer;
    float *head_buffer = mhsa_args->head_buffer->data;
    float *qkv = mhsa_args->qkv->data;
    float *q = mhsa_args->qkv->data;
    float *k = mhsa_args->qkv->data;
    float *v = mhsa_args->qkv->data;
    int n_heads = mhsa_args->n_heads;

    int L = mhsa_args->input->H; // Input/Output Sequence length
    int E = mhsa_args->input->W; // Input Sequence element size

    int H = E / n_heads; // Size of head chunks

    // Projecting input sequence into Q, K, V
    struct matMul_args matMul_args1;
    matMul_args1.A = inputData;
    matMul_args1.B = coeffDataWin; 
    matMul_args1.C = qkv; // Q, K, V are saved contiguously, in the same matrix
    matMul_args1.N = L;
    matMul_args1.K = E;
    matMul_args1.M = 3*E;
    matMul_args1.trans_B = 0;

    #ifdef DEBUG
    printf("\ninputData: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args1.A[j]);
    }
    printf("\n");

    printf("\nWin: %d %d\n", E, 3*E);
    for (int j=0; j<E*3*E; j++){
        if(!(j%(3*E))) printf("\n");
        printf("%.8f ",  matMul_args1.B[j]);
    }
    printf("\n");
    #endif

    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);


    #ifdef DEBUG
    printf("\nQKV Data: %d %d\n", L, 3*E);
    for (int j=0; j<L*3*E; j++){
        if(!(j%(3*E))) printf("\n");
        printf("%.8f ", matMul_args1.C[j]);
    }
    printf("\n");
    #endif

    // Transpose Projections (L x 3E -> 3E x L) to facilitate division in chunks for the multiple heads. Copy of temporary buffer required because transpose is NOT inplace
    struct transp_args transp_args1;
    transp_args1.matrix = qkv;
    transp_args1.transp_matrix = temp;
    transp_args1.N = L;
    transp_args1.M = 3*E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);

    struct copy_args copy_args1;
    copy_args1.from = temp;
    copy_args1.to = qkv;
    copy_args1.size = L*3*E;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args1);

    // Separate Q, K and V entry points in the QKV matrix
    q = qkv;
    k = qkv + L*E;
    v = qkv + L*2*E;

    // Cycle on the different heads
    for(int i = 0; i < n_heads; i++){
        // Transpose i-th head's K chunk
        struct transp_args transp_args2;
        transp_args2.matrix = k + L*i*H;
        transp_args2.transp_matrix = temp;
        transp_args2.N = H;
        transp_args2.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);

        // Multiply it with the i-th head's Q chunk
        struct matMul_args matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = q + L*i*H;
        matMul_args2.C = head_buffer + i*L*L;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;

        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args2);

        struct act_args softmax_arg;
        struct blob input;
        struct blob output;
        input.data = head_buffer + i*L*L;
        input.dim = L*L;
        output.data = head_buffer + i*L*L;
        softmax_arg.input = &input;
        softmax_arg.output = &output;

        pi_cl_team_fork(NUM_CORES, pulp_softmax_fp32_fw_cl, &softmax_arg);

        // Multiply softmax result with the i-th head's V chunk
        struct matMul_args matMul_args3;
        matMul_args3.A = v + L*i*H;
        matMul_args3.B = head_buffer + i*L*L;
        matMul_args3.C = attention_map + L*i*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args3);
    }

    // Final attention map projection
    struct matMul_args matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = outData;
    matMul_args4.N = E;
    matMul_args4.K = E;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;

    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args4);

    #ifdef DEBUG
    printf("\nTransposed Attention map Data: %d %d\n", E, L);
    for (int j=0; j<L*E; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args4.C[j]);
    }
    printf("\n");
    #endif

    // Transpose back to original dimension
    struct transp_args transp_args3;
    transp_args3.matrix = outData;
    transp_args3.transp_matrix = temp;
    transp_args3.N = E;
    transp_args3.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args3);

    struct copy_args copy_args2;
    copy_args2.from = temp;
    copy_args2.to = outData;
    copy_args2.size = L*E;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args2);

    #ifdef DEBUG
    printf("\nOutput Data map Data: %d %d\n", E, L);
    for (int j=0; j<L*E; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", outData[j]);
    }
    printf("\n");
    #endif
}





//BACKWARD
void pulp_mhsa_fp32_bw_cl(void * Mhsa_args) {
    struct Mhsa_args *mhsa_args = (struct Mhsa_args *) Mhsa_args;

    float *coeffDataWin = mhsa_args->coeff_in->data;
    float *coeffDataWout = mhsa_args->coeff_out->data;
    float *inData = mhsa_args->input->data;
    float *temp = mhsa_args->temp_buffer; // Temporary buffer to save transposed matrices
    float *grad = mhsa_args->grad;
    float *outData = mhsa_args->output->data;
    float *attention_map = mhsa_args->attention_map->data;
    float *diff_attention_map = mhsa_args->attention_map->diff;
    float *coeffDiffWin = mhsa_args->coeff_in->diff;
    float *coeffDiffWout = mhsa_args->coeff_out->diff;
    float *head_buffer = mhsa_args->head_buffer->data;

    int total_dim = mhsa_args->output->dim;
    int L = mhsa_args->input->H; // Input sequence length
    int E = mhsa_args->input->W; // Input sequence element size
    int n_heads = mhsa_args->n_heads; // Number of heads of the mhsa
    int H = E / n_heads;

    float *q = mhsa_args->qkv->data;
    float *k = mhsa_args->qkv->data + E*L;
    float *v = mhsa_args->qkv->data + 2*E*L;

    float *q_diff = mhsa_args->qkv->diff;
    float *k_diff = mhsa_args->qkv->diff + E*L;
    float *v_diff = mhsa_args->qkv->diff + 2*E*L;


    float *outDiff = mhsa_args->output->diff;  
    float *inDiff = mhsa_args->input->diff;
    float *attention_map_diff = mhsa_args->attention_map->diff;
    float *head_buffer_diff = mhsa_args->head_buffer->diff;

    
    
    #ifdef DEBUG
    printf("\noutData\n");
    for(int i=0; i<total_dim; i++){
    if(!(i%E)) printf("\n");
      printf("%4.2e  ",outData[i]);
    }
    printf("\n");

    printf("\nLinear outDiff\n");
    for(int i=0; i<total_dim; i++){
    if(!(i%E)) printf("\n");
      printf("%4.2e  ",outDiff[i]);
    }
    printf("\n");
    #endif


    // Calculate gradient for Output Weights and Attention Map

    // Attention Map

    // matmul setup 1
    struct matMul_args matMul_args1;
    matMul_args1.A = outDiff; 
    matMul_args1.B = coeffDataWout; 
    matMul_args1.C = attention_map_diff;
    matMul_args1.N = L;
    matMul_args1.K = E;
    matMul_args1.M = E;
    matMul_args1.trans_B = 1;

    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args1);

    //Transpose map gradients
    struct transp_args transp_args1;
    transp_args1.matrix = attention_map_diff;
    transp_args1.transp_matrix = attention_map_diff;
    transp_args1.N = L;
    transp_args1.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);


    // Output Projection Weights

    // Transpose output gradients
    struct transp_args transp_args2;
    transp_args2.matrix = outDiff;
    transp_args2.transp_matrix = outDiff;
    transp_args2.N = L;
    transp_args2.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);


    // matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = outDiff; 
    matMul_args2.B = attention_map; 
    matMul_args2.C = coeffDiffWout;
    matMul_args2.N = E;
    matMul_args2.K = L;
    matMul_args2.M = E;
    matMul_args2.trans_B = 1;


    #ifdef DEBUG
    printf("\ngrad transposed\n");
    for (int i=0; i<total_dim; i++){
        if(!(i%L)) printf("\n");
        printf("%4.2e  ", outDiff[i]);
    }
    printf("\n");

    printf("\nTransposed Attention map\n");
    for (int i=0; i<E*L; i++){
        if(!(i%L)) printf("\n");
        printf("%4.2e  ", attention_map[i]);
    }
    printf("\n");
    #endif

  
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args2);


    #ifdef DEBUG
    printf("\nLinear coeffDiffWout");
    for (int i=0; i<E*E; i++){
      if(!(i%E)) printf("\n");
      printf("%4.2e (i=%d)", matMul_args1.C[i], i);
    }
    printf("\n");
    #endif

    // Cycle on the heads
    for(int i=0; i<n_heads; i++){
        // I-th head Value Gradient

        // matmul setup 3
        struct matMul_args matMul_args3;
        matMul_args3.A = attention_map_diff + i*L*H; 
        matMul_args3.B = head_buffer + i*L*L; 
        matMul_args3.C = v_diff + i*L*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 1;

        pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args3);


        // I-th head Buffer Gradient

        // Transpose output gradients
        struct transp_args transp_args3;
        transp_args3.matrix = attention_map_diff + i*L*H;
        transp_args3.transp_matrix = temp;
        transp_args3.N = H;
        transp_args3.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args3);


        // matmul setup 4
        struct matMul_args matMul_args4;
        matMul_args4.A = temp; 
        matMul_args4.B = v + i*L*H; 
        matMul_args4.C = head_buffer_diff + i*L*L;
        matMul_args4.N = L;
        matMul_args4.K = H;
        matMul_args4.M = L;
        matMul_args4.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args4);


        // Back propagation of i-th head Buffer gradient through the softmax operation

        for(int j = 0; j < L*L; j++){
            float sum = 0.0;
            const float neg_sft_j  =  -(head_buffer + i*L*L)[j];
            for(int z = 0; z < L*L; ++z){
                float mul =  (head_buffer_diff + i*L*L)[z] * (head_buffer + i*L*L)[z] * neg_sft_j;
                sum +=  mul;//adding to the total sum of this row.
            }
            grad[j] = sum;
        }

        for(int j=0; j<L*L; j++){
            grad[j] += (head_buffer + i*L*L)[j] * (head_buffer_diff + i*L*L)[j];
        }


        // I-th head Query Gradient

        // Transpose softmax gradients
        struct transp_args transp_args4;
        transp_args4.matrix = grad;
        transp_args4.transp_matrix = grad;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args4);

        // matmul setup 5
        struct matMul_args matMul_args5;
        matMul_args5.A = k + i*L*H; 
        matMul_args5.B = grad; 
        matMul_args5.C = q_diff + i*L*H;
        matMul_args5.N = H;
        matMul_args5.K = L;
        matMul_args5.M = L;
        matMul_args5.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args5);


        // I-th head Key Gradients

        // matmul setup 6
        struct matMul_args matMul_args6;
        matMul_args6.A = q + i*L*H; 
        matMul_args6.B = grad; 
        matMul_args6.C = k_diff + i*L*H;
        matMul_args6.N = H;
        matMul_args6.K = L;
        matMul_args6.M = L;
        matMul_args6.trans_B = 1;

        pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args6);
    }

    // Input projection Gradients
    
    // matmul setup 7
    struct matMul_args matMul_args7;
    matMul_args7.A = q_diff; 
    matMul_args7.B = inData; 
    matMul_args7.C = coeffDiffWin;
    matMul_args7.N = 3*E;
    matMul_args7.K = L;
    matMul_args7.M = E;
    matMul_args7.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args7);

    // Transpose input weight gradients
    struct transp_args transp_args5;
    transp_args5.matrix = coeffDiffWin;
    transp_args5.transp_matrix = coeffDiffWin;
    transp_args5.N = 3*E;
    transp_args5.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args5);


    // Input Gradients

    // matmul setup 8
    struct matMul_args matMul_args8;
    matMul_args8.A = coeffDataWin; 
    matMul_args8.B = q_diff; 
    matMul_args8.C = inDiff;
    matMul_args8.N = E;
    matMul_args8.K = 3*E;
    matMul_args8.M = L;
    matMul_args8.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args8);

    // Transpose input weight gradients
    struct transp_args transp_args6;
    transp_args6.matrix = inDiff;
    transp_args6.transp_matrix = inDiff;
    transp_args6.N = E;
    transp_args6.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args6);


}
  



