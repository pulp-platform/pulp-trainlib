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
#include <math.h>


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
    int F = mhsa_args->attention_map->W; // Hidden dimension of attention

    printf("\nPrinting the parameters: L-%d, E-%d, F-%d", L, E, F);

    int H = F / n_heads; // Size of head chunks
    float scaling = 1/sqrt(H);

    // Projecting input sequence into Q, K, V
    struct matMul_args matMul_args1;
    matMul_args1.A = inputData;
    matMul_args1.B = coeffDataWin; 
    matMul_args1.C = qkv; // Q, K, V are saved contiguously, in the same matrix
    matMul_args1.N = L;
    matMul_args1.K = E;
    matMul_args1.M = 3*F;
    matMul_args1.trans_B = 0;

    #ifdef DEBUG
    printf("\ninputData: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args1.A[j]);
    }
    printf("\n");

    printf("\nWin: %d %d\n", E, 3*F);
    for (int j=0; j<E*3*F; j++){
        if(!(j%(3*F))) printf("\n");
        printf("%.8f ",  matMul_args1.B[j]);
    }
    printf("\n");
    #endif

    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);


    #ifdef DEBUG
    printf("\nQKV Data: %d %d\n", L, 3*F);
    for (int j=0; j<L*3*F; j++){
        if(!(j%(3*F))) printf("\n");
        printf("%.8f ", matMul_args1.C[j]);
    }
    printf("\n");
    #endif

    // Transpose Projections (L x 3F -> 3F x L) to facilitate division in chunks for the multiple heads. Copy of temporary buffer required because transpose is NOT inplace
    struct transp_args transp_args1;
    transp_args1.matrix = qkv;
    transp_args1.transp_matrix = temp;
    transp_args1.N = L;
    transp_args1.M = 3*F;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);

    struct copy_args copy_args1;
    copy_args1.from = temp;
    copy_args1.to = qkv;
    copy_args1.size = L*3*F;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args1);

    // Separate Q, K and V entry points in the QKV matrix
    q = qkv;
    k = qkv + L*F;
    v = qkv + L*2*F;

    // Cycle on the different heads
    for(int i = 0; i < n_heads; i++){
        float* current_head_buffer = head_buffer + i*L*L;
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
        matMul_args2.C = current_head_buffer;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;

        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args2);

        for(int j = 0; j < (L*L); j++)
            current_head_buffer[j] = current_head_buffer[j]*scaling;

        struct act_args softmax_arg;
        struct blob input;
        struct blob output;
        input.data = current_head_buffer;
        input.dim = L*L;
        output.data = current_head_buffer;
        softmax_arg.input = &input;
        softmax_arg.output = &output;

        pi_cl_team_fork(1, pulp_softmax_fp32_fw_cl, &softmax_arg); //TODO: actually parallelize this function

        // Multiply softmax result with the i-th head's V chunk
        struct matMul_args matMul_args3;
        matMul_args3.A = v + L*i*H;
        matMul_args3.B = current_head_buffer;
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
    matMul_args4.K = F;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;

    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args4);

    #ifdef DEBUG
    printf("\nTransposed Output sequence Data: %d %d\n", E, L);
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

    float *coeffDataWin = mhsa_args->coeff_in->data; // E x 3F
    float *coeffDataWout = mhsa_args->coeff_out->data; // E x F
    float *inData = mhsa_args->input->data; // L x E
    float *temp = mhsa_args->temp_buffer; // Temporary buffer to save transposed matrices // TODO: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
    float *grad = mhsa_args->grad; // L x L
    float *outData = mhsa_args->output->data; // L x E
    float *attention_map = mhsa_args->attention_map->data; // F x L
    float *diff_attention_map = mhsa_args->attention_map->diff; // F x L
    float *coeffDiffWin = mhsa_args->coeff_in->diff; // E x 3F
    float *coeffDiffWout = mhsa_args->coeff_out->diff; // E x F
    float *head_buffer = mhsa_args->head_buffer->data; // L x L

    int total_dim = mhsa_args->output->dim;
    int L = mhsa_args->input->H; // Input sequence length
    int E = mhsa_args->input->W; // Input sequence element size
    int F = mhsa_args->attention_map->W; // Attention block hidden size
    int n_heads = mhsa_args->n_heads; // Number of heads of the mhsa
    int H = F / n_heads;

    float *q = mhsa_args->qkv->data; // 3F x L
    float *k = mhsa_args->qkv->data + F*L;
    float *v = mhsa_args->qkv->data + 2*F*L;

    float *q_diff = mhsa_args->qkv->diff; // 3F x L
    float *k_diff = mhsa_args->qkv->diff + F*L;
    float *v_diff = mhsa_args->qkv->diff + 2*F*L;


    float *outDiff = mhsa_args->output->diff; // L x E
    float *inDiff = mhsa_args->input->diff; // L x E
    float *attention_map_diff = mhsa_args->attention_map->diff; // F x L
    float *head_buffer_diff = mhsa_args->head_buffer->diff; // L x L

    
    
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
    matMul_args1.M = F;
    matMul_args1.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm, &matMul_args1); // Gradient of attention map: (L x E)*(E x F) - > (L x F)

    //Transpose map gradients (copy required because transpose can't be done inplace)
    struct transp_args transp_args1;
    transp_args1.matrix = attention_map_diff;
    transp_args1.transp_matrix = temp;
    transp_args1.N = L;
    transp_args1.M = F;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);

    struct copy_args copy_args1;
    copy_args1.from = temp;
    copy_args1.to = attention_map_diff;
    copy_args1.size = F*L;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args1); // Transposed gradient of attention map: (L x F) - > (F x L)

    // Output Projection Weights

    // Transpose output gradients
    struct transp_args transp_args2;
    transp_args2.matrix = outDiff;
    transp_args2.transp_matrix = temp;
    transp_args2.N = L;
    transp_args2.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);

    struct copy_args copy_args2;
    copy_args2.from = temp;
    copy_args2.to = outDiff;
    copy_args2.size = E*L;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args2); // Transposed output gradient: (L x E) - > (E x L)

    // Transpose Attention Map
    struct transp_args transp_args7;
    transp_args7.matrix = attention_map;
    transp_args7.transp_matrix = temp;
    transp_args7.N = F;
    transp_args7.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args7); 

    struct copy_args copy_args6;
    copy_args6.from = temp;
    copy_args6.to = attention_map;
    copy_args6.size = L*F;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args6); // Attention map: (F x L) - > (L x F)



    // matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = outDiff; 
    matMul_args2.B = attention_map; 
    matMul_args2.C = coeffDiffWout;
    matMul_args2.N = E;
    matMul_args2.K = L;
    matMul_args2.M = F;
    matMul_args2.trans_B = 0;


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

  
    pi_cl_team_fork(NUM_CORES, mm, &matMul_args2); // Output weight gradient: (E x L)*(L x F) - > (E x F)


    #ifdef DEBUG
    printf("\nLinear coeffDiffWout");
    for (int i=0; i<E*F; i++){
      if(!(i%F)) printf("\n");
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
        matMul_args3.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm, &matMul_args3); // i-th head Value gradient: (H x L)*(L x L) - > (H x L)


        // I-th head Buffer Gradient

        // Transpose output gradients
        struct transp_args transp_args3;
        transp_args3.matrix = attention_map_diff + i*L*H;
        transp_args3.transp_matrix = temp;
        transp_args3.N = H;
        transp_args3.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args3); // Transpose i-th head attention map gradient: (H x L) - > (L x H) 


        // matmul setup 4
        struct matMul_args matMul_args4;
        matMul_args4.A = temp; 
        matMul_args4.B = v + i*L*H; 
        matMul_args4.C = head_buffer_diff + i*L*L;
        matMul_args4.N = L;
        matMul_args4.K = H;
        matMul_args4.M = L;
        matMul_args4.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm, &matMul_args4); // i-th head Buffer gradient: (L x H)*(H x L) - > (L x L)


        // Back propagation of i-th head Buffer gradient through the softmax operation

        for(int j = 0; j < L*L; j++){ // Cycle over the elements of the i-th head buffer
            float sum = 0.0;
            const float neg_sft_j  =  -(head_buffer + i*L*L)[j]; 
            for(int z = 0; z < L*L; ++z){ // Softmax involves all the elements of the i-th head buffer
                float mul =  (head_buffer_diff + i*L*L)[z] * (head_buffer + i*L*L)[z] * neg_sft_j;
                sum +=  mul; // adding to the total sum of this row.
            }
            grad[j] = sum;
        }

        for(int j=0; j<L*L; j++){
            grad[j] += (head_buffer + i*L*L)[j] * (head_buffer_diff + i*L*L)[j]; // Gradient of pre-softmax head buffer: (L x L)
        }




        // I-th head Query Gradient
        
        // Transpose softmax gradients
        struct transp_args transp_args4;
        transp_args4.matrix = grad;
        transp_args4.transp_matrix = temp;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args4); 

        struct copy_args copy_args3;
        copy_args3.from = temp;
        copy_args3.to = grad;
        copy_args3.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args3); // Still (L x L). Unsure if it is needed.

        // matmul setup 5
        struct matMul_args matMul_args5;
        matMul_args5.A = k + i*L*H; 
        matMul_args5.B = grad; 
        matMul_args5.C = q_diff + i*L*H;
        matMul_args5.N = H;
        matMul_args5.K = L;
        matMul_args5.M = L;
        matMul_args5.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm, &matMul_args5); // i-th head Query gradient: (H x L)*(L x L) - > (H x L)


        // I-th head Key Gradients

        // matmul setup 6
        struct matMul_args matMul_args6;
        matMul_args6.A = q + i*L*H; 
        matMul_args6.B = grad; 
        matMul_args6.C = k_diff + i*L*H;
        matMul_args6.N = H;
        matMul_args6.K = L;
        matMul_args6.M = L;
        matMul_args6.trans_B = 0;

        pi_cl_team_fork(NUM_CORES, mm, &matMul_args6); // i-th head Key gradient: (H x L)*(L x L) - > (H x L)
    }

    // Input projection Gradients
    
    // matmul setup 7
    struct matMul_args matMul_args7;
    matMul_args7.A = q_diff; 
    matMul_args7.B = inData; 
    matMul_args7.C = coeffDiffWin;
    matMul_args7.N = 3*F;
    matMul_args7.K = L;
    matMul_args7.M = E;
    matMul_args7.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm, &matMul_args7); // Input weight gradient: (3F x L)*(L x E) - > (3F x E)

    // Transpose input weight gradients
    struct transp_args transp_args5;
    transp_args5.matrix = coeffDiffWin;
    transp_args5.transp_matrix = temp;
    transp_args5.N = 3*F;
    transp_args5.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args5); 

    struct copy_args copy_args4;
    copy_args4.from = temp;
    copy_args4.to = coeffDiffWin;
    copy_args4.size = E*3*F;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args4); // Transpose input weight gradient: (3F x E) - > (E x 3F)




    // Input Gradients

    // matmul setup 8
    struct matMul_args matMul_args8;
    matMul_args8.A = coeffDataWin; 
    matMul_args8.B = q_diff; 
    matMul_args8.C = inDiff;
    matMul_args8.N = E;
    matMul_args8.K = 3*F;
    matMul_args8.M = L;
    matMul_args8.trans_B = 0;

    pi_cl_team_fork(NUM_CORES, mm, &matMul_args8); // Input gradients: (E x 3F)*(3F x L) - > (E x L)

    // Transpose input weight gradients
    struct transp_args transp_args6;
    transp_args6.matrix = inDiff;
    transp_args6.transp_matrix = temp;
    transp_args6.N = E;
    transp_args6.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args6); 

    struct copy_args copy_args5;
    copy_args5.from = temp;
    copy_args5.to = inDiff;
    copy_args5.size = L*E;

    pi_cl_team_fork(NUM_CORES, copy, &copy_args5); // Input gradients transpose: (E x L) - > (L x E)


}
  



