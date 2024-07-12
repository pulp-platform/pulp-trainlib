/*
 * Copyright (C) 2023 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alberto Dequino (alberto.dequino@unibo.it), Calin Diaconu (calin.diaconu@studio.unibo.it)
 */


#include "pulp_mhsa_fp32.h"
#include "pulp_matmul_fp32.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_act_fp32.h"
#include <math.h>


//FORWARD
void pulp_mhsa_fp32_fw_cl(void* Mhsa_args) {
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args *mhsa_args = (struct Mhsa_args *) Mhsa_args;
    float *coeffDataWin = mhsa_args->coeff_in->data;            //  Input Projection Weights
    float *coeffDataWout = mhsa_args->coeff_out->data;          //  Output Projection Weights (Already transposed from GM)
    float *attention_map = mhsa_args->attention_map->data;      //  Buffer saving the MHSA map before output projection
    float *outData = mhsa_args->output->data;                   //  Output sequence (Transposed, E x L)
    float *inputData = mhsa_args->input->data;                  //  Input vector (Transposed, E x L)
    float *temp = mhsa_args->temp_buffer;                       //  Support buffer used in the attention head loop
    //float *head_buffer = mhsa_args->head_buffer->data;        //  Buffer containing the Q*Kt result (necessary to save for backward pass)
    float *softmax_buffer = mhsa_args->softmax_buffer->data;    //  Buffer containing the softmax results (necessary to save for backward pass)
    float *maxes = mhsa_args->maxes;                            //  Buffer containing the row-wise maxes in the softmax process
    float *sums = mhsa_args->sums;                              //  Buffer containing the row-wise exponential sums in the softmax process
    float *qkv = mhsa_args->qkv->data;                          //  Matrix containing the transposed Q, K and V (3*F x L)
    float *q = mhsa_args->qkv->data;                            //  Pointer to the first element of Q
    float *k = mhsa_args->qkv->data;                            //  Pointer to the first element of K 
    float *v = mhsa_args->qkv->data;                            //  Pointer to the first element of V
    int n_heads = mhsa_args->n_heads;                           //  Number of heads used for MHSA

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
    int E = mhsa_args->input->W;                                //  Input Sequence element size
    int F = mhsa_args->attention_map->W;                        //  Hidden dimension of attention (N. Heads * Head dimension)

    #ifdef DEBUG
    printf("\nPrinting the parameters: L-%d, E-%d, F-%d", L, E, F);
    #endif

    int H = F / n_heads;                                        //  Head dimension
    float scaling = q_rsqrt((float)H);                          //  Scaling factor to avoid vanishing gradients
    //float scaling = 1/sqrt(H);


    // ================================================== OP 1 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ coeffDataWin @ inputData ->  qkv   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    3F x E    @   E x L   -> 3F x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Projecting input sequence into Q, K, V
    struct matMul_args matMul_args1;
    matMul_args1.A = coeffDataWin;                              //  3F x E
    matMul_args1.B = inputData;                                 //  E x L 
    matMul_args1.C = qkv;                                       //  Q, K, V are saved contiguously, in the same matrix. 
                                                                //  It is transposed so that the elements of the same head are contiguous in memory. 
    matMul_args1.N = 3*F;                                       
    matMul_args1.K = E;
    matMul_args1.M = L;
    matMul_args1.trans_B = 0;

    #ifdef DEBUG
    printf("\ninputData: %d %d\n", E, L);
    for (int j=0; j<L*E; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1.B[j]);
    }
    printf("\n");

    printf("\nWin: %d %d\n", 3*F, E);
    for (int j=0; j<E*3*F; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ",  matMul_args1.A[j]);
    }
    printf("\n");
    #endif

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);
    #else
    struct mm_manager_args man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args1);
    #endif

    #ifdef DEBUG
    printf("\nQKV Data: %d %d\n", 3*F, L);
    for (int j=0; j<L*3*F; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1.C[j]);
    }
    printf("\n");
    #endif

    #ifdef DEBUG
    printf("\nWout: %d %d\n", E, F);
    for(int i=0; i<E*F; i++){
        if(!(i%F))
            printf("\n");
        printf("%.8f ", coeffDataWout[i]);
    }   
    printf("\n");
    #endif

    // ================================================== OP 2 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   qkv  ->  q, k, v  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3F x L -> 3 (F x L) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // Separate Q, K and V entry points in the QKV matrix. Q, K and V are F x L vectors.
    // TODO 0006: Maybe different key size (from q and v)
    q = qkv;
    k = qkv + L*F;
    v = qkv + L*2*F;

    //  Cycle on the different heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for(int i = 0; i < n_heads; i++){
        // ================================================== OP 3 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   -T-> temp [k ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L  ->     L x H     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [k ^ T] @   q   -> softmax_buffer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    L x H     @ H x L ->      L x L     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //  Transpose i-th head's K chunk
        struct transp_args transp_args2;
        transp_args2.matrix = k + L*i*H;
        transp_args2.transp_matrix = temp;
        transp_args2.N = H;
        transp_args2.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);

        //  Multiply it with the i-th head's Q chunk
        struct matMul_args matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = q + L*i*H;
        matMul_args2.C = softmax_buffer + i * L * L;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args2);
        #else
        struct mm_manager_args man_args2;
        man_args2.mm_args = &matMul_args2;
        man_args2.layer_type = LAYER_LINEAR;
        man_args2.step_type = STEP_FW;
        man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args2);
        #endif


        // ================================================== OP 4 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //  Scale the current head values by a factor proportional to the head dimension
        struct scalar_mul_args s_m_args;
        s_m_args.input = softmax_buffer + i * L * L;
        s_m_args.scalar = scaling;
        s_m_args.dim = L*L;

        pi_cl_team_fork(NUM_CORES,  pulp_scalar_mul_fp32_cl, &s_m_args);


        // ================================================== OP 5 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
        //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
        //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
        struct transp_args transp_args4;
        transp_args4.matrix = softmax_buffer + i * L * L;
        transp_args4.transp_matrix = temp;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args4);

        #ifdef DEBUG
        printf("\nCurrent head buffer Data: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", temp[j]);
        }
        printf("\n");
        #endif

        //  Softmax algorithm
        struct softmax_args softmax_arg;
        struct blob input;
        struct blob output;
        input.data = temp;
        input.dim = L;
        output.data = softmax_buffer + i * L * L;
        softmax_arg.input = &input;
        softmax_arg.output = &output;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;

        pulp_softmax_fp32_fw_cl(&softmax_arg);


        // ================================================== OP 6 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~

        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
        struct transp_args transp_args5;
        transp_args5.matrix = softmax_buffer + i * L * L;
        transp_args5.transp_matrix = temp;
        transp_args5.N = L;
        transp_args5.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args5);

        //  Multiply softmax result with the i-th head's Vt chunk
        struct matMul_args matMul_args3;
        matMul_args3.A = v + L*i*H;
        matMul_args3.B = temp;
        matMul_args3.C = attention_map + L*i*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args3);
        #else
        struct mm_manager_args man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args3);
        #endif
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #ifdef DEBUG
    printf("\nAttention results: %d %d\n", F, L);
    for(int i=0; i<F*L; i++){
        if(!(i%L))
            printf("\n");
        printf("%.8f ", attention_map[i]);
    }   
    printf("\n");
    #endif

    #ifdef DEBUG
    printf("\nWout: %d %d\n", E, F);
    for(int i=0; i<E*F; i++){
        if(!(i%F))
            printf("\n");
        printf("%.8f ", coeffDataWout[i]);
    }   
    printf("\n");
    #endif


    // ================================================== OP 7 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> outData ~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     @     F x L     ->  E x L  ~~~~~~~~~~~~~~~~~~~~~~

    //  Final attention map projection
    struct matMul_args matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = outData;
    matMul_args4.N = E;
    matMul_args4.K = F;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args4);
    #else
    struct mm_manager_args man_args4;
    man_args4.mm_args = &matMul_args4;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args4);
    #endif

    #ifdef DEBUG
    printf("\nTransposed Output sequence Data: %d %d\n", E, L);
    for (int j=0; j<L*E; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args4.C[j]);
    }
    printf("\n");
    #endif
}


//FORWARD FOR OLD PARTIAL SOFTMAX VERSION
void pulp_mhsa_fp32_fw_cl_2(void* Mhsa_args){
    
    struct Mhsa_args *mhsa_args = (struct Mhsa_args *) Mhsa_args;
    float *coeffDataWin = mhsa_args->coeff_in->data; // Input Projection Weights
    float *coeffDataWout = mhsa_args->coeff_out->data; // Output Projection Weights
    float *attention_map = mhsa_args->attention_map->data; // Buffer saving the MHSA map before projection
    float *outData = mhsa_args->output->data;  
    float *inputData = mhsa_args->input->data;
    float *temp = mhsa_args->temp_buffer;
    float *head_buffer = mhsa_args->head_buffer->data;
    float *softmax_buffer = mhsa_args->softmax_buffer->data;
    float *qkv = mhsa_args->qkv->data;
    float *q = mhsa_args->qkv->data;
    float *k = mhsa_args->qkv->data;
    float *v = mhsa_args->qkv->data;
    float *global_max = mhsa_args->global_max;
    float *partial_exp_sum = mhsa_args->partial_exp_sum;
    int n_heads = mhsa_args->n_heads;

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;

    int L = mhsa_args->input->H; // Input/Output Sequence length
    int E = mhsa_args->input->W; // Input Sequence element size
    int F = mhsa_args->attention_map->W; // Hidden dimension of attention

    #ifdef DEBUG
    printf("\nPrinting the parameters: L-%d, E-%d, F-%d", L, E, F);
    #endif

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

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);
    #else
    struct mm_manager_args man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args1);
    #endif

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

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args2);
        #else
        struct mm_manager_args man_args2;
        man_args2.mm_args = &matMul_args2;
        man_args2.layer_type = LAYER_LINEAR;
        man_args2.step_type = STEP_FW;
        man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args2);
        #endif


        struct scalar_mul_args s_m_args;
        s_m_args.input = current_head_buffer;
        s_m_args.scalar = scaling;
        s_m_args.dim = L*L;

        pi_cl_team_fork(NUM_CORES,  pulp_scalar_mul_fp32_cl, &s_m_args);

        #ifdef DEBUG
        printf("\nCurrent head buffer Data: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", current_head_buffer[j]);
        }
        printf("\n");
        #endif
    }

    float exp_max = 0.03125f;

    struct div_args d_args;
    d_args.input = head_buffer;
    d_args.n = exp_max;
    d_args.dim = L*L*n_heads;

    pi_cl_team_fork(NUM_CORES, pulp_div_fp32_cl, &d_args);

    #ifdef DEBUG
    printf("\nSoftmax inputs: %d %d %d\n", L, L, n_heads);
    for (int j=0; j<n_heads; j++){
        printf("\n\n");
        for(int i=0; i<L*L; i++){
            if(!(i%L))
                printf("\n");
            printf("%.8f ", head_buffer[j*L*L+i]);
        }
        
    }
    printf("\n");
    #endif

    

    struct softmax_args softmax_arg;
    struct blob input;
    struct blob output;
    input.data = head_buffer;
    input.dim = L*L*n_heads;
    output.data = softmax_buffer;
    output.dim = L*L*n_heads;
    softmax_arg.input = &input;
    softmax_arg.output = &output;
    softmax_arg.L = L;
    softmax_arg.n_heads = n_heads;
    softmax_arg.global_max = global_max;
    softmax_arg.partial_exp_sum = partial_exp_sum;

    /*
    ///------------------------------------------------------------------///
    printf("\nSoftmax stats\n");
    pi_perf_conf((1<<PI_PERF_CYCLES)); 
    
    
    unsigned long _cycles = 0; 
    int id = 0; 

    pi_perf_stop();
    pi_perf_reset(); 
    pi_perf_start();
    */

    pi_cl_team_fork(NUM_CORES, pulp_partial_softmax_fp32_fw_cl, &softmax_arg);

    /*
    pi_perf_stop(); 
    _cycles   += pi_perf_read (PI_PERF_CYCLES); 
    id = pi_core_id(); 
    ///----------------------------------------------------------------///
    
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    printf("\n"); 
    printf("[%d] TOTAL SOFTMAX cycles = %lu\n", id, _cycles); 
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    */

    #ifdef DEBUG
    printf("\nSoftmax results: %d %d %d\n", L, L, n_heads);
    for (int j=0; j<n_heads; j++){
        printf("\n\n");
        for(int i=0; i<L*L; i++){
            if(!(i%L))
                printf("\n");
            printf("%.8f ", softmax_buffer[j*L*L+i]);
        }
        
    }
    printf("\n");
    #endif


    for(int i = 0; i < n_heads; i++){
        // Multiply softmax result with the i-th head's V chunk
        struct matMul_args matMul_args3;
        matMul_args3.A = v + L*i*H;
        matMul_args3.B = softmax_buffer + i*L*L;
        matMul_args3.C = attention_map + L*i*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args3);
        #else
        struct mm_manager_args man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args3);
        #endif
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

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args4);
    #else
    struct mm_manager_args man_args4;
    man_args4.mm_args = &matMul_args4;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args4);
    #endif

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
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args *mhsa_args = (struct Mhsa_args *) Mhsa_args;

    float *coeffDataWin = mhsa_args->coeff_in->data; // E x 3F
    float *coeffDataWout = mhsa_args->coeff_out->data; // E x F
    float *inputData = mhsa_args->input->data; // L x E
    // TODO 0007: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
    float *temp = mhsa_args->temp_buffer; // Temporary buffer to save transposed matrices
    float *grad = mhsa_args->grad; // L x L
    float *outData = mhsa_args->output->data; // L x E
    float *attention_map = mhsa_args->attention_map->data; // F x L
    float *diff_attention_map = mhsa_args->attention_map->diff; // F x L
    float *coeffDiffWin = mhsa_args->coeff_in->diff; // E x 3F
    float *coeffDiffWout = mhsa_args->coeff_out->diff; // E x F
    float *head_buffer = mhsa_args->head_buffer->data; // L x L
    float *softmax_buffer = mhsa_args->softmax_buffer->data;

    int total_dim = mhsa_args->output->dim;
    int L = mhsa_args->input->H; // Input sequence length
    int E = mhsa_args->input->W; // Input sequence element size
    int F = mhsa_args->attention_map->W; // Attention block hidden size
    int n_heads = mhsa_args->n_heads; // Number of heads of the mhsa
    int H = F / n_heads;
    int opt_matmul_type = mhsa_args->opt_matmul_type_wg;

    // qkv ~ 3F x L
    float *q = mhsa_args->qkv->data;            // F x L
    float *k = mhsa_args->qkv->data + F*L;      // F x L
    float *v = mhsa_args->qkv->data + 2*F*L;    // F x L

    float *q_diff = mhsa_args->qkv->diff; // 3F x L
    float *k_diff = mhsa_args->qkv->diff + F*L;
    float *v_diff = mhsa_args->qkv->diff + 2*F*L;


    float *outDiff = mhsa_args->output->diff; // L x E
    float *inputDiff = mhsa_args->input->diff; // L x E
    float *attention_map_diff = mhsa_args->attention_map->diff; // F x L
    float *softmax_buffer_diff = mhsa_args->softmax_buffer->diff;

    float scaling = q_rsqrt((float)H);

    // ================================================== BACKPROP 7 ==================================================
    // INITIAL OP [A @ B -> C]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> outData ~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     @     F x L     ->  E x L  ~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 8.1 [dC @ B^T -> dA]
    // ~~~~~~~~~~~~~~~~~~~~~~ attention_map -T-> temp [attention_map ^ T] ~~~~~~~~~~~~~~~~~~~~~~                (T1)
    // ~~~~~~~~~~~~~~~~~~~~~~     F x L     -T->         L x F            ~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ outDiff @ temp [attention_map ^ T] -> coeffDiffWout ~~~~~~~~~~~~~~~~~~~~~~        (M1)
    // ~~~~~~~~~~~~~~~~~~~~~~  E x L  @           L x F          ->     E x F    ~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 8.2 [A^T @ dC -> dB]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout -T-> temp [coeffDataWout ^ T] ~~~~~~~~~~~~~~~~~~~~~~                (T2)
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     -T->         F x E            ~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWout ^ T] @ outDiff -> attention_map_diff ~~~~~~~~~~~~~~~~~~~~~~   (M2)
    // ~~~~~~~~~~~~~~~~~~~~~~         F x E            @  E x L  ->       F x L        ~~~~~~~~~~~~~~~~~~~~~~

    // T1
    struct transp_args transp_args1;
    transp_args1.matrix = attention_map;
    transp_args1.transp_matrix = temp;
    transp_args1.N = F;
    transp_args1.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args1);

    // M1
    struct matMul_args matMul_args1;
    matMul_args1.A = outDiff;
    matMul_args1.B = temp;
    matMul_args1.C = coeffDiffWout;
    matMul_args1.N = E;
    matMul_args1.K = L;
    matMul_args1.M = F;
    matMul_args1.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);
    #else
    struct mm_manager_args man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args1);
    #endif

    // T2
    struct transp_args transp_args2;
    transp_args2.matrix = coeffDataWout;
    transp_args2.transp_matrix = temp;
    transp_args2.N = E;
    transp_args2.M = F;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args2);

    // M2
    struct matMul_args matMul_args2;
    matMul_args2.A = temp;
    matMul_args2.B = outDiff;
    matMul_args2.C = attention_map_diff;
    matMul_args2.N = F;
    matMul_args2.K = E;
    matMul_args2.M = L;
    matMul_args2.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args2);
    #else
    struct mm_manager_args man_args2;
    man_args2.mm_args = &matMul_args2;
    man_args2.layer_type = LAYER_LINEAR;
    man_args2.step_type = STEP_FW;
    man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args2);
    #endif

    // Cycle on the heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for(int i=0; i<n_heads; i++){
        // ================================================ BACKPROP 6 ================================================
        // INITIAL OPS
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~
        // [A @ B -> C]
        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~
        //
        // BACKPROP 6.1 [dC @ B^T -> dA]
        // ~~~~~~~~~~~~~~~~~~~~~~ attention_map_diff @ softmax_buffer [softmax_buffer ^ T] -> v_diff ~~~~~~~~~~~~~~~~~~~~~~   (M3)
        // ~~~~~~~~~~~~~~~~~~~~~~       H x L        @               L x L                 -> H x L  ~~~~~~~~~~~~~~~~~~~~~~
        //
        // BACKPROP 6.2 [A^T @ dC -> dB]
        // ~~~~~~~~~~~~~~~~~~~~~~   v   -T-> temp [v ^ T] ~~~~~~~~~~~~~~~~~~~~~~                                    (T4)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L -T->   L x H      ~~~~~~~~~~~~~~~~~~~~~~
        //
        // ~~~~~~~~~ temp [v ^ T] @ attention_map_diff -> softmax_buffer_diff [softmax_buffer_diff ^ T] ~~~~~~~~~   (M4)
        // ~~~~~~~~~    L X H     @      H x L         ->                    L x L                      ~~~~~~~~~
        // M3
        struct matMul_args matMul_args3;
        matMul_args3.A = attention_map_diff + i * L * H;
        matMul_args3.B = softmax_buffer + i * L * L;
        matMul_args3.C = v_diff + i * L * H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args3);
        #else
        struct mm_manager_args man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args3);
        #endif

        // T4
        struct transp_args transp_args4;
        transp_args4.matrix = v + i * L * H;
        transp_args4.transp_matrix = temp;
        transp_args4.N = H;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args4);

        // M4
        struct matMul_args matMul_args4;
        matMul_args4.A = temp;
        matMul_args4.B = attention_map_diff + i * L * H;
        matMul_args4.C = softmax_buffer_diff + i * L * L;
        matMul_args4.N = L;
        matMul_args4.K = H;
        matMul_args4.M = L;
        matMul_args4.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args4);
        #else
        struct mm_manager_args man_args4;
        man_args4.mm_args = &matMul_args4;
        man_args4.layer_type = LAYER_LINEAR;
        man_args4.step_type = STEP_FW;
        man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args4);
        #endif


        // ================================================ BACKPROP 5 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~
        // BACKPROP
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (softmax_buffer_diff) [L x L] ~~~~~~~~~~~~~~~~~~                                                                 (T5 & C1)
        // ~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] & softmax_buffer_diff [softmax_buffer_diff ^ T] -BW_SM->  grad [sm_diff ^ T] ~~~~~~~~~~~~~~~~~~   (SM1)
        // ~~~~~~~~~~~~~~~~~~               L x L                 &                  L x L                        -BW_SM->       L x L         ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (grad) [L x L] ~~~~~~~~~~~~~~~~~~                                                                                (T7 & C3)
        // T5
        struct transp_args transp_args5;
        transp_args5.matrix = softmax_buffer_diff + i * L * L;
        transp_args5.transp_matrix = temp;
        transp_args5.N = L;
        transp_args5.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args5);

        // C1
        struct copy_args copy_args1;
        copy_args1.from = temp;
        copy_args1.to = softmax_buffer_diff + i * L * L;
        copy_args1.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args1);

        // SM1
        struct act_args softmax_arg;
        struct blob input;
        struct blob output;
        input.diff = grad;
        input.dim = i;
        output.data = softmax_buffer + i * L * L;
        output.diff = softmax_buffer_diff + i * L * L;
        output.dim = L;
        softmax_arg.input = &input;
        softmax_arg.output = &output;

        pi_cl_team_fork(1, pulp_softmax_fp32_bw_cl, &softmax_arg);

        // T7
        struct transp_args transp_args7;
        transp_args7.matrix = grad;
        transp_args7.transp_matrix = temp;
        transp_args7.N = L;
        transp_args7.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args7);

        // C3
        struct copy_args copy_args3;
        copy_args3.from = temp;
        copy_args3.to = grad;
        copy_args3.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args3);


        // ================================================ BACKPROP 4 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // BACKPROP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ grad [softmax_buffer] *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~               L x L             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        struct scalar_mul_args s_m_args;
        s_m_args.input = grad;
        s_m_args.scalar = scaling;
        s_m_args.dim = L*L;

        pi_cl_team_fork(NUM_CORES,  pulp_scalar_mul_fp32_cl, &s_m_args);


        // ================================================ BACKPROP 3 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   -T-> temp [k ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L  ->     L x H     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // [A @ B -> C]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [k ^ T] @   q   -> softmax_buffer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    L x H     @ H x L ->     L x L      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // BACKPROP 3.1 [dC @ B^T -> dA]
        // ~~~~~~~~~~~~~~~~~~~~~~   q   -T-> temp [q ^ T] ~~~~~~~~~~~~~~~~~~~~~~                                     (T8)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L -T->    L x H     ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  grad @ temp [q ^ T] -> k_diff [k_diff ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (M5)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ L x L @     L x H    ->      L x H          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (k_diff) [L x H] -> [H x L] ~~~~~~~~~~~~~~~~~~                    (T9 & C4)
        //
        // BACKPROP 3.2 [A^T @ dC -> dB]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   @ grad  -> q_diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                     (M6)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L @ L x L -> H x L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // T8
        struct transp_args transp_args8;
        transp_args8.matrix = q + i * L * H;
        transp_args8.transp_matrix = temp;
        transp_args8.N = H;
        transp_args8.M = L;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args8);

        // M5
        struct matMul_args matMul_args5;
        matMul_args5.A = grad;
        matMul_args5.B = temp;
        matMul_args5.C = k_diff + i * L * H;
        matMul_args5.N = L;
        matMul_args5.K = L;
        matMul_args5.M = H;
        matMul_args5.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm, &matMul_args5);
        #else
        struct mm_manager_args man_args5;
        man_args5.mm_args = &matMul_args5;
        man_args5.layer_type = LAYER_LINEAR;
        man_args5.step_type = STEP_FW;
        man_args5.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args5);
        #endif

        // T9
        struct transp_args transp_args9;
        transp_args9.matrix = k_diff + i * L * H;
        transp_args9.transp_matrix = temp;
        transp_args9.N = L;
        transp_args9.M = H;

        pi_cl_team_fork(NUM_CORES, transpose, &transp_args9);

        // C4
        struct copy_args copy_args4;
        copy_args4.from = temp;
        copy_args4.to = k_diff + i * L * H;
        copy_args4.size = L*H;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args4);

        // M6
        struct matMul_args matMul_args6;
        matMul_args6.A = k + i * L * H;
        matMul_args6.B = grad;
        matMul_args6.C = q_diff + i * L * H;
        matMul_args6.N = H;
        matMul_args6.K = L;
        matMul_args6.M = L;
        matMul_args6.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm, &matMul_args6);
        #else
        struct mm_manager_args man_args6;
        man_args6.mm_args = &matMul_args6;
        man_args6.layer_type = LAYER_LINEAR;
        man_args6.step_type = STEP_FW;
        man_args6.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager, &man_args6);
        #endif
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ================================================== BACKPROP 1 ==================================================
    // INITIAL OP [A @ B -> C]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ coeffDataWin @ inputData ->  qkv   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    3F x E    @   E x L   -> 3F x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 1.1 [dC @ B^T -> dA]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ q_diff [qkvDiff] @ inputData ^ T -> coeffDiffWin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    (M7)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      3F x L      @     L x E     ->   3F x E     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 1.2 [A^T @ dC -> dB]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWin -T-> temp [coeffDataWin ^ T] ~~~~~~~~~~~~~~~~~~~~~~                           (T10)
    // ~~~~~~~~~~~~~~~~~~~~~~    3F x E    -T->        E x 3F           ~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWin ^ T] @ q_diff [qkvDiff] -> inputDiff ~~~~~~~~~~~~~~~~~~~~~~~~~~    (M8)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~         E x 3F          @      3F x L      ->   E x L   ~~~~~~~~~~~~~~~~~~~~~~~~~~
    // TX
    struct transp_args transp_argsX;
    transp_argsX.matrix = inputData;
    transp_argsX.transp_matrix = temp;
    transp_argsX.N = E;
    transp_argsX.M = L;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_argsX);

    // M7
    struct matMul_args matMul_args7;
    matMul_args7.A = q_diff;
    matMul_args7.B = temp;
    matMul_args7.C = coeffDiffWin;
    matMul_args7.N = 3 * F;
    matMul_args7.K = L;
    matMul_args7.M = E;
    matMul_args7.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args7);
    #else
    struct mm_manager_args man_args7;
    man_args7.mm_args = &matMul_args7;
    man_args7.layer_type = LAYER_LINEAR;
    man_args7.step_type = STEP_FW;
    man_args7.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args7);
    #endif

    // T10
    struct transp_args transp_args10;
    transp_args10.matrix = coeffDataWin;
    transp_args10.transp_matrix = temp;
    transp_args10.N = 3 * F;
    transp_args10.M = E;

    pi_cl_team_fork(NUM_CORES, transpose, &transp_args10);

    // M8
    struct matMul_args matMul_args8;
    matMul_args8.A = temp;
    matMul_args8.B = q_diff;
    matMul_args8.C = inputDiff;
    matMul_args8.N = E;
    matMul_args8.K = 3 * F;
    matMul_args8.M = L;
    matMul_args8.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &matMul_args8);
    #else
    struct mm_manager_args man_args8;
    man_args8.mm_args = &matMul_args8;
    man_args8.layer_type = LAYER_LINEAR;
    man_args8.step_type = STEP_FW;
    man_args8.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args8);
    #endif
}
