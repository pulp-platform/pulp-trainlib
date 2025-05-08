/*
 * Copyright (C) 2023-2025 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alberto Dequino (alberto.dequino@unibo.it), Calin Diaconu (calin.diaconu@studio.unibo.it)
 */


#include "pulp_mhsa_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_act_fp16.h"
#include <math.h>


//FORWARD
void pulp_mhsa_fp16_fw_cl(void *Mhsa_args) {
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;

    fp16 *coeffDataWinQ = mhsa_args->coeff_in_q->data;          //  TRANSPOSE of Input Projection Weights Query
    fp16 *coeffDataWinK = mhsa_args->coeff_in_k->data;          //  TRANSPOSE of Input Projection Weights Key
    fp16 *coeffDataWinV = mhsa_args->coeff_in_v->data;          //  TRANSPOSE of Input Projection Weights Value

    fp16 *coeffBiasWinQ = mhsa_args->bias_in_q->data;           //  Input Projection Biases Query
    fp16 *coeffBiasWinK = mhsa_args->bias_in_k->data;           //  Input Projection Biases Key
    fp16 *coeffBiasWinV = mhsa_args->bias_in_v->data;           //  Input Projection Biases Value

    fp16 *coeffDataWout = mhsa_args->coeff_out->data;           //  Output Projection Weights (Already transposed from GM)
    fp16 *attention_map = mhsa_args->attention_map->data;       //  Buffer saving the MHSA map before output projection
    fp16 *inputData = mhsa_args->input->data;                   //  Input vector (Transposed, E x L)
    fp16 *outData = mhsa_args->output->data;                    //  Output sequence (Transposed, E x L)
    fp16 *temp = mhsa_args->temp_buffer;                        //  Support buffer used in the attention head loop
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;     //  Buffer containing the softmax results (necessary to save for backward pass)
    fp16 *maxes = mhsa_args->maxes;                             //  Buffer containing the row-wise maxes in the softmax process
    fp16 *sums = mhsa_args->sums;                               //  Buffer containing the row-wise exponential sums in the softmax process
    fp16 *q = mhsa_args->q->data;                               //  Pointer to the first element of Q
    fp16 *k = mhsa_args->k->data;                               //  Pointer to the first element of K
    fp16 *v = mhsa_args->v->data;                               //  Pointer to the first element of V
    int n_heads = mhsa_args->n_heads;                           //  Number of heads used for MHSA

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
    int E = mhsa_args->input->W;                                //  Input Sequence element size
    int F = mhsa_args->attention_map->W;                        //  Hidden dimension of attention (N. Heads * Head dimension)

#ifdef DEBUG
    printf("\n~~~~~~~~~~~~~~~FORWARD PASS~~~~~~~~~~~~~~~\n\nPrinting the parameters: L-%d, E-%d, F-%d", L, E, F);
#endif

    int H = F / n_heads;                                        //  Head dimension
    fp16 scaling = (fp16)(1 / sqrt(H));                               //  Scaling factor to avoid vanishing gradients
    //float scaling = q_rsqrt_fp16((float)H);


    // ================================================== OP 1 ==================================================
    // Custom bias addition needed, since original operation is a linear layer, applied here as a matrix multiplication
    // ~~~~~~~~~~~~~~~~~~~ coeffDataWinQ [^T] -T-> temp [coeffDataWinQ]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         (T0_q)
    // ~~~~~~~~~~~~~~~~~~~       E x F        -T->      F x E            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWinQ] @ inputData ->   q   ~~~~~~~~~~~~~~~~~~~~~~~         (M1_q)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         F x E        @   E x L   -> F x L ~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~ coeffDataWinK [^T] -T-> temp [coeffDataWinK]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         (T0_k)
    // ~~~~~~~~~~~~~~~~~~~       E x F        -T->      F x E            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWinK] @ inputData ->   k   ~~~~~~~~~~~~~~~~~~~~~~~         (M1_k)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         F x E        @  E x L    -> F x L ~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~ coeffDataWinV [^T] -T-> temp [coeffDataWinV]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         (T0_v)
    // ~~~~~~~~~~~~~~~~~~~       E x F        -T->      F x E            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWinV] @ inputData ->   v   ~~~~~~~~~~~~~~~~~~~~~~~         (M1_v)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         F x E        @  E x L    -> F x L ~~~~~~~~~~~~~~~~~~~~~~~
    // TODO 0001: Maybe different key size (from q and v)

    // T0_q
    struct transp_args_fp16 transp_args0_q;

    int dim[] = {E, F};
    int tr_axes[] = {1, 0};

    transp_args0_q.in_matrix = coeffDataWinQ;
    transp_args0_q.out_matrix = temp;
    transp_args0_q.dim = dim;
    transp_args0_q.transposed_axes = tr_axes;
    transp_args0_q.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args0_q);

#ifdef DEBUG
    printf("\n\n\nT0_q result\n\ncoeffDataWinQ [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  transp_args0_q.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", transp_args0_q.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M1_q
    // Projecting input sequence into Q
    struct matMul_args_fp16 matMul_args1_q;
    matMul_args1_q.A = temp;                                       //  F x E
    matMul_args1_q.B = inputData;                                  //  E x L
    matMul_args1_q.C = q;
    matMul_args1_q.N = F;
    matMul_args1_q.K = E;
    matMul_args1_q.M = L;
    matMul_args1_q.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_q);
#else
    struct mm_manager_args_fp16 man_args1_q;
    man_args1_q.mm_args = &matMul_args1_q;
    man_args1_q.layer_type = LAYER_LINEAR;
    man_args1_q.step_type = STEP_FW;
    man_args1_q.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_q);
#endif

#ifdef DEBUG
    printf("\n\n\nM1_q result\n\ncoeffDataWinQ: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ",  matMul_args1_q.A[j]);
    }
    printf("\n");

    printf("\ninputData: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_q.B[j]);
    }
    printf("\n");

    printf("\nq: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_q.C[j]);
    }
    printf("\n\n");
#endif

    // Bias_addition_q
    struct mm_bias_add_args_fp16 mm_bias_add_args_q;
    mm_bias_add_args_q.mat = q;
    mm_bias_add_args_q.bias = coeffBiasWinQ;
    mm_bias_add_args_q.H = F;
    mm_bias_add_args_q.W = L;

    pi_cl_team_fork(NUM_CORES, mm_bias_add_transposed_fp16, &mm_bias_add_args_q);

    // T0_k
    struct transp_args_fp16 transp_args0_k;

    dim[] = {E, F};

    transp_args0_k.in_matrix = coeffDataWinK;
    transp_args0_k.out_matrix = temp;
    transp_args0_k.dim = dim;
    transp_args0_k.transposed_axes = tr_axes;
    transp_args0_k.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args0_k);

#ifdef DEBUG
    printf("\n\n\nT0_k result\n\ncoeffDataWinK [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  transp_args0_k.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", transp_args0_k.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M1_k
    // Projecting input sequence into K
    struct matMul_args_fp16 matMul_args1_k;
    matMul_args1_k.A = temp;                                       //  F x E
    matMul_args1_k.B = inputData;                                  //  E x L
    matMul_args1_k.C = k;
    matMul_args1_k.N = F;
    matMul_args1_k.K = E;
    matMul_args1_k.M = L;
    matMul_args1_k.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_k);
#else
    struct mm_manager_args_fp16 man_args1_k;
    man_args1_k.mm_args = &matMul_args1_k;
    man_args1_k.layer_type = LAYER_LINEAR;
    man_args1_k.step_type = STEP_FW;
    man_args1_k.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_k);
#endif

#ifdef DEBUG
    printf("\n\n\nM1_k result\n\ncoeffDataWinK: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ",  matMul_args1_k.A[j]);
    }
    printf("\n");

    printf("\ninputData: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_k.B[j]);
    }
    printf("\n");

    printf("\nk: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_k.C[j]);
    }
    printf("\n\n");
#endif

    // Bias_addition_k
    struct mm_bias_add_args_fp16 mm_bias_add_args_k;
    mm_bias_add_args_k.mat = k;
    mm_bias_add_args_k.bias = coeffBiasWinK;
    mm_bias_add_args_k.H = F;
    mm_bias_add_args_k.W = L;

    pi_cl_team_fork(NUM_CORES, mm_bias_add_transposed_fp16, &mm_bias_add_args_k);

    // T0_v
    struct transp_args_fp16 transp_args0_v;

    dim = {E, F};

    transp_args0_v.in_matrix = coeffDataWinV;
    transp_args0_v.out_matrix = temp;
    transp_args0_v.dim = dim;
    transp_args0_v.transposed_axes = tr_axes;
    transp_args0_v.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args0_v);

#ifdef DEBUG
    printf("\n\n\nT0_v result\n\ncoeffDataWinV [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  transp_args0_v.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", transp_args0_v.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M1_v
    // Projecting input sequence into V
    struct matMul_args_fp16 matMul_args1_v;
    matMul_args1_v.A = temp;                                       //  F x E
    matMul_args1_v.B = inputData;                                  //  E x L
    matMul_args1_v.C = v;
    matMul_args1_v.N = F;
    matMul_args1_v.K = E;
    matMul_args1_v.M = L;
    matMul_args1_v.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_v);
#else
    struct mm_manager_args_fp16 man_args1_v;
    man_args1_v.mm_args = &matMul_args1_v;
    man_args1_v.layer_type = LAYER_LINEAR;
    man_args1_v.step_type = STEP_FW;
    man_args1_v.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_v);
#endif

#ifdef DEBUG
    printf("\n\n\nM1_v result\n\ncoeffDataWinV: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ",  matMul_args1_v.A[j]);
    }
    printf("\n");

    printf("\ninputData: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_v.B[j]);
    }
    printf("\n");

    printf("\nv: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1_v.C[j]);
    }
    printf("\n\n");
#endif

    // Bias_addition_v
    struct mm_bias_add_args_fp16 mm_bias_add_args_v;
    mm_bias_add_args_v.mat = v;
    mm_bias_add_args_v.bias = coeffBiasWinV;
    mm_bias_add_args_v.H = F;
    mm_bias_add_args_v.W = L;

    pi_cl_team_fork(NUM_CORES, mm_bias_add_transposed_fp16, &mm_bias_add_args_v);

    //  Cycle on the different heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 0; i < n_heads; i++) {
        // ================================================== OP 3 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   -T-> temp [k ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                    (T1)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L -T->     L x H    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [k ^ T] @   q   -> softmax_buffer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     (M2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    L x H     @ H x L ->      L x L     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //  T1
        struct transp_args_fp16 transp_args1;

        dim = {H, L};

        transp_args1.in_matrix = k + L * i * H;
        transp_args1.out_matrix = temp;
        transp_args1.dim = dim;
        transp_args1.transposed_axes = tr_axes;
        transp_args1.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args1);

#ifdef DEBUG
        printf("\n\n\nHead %d - T1 result\n\nk: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args1.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", transp_args1.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // M2
        // Multiply it with the i-th head's Q chunk
        struct matMul_args_fp16 matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = q + L * i * H;
        matMul_args2.C = softmax_buffer + i * L * L;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args2);
#else
        struct mm_manager_args_fp16 man_args2;
        man_args2.mm_args = &matMul_args2;
        man_args2.layer_type = LAYER_LINEAR;
        man_args2.step_type = STEP_FW;
        man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args2);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M2 result\n\ntemp: %d %d\n", i, L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ",  matMul_args2.A[j]);
        }
        printf("\n");

        printf("\nq: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args2.B[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args2.C[j]);
        }
        printf("\n\n");
#endif


        // ================================================== OP 4 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = softmax_buffer + i * L * L;
        s_m_args.scalar = scaling;
        s_m_args.dim = L * L;

#ifdef DEBUG
        printf("\n\n\nHead %d - scalar result\n\nsoftmax_buffer (BEFORE scaling): %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", s_m_args.input[j]);
        }
        printf("\n\n");
#endif

        pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp16_cl, &s_m_args);

#ifdef DEBUG
        printf("\n\n\nsoftmax_buffer (AFTER scaling): %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", s_m_args.input[j]);
        }
        printf("\n\n");
#endif


        // ================================================== OP 5 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (T2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~

        //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
        //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
        //  row-wise max and sums, therefore it is necessary to transpose_fp16 the current head buffer.
        // T2
        struct transp_args_fp16 transp_args2;

        dim = {L, L};

        transp_args2.in_matrix = softmax_buffer + i * L * L;
        transp_args2.out_matrix = temp;
        transp_args2.dim = dim;
        transp_args2.transposed_axes = tr_axes;
        transp_args2.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args2);

#ifdef DEBUG
        printf("\n\n\nHead %d - T2 result\n\nsoftmax_buffer: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args2.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", transp_args2.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        //  Softmax algorithm
        struct softmax_args_fp16 softmax_arg;
        softmax_arg.input_data = temp;
        softmax_arg.output_data = softmax_buffer + i * L * L;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;
        softmax_arg.H = L;
        softmax_arg.W = L;

        pulp_softmax_fp16_fw_cl(&softmax_arg);

#ifdef DEBUG
        printf("\n\n\nHead %d - softmax result\n\ntemp: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  softmax_arg.input_data[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", softmax_arg.output_data[j]);
        }
        printf("\n\n");
#endif


        // ================================================== OP 6 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~     (T3)
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~                     (M3)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~

        // T3
        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose_fp16 the memory buffer again.
        struct transp_args_fp16 transp_args3;

        dim = {L, L};

        transp_args3.in_matrix = softmax_buffer + i * L * L;
        transp_args3.out_matrix = temp;
        transp_args3.dim = dim;
        transp_args3.transposed_axes = tr_axes;
        transp_args3.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args3);

#ifdef DEBUG
        printf("\n\n\nHead %d - T2 result\n\nsoftmax_buffer: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args3.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", transp_args3.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // M3
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = v + L * i * H;
        matMul_args3.B = temp;
        matMul_args3.C = attention_map + L * i * H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args3);
#else
        struct mm_manager_args_fp16 man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args3);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M3 result\n\nv: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  matMul_args3.A[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args3.B[j]);
        }
        printf("\n");

        printf("\nattention_map: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args3.C[j]);
        }
        printf("\n\n");
#endif
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ================================================== OP 7 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> outData ~~~~~~~~~~~~~~~~~~~~~~       (M4)
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     @     F x L     ->  E x L  ~~~~~~~~~~~~~~~~~~~~~~

    // M4
    //  Final attention map projection
    struct matMul_args_fp16 matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = outData;
    matMul_args4.N = E;
    matMul_args4.K = F;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args4);
#else
    struct mm_manager_args_fp16 man_args4;
    man_args4.mm_args = &matMul_args4;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args4);
#endif

#ifdef DEBUG
    printf("\n\n\nM4 result\n\ncoeffDataWout: %d %d\n", E, F);
        for (int j=0; j<E*F; j++){
            if(!(j%(F))) printf("\n");
            printf("%.8f ",  matMul_args4.A[j]);
        }
        printf("\n");

        printf("\nq: %d %d\n", F, L);
        for (int j=0; j<F*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args4.B[j]);
        }
        printf("\n");

        printf("\noutData: %d %d\n", E, L);
        for (int j=0; j<E*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args4.C[j]);
        }
        printf("\n\n");
#endif
}


//BACKWARD
void pulp_mhsa_fp16_bw_cl(void *Mhsa_args) {
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;

    fp16 *coeffDataWinQ = mhsa_args->coeff_in_q->data; // E x F
    fp16 *coeffDataWinK = mhsa_args->coeff_in_k->data; // E x F
    fp16 *coeffDataWinV = mhsa_args->coeff_in_v->data; // E x F
    fp16 *coeffDataWout = mhsa_args->coeff_out->data; // E x F
    fp16 *attention_map = mhsa_args->attention_map->data; // F x L
    fp16 *inputData = mhsa_args->input->data; // L x E
    fp16 *temp = mhsa_args->temp_buffer; // Temporary buffer to save transposed matrices
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;
    fp16 *sums = mhsa_args->sums;
    fp16 *grad = mhsa_args->grad; // L x L

    int L = mhsa_args->input->H; // Input sequence length
    int E = mhsa_args->input->W; // Input sequence element size
    int F = mhsa_args->attention_map->W; // Attention block hidden size
    int n_heads = mhsa_args->n_heads; // Number of heads of the mhsa
    int H = F / n_heads;
    int opt_matmul_type = mhsa_args->opt_matmul_type_wg;

    fp16 *q = mhsa_args->q->data;                         // F x L
    fp16 *k = mhsa_args->k->data;                         // F x L
    fp16 *v = mhsa_args->v->data;                         // F x L

    fp16 *q_diff = mhsa_args->q->diff;                    // F x L
    fp16 *k_diff = mhsa_args->k->diff;                    // F x L
    fp16 *v_diff = mhsa_args->v->diff;                    // F x L

    fp16 *outDiff = mhsa_args->output->diff; // L x E
    fp16 *inputDiff = mhsa_args->input->diff; // L x E
    fp16 *attention_map_diff = mhsa_args->attention_map->diff; // F x L
    fp16 *softmax_buffer_diff = mhsa_args->softmax_buffer->diff;
    fp16 *coeffDiffWinQ = mhsa_args->coeff_in_q->diff; // E x F
    fp16 *coeffDiffWinK = mhsa_args->coeff_in_k->diff; // E x F
    fp16 *coeffDiffWinV = mhsa_args->coeff_in_v->diff; // E x F
    fp16 *coeffDiffWout = mhsa_args->coeff_out->diff; // E x F

    fp16 scaling = (fp16) q_rsqrt_fp16((float) H);

    // ================================================== BACKPROP 7 ==================================================
    // INITIAL OP [A @ B -> C]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> outData ~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     @     F x L     ->  E x L  ~~~~~~~~~~~~~~~~~~~~~~
    //
    // ----------------------------------------------------------------------------------------------------------------
    // BACKPROP 7.1 [dC @ B^T -> dA]
    // ~~~~~~~~~~~~~~~~~~~~~~ attention_map -T-> temp [attention_map ^ T] ~~~~~~~~~~~~~~~~~~~~~~                (T1)
    // ~~~~~~~~~~~~~~~~~~~~~~     F x L     -T->         L x F            ~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ outDiff @ temp [attention_map ^ T] -> coeffDiffWout ~~~~~~~~~~~~~~~~~~~~~~        (M1)
    // ~~~~~~~~~~~~~~~~~~~~~~  E x L  @           L x F          ->     E x F    ~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 7.2 [A^T @ dC -> dB]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout -T-> temp [coeffDataWout ^ T] ~~~~~~~~~~~~~~~~~~~~~~                (T2)
    // ~~~~~~~~~~~~~~~~~~~~~~     E x F     -T->         F x E            ~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ temp [coeffDataWout ^ T] @ outDiff -> attention_map_diff ~~~~~~~~~~~~~~~~~~~~~~   (M2)
    // ~~~~~~~~~~~~~~~~~~~~~~         F x E            @  E x L  ->       F x L        ~~~~~~~~~~~~~~~~~~~~~~

    // T1
    struct transp_args_fp16 transp_args1;

    dim = {F, L};

    transp_args1.in_matrix = attention_map;
    transp_args1.out_matrix = temp;
    transp_args1.dim = dim;
    transp_args1.transposed_axes = tr_axes;
    transp_args1.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args1);

#ifdef DEBUG
    printf("\n\n\n~~~~~~~~~~~~~~~BACKWARD PASS~~~~~~~~~~~~~~~\n\n");
    printf("\nT1 result\n\nattention_map: %d %d\n", F, L);
    for (int j=0; j<H*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  transp_args1.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, F);
    for (int j=0; j<L*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ", transp_args1.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M1
    struct matMul_args_fp16 matMul_args1;
    matMul_args1.A = outDiff;
    matMul_args1.B = temp;
    matMul_args1.C = coeffDiffWout;
    matMul_args1.N = E;
    matMul_args1.K = L;
    matMul_args1.M = F;
    matMul_args1.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1);
#else
    struct mm_manager_args_fp16 man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1);
#endif

#ifdef DEBUG
    printf("\n\n\nM1 result\n\noutDiff: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  matMul_args1.A[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, F);
    for (int j=0; j<L*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ", matMul_args1.B[j]);
    }
    printf("\n");

    printf("\ncoeffDiffWout: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ", matMul_args1.C[j]);
    }
    printf("\n\n");
#endif

    // T2
    struct transp_args_fp16 transp_args2;

    dim = {E, F};

    transp_args2.in_matrix = coeffDataWout;
    transp_args2.out_matrix = temp;
    transp_args2.dim = dim;
    transp_args2.transposed_axes = tr_axes;
    transp_args2.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args2);

#ifdef DEBUG
    printf("\n\n\ncoeffDataWout: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  transp_args2.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", transp_args2.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M2
    struct matMul_args_fp16 matMul_args2;
    matMul_args2.A = temp;
    matMul_args2.B = outDiff;
    matMul_args2.C = attention_map_diff;
    matMul_args2.N = F;
    matMul_args2.K = E;
    matMul_args2.M = L;
    matMul_args2.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args2);
#else
    struct mm_manager_args_fp16 man_args2;
    man_args2.mm_args = &matMul_args2;
    man_args2.layer_type = LAYER_LINEAR;
    man_args2.step_type = STEP_FW;
    man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args2);
#endif

#ifdef DEBUG
    printf("\n\n\nM2 result\n\ntemp: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ",  matMul_args2.A[j]);
    }
    printf("\n");

    printf("\noutDiff: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args2.B[j]);
    }
    printf("\n");

    printf("\nattention_map_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args2.C[j]);
    }
    printf("\n\n");
#endif

    // Cycle on the heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 0; i < n_heads; i++) {
        // ================================================ BACKPROP 6 ================================================
        // INITIAL OPS
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~
        // [A @ B -> C]
        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~
        //
        // ------------------------------------------------------------------------------------------------------------
        // BACKPROP 6.1 [dC @ B^T -> dA]
        // ~~~~~~~~~~~~~~~~~ attention_map_diff @ softmax_buffer [softmax_buffer ^ T] -> v_diff ~~~~~~~~~~~~~~~~~~  (M3)
        // ~~~~~~~~~~~~~~~~~       H x L        @               L x L                 -> H x L  ~~~~~~~~~~~~~~~~~~
        //
        // BACKPROP 6.2 [A^T @ dC -> dB]
        // ~~~~~~~~~~~~~~~~~~~~~~   v   -T-> temp [v ^ T] ~~~~~~~~~~~~~~~~~~~~~~                                    (T3)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L -T->   L x H      ~~~~~~~~~~~~~~~~~~~~~~
        //
        // ~~~~~~~~~ temp [v ^ T] @ attention_map_diff -> softmax_buffer_diff [softmax_buffer_diff ^ T] ~~~~~~~~~   (M4)
        // ~~~~~~~~~    L X H     @      H x L         ->                    L x L                      ~~~~~~~~~

        // M3
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = attention_map_diff + i * L * H;
        matMul_args3.B = softmax_buffer + i * L * L;
        matMul_args3.C = v_diff + i * L * H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args3);
#else
        struct mm_manager_args_fp16 man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args3);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M3 result\n\nattention_map_diff: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  matMul_args3.A[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args3.B[j]);
        }
        printf("\n");

        printf("\nv_diff: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args3.C[j]);
        }
        printf("\n\n");
#endif

        // T3
        struct transp_args_fp16 transp_args3;

        dim = {H, L};

        transp_args3.in_matrix = v + i * L * H;
        transp_args3.out_matrix = temp;
        transp_args3.dim = dim;
        transp_args3.transposed_axes = tr_axes;
        transp_args3.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args3);

#ifdef DEBUG
        printf("\n\n\nHead %d - T3 result\n\nv: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args3.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", transp_args3.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // M4
        struct matMul_args_fp16 matMul_args4;
        matMul_args4.A = temp;
        matMul_args4.B = attention_map_diff + i * L * H;
        matMul_args4.C = softmax_buffer_diff + i * L * L;
        matMul_args4.N = L;
        matMul_args4.K = H;
        matMul_args4.M = L;
        matMul_args4.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args4);
#else
        struct mm_manager_args_fp16 man_args4;
        man_args4.mm_args = &matMul_args4;
        man_args4.layer_type = LAYER_LINEAR;
        man_args4.step_type = STEP_FW;
        man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args4);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M4 result\n\ntemp: %d %d\n", i, L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ",  matMul_args4.A[j]);
        }
        printf("\n");

        printf("\nattention_map_diff: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args4.B[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer_diff: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args4.C[j]);
        }
        printf("\n\n");
#endif


        // ================================================ BACKPROP 5 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~
        //
        // ------------------------------------------------------------------------------------------------------------
        // BACKPROP
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (softmax_buffer_diff) [L x L] ~~~~~~~~~~~~~~~~~~                               (T4 & C1)
        // ~ softmax_buffer [softmax_buffer ^ T] & softmax_buffer_diff [softmax_buffer_diff ^ T] -BW_SM->  grad [sm_diff ^ T] ~   (SM)
        // ~               L x L                 &                  L x L                        -BW_SM->       L x L         ~
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (grad) [L x L] ~~~~~~~~~~~~~~~~~~                                              (T5 & C2)

        // T4
        dim = {L, L};

        struct transp_args_fp16 transp_args4;

        transp_args4.in_matrix = softmax_buffer_diff + i * L * L;
        transp_args4.out_matrix = temp;
        transp_args4.dim = dim;
        transp_args4.transposed_axes = tr_axes;
        transp_args4.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args4);

#ifdef DEBUG
        printf("\n\n\nHead %d - T4 result\n\nsoftmax_buffer: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args4.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", transp_args4.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // C1
        struct copy_args_fp16 copy_args1;
        copy_args1.from = temp;
        copy_args1.to = softmax_buffer_diff + i * L * L;
        copy_args1.size = L * L;

        pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args1);

#ifdef DEBUG
        printf("\n\n\nHead %d - C1 result\n\ntemp: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  copy_args1.from[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer_diff: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", copy_args1.to[j]);
        }
        printf("\n\n");
#endif

        // SM
        struct softmax_args_fp16 softmax_arg;
        softmax_arg.input_diff = grad;
        softmax_arg.output_data = softmax_buffer + i * L * L;
        softmax_arg.output_diff = softmax_buffer_diff + i * L * L;
        softmax_arg.sums = sums;
        softmax_arg.H = L;
        softmax_arg.W = L;

        pulp_softmax_fp16_bw_cl(&softmax_arg);

#ifdef DEBUG
        printf("\n\n\nHead %d - softmax backprop result\n\nsoftmax_buffer: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  softmax_arg.output_data[j]);
        }
        printf("\n");

        printf("\nsoftmax_buffer_diff: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", softmax_arg.output_diff[j]);
        }
        printf("\n");

        printf("\ngrad: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", softmax_arg.input_diff[j]);
        }
        printf("\n\n");
#endif

        // T5
        struct transp_args_fp16 transp_args5;

        dim = {L, L};

        transp_args5.in_matrix = grad;
        transp_args5.out_matrix = temp;
        transp_args5.dim = dim;
        transp_args5.transposed_axes = tr_axes;
        transp_args5.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args5);

#ifdef DEBUG
        printf("\n\n\nHead %d - T5 result\n\ngrad: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args5.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", transp_args5.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // C2
        struct copy_args_fp16 copy_args2;
        copy_args2.from = temp;
        copy_args2.to = grad;
        copy_args2.size = L * L;

        pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args2);

#ifdef DEBUG
        printf("\n\n\nHead %d - C2 result\n\ntemp: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  copy_args2.from[j]);
        }
        printf("\n");

        printf("\ngrad: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", copy_args1.to[j]);
        }
        printf("\n\n");
#endif


        // ================================================ BACKPROP 4 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // ------------------------------------------------------------------------------------------------------------
        // BACKPROP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ grad [softmax_buffer] *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~               L x L             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = grad;
        s_m_args.scalar = scaling;
        s_m_args.dim = L * L;

#ifdef DEBUG
        printf("\n\n\nHead %d - backprop scalar result\n\ngrad (BEFORE scaling): %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", s_m_args.input[j]);
        }
        printf("\n\n");
#endif

        pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp16_cl, &s_m_args);

#ifdef DEBUG
        printf("\n\n\ngrad (AFTER scaling): %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", s_m_args.input[j]);
        }
        printf("\n\n");
#endif


        // ================================================ BACKPROP 3 ================================================
        // INITIAL OP
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   -T-> temp [k ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L  ->     L x H     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // [A @ B -> C]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ temp [k ^ T] @   q   -> softmax_buffer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    L x H     @ H x L ->     L x L      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // ------------------------------------------------------------------------------------------------------------
        // BACKPROP 3.1 [dC @ B^T -> dA]
        // ~~~~~~~~~~~~~~~~~~~~~~   q   -T-> temp [q ^ T] ~~~~~~~~~~~~~~~~~~~~~~                                     (T6)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L -T->    L x H     ~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  grad @ temp [q ^ T] -> k_diff [k_diff ^ T] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (M5)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ L x L @     L x H    ->      L x H          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~ "IN-PLACE"_TRANSFORM (k_diff) [L x H] -> [H x L] ~~~~~~~~~~~~~~~~~~                    (T7 & C3)
        //
        // BACKPROP 3.2 [A^T @ dC -> dB]
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   k   @ grad  -> q_diff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                     (M6)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H x L @ L x L -> H x L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // T6
        struct transp_args_fp16 transp_args6;

        dim = {H, L};

        transp_args6.in_matrix = q + i * L * H;
        transp_args6.out_matrix = temp;
        transp_args6.dim = dim;
        transp_args6.transposed_axes = tr_axes;
        transp_args6.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args6);

#ifdef DEBUG
        printf("\n\n\nHead %d - T6 result\n\nq: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  transp_args6.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", transp_args6.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // M5
        struct matMul_args_fp16 matMul_args5;
        matMul_args5.A = grad;
        matMul_args5.B = temp;
        matMul_args5.C = k_diff + i * L * H;
        matMul_args5.N = L;
        matMul_args5.K = L;
        matMul_args5.M = H;
        matMul_args5.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args5);
#else
        struct mm_manager_args_fp16 man_args5;
        man_args5.mm_args = &matMul_args5;
        man_args5.layer_type = LAYER_LINEAR;
        man_args5.step_type = STEP_FW;
        man_args5.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args5);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M5 result\n\ngrad: %d %d\n", i, L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  matMul_args5.A[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", matMul_args5.B[j]);
        }
        printf("\n");

        printf("\nk_diff: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", matMul_args5.C[j]);
        }
        printf("\n\n");
#endif

        // T7
        struct transp_args_fp16 transp_args7;

        dim = {L, H};

        transp_args7.in_matrix = k_diff + i * L * H;
        transp_args7.out_matrix = temp;
        transp_args7.dim = dim;
        transp_args7.transposed_axes = tr_axes;
        transp_args7.n_dim = 2;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args7);

#ifdef DEBUG
        printf("\n\n\nHead %d - T7 result\n\nk_diff: %d %d\n", i, L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ",  transp_args7.matrix[j]);
        }
        printf("\n");

        printf("\ntemp: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", transp_args7.transp_matrix[j]);
        }
        printf("\n\n");
#endif

        // C3
        struct copy_args_fp16 copy_args3;
        copy_args3.from = temp;
        copy_args3.to = k_diff + i * L * H;
        copy_args3.size = L * H;

        pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args3);

#ifdef DEBUG
        printf("\n\n\nHead %d - C3 result\n\ntemp: %d %d\n", i, L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ",  copy_args3.from[j]);
        }
        printf("\n");

        printf("\nk_diff: %d %d\n", L, H);
        for (int j=0; j<L*H; j++){
            if(!(j%(H))) printf("\n");
            printf("%.8f ", copy_args2.to[j]);
        }
        printf("\n\n");
#endif

        // M6
        struct matMul_args_fp16 matMul_args6;
        matMul_args6.A = k + i * L * H;
        matMul_args6.B = grad;
        matMul_args6.C = q_diff + i * L * H;
        matMul_args6.N = H;
        matMul_args6.K = L;
        matMul_args6.M = L;
        matMul_args6.trans_B = 0;

#ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args6);
#else
        struct mm_manager_args_fp16 man_args6;
        man_args6.mm_args = &matMul_args6;
        man_args6.layer_type = LAYER_LINEAR;
        man_args6.step_type = STEP_FW;
        man_args6.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args6);
#endif

#ifdef DEBUG
        printf("\n\n\nHead %d - M6 result\n\nk: %d %d\n", i, H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ",  matMul_args6.A[j]);
        }
        printf("\n");

        printf("\ngrad: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args6.B[j]);
        }
        printf("\n");

        printf("\nq_diff: %d %d\n", H, L);
        for (int j=0; j<H*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", matMul_args6.C[j]);
        }
        printf("\n\n");
#endif
    }
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ================================================== BACKPROP 1 ==================================================
    // INITIAL OP [A @ B -> C]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinQ @ inputData ->   q   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     F x E     @  E x L    -> F x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinK @ inputData ->   k   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     F x E     @  E x L    -> F x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinV @ inputData ->   v   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     F x E     @  E x L    -> F x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // ------------------------------------------------------------------------------------------------------------
    // BACKPROP 1.1 [dC @ B^T -> dA]
    // ~~~~~~~~~~~~~~~~~~~~~~ inputData -T-> temp [inputData ^ T] ~~~~~~~~~~~~~~~~~~~~~~                         (T8)
    // ~~~~~~~~~~~~~~~~~~~~~~   E x L   -T->        L x E         ~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ q_diff  @ temp [inputData ^ T] -> coeffDiffWinQ ~~~~~~~~~~~~~~~~~~~~~~~~~~~   (M7_q)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~  F x L  @       L x E          ->     F x E     ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ k_diff  @ temp [inputData ^ T] -> coeffDiffWinK ~~~~~~~~~~~~~~~~~~~~~~~~~~~   (M7_k)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~  F x L  @       L x E          ->     F x E     ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ v_diff  @ temp [inputData ^ T] -> coeffDiffWinV ~~~~~~~~~~~~~~~~~~~~~~~~~~~   (M7_v)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~  F x L  @       L x E          ->     F x E     ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // BACKPROP 1.2 [A^T @ dC -> dB]
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinQ [coeffDataWinQ ^ T] @ q_diff -> temp  ~~~~~~~~~~~~~~~~~~~~~~~~~~~    (M8_q)
    // ~~~~~~~~~~~~~~~~~~~~~~              E x F                @ F x L  -> E x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~ inputDiff + temp  -> inputDiff  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    (SUM_q)
    // ~~~~~~~~~~~~~~~~~~~~~~   E x L   + E x L ->   E x L    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinK [coeffDataWinK ^ T] @ k_diff -> temp  ~~~~~~~~~~~~~~~~~~~~~~~~~~~    (M8_k)
    // ~~~~~~~~~~~~~~~~~~~~~~              E x F                @ F x L  -> E x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~ inputDiff + temp  -> inputDiff  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    (SUM_k)
    // ~~~~~~~~~~~~~~~~~~~~~~   E x L   + E x L ->   E x L    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWinV [coeffDataWinV ^ T] @ v_diff -> temp  ~~~~~~~~~~~~~~~~~~~~~~~~~~~    (M8_v)
    // ~~~~~~~~~~~~~~~~~~~~~~              E x F                @ F x L  -> E x L ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~ inputDiff + temp  -> inputDiff  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    (SUM_v)
    // ~~~~~~~~~~~~~~~~~~~~~~   E x L   + E x L ->   E x L    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // T8
    struct transp_args_fp16 transp_args8;

    dim = {E, L};

    transp_args8.in_matrix = inputData;
    transp_args8.out_matrix = temp;
    transp_args8.dim = dim;
    transp_args8.transposed_axes = tr_axes;
    transp_args8.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args8);

#ifdef DEBUG
    printf("\n\n\nT8 result\n\ninputData: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  transp_args8.matrix[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", transp_args8.transp_matrix[j]);
    }
    printf("\n\n");
#endif

    // M7_q
    struct matMul_args_fp16 matMul_args7_q;
    matMul_args7_q.A = q_diff;
    matMul_args7_q.B = temp;
    matMul_args7_q.C = coeffDiffWinQ;
    matMul_args7_q.N = F;
    matMul_args7_q.K = L;
    matMul_args7_q.M = E;
    matMul_args7_q.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args7_q);
#else
    struct mm_manager_args_fp16 man_args7_q;
    man_args7_q.mm_args = &matMul_args7_q;
    man_args7_q.layer_type = LAYER_LINEAR;
    man_args7_q.step_type = STEP_FW;
    man_args7_q.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args7_q);
#endif

#ifdef DEBUG
    printf("\n\n\nM7_q result\n\nq_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  matMul_args7_q.A[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_q.B[j]);
    }
    printf("\n");

    printf("\ncoeffDiffWinQ: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_q.C[j]);
    }
    printf("\n\n");
#endif

    // M7_k
    struct matMul_args_fp16 matMul_args7_k;
    matMul_args7_k.A = k_diff;
    matMul_args7_k.B = temp;
    matMul_args7_k.C = coeffDiffWinK;
    matMul_args7_k.N = F;
    matMul_args7_k.K = L;
    matMul_args7_k.M = E;
    matMul_args7_k.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args7_k);
#else
    struct mm_manager_args_fp16 man_args7_k;
    man_args7_k.mm_args = &matMul_args7_k;
    man_args7_k.layer_type = LAYER_LINEAR;
    man_args7_k.step_type = STEP_FW;
    man_args7_k.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args7_k);
#endif

#ifdef DEBUG
    printf("\n\n\nM7_k result\n\nk_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  matMul_args7_k.A[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_k.B[j]);
    }
    printf("\n");

    printf("\ncoeffDiffWinK: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_k.C[j]);
    }
    printf("\n\n");
#endif

    // M7_v
    struct matMul_args_fp16 matMul_args7_v;
    matMul_args7_v.A = v_diff;
    matMul_args7_v.B = temp;
    matMul_args7_v.C = coeffDiffWinV;
    matMul_args7_v.N = F;
    matMul_args7_v.K = L;
    matMul_args7_v.M = E;
    matMul_args7_v.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args7_v);
#else
    struct mm_manager_args_fp16 man_args7_v;
    man_args7_v.mm_args = &matMul_args7_v;
    man_args7_v.layer_type = LAYER_LINEAR;
    man_args7_v.step_type = STEP_FW;
    man_args7_v.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args7_v);
#endif

#ifdef DEBUG
    printf("\n\n\nM7_v result\n\nv_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ",  matMul_args7_v.A[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", L, E);
    for (int j=0; j<L*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_v.B[j]);
    }
    printf("\n");

    printf("\ncoeffDiffWinV: %d %d\n", F, E);
    for (int j=0; j<F*E; j++){
        if(!(j%(E))) printf("\n");
        printf("%.8f ", matMul_args7_v.C[j]);
    }
    printf("\n\n");
#endif

    // M8_q
    struct matMul_args_fp16 matMul_args8_q;
    matMul_args8_q.A = coeffDataWinQ;
    matMul_args8_q.B = q_diff;
    matMul_args8_q.C = temp;
    matMul_args8_q.N = E;
    matMul_args8_q.K = F;
    matMul_args8_q.M = L;
    matMul_args8_q.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args8_q);
#else
    struct mm_manager_args_fp16 man_args8_q;
    man_args8_q.mm_args = &matMul_args8_q;
    man_args8_q.layer_type = LAYER_LINEAR;
    man_args8_q.step_type = STEP_FW;
    man_args8_q.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args8_q);
#endif

#ifdef DEBUG
    printf("\n\n\nM8_q result\n\ncoeffDataWinQ [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  matMul_args8_q.A[j]);
    }
    printf("\n");

    printf("\nq_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_q.B[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_q.C[j]);
    }
    printf("\n\n");
#endif

    // SUM_q
    int dims[1] = {E * L};

    struct array_broadcast_sum_fp16_args vect_sum_args;

    vect_sum_args.op_1 = inputDiff;
    vect_sum_args.op_2 = temp;
    vect_sum_args.dest = inputDiff;

    vect_sum_args.op_1_dims = dims;
    vect_sum_args.op_2_dims = dims;

    vect_sum_args.op_1_dims_len = 1;
    vect_sum_args.op_2_dims_len = 1;

    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp16, &vect_sum_args);

    // M8_k
    struct matMul_args_fp16 matMul_args8_k;
    matMul_args8_k.A = coeffDataWinK;
    matMul_args8_k.B = k_diff;
    matMul_args8_k.C = temp;
    matMul_args8_k.N = E;
    matMul_args8_k.K = F;
    matMul_args8_k.M = L;
    matMul_args8_k.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args8_k);
#else
    struct mm_manager_args_fp16 man_args8_k;
    man_args8_k.mm_args = &matMul_args8_k;
    man_args8_k.layer_type = LAYER_LINEAR;
    man_args8_k.step_type = STEP_FW;
    man_args8_k.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args8_k);
#endif

#ifdef DEBUG
    printf("\n\n\nM8_k result\n\ncoeffDataWinK [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  matMul_args8_k.A[j]);
    }
    printf("\n");

    printf("\nk_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_k.B[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_k.C[j]);
    }
    printf("\n\n");
#endif

    // SUM_k
    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp16, &vect_sum_args);

    // M8_v
    struct matMul_args_fp16 matMul_args8_v;
    matMul_args8_v.A = coeffDataWinV;
    matMul_args8_v.B = v_diff;
    matMul_args8_v.C = temp;
    matMul_args8_v.N = E;
    matMul_args8_v.K = F;
    matMul_args8_v.M = L;
    matMul_args8_v.trans_B = 0;

#ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args8_v);
#else
    struct mm_manager_args_fp16 man_args8_v;
    man_args8_v.mm_args = &matMul_args8_v;
    man_args8_v.layer_type = LAYER_LINEAR;
    man_args8_v.step_type = STEP_FW;
    man_args8_v.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args8_v);
#endif

#ifdef DEBUG
    printf("\n\n\nM8_v result\n\ncoeffDataWinV [^T]: %d %d\n", E, F);
    for (int j=0; j<E*F; j++){
        if(!(j%(F))) printf("\n");
        printf("%.8f ",  matMul_args8_v.A[j]);
    }
    printf("\n");

    printf("\nv_diff: %d %d\n", F, L);
    for (int j=0; j<F*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_v.B[j]);
    }
    printf("\n");

    printf("\ntemp: %d %d\n", E, L);
    for (int j=0; j<E*L; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args8_v.C[j]);
    }
    printf("\n\n");
#endif

    // SUM_v
    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp16, &vect_sum_args);
}

//FORWARD INFERENCE
void pulp_mhsa_mobilebert_inference_fp16_fw_cl(void *Mhsa_args) {
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;

    fp16 *coeffDataWinQ = mhsa_args->coeff_in_q->data;         //  Input Projection Weights for Query (transposed)
    fp16 *coeffDataWinK = mhsa_args->coeff_in_k->data;         //  Input Projection Weights for Key (transposed)
    fp16 *coeffDataWinV = mhsa_args->coeff_in_v->data;         //  Input Projection Weights for Value (transposed)

    fp16 *coeffBiasWinQ = mhsa_args->bias_in_q->data;          //  Input Projection Biases for Query
    fp16 *coeffBiasWinK = mhsa_args->bias_in_k->data;          //  Input Projection Biases for Key
    fp16 *coeffBiasWinV = mhsa_args->bias_in_v->data;          //  Input Projection Biases for Value

    fp16 *coeffDataWout = mhsa_args->coeff_out->data;          //  Output Projection Weights (Already transposed from GM)
    fp16 *coeffBiasWout = mhsa_args->bias_out->data;           //  Output Projection Biases
    
    fp16 *attention_map = mhsa_args->attention_map->data;      //  Buffer saving the MHSA map before output projection
    fp16 *outData = mhsa_args->output->data;                   //  Output sequence (Transposed, E x L)
    fp16 *inputData = mhsa_args->input->data;                  //  Input vector (L x E)
    fp16 *inputDataBn = mhsa_args->input_bn->data;             //  Input vector bottlenecked (L x F)
    fp16 *temp = mhsa_args->temp_buffer;                       //  Support buffer used in the attention head loop
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;    //  Buffer containing the softmax results (necessary to save for backward pass)
    fp16 *maxes = mhsa_args->maxes;                            //  Buffer containing the row-wise maxes in the softmax process
    fp16 *sums = mhsa_args->sums;                              //  Buffer containing the row-wise exponential sums in the softmax process
    fp16 *qt = mhsa_args->q->data;                             //  Pointer to the first element of Q transposed
    fp16 *kt = mhsa_args->k->data;                             //  Pointer to the first element of K transposed
    fp16 *vt = mhsa_args->v->data;                             //  Pointer to the first element of V transposed
    int n_heads = mhsa_args->n_heads;                           //  Number of heads used for MHSA

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
    int E = mhsa_args->input->W;                                //  Input Sequence element size
    int F = mhsa_args->input_bn->W;                             //  Hidden dimension of attention (N. Heads * Head dimension)

    //printf("\n~~~~~~~~~~~~~~~FORWARD PASS~~~~~~~~~~~~~~~\n\nPrinting the parameters: L-%d, E-%d, F-%d\n", L, E, F);

    int H = F / n_heads;                                        //  Head dimension
    fp16 scaling = q_rsqrt_fp16((fp16) H);                         //  Scaling factor to avoid vanishing gradients
    //float scaling = 1/sqrt(H);

    // M1_q
    // (Wq)t * input_bn
    struct matMul_args_fp16 matMul_args1_q;
    matMul_args1_q.A = coeffDataWinQ;                              //  F x F
    matMul_args1_q.B = inputDataBn;                                //  L x F
    matMul_args1_q.C = qt;                                         //  F x L
    matMul_args1_q.N = F;
    matMul_args1_q.K = F;
    matMul_args1_q.M = L;
    matMul_args1_q.trans_B = 1;
    matMul_args1_q.bias = coeffBiasWinQ;
    matMul_args1_q.USE_BIASES = 1;
    matMul_args1_q.bias_transposed = 1;
    matMul_args1_q.bias_dim = F;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_q);
    #else
    struct mm_manager_args_fp16 man_args1_q;
    man_args1_q.mm_args = &matMul_args1_q;
    man_args1_q.layer_type = LAYER_LINEAR;
    man_args1_q.step_type = STEP_FW;
    man_args1_q.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_q);
    #endif

    // M1_k
    // (Wk)t * input_bn
    struct matMul_args_fp16 matMul_args1_k;
    matMul_args1_k.A = coeffDataWinK;                              //  F x F
    matMul_args1_k.B = inputDataBn;                                //  L x F
    matMul_args1_k.C = kt;                                         //  F x L
    matMul_args1_k.N = F;
    matMul_args1_k.K = F;
    matMul_args1_k.M = L;
    matMul_args1_k.trans_B = 1;
    matMul_args1_k.bias = coeffBiasWinK;
    matMul_args1_k.USE_BIASES = 1;
    matMul_args1_k.bias_transposed = 1;
    matMul_args1_k.bias_dim = F;


    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_k);
    #else
    struct mm_manager_args_fp16 man_args1_k;
    man_args1_k.mm_args = &matMul_args1_k;
    man_args1_k.layer_type = LAYER_LINEAR;
    man_args1_k.step_type = STEP_FW;
    man_args1_k.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_k);
    #endif

    //  Cycle on the different heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 0; i < n_heads; i++) {
        //  T1
        struct transp_args_fp16 transp_args1;
        transp_args1.matrix = kt + L * i * H;
        transp_args1.transp_matrix = temp;
        transp_args1.N = H;
        transp_args1.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args1);

        // M2
        // Multiply it with the i-th head's transposed Q chunk
        struct matMul_args_fp16 matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = qt + L * i * H;
        matMul_args2.C = softmax_buffer + i * L * L;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;
        matMul_args2.USE_BIASES = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args2);
        #else
        struct mm_manager_args_fp16 man_args2;
        man_args2.mm_args = &matMul_args2;
        man_args2.layer_type = LAYER_LINEAR;
        man_args2.step_type = STEP_FW;
        man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args2);
        #endif


        // ================================================== OP 4 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = softmax_buffer + i * L * L;
        s_m_args.scalar = scaling;
        s_m_args.dim = L * L;

        pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp16_cl, &s_m_args);

        // ================================================== OP 5 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (T2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~

        //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
        //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
        //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
        // T2
        struct transp_args_fp16 transp_args2;
        transp_args2.matrix = softmax_buffer + i * L * L;
        transp_args2.transp_matrix = temp;
        transp_args2.N = L;
        transp_args2.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args2);

        //  Softmax algorithm
        struct softmax_args_fp16 softmax_arg;
        softmax_arg.input_data = temp;
        softmax_arg.output_data = softmax_buffer + i * L * L;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;
        softmax_arg.H = L;
        softmax_arg.W = L;

        pulp_softmax_fp16_fw_cl(&softmax_arg);
    }


    // M1_v
    // (Wv)t * input
    struct matMul_args_fp16 matMul_args1_v;
    matMul_args1_v.A = coeffDataWinV;                              //  F x E
    matMul_args1_v.B = inputData;                                  //  L x E
    matMul_args1_v.C = vt;                                         //  F x L
    matMul_args1_v.N = F;
    matMul_args1_v.K = E;
    matMul_args1_v.M = L;
    matMul_args1_v.trans_B = 1;
    matMul_args1_v.bias = coeffBiasWinV;
    matMul_args1_v.USE_BIASES = 1;
    matMul_args1_v.bias_transposed = 1;
    matMul_args1_v.bias_dim = F;


    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args1_v);
    #else
    struct mm_manager_args_fp16 man_args1_v;
    man_args1_v.mm_args = &matMul_args1_v;
    man_args1_v.layer_type = LAYER_LINEAR;
    man_args1_v.step_type = STEP_FW;
    man_args1_v.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1_v);
    #endif


    for (int i = 0; i < n_heads; i++) {
        // ================================================== OP 6 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~     (T3)
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~                     (M3)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~

        // T3
        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
        struct transp_args_fp16 transp_args3;
        transp_args3.matrix = softmax_buffer + i * L * L;
        transp_args3.transp_matrix = temp;
        transp_args3.N = L;
        transp_args3.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args3);

        // M3
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = vt + L * i * H;
        matMul_args3.B = temp;
        matMul_args3.C = attention_map + L * i * H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args3);
        #else
        struct mm_manager_args_fp16 man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args3);
        #endif
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ================================================== OP 7 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> temp ~~~~~~~~~~~~~~~~~~~~~~       (M4)
    // ~~~~~~~~~~~~~~~~~~~~~~     F x F     @     F x L     ->  F x L  ~~~~~~~~~~~~~~~~~~~~~~

    // M4
    //  Final attention map projection
    struct matMul_args_fp16 matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = temp;
    matMul_args4.N = F;
    matMul_args4.K = F;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;
    matMul_args4.bias = coeffBiasWout;
    matMul_args4.USE_BIASES = 1;
    matMul_args4.bias_transposed = 1;
    matMul_args4.bias_dim = F;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args4);
    #else
    struct mm_manager_args_fp16 man_args4;
    man_args4.mm_args = &matMul_args4;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args4);
    #endif

    // T4
    // The last transpose to original shape
    struct transp_args_fp16 transp_args4;
    transp_args4.matrix = temp;
    transp_args4.transp_matrix = outData;
    transp_args4.N = F;
    transp_args4.M = L;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args4);
}


void tiled_matmul_mhsa_fp16(void* matmul_args, void* tiled_matmul_mhsa_args, int projection){
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *)matmul_args;
    struct Tiled_Matmul_Mhsa_args_fp16 * tiled_args = (struct Tiled_Matmul_Mhsa_args_fp16*) tiled_matmul_mhsa_args;
    int tile_h;
    int tile_w;
    int tile_dim;
    if(projection == 1){
        tile_h = tiled_args->tile_h_p;
        tile_w = tiled_args->tile_w_p;
        tile_dim = tiled_args->tile_dim_p;
    }
    else if(projection == 0){
        tile_h = tiled_args->tile_h_sm;
        tile_w = tiled_args->tile_w_sm;
        tile_dim = tiled_args->tile_dim_sm;
    }
    else if(projection == 2){
        tile_h = tiled_args->tile_h_attv;
        tile_w = tiled_args->tile_w_attv;
        tile_dim = tiled_args->tile_dim_attv;
    }
    fp16* BUFF = tiled_args->BUFF;
    pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
    pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;
    
    int n_tiles_i = (args->N) / tile_h;
    int n_tiles_j = (args->M) / tile_w;
    int K = args->K;
    
    fp16* IN_DATA = BUFF;
    fp16* W_DATA = BUFF + K * tile_h;
    fp16* OUT_DATA = W_DATA + K * tile_w;
    fp16* BIAS_DATA = OUT_DATA + tile_dim;
    
    tiled_args->mm_args->A = IN_DATA;
    tiled_args->mm_args->B = W_DATA;
    tiled_args->mm_args->C = OUT_DATA;
    tiled_args->mm_args->N = tile_h;
    tiled_args->mm_args->K = K;
    tiled_args->mm_args->M = tile_w;
    tiled_args->mm_args->trans_B = args->trans_B;
    tiled_args->mm_args->bias = BIAS_DATA;
    tiled_args->mm_args->bias_dim = tile_w;
    tiled_args->mm_args->USE_BIASES = args->USE_BIASES;
    tiled_args->mm_args->bias_transposed = args->bias_transposed;
    
    for(int j = 0; j < n_tiles_j; j++){
        if(!tiled_args->mm_args->trans_B){
            pi_cl_dma_cmd_2d((uint32_t) (args->B + j * tile_w), (uint32_t) (W_DATA), 2 * tile_w * K, 2 * (args->M), 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            if(args->USE_BIASES == 1){
                pi_cl_dma_cmd((uint32_t) (args->bias + j * tile_w), (uint32_t) (BIAS_DATA), 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
                pi_cl_dma_cmd_wait(cmd_load);
            }   
        }
        else{
            pi_cl_dma_cmd_2d((uint32_t) (args->B + j * tile_w * K), (uint32_t) (W_DATA), 2 * tile_w * K, 2 * K, 2 * K, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
        }
        for(int i = 0; i < n_tiles_i; i++){
            if(tiled_args->mm_args->bias_transposed && args->USE_BIASES == 1){
                pi_cl_dma_cmd((uint32_t) (args->bias + i * tile_h), (uint32_t) (BIAS_DATA), 2 * tile_h, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
                pi_cl_dma_cmd_wait(cmd_load);
            }
            pi_cl_dma_cmd_2d((uint32_t) (args->A + i * K * tile_h), (uint32_t) (IN_DATA), 2 * K * tile_h, 2 * K, 2 * K, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            tiled_args->man_args->mm_args = tiled_args->mm_args;
            pi_cl_team_fork(NUM_CORES, mm_manager_fp16, tiled_args->man_args);
            pi_cl_dma_cmd_2d((uint32_t) (args->C + i * args->M * tile_h + tile_w * j), (uint32_t) (OUT_DATA), 2 * tile_dim, 2 * (args->M), 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
        }
    }
}

void tiled_transpose_mhsa_fp16(void *transpose_args, void* Tiled_matmul_mhsa_args, int projection){
    struct transp_args_fp16 *args = (struct transp_args_fp16 *) transpose_args;
    struct Tiled_Matmul_Mhsa_args_fp16 * tiled_args = (struct Tiled_Matmul_Mhsa_args_fp16*) Tiled_matmul_mhsa_args;
    struct transp_args_fp16 args_l1;

    int tile_h;
    int tile_w;
    int tile_dim;
    if(projection == 0){
        tile_h = tiled_args->tile_h_sm;
        tile_w = tiled_args->tile_w_sm;
        tile_dim = tiled_args->tile_dim_sm;
    }
    else if(projection == 1){
        tile_h = tiled_args->tile_h_out_tr;
        tile_w = tiled_args->tile_w_out_tr;
        tile_dim = tiled_args->tile_dim_out_tr;
    }
    else if(projection == 2){
        tile_h = tiled_args->tile_h_tr;
        tile_w = tiled_args->tile_w_tr;
        tile_dim = tiled_args->tile_dim_tr;
    }
    fp16* BUFF = tiled_args->BUFF;
    pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
    pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;

    int N = args->N;
    int M = args->M;
    int n_tiles_i = M / tile_h;
    int n_tiles_j = N / tile_w;

    fp16* IN_DATA = BUFF;
    fp16* OUT_DATA = BUFF + tile_dim;

    args_l1.matrix = IN_DATA;
    args_l1.transp_matrix = OUT_DATA;
    args_l1.N = tile_w;
    args_l1.M = tile_h;
    
    for(int i = 0; i < n_tiles_i; i++){
        for(int j = 0; j < n_tiles_j; j++){
            pi_cl_dma_cmd_2d((uint32_t) (args->matrix + i * tile_h + j * tile_w * M), (uint32_t) (IN_DATA), 2 * tile_dim, 2 * M, 2 * tile_h, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            pi_cl_team_fork(NUM_CORES, transpose_fp16, &args_l1);
            pi_cl_dma_cmd_2d((uint32_t) (args->transp_matrix + j * tile_w + i * tile_h * N), (uint32_t) (OUT_DATA), 2 * tile_dim, 2 * N, 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
        }
    }
}

void tiled_mhsa_fp16(void *Mhsa_args, void* Tiled_mhsa_matmul_args){
    // ======================================== DECLARATIONS ========================================
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;
    struct Tiled_Matmul_Mhsa_args_fp16 *tiled_args = (struct Tiled_Matmul_Mhsa_args_fp16 *) Tiled_mhsa_matmul_args; 

    int tile_h = tiled_args->tile_h_sm;
    int tile_w = tiled_args->tile_w_sm;
    int tile_dim = tiled_args->tile_dim_sm;
    fp16* BUFF = tiled_args->BUFF;
    pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
    pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;

    fp16 *coeffDataWinQ = mhsa_args->coeff_in_q->data;         //  Input Projection Weights for Query (transposed)
    fp16 *coeffDataWinK = mhsa_args->coeff_in_k->data;         //  Input Projection Weights for Key (transposed)
    fp16 *coeffDataWinV = mhsa_args->coeff_in_v->data;         //  Input Projection Weights for Value (transposed)

    fp16 *coeffBiasWinQ = mhsa_args->bias_in_q->data;          //  Input Projection Biases for Query
    fp16 *coeffBiasWinK = mhsa_args->bias_in_k->data;          //  Input Projection Biases for Key
    fp16 *coeffBiasWinV = mhsa_args->bias_in_v->data;          //  Input Projection Biases for Value

    fp16 *coeffDataWout = mhsa_args->coeff_out->data;          //  Output Projection Weights (Already transposed from GM)
    fp16 *coeffBiasWout = mhsa_args->bias_out->data;           //  Output Projection Biases
    
    fp16 *attention_map = mhsa_args->attention_map->data;      //  Buffer saving the MHSA map before output projection
    fp16 *outData = mhsa_args->output->data;                   //  Output sequence (Transposed, E x L)
    fp16 *inputData = mhsa_args->input->data;                  //  Input vector (L x E)
    fp16 *inputDataBn = mhsa_args->input_bn->data;             //  Input vector bottlenecked (L x F)
    fp16 *temp = mhsa_args->temp_buffer;                       //  Support buffer used in the attention head loop
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;    //  Buffer containing the softmax results (necessary to save for backward pass)
    fp16 *maxes = mhsa_args->maxes;                            //  Buffer containing the row-wise maxes in the softmax process
    fp16 *sums = mhsa_args->sums;                              //  Buffer containing the row-wise exponential sums in the softmax process
    fp16 *qt = mhsa_args->q->data;                             //  Pointer to the first element of Q transposed
    fp16 *kt = mhsa_args->k->data;                             //  Pointer to the first element of K transposed
    fp16 *vt = mhsa_args->v->data;                             //  Pointer to the first element of V transposed
    int n_heads = mhsa_args->n_heads;                           //  Number of heads used for MHSA

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
    int E = mhsa_args->input->W;                                //  Input Sequence element size
    int F = mhsa_args->coeff_in_q->H;                           //  Hidden dimension of attention (N. Heads * Head dimension)

    //printf("\n~~~~~~~~~~~~~~~FORWARD PASS~~~~~~~~~~~~~~~\n\nPrinting the parameters: L-%d, E-%d, F-%d\n", L, E, F);

    int H = F / n_heads;                                        //  Head dimension
    float scaling = q_rsqrt_fp16((float) H);                    //  Scaling factor to avoid vanishing gradients
    //float scaling = 1/sqrt(H);

    // M1_q
    // (Wq)t * input_bn
    struct matMul_args_fp16 matMul_args1_q;
    matMul_args1_q.A = coeffDataWinQ;                              //  F x F
    matMul_args1_q.B = inputDataBn;                                //  L x F
    matMul_args1_q.C = qt;                                         //  F x L
    matMul_args1_q.N = mhsa_args->coeff_in_q->H;
    matMul_args1_q.K = mhsa_args->coeff_in_q->W;
    matMul_args1_q.M = L;
    matMul_args1_q.trans_B = 1;
    matMul_args1_q.bias = coeffBiasWinQ;
    matMul_args1_q.USE_BIASES = 1;
    matMul_args1_q.bias_transposed = 1;
    matMul_args1_q.bias_dim = F;
    
    tiled_matmul_mhsa_fp16(&matMul_args1_q, Tiled_mhsa_matmul_args, 1);

    // M1_k
    // (Wk)t * input_bn
    struct matMul_args_fp16 matMul_args1_k;
    matMul_args1_k.A = coeffDataWinK;                              //  F x F
    matMul_args1_k.B = inputDataBn;                                //  L x F
    matMul_args1_k.C = kt;                                         //  F x L
    matMul_args1_k.N = mhsa_args->coeff_in_k->H;
    matMul_args1_k.K = mhsa_args->coeff_in_k->W;
    matMul_args1_k.M = L;
    matMul_args1_k.trans_B = 1;
    matMul_args1_k.bias = coeffBiasWinK;
    matMul_args1_k.USE_BIASES = 1;
    matMul_args1_k.bias_transposed = 1;
    matMul_args1_k.bias_dim = F;

    tiled_matmul_mhsa_fp16(&matMul_args1_k, Tiled_mhsa_matmul_args, 1);
    
    //  Cycle on the different heads
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for (int i = 0; i < n_heads; i++) {
        //  T1
        struct transp_args_fp16 transp_args1;
        transp_args1.matrix = kt + L * i * H;
        transp_args1.transp_matrix = temp;
        transp_args1.N = H;
        transp_args1.M = L;
        
        tiled_transpose_mhsa_fp16(&transp_args1, Tiled_mhsa_matmul_args, 2);


        // M2
        // Multiply it with the i-th head's transposed Q chunk
        struct matMul_args_fp16 matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = qt + L * i * H;
        matMul_args2.C = softmax_buffer + i * L * L;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;
        matMul_args2.USE_BIASES = 0;
        
        tiled_matmul_mhsa_fp16(&matMul_args2, Tiled_mhsa_matmul_args, 0);


        // ================================================== OP 4 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = BUFF;
        s_m_args.scalar = scaling;
        s_m_args.dim = tile_dim;

        for(int k = 0; k < (L / tile_h); k++){
            for(int j = 0; j < (L / tile_w); j++){
                pi_cl_dma_cmd_2d((uint32_t) ((softmax_buffer + i * L * L) + k * L * tile_h + j * tile_w), (uint32_t) (BUFF), 2 * tile_dim, 2 * L, 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
                pi_cl_dma_cmd_wait(cmd_load);
                pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp16_cl, &s_m_args);
                pi_cl_dma_cmd_2d((uint32_t) ((softmax_buffer + i * L * L) + k * L * tile_h + j * tile_w), (uint32_t) (BUFF), 2 * tile_dim, 2 * L, 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
                pi_cl_dma_cmd_wait(cmd_store);
            }
        }

        // ================================================== OP 5 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (T2)
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //
        // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
        // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~

        //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
        //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
        //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
        // T2
        struct transp_args_fp16 transp_args2;
        transp_args2.matrix = softmax_buffer + i * L * L;
        transp_args2.transp_matrix = temp;
        transp_args2.N = L;
        transp_args2.M = L;

        tiled_transpose_mhsa_fp16(&transp_args2, Tiled_mhsa_matmul_args, 0);

        //  Softmax algorithm
        struct softmax_args_fp16 softmax_arg;
        softmax_arg.input_data = temp;
        softmax_arg.output_data = softmax_buffer + i * L * L;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;
        softmax_arg.H = L;
        softmax_arg.W = L;
        
        pulp_softmax_fp16_fw_cl_tiled(&softmax_arg, Tiled_mhsa_matmul_args);
    }

    
    // M1_v
    // (Wv)t * input
    struct matMul_args_fp16 matMul_args1_v;
    matMul_args1_v.A = coeffDataWinV;                              //  F x E
    matMul_args1_v.B = inputData;                                  //  L x E
    matMul_args1_v.C = vt;                                         //  F x L
    matMul_args1_v.N = F;
    matMul_args1_v.K = E;
    matMul_args1_v.M = L;
    matMul_args1_v.trans_B = 1;
    matMul_args1_v.bias = coeffBiasWinV;
    matMul_args1_v.USE_BIASES = 1;
    matMul_args1_v.bias_transposed = 1;
    matMul_args1_v.bias_dim = F;
    
    tiled_matmul_mhsa_fp16(&matMul_args1_v, Tiled_mhsa_matmul_args, 1);

    
    for (int i = 0; i < n_heads; i++) {
        // ================================================== OP 6 ==================================================
        // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~     (T3)
        // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~

        // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~                     (M3)
        // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~

        // T3
        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
        struct transp_args_fp16 transp_args3;
        transp_args3.matrix = softmax_buffer + i * L * L;
        transp_args3.transp_matrix = temp;
        transp_args3.N = L;
        transp_args3.M = L;

        tiled_transpose_mhsa_fp16(&transp_args3, Tiled_mhsa_matmul_args, 0);

        // M3
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = vt + L * i * H;
        matMul_args3.B = temp;
        matMul_args3.C = attention_map + L * i * H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;
        
        tiled_matmul_mhsa_fp16(&matMul_args3, Tiled_mhsa_matmul_args, 2);
    }
    

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ================================================== OP 7 ==================================================
    // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> temp ~~~~~~~~~~~~~~~~~~~~~~       (M4)
    // ~~~~~~~~~~~~~~~~~~~~~~     F x F     @     F x L     ->  F x L  ~~~~~~~~~~~~~~~~~~~~~~

    // M4
    //  Final attention map projection
    struct matMul_args_fp16 matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = temp;
    matMul_args4.N = mhsa_args->coeff_out->H;
    matMul_args4.K = mhsa_args->coeff_out->W;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;
    matMul_args4.bias = coeffBiasWout;
    matMul_args4.USE_BIASES = 1;
    matMul_args4.bias_transposed = 1;
    matMul_args4.bias_dim = E;
    
    tiled_matmul_mhsa_fp16(&matMul_args4, Tiled_mhsa_matmul_args, 1);

    // T4
    // The last transpose to original shape
    struct transp_args_fp16 transp_args4;
    transp_args4.matrix = temp;
    transp_args4.transp_matrix = outData;
    transp_args4.N = mhsa_args->input_bn->W;
    transp_args4.M = L;

    tiled_transpose_mhsa_fp16(&transp_args4, Tiled_mhsa_matmul_args, 1);
}

// void tiled_mhsa_fp16_flash(void *Mhsa_args, void* Tiled_mhsa_matmul_args, fp16* BUFF_L2, pi_fs_file_t *file_p, int size){
//     // ======================================== DECLARATIONS ========================================
//     // BUFF_L2 = (fp16*) pi_l2_malloc(size * 2);
//     struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;
//     struct Tiled_Matmul_Mhsa_args_fp16 *tiled_args = (struct Tiled_Matmul_Mhsa_args_fp16 *) Tiled_mhsa_matmul_args;

//     int tile_h = tiled_args->tile_h;
//     int tile_w = tiled_args->tile_w;
//     int tile_dim = tiled_args->tile_dim;
//     fp16* BUFF = tiled_args->BUFF;
//     pi_cl_dma_cmd_t * cmd_store = tiled_args->cmd_store;
//     pi_cl_dma_cmd_t * cmd_load = tiled_args->cmd_load;

//     int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
//     int E = mhsa_args->input->W;                                //  Input Sequence element size
//     int F = mhsa_args->input_bn->W;                             //  Hidden dimension of attention (N. Heads * Head dimension)

//     pi_cl_fs_req_t req1;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->coeff_in_q->data, (void*) (BUFF_L2), (F * F * 2), 1, &req1);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req1);
//     fp16 *coeffDataWinQ = BUFF_L2; 

//     pi_cl_fs_req_t req2;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->coeff_in_k->data, (void*) (BUFF_L2 + F * F), (F * F * 2), 1, &req2);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req2);
//     fp16 *coeffDataWinK = BUFF_L2 + F * F; 

//     pi_cl_fs_req_t req3;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->coeff_in_v->data, (void*) (BUFF_L2 + F * F * 2), (F * E * 2), 1, &req3);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req3);
//     fp16 *coeffDataWinV = BUFF_L2 + F * F * 2; 

//     pi_cl_fs_req_t req4;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->bias_in_q->data, (void*) (BUFF_L2 + (F * F * 2) + (F * E)), (F * 2), 1, &req4);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req4);
//     fp16 *coeffBiasWinQ = BUFF_L2 + F * F * 2 + (F * E); 

//     pi_cl_fs_req_t req5;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->bias_in_k->data, (void*) (BUFF_L2 + (F * F * 2) + (F * E) + F), (F * 2), 1, &req5);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req5);
//     fp16 *coeffBiasWinK = BUFF_L2 + (F * F * 2) + (F * E) + F; 

//     pi_cl_fs_req_t req6;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->bias_in_v->data, (void*) (BUFF_L2 + (F * F * 2) + (F * E) + (F * 2)), (F * 2), 1, &req6);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req6);
//     fp16 *coeffBiasWinV = BUFF_L2 + (F * F * 2) + (F * E) + (F * 2); 

//     pi_cl_fs_req_t req7;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->coeff_out->data, (void*) (BUFF_L2 + (F * F * 2) + (F * E) + (F * 3)), (F * F * 2), 1, &req7);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req7);
//     fp16 *coeffDataWout = BUFF_L2 + (F * F * 2) + (F * E) + (F * 3);

//     pi_cl_fs_req_t req8;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->bias_out->data, (void*) (BUFF_L2 + (F * F * 3) + (F * E) + (F * 3)), (F * 2), 1, &req8);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req8);
//     fp16 *coeffBiasWout = BUFF_L2 + (F * F * 3) + (F * E) + (F * 3);

//     pi_cl_fs_req_t req9;
//     pi_cl_fs_copy(file_p, (uint32_t) mhsa_args->input->data, (void*) (BUFF_L2 + (F * F * 3) + (F * E) + (F * 4)), (L * E * 2), 1, &req9);
//     // pi_cl_fs_read(file_p, (void*) (W_DATA_L2), (args->M * args->K * 2), &req2);
//     pi_cl_fs_wait(&req9);
//     fp16 *inputData = BUFF_L2 + (F * F * 3) + (F * E) + (F * 4);

    
//     fp16 *attention_map = mhsa_args->attention_map->data;      //  Buffer saving the MHSA map before output projection
//     fp16 *outData = mhsa_args->output->data;                   //  Output sequence (Transposed, E x L)
//     fp16 *inputDataBn = mhsa_args->input_bn->data;             //  Input vector bottlenecked (L x F)
//     fp16 *temp = mhsa_args->temp_buffer;                       //  Support buffer used in the attention head loop
//     fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;    //  Buffer containing the softmax results (necessary to save for backward pass)
//     fp16 *maxes = mhsa_args->maxes;                            //  Buffer containing the row-wise maxes in the softmax process
//     fp16 *sums = mhsa_args->sums;                              //  Buffer containing the row-wise exponential sums in the softmax process
//     fp16 *qt = mhsa_args->q->data;                             //  Pointer to the first element of Q transposed
//     fp16 *kt = mhsa_args->k->data;                             //  Pointer to the first element of K transposed
//     fp16 *vt = mhsa_args->v->data;                             //  Pointer to the first element of V transposed
//     int n_heads = mhsa_args->n_heads;                          //  Number of heads used for MHSA

//     int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

//     //printf("\n~~~~~~~~~~~~~~~FORWARD PASS~~~~~~~~~~~~~~~\n\nPrinting the parameters: L-%d, E-%d, F-%d\n", L, E, F);

//     int H = F / n_heads;                                        //  Head dimension
//     float scaling = q_rsqrt_fp16((float) H);                    //  Scaling factor to avoid vanishing gradients
//     //float scaling = 1/sqrt(H);

//     // M1_q
//     // (Wq)t * input_bn
//     struct matMul_args_fp16 matMul_args1_q;
//     matMul_args1_q.A = coeffDataWinQ;                              //  F x F
//     matMul_args1_q.B = inputDataBn;                                //  L x F
//     matMul_args1_q.C = qt;                                         //  F x L
//     matMul_args1_q.N = F;
//     matMul_args1_q.K = F;
//     matMul_args1_q.M = L;
//     matMul_args1_q.trans_B = 1;
//     matMul_args1_q.bias = coeffBiasWinQ;
//     matMul_args1_q.USE_BIASES = 1;
//     matMul_args1_q.bias_transposed = 1;
//     matMul_args1_q.bias_dim = F;

//     tiled_matmul_mhsa_fp16(&matMul_args1_q, Tiled_mhsa_matmul_args);

//     // M1_k
//     // (Wk)t * input_bn
//     struct matMul_args_fp16 matMul_args1_k;
//     matMul_args1_k.A = coeffDataWinK;                              //  F x F
//     matMul_args1_k.B = inputDataBn;                                //  L x F
//     matMul_args1_k.C = kt;                                         //  F x L
//     matMul_args1_k.N = F;
//     matMul_args1_k.K = F;
//     matMul_args1_k.M = L;
//     matMul_args1_k.trans_B = 1;
//     matMul_args1_k.bias = coeffBiasWinK;
//     matMul_args1_k.USE_BIASES = 1;
//     matMul_args1_k.bias_transposed = 1;
//     matMul_args1_k.bias_dim = F;

//     tiled_matmul_mhsa_fp16(&matMul_args1_k, Tiled_mhsa_matmul_args);


//     //  Cycle on the different heads
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ F -> H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//     for (int i = 0; i < n_heads; i++) {
//         //  T1
//         struct transp_args_fp16 transp_args1;
//         transp_args1.matrix = kt + L * i * H;
//         transp_args1.transp_matrix = temp;
//         transp_args1.N = H;
//         transp_args1.M = L;

//         tiled_transpose_mhsa_fp16(&transp_args1, Tiled_mhsa_matmul_args);

//         // M2
//         // Multiply it with the i-th head's transposed Q chunk
//         struct matMul_args_fp16 matMul_args2;
//         matMul_args2.A = temp;
//         matMul_args2.B = qt + L * i * H;
//         matMul_args2.C = softmax_buffer + i * L * L;
//         matMul_args2.N = L;
//         matMul_args2.K = H;
//         matMul_args2.M = L;
//         matMul_args2.trans_B = 0;
//         matMul_args2.USE_BIASES = 0;

//         tiled_matmul_mhsa_fp16(&matMul_args2, Tiled_mhsa_matmul_args);


//         // ================================================== OP 4 ==================================================
//         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer *= scalar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~           L x L          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//         struct scalar_mul_args_fp16 s_m_args;
//         s_m_args.input = BUFF;
//         s_m_args.scalar = scaling;
//         s_m_args.dim = tile_dim;

//         for(int k = 0; k < L / tile_h; k++){
//             for(int j = 0; j < L / tile_w; j++){
//                 pi_cl_dma_cmd_2d((uint32_t) ((softmax_buffer + i * L * L) + k * L * tile_h + j * tile_w), (uint32_t) (BUFF), 2 * tile_dim, 2 * L, 2 * tile_w, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
//                 pi_cl_dma_cmd_wait(cmd_load);
//                 pi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp16_cl, &s_m_args);
//                 pi_cl_dma_cmd_2d((uint32_t) ((softmax_buffer + i * L * L) + k * L * tile_h + j * tile_w), (uint32_t) (BUFF), 2 * tile_dim, 2 * L, 2 * tile_w, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
//                 pi_cl_dma_cmd_wait(cmd_store);
//             }
//         }

//         // ================================================== OP 5 ==================================================
//         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer -T-> temp [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (T2)
//         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      L x L     -T->         L x L              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//         //
//         // ~~~~~~~~~~~~~~~~~~ temp [softmax_buffer ^ T] -SM-> softmax_buffer [softmax_buffer ^ T]  ~~~~~~~~~~~~~~~~~~
//         // ~~~~~~~~~~~~~~~~~~          L x L            -SM->               L x L                  ~~~~~~~~~~~~~~~~~~

//         //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
//         //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
//         //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
//         // T2
//         struct transp_args_fp16 transp_args2;
//         transp_args2.matrix = softmax_buffer + i * L * L;
//         transp_args2.transp_matrix = temp;
//         transp_args2.N = L;
//         transp_args2.M = L;

//         tiled_transpose_mhsa_fp16(&transp_args2, Tiled_mhsa_matmul_args);

//         //  Softmax algorithm
//         struct softmax_args_fp16 softmax_arg;
//         softmax_arg.input_data = temp;
//         softmax_arg.output_data = softmax_buffer + i * L * L;
//         softmax_arg.maxes = maxes;
//         softmax_arg.sums = sums;
//         softmax_arg.H = L;
//         softmax_arg.W = L;

//         pulp_softmax_fp16_fw_cl_tiled(&softmax_arg, Tiled_mhsa_matmul_args);
//     }


//     // M1_v
//     // (Wv)t * input
//     struct matMul_args_fp16 matMul_args1_v;
//     matMul_args1_v.A = coeffDataWinV;                              //  F x E
//     matMul_args1_v.B = inputData;                                  //  L x E
//     matMul_args1_v.C = vt;                                         //  F x L
//     matMul_args1_v.N = F;
//     matMul_args1_v.K = E;
//     matMul_args1_v.M = L;
//     matMul_args1_v.trans_B = 1;
//     matMul_args1_v.bias = coeffBiasWinV;
//     matMul_args1_v.USE_BIASES = 1;
//     matMul_args1_v.bias_transposed = 1;
//     matMul_args1_v.bias_dim = F;

//     tiled_matmul_mhsa_fp16(&matMul_args1_v, Tiled_mhsa_matmul_args);


//     for (int i = 0; i < n_heads; i++) {
//         // ================================================== OP 6 ==================================================
//         // ~~~~~~~~~~~~~~~~~~~~~~ softmax_buffer [softmax_buffer ^ T] -T-> temp [softmax_buffer] ~~~~~~~~~~~~~~~~~~~~~~     (T3)
//         // ~~~~~~~~~~~~~~~~~~~~~~               L x L                 -T->        L x L          ~~~~~~~~~~~~~~~~~~~~~~

//         // ~~~~~~~~~~~~~~~~~~~~~~   v   @ temp [softmax_buffer] -> attention_map ~~~~~~~~~~~~~~~~~~~~~~                     (M3)
//         // ~~~~~~~~~~~~~~~~~~~~~~ H x L @        L x L          ->     H x L     ~~~~~~~~~~~~~~~~~~~~~~

//         // T3
//         //  Each head result has to be appended to the full attention map, to do so we require to store the current
//         //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
//         struct transp_args_fp16 transp_args3;
//         transp_args3.matrix = softmax_buffer + i * L * L;
//         transp_args3.transp_matrix = temp;
//         transp_args3.N = L;
//         transp_args3.M = L;

//         tiled_transpose_mhsa_fp16(&transp_args3, Tiled_mhsa_matmul_args);

//         // M3
//         struct matMul_args_fp16 matMul_args3;
//         matMul_args3.A = vt + L * i * H;
//         matMul_args3.B = temp;
//         matMul_args3.C = attention_map + L * i * H;
//         matMul_args3.N = H;
//         matMul_args3.K = L;
//         matMul_args3.M = L;
//         matMul_args3.trans_B = 0;

//         tiled_matmul_mhsa_fp16(&matMul_args3, Tiled_mhsa_matmul_args);
//     }

//     // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ H -> F ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//     // ================================================== OP 7 ==================================================
//     // ~~~~~~~~~~~~~~~~~~~~~~ coeffDataWout @ attention_map -> temp ~~~~~~~~~~~~~~~~~~~~~~       (M4)
//     // ~~~~~~~~~~~~~~~~~~~~~~     F x F     @     F x L     ->  F x L  ~~~~~~~~~~~~~~~~~~~~~~

//     // M4
//     //  Final attention map projection
//     struct matMul_args_fp16 matMul_args4;
//     matMul_args4.A = coeffDataWout;
//     matMul_args4.B = attention_map;
//     matMul_args4.C = temp;
//     matMul_args4.N = F;
//     matMul_args4.K = F;
//     matMul_args4.M = L;
//     matMul_args4.trans_B = 0;
//     matMul_args4.bias = coeffBiasWout;
//     matMul_args4.USE_BIASES = 1;
//     matMul_args4.bias_transposed = 1;
//     matMul_args4.bias_dim = F;

//     tiled_matmul_mhsa_fp16(&matMul_args4, Tiled_mhsa_matmul_args);

//     // T4
//     // The last transpose to original shape
//     struct transp_args_fp16 transp_args4;
//     transp_args4.matrix = temp;
//     transp_args4.transp_matrix = outData;
//     transp_args4.N = F;
//     transp_args4.M = L;

//     tiled_transpose_mhsa_fp16(&transp_args4, Tiled_mhsa_matmul_args);
//     // pi_l2_free(BUFF_L2, size * 2);
// }




/*
// Instantiate zero_tensor support function
static inline void zero_tensor(void *args);


//FORWARD W/ DOUBLE BUFFERING L1 DMA
void pulp_mhsa_fp16_fw_cl_dblbuffer(void* Mhsa_args){
    struct Mhsa_args_fp16_db *mhsa_args_db = (struct Mhsa_args_fp16_db *) Mhsa_args;
    fp16 *coeffDataWin_l2 = mhsa_args_db->coeff_in->data;          //  Input Projection Weights in L2 (3F X E)
    fp16 *coeffDataWout = mhsa_args_db->coeff_out->data;           //  Output Projection Weights (Already transposed from GM)
    fp16 *attention_map = mhsa_args_db->attention_map->data;       //  Buffer saving the MHSA map before output projection
    fp16 *attention_map_l2 = mhsa_args_db->attention_map_l2->data;       //  Buffer saving the MHSA map before output projection
    fp16 *outData = mhsa_args_db->output->data;                    //  Output sequence (Transposed, E x L)
    fp16 *temp = mhsa_args_db->temp_buffer;                        //  Support buffer used in the attention head loop
    //float *head_buffer = mhsa_args->head_buffer->data;        //  Buffer containing the Q*Kt result (necessary to save for backward pass)
    fp16 *softmax_buffer = mhsa_args_db->softmax_buffer->data;     //  Buffer containing the softmax results (necessary to save for backward pass)
    fp16 *maxes = mhsa_args_db->maxes;                             //  Buffer containing the row-wise maxes in the softmax process
    fp16 *sums = mhsa_args_db->sums;                               //  Buffer containing the row-wise exponential sums in the softmax process
    fp16 *qkv_l2 = mhsa_args_db->qkv->data;                        //  Matrix in L2 containing the transposed Q, K and V (3*F x L)
    fp16 *q = mhsa_args_db->buff2_a->data;                        //  Pointer to the first element of Q
    fp16 *k = mhsa_args_db->buff2_a->data;                        //  Pointer to the first element of K
    fp16 *v = mhsa_args_db->buff2_a->data;                        //  Pointer to the first element of V
    fp16 *out = mhsa_args_db->attention_map_l2->data;
    int n_heads = mhsa_args_db->n_heads;                           //  Number of heads used for MHSA
    int n_tiles = mhsa_args_db->n_tiles;                           //  Number of tiles used for QKV projection

    fp16 *buff0 = mhsa_args_db->input->data;
    fp16 *buff1_a = mhsa_args_db->buff1_a->data;
    fp16 *buff2_a = mhsa_args_db->buff2_a->data;
    fp16 *buff1_b = mhsa_args_db->buff1_b->data;
    fp16 *buff2_b = mhsa_args_db->buff2_b->data;


    int opt_matmul_type = mhsa_args_db->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args_db->input->H;                                //  Input/Output Sequence length
    int E = mhsa_args_db->input->W;                                //  Input Sequence element size
    int F = mhsa_args_db->attention_map->W;                        //  Hidden dimension of attention (N. Heads * Head dimension)
    int TILE = mhsa_args_db->buff1_a->W;


    #ifdef DEBUG
    printf("\nPrinting the parameters: L-%d, E-%d, F-%d, TILE-%d, n_tiles-%d\n", L, E, F, TILE, n_tiles);
    #endif

    int H = F / n_heads;                                        //  Head dimension
    fp16 scaling = (fp16) (1/sqrt(H));                          //  Scaling factor to avoid vanishing gradients
    //float scaling = q_rsqrt_fp16((float)H);


    // PART 1: DOUBLE BUFFERING FOR QKV PROJECTION //

    // Set the current L1 buffer (0: L1A, 1: L1B)
    int curr_L1_buffer = 0;

    // Index for input transfers into L1
    int in_idx = 0;

    // Define the copy structures
    pi_cl_dma_copy_t copy_inA, copy_inB;
    pi_cl_dma_copy_t copy_outA, copy_outB;

    struct zero_tensor_args zero_args;
    zero_args.size = TILE*L;
    zero_args.zero_init = 0.0f;

    // Load first data into BUFFER 1.A in L1
    copy_inA.ext = (uint32_t) &coeffDataWin_l2[0];
    copy_inA.loc = (uint32_t) buff1_a;
    copy_inA.size = 2*TILE*E;
    copy_inA.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_inA.id = 0;
    copy_inA.merge = 0;
    pi_cl_dma_memcpy(&copy_inA);

    // Copy struct for BUFFER 1.B in L1
    copy_inB.loc = (uint32_t) buff1_b;
    copy_inB.size = 2*TILE*E;
    copy_inB.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_inB.id = 0;
    copy_inB.merge = 0;

    // Output buffer from BUFFER 2.A
    copy_outA.dir = PI_CL_DMA_DIR_LOC2EXT;
    copy_outA.merge = 0;
    copy_outA.size = 2*TILE*L;
    copy_outA.id = 0;
    copy_outA.loc = (uint32_t) buff2_a;

    // Output buffer from BUFFER 2.B
    copy_outB.dir = PI_CL_DMA_DIR_LOC2EXT;
    copy_outB.merge = 0;
    copy_outB.size = 2*TILE*L;
    copy_outB.id = 0;
    copy_outB.loc = (uint32_t) buff2_b;

    // Projecting input sequence into Q, K, V, one tile at a time
    struct matMul_args_fp16 matMul_args1;
    matMul_args1.B = buff0;                                 //  E x L
    matMul_args1.N = TILE;
    matMul_args1.K = E;
    matMul_args1.M = L;
    matMul_args1.trans_B = 0;

    //  Cycle on the tiles
    for(int tile = 0; tile < n_tiles; tile++){
        //printf("\nTILE %d\n", tile);
        //printf("\nWaiting for computation...");
        // Wait for input data & connect L1 data to computation
        if (curr_L1_buffer == 0) {
            pi_cl_dma_wait(&copy_inA);
            pi_cl_dma_wait(&copy_outA);
            matMul_args1.A = buff1_a;                       //  TILE x E Tile of weight matrix
            matMul_args1.C = buff2_a;                       //  TILE x L Q, K, V are saved contiguously, in the same matrix.
        }
        else {
            pi_cl_dma_wait(&copy_inB);
            pi_cl_dma_wait(&copy_outB);
            matMul_args1.A = buff1_b;                       //  TILE x E Tile of weight matrix
            matMul_args1.C = buff2_b;                       //  TILE x L Q, K, V are saved contiguously, in the same matrix.
        }

        //  printf("\nLoading next data...");
        //  Begin to load next data
        if (tile < n_tiles-1) {
          in_idx+=TILE*E;
          if (curr_L1_buffer == 0) {
            copy_inB.ext = (uint32_t) &coeffDataWin_l2[in_idx];
            pi_cl_dma_memcpy(&copy_inB);
          }
          else {
            copy_inA.ext = (uint32_t) &coeffDataWin_l2[in_idx];
            pi_cl_dma_memcpy(&copy_inA);
          }
        }

        //printf("\nNull output tensor...");
        // Null output tensor before computation
        if (curr_L1_buffer == 0) {
          //for (int i=0; i<L1_OUT_SIZ; i++)  L1A_out[i] = 0;
          zero_args.tensor = buff2_a;
          pi_cl_team_fork(NUM_CORES, zero_tensor, &zero_args);
        }
        else {
          //for (int i=0; i<L1_OUT_SIZ; i++)  L1B_out[i] = 0;
          zero_args.tensor = buff2_b;
          pi_cl_team_fork(NUM_CORES, zero_tensor, &zero_args);
        }

        // Compute MatMul for QKV projection
        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args1);
        #else
        struct mm_manager_args_fp16 man_args1;
        man_args1.mm_args = &matMul_args1;
        man_args1.layer_type = LAYER_LINEAR;
        man_args1.step_type = STEP_FW;
        man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1);
        #endif

        //printf("\nStore result...");
        // Store results into L2
        if (curr_L1_buffer == 0) {
          copy_outA.ext = (uint32_t) &qkv_l2[tile * (TILE * L)];
          pi_cl_dma_memcpy(&copy_outA);
        }
        else {
          copy_outB.ext = (uint32_t) &qkv_l2[tile * (TILE * L)];
          pi_cl_dma_memcpy(&copy_outB);
        }

        //printf("\nSwitch buffers...\n");
        // Switch buffer roles
        curr_L1_buffer = 1 - curr_L1_buffer;
    }

    // Select correct buffer to be waited
    curr_L1_buffer = 1 - curr_L1_buffer;
    //printf("\nWaiting last buffer...\n");
    //curr_L1_buffer != curr_L1_buffer;
    // Wait for the last transfer
    if (curr_L1_buffer == 0) {
        pi_cl_dma_wait(&copy_inA);
        pi_cl_dma_wait(&copy_outA);
    }
    else {
        pi_cl_dma_wait(&copy_inB);
        pi_cl_dma_wait(&copy_outB);
    }

    // PART 2: MULTI-HEAD SELF ATTENTION //

    // Reset the current L1 buffer (0: L1A, 1: L1B)
    curr_L1_buffer = 0;

    // Reset index for input transfers into L1
    in_idx = 0;

    zero_args.size = H * L;

    // Define the NEW 2D copy structures
    pi_cl_dma_copy_2d_t copy2d_inA, copy2d_inB;

    // Load first data into L1 buffer 1.A (QKV is 3F x L)
    copy2d_inA.ext = (uint32_t) qkv_l2;
    copy2d_inA.loc = (uint32_t) buff1_a;
    copy2d_inA.size = 2 * 3 * H * L;
    copy2d_inA.stride = 2 * ((F) * L);
    copy2d_inA.length = 2 * H * L;
    copy2d_inA.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy2d_inA.id = 0;
    copy2d_inA.merge = 0;
    pi_cl_dma_memcpy_2d(&copy2d_inA);

    // Copy struct for BUFFER 1.B in L1
    copy2d_inB.loc = (uint32_t) buff1_b;
    copy2d_inB.size = 2 * 3 * H * L;
    copy2d_inB.stride = 2 * ((F) * L);
    copy2d_inB.length = 2 * H * L;
    copy2d_inB.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy2d_inB.id = 0;
    copy2d_inB.merge = 0;

    // Output buffer from BUFFER 2.A
    copy_outA.dir = PI_CL_DMA_DIR_LOC2EXT;
    copy_outA.merge = 0;
    copy_outA.size = 2 * H * L;
    copy_outA.id = 0;
    copy_outA.loc = (uint32_t) buff2_a;

    // Output buffer from BUFFER 2.B
    copy_outB.dir = PI_CL_DMA_DIR_LOC2EXT;
    copy_outB.merge = 0;
    copy_outB.size = 2 * H * L;
    copy_outB.id = 0;
    copy_outB.loc = (uint32_t) buff2_b;

    //  Cycle on the different heads
    for(int i = 0; i < n_heads; i++){
        //printf("\nHEAD %d\n", i);
        //printf("\nWaiting for computation...");
        // Wait for input data & connect L1 data to computation
        if (curr_L1_buffer == 0) {
            pi_cl_dma_wait(&copy2d_inA);
            pi_cl_dma_wait(&copy_outA);
            q = buff1_a;
            k = buff1_a + (L * H);
            v = buff1_a + (L * 2 * H);
        }
        else {
            pi_cl_dma_wait(&copy2d_inB);
            pi_cl_dma_wait(&copy_outB);
            q = buff1_b;
            k = buff1_b + (L * H);
            v = buff1_b + (L * 2 * H);
        }

        //printf("\nLoading next data...");
        // Begin to load next data
        if (i < n_heads-1) {
          if (curr_L1_buffer == 0) {
            in_idx+=(H * L);
            copy2d_inB.ext = (uint32_t) &qkv_l2[in_idx];
            pi_cl_dma_memcpy_2d(&copy2d_inB);
          }
          else {
            in_idx+=(H * L);
            copy2d_inA.ext = (uint32_t) &qkv_l2[in_idx];
            pi_cl_dma_memcpy_2d(&copy2d_inA);
          }
        }

        //printf("\nNull output tensor...");
        // Null output tensor before computation
        if (curr_L1_buffer == 0) {
          //for (int i=0; i<L1_OUT_SIZ; i++)  L1A_out[i] = 0;
          zero_args.tensor = buff2_a;
          pi_cl_team_fork(NUM_CORES, zero_tensor, &zero_args);
        }
        else {
          //for (int i=0; i<L1_OUT_SIZ; i++)  L1B_out[i] = 0;
          zero_args.tensor = buff2_b;
          pi_cl_team_fork(NUM_CORES, zero_tensor, &zero_args);
        }

        //printf("\nExecute computation...");
        //  Transpose i-th head's K chunk
        struct transp_args_fp16 transp_args2;
        transp_args2.matrix = k;
        transp_args2.transp_matrix = temp;
        transp_args2.N = H;
        transp_args2.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args2);

        //  Multiply it with the i-th head's Q chunk
        struct matMul_args_fp16 matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = q;
        matMul_args2.C = buff0;
        matMul_args2.N = L;
        matMul_args2.K = H;
        matMul_args2.M = L;
        matMul_args2.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args2);
        #else
        struct mm_manager_args_fp16 man_args2;
        man_args2.mm_args = &matMul_args2;
        man_args2.layer_type = LAYER_LINEAR;
        man_args2.step_type = STEP_FW;
        man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args2);
        #endif

        //  Due to the fact that we multiplied K * Qt instead of Q * Kt like in the original MHSA model, the current
        //  head buffer is transposed. To achieve the best experimental accuracy, the Softmax algorithm requires to compute
        //  row-wise max and sums, therefore it is necessary to transpose_fp16 the current head buffer.
        struct transp_args_fp16 transp_args4;
        transp_args4.matrix = buff0;
        transp_args4.transp_matrix = temp;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args4);


        //  Scale the current head values by a factor proportional to the head dimension
        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = temp;
        s_m_args.scalar = (fp16) scaling;
        s_m_args.dim = L*L;

        pi_cl_team_fork(NUM_CORES,  pulp_scalar_mul_fp16_cl, &s_m_args);

        //  Softmax operation
        struct softmax_args_fp16 softmax_arg;
        struct blob_fp16 input;
        struct blob_fp16 output;
        input.data = temp;
        input.dim = L;
        output.data = buff0;
        softmax_arg.input = &input;
        softmax_arg.output = &output;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;

        pulp_softmax_fp16_fw_cl(&softmax_arg);

        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose_fp16 the memory buffer again.
        struct transp_args_fp16 transp_args5;
        transp_args5.matrix = buff0;
        transp_args5.transp_matrix = temp;
        transp_args5.N = L;
        transp_args5.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args5);

        struct matMul_args_fp16 matMul_args3;
        if (curr_L1_buffer == 0) {
            //  Multiply softmax result with the i-th head's Vt chunk
            matMul_args3.A = v;
            matMul_args3.B = temp;
            matMul_args3.C = buff2_a;
            matMul_args3.N = H;
            matMul_args3.K = L;
            matMul_args3.M = L;
            matMul_args3.trans_B = 0;
        }
        else{
            //  Multiply softmax result with the i-th head's Vt chunk
            matMul_args3.A = v;
            matMul_args3.B = temp;
            matMul_args3.C = buff2_b;
            matMul_args3.N = H;
            matMul_args3.K = L;
            matMul_args3.M = L;
            matMul_args3.trans_B = 0;
        }

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args3);
        #else
        struct mm_manager_args_fp16 man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args3);
        #endif

        //printf("\nStore result...");
        // Store results into L2
        if (curr_L1_buffer == 0) {
          copy_outA.ext = (uint32_t) &attention_map_l2[i * (H * L)];
          pi_cl_dma_memcpy(&copy_outA);
        }
        else {
          copy_outB.ext = (uint32_t) &attention_map_l2[i * (H * L)];
          pi_cl_dma_memcpy(&copy_outB);
        }

        //printf("\nSwitch buffers...\n");
        // Switch buffer roles
        curr_L1_buffer = 1 - curr_L1_buffer;
    }

    // Select correct buffer to be waited
    curr_L1_buffer = 1 - curr_L1_buffer;
    //curr_L1_buffer != curr_L1_buffer;
    // Wait for the last transfer
    if (curr_L1_buffer == 0) {
        pi_cl_dma_wait(&copy2d_inA);
        pi_cl_dma_wait(&copy_outA);
    }
    else {
        pi_cl_dma_wait(&copy2d_inB);
        pi_cl_dma_wait(&copy_outB);
    }

    // DA RIMUOVERE: metto tutto attention map in l1
    pi_cl_dma_copy_t copy_attmap;

    copy_attmap.ext = (uint32_t) attention_map_l2;
    copy_attmap.loc = (uint32_t) attention_map;
    copy_attmap.size = 2*F*L;
    copy_attmap.dir = PI_CL_DMA_DIR_EXT2LOC;
    copy_attmap.id = 0;
    copy_attmap.merge = 0;
    pi_cl_dma_memcpy(&copy_attmap);
    pi_cl_dma_wait(&copy_attmap);

    //  Final attention map projection
    struct matMul_args_fp16 matMul_args4;
    matMul_args4.A = coeffDataWout;
    matMul_args4.B = attention_map;
    matMul_args4.C = outData;
    matMul_args4.N = E;
    matMul_args4.K = F;
    matMul_args4.M = L;
    matMul_args4.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args4);
    #else
    struct mm_manager_args_fp16 man_args4;
    man_args4.mm_args = &matMul_args4;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args4);
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


 static inline void zero_tensor(void * args) {

  struct zero_tensor_args * zero_args = (struct zero_tensor_args *) args;

  fp16 * __restrict__ tensor = zero_args->tensor;
  int size = zero_args->size;

  int blockSize = (size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > size ? size : start+blockSize;

  for (int i=start; i<stop; i++) {
    tensor[i] = zero_args->zero_init;
  }
}
*/
