/*
 * Copyright (C) 2023 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alberto Dequino (alberto.dequino@unibo.it)
 */


#include "pulp_mhsa_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_act_fp16.h"
#include <math.h>

// Instantiate zero_tensor support function
static inline void zero_tensor(void *args);

//FORWARD
void pulp_mhsa_fp16_fw_cl(void* Mhsa_args){
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;
    fp16 *coeffDataWin = mhsa_args->coeff_in->data;             //  Input Projection Weights
    fp16 *coeffDataWout = mhsa_args->coeff_out->data;           //  Output Projection Weights (Already transposed from GM)
    fp16 *attention_map = mhsa_args->attention_map->data;       //  Buffer saving the MHSA map before output projection
    fp16 *outData = mhsa_args->output->data;                    //  Output sequence (Transposed, E x L)
    fp16 *inputData = mhsa_args->input->data;                   //  Input vector (Transposed, E x L)
    fp16 *temp = mhsa_args->temp_buffer;                        //  Support buffer used in the attention head loop
    //float *head_buffer = mhsa_args->head_buffer->data;        //  Buffer containing the Q*Kt result (necessary to save for backward pass)
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;     //  Buffer containing the softmax results (necessary to save for backward pass)
    fp16 *maxes = mhsa_args->maxes;                             //  Buffer containing the row-wise maxes in the softmax process
    fp16 *sums = mhsa_args->sums;                               //  Buffer containing the row-wise exponential sums in the softmax process
    fp16 *qkv = mhsa_args->qkv->data;                           //  Matrix containing the transposed Q, K and V (3*F x L)
    fp16 *q = mhsa_args->qkv->data;                             //  Pointer to the first element of Q
    fp16 *k = mhsa_args->qkv->data;                             //  Pointer to the first element of K 
    fp16 *v = mhsa_args->qkv->data;                             //  Pointer to the first element of V
    fp16 *out;
    int n_heads = mhsa_args->n_heads;                           //  Number of heads used for MHSA

    int opt_matmul_type = mhsa_args->opt_matmul_type_fw;        //  Matmul type used

    int L = mhsa_args->input->H;                                //  Input/Output Sequence length    
    int E = mhsa_args->input->W;                                //  Input Sequence element size
    int F = mhsa_args->attention_map->W;                        //  Hidden dimension of attention (N. Heads * Head dimension)

    #ifdef DEBUG
    printf("\nPrinting the parameters: L-%d, E-%d, F-%d", L, E, F);
    #endif

    int H = F / n_heads;                                        //  Head dimension
    float scaling = (1/sqrt(H));                        //  Scaling factor to avoid vanishing gradients
    //float scaling = q_rsqrt_fp16((float)H);

    // Projecting input sequence into Q, K, V
    struct matMul_args_fp16 matMul_args1;
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
    pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args1);
    #else
    struct mm_manager_args_fp16 man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1);
    #endif

    #ifdef DEBUG
    printf("\nQKV Data: %d %d\n", 3*F, L);
    for (int j=0; j<L*3*F; j++){
        if(!(j%(L))) printf("\n");
        printf("%.8f ", matMul_args1.C[j]);
    }
    printf("\n");
    #endif

    // Separate Q, K and V entry points in the QKV matrix. Q, K and V are F x L vectors.
    q = qkv;
    k = qkv + L*F;
    v = qkv + L*2*F;

    /*
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    unsigned long _cycles = 0; 
    int id = 0;
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    */



    //  Cycle on the different heads
    for(int i = 0; i < n_heads; i++){
        /*
        //  Initialize the pointers to the correct starting positions for the i-th head
        float* current_head_buffer = head_buffer + i*L*L;
        float* current_softmax_buffer = softmax_buffer + i*L*L;
        */

        //  Transpose i-th head's K chunk
        struct transp_args_fp16 transp_args2;
        transp_args2.matrix = k + L*i*H;
        transp_args2.transp_matrix = temp;
        transp_args2.N = H;
        transp_args2.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args2);

        //  Multiply it with the i-th head's Q chunk
        struct matMul_args_fp16 matMul_args2;
        matMul_args2.A = temp;
        matMul_args2.B = q + L*i*H;
        matMul_args2.C = softmax_buffer;
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
        //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
        struct transp_args_fp16 transp_args4;
        transp_args4.matrix = softmax_buffer;
        transp_args4.transp_matrix = temp;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args4);

        #ifdef DEBUG
        printf("\nCurrent head buffer Data: %d %d\n", L, L);
        for (int j=0; j<L*L; j++){
            if(!(j%(L))) printf("\n");
            printf("%.8f ", temp[j]);
        }
        printf("\n");
        #endif

        /*
        struct copy_args copy_args3;
        copy_args3.from = temp;
        copy_args3.to = current_head_buffer;
        copy_args3.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args3);
        */
        
        /*
        if(i==(n_heads-1)){
            printf("\nCurrent head buffer Data: %d %d\n", L, L);
            for (int j=0; j<L*L; j++){
                if(!(j%(L))) printf("\n\n");
                printf("%.8f ", temp[j]);
            }
            printf("\n");
            exit(1);
        }*/

        //  Scale the current head values by a factor proportional to the head dimension
        struct scalar_mul_args_fp16 s_m_args;
        s_m_args.input = temp;
        s_m_args.scalar = (fp16) scaling;
        s_m_args.dim = L*L;

        pi_cl_team_fork(NUM_CORES,  pulp_scalar_mul_fp16_cl, &s_m_args);

        

        //  Softmax algorithm
        struct softmax_args_fp16 softmax_arg;
        struct blob_fp16 input;
        struct blob_fp16 output;
        input.data = temp;
        input.dim = L;
        output.data = softmax_buffer;
        softmax_arg.input = &input;
        softmax_arg.output = &output;
        softmax_arg.maxes = maxes;
        softmax_arg.sums = sums;

        /*
        ///------------------------------------------------------------------///
        //printf("\nSoftmax stats\n");
        pi_perf_conf((1<<PI_PERF_CYCLES)); 


        pi_perf_stop();
        pi_perf_reset(); 
        pi_perf_start();
        */

        pulp_softmax_fp16_fw_cl(&softmax_arg);
        //pulp_partial_softmax_simple_fp32_fw_cl(&softmax_arg);

        /*
        pi_perf_stop(); 
        _cycles   += pi_perf_read (PI_PERF_CYCLES); 
        id = pi_core_id(); 
        ///----------------------------------------------------------------///
        */

        //  Each head result has to be appended to the full attention map, to do so we require to store the current
        //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
        struct transp_args_fp16 transp_args5;
        transp_args5.matrix = softmax_buffer;
        transp_args5.transp_matrix = temp;
        transp_args5.N = L;
        transp_args5.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args5);

        /*
        struct copy_args copy_args4;
        copy_args4.from = temp;
        copy_args4.to = current_softmax_buffer;
        copy_args4.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy, &copy_args4);
        */

        //  Multiply softmax result with the i-th head's Vt chunk
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = v + L*i*H;
        matMul_args3.B = temp;
        matMul_args3.C = attention_map + L*i*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

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
    }

    #ifdef DEBUG
    printf("\nAttention results: %d %d\n", F, L);
    printf("\n\n");
    for(int i=0; i<F*L; i++){
        if(!(i%L))
            printf("\n");
        printf("%.8f ", attention_map[i]);
    } 
    printf("\n");
    #endif

    /*
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    printf("\n"); 
    printf("[%d] TOTAL softmax cycles = %lu\n", id, _cycles); 
    ///////////////////////////DELETE THIS////////////////////////////////////////////////////
    */

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
        //  row-wise max and sums, therefore it is necessary to transpose the current head buffer.
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
        //  softmax buffer data following the H x L convention, therefore we need to transpose the memory buffer again.
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





//BACKWARD
void pulp_mhsa_fp16_bw_cl(void * Mhsa_args) {
    struct Mhsa_args_fp16 *mhsa_args = (struct Mhsa_args_fp16 *) Mhsa_args;

    fp16 *coeffDataWin = mhsa_args->coeff_in->data; // E x 3F
    fp16 *coeffDataWout = mhsa_args->coeff_out->data; // E x F
    fp16 *inData = mhsa_args->input->data; // L x E
    fp16 *temp = mhsa_args->temp_buffer; // Temporary buffer to save transposed matrices // TODO: THIS HAS TO BE DYNAMIC (calculate the max capacity required)
    fp16 *grad = mhsa_args->grad; // L x L
    fp16 *outData = mhsa_args->output->data; // L x E
    fp16 *attention_map = mhsa_args->attention_map->data; // F x L
    fp16 *diff_attention_map = mhsa_args->attention_map->diff; // F x L
    fp16 *coeffDiffWin = mhsa_args->coeff_in->diff; // E x 3F
    fp16 *coeffDiffWout = mhsa_args->coeff_out->diff; // E x F
    fp16 *head_buffer = mhsa_args->head_buffer->data; // L x L
    fp16 *softmax_buffer = mhsa_args->softmax_buffer->data;

    int total_dim = mhsa_args->output->dim;
    int L = mhsa_args->input->H; // Input sequence length
    int E = mhsa_args->input->W; // Input sequence element size
    int F = mhsa_args->attention_map->W; // Attention block hidden size
    int n_heads = mhsa_args->n_heads; // Number of heads of the mhsa
    int H = F / n_heads;
    int opt_matmul_type = mhsa_args->opt_matmul_type_wg;

    fp16 *q = mhsa_args->qkv->data; // 3F x L
    fp16 *k = mhsa_args->qkv->data + F*L;
    fp16 *v = mhsa_args->qkv->data + 2*F*L;

    fp16 *q_diff = mhsa_args->qkv->diff; // 3F x L
    fp16 *k_diff = mhsa_args->qkv->diff + F*L;
    fp16 *v_diff = mhsa_args->qkv->diff + 2*F*L;


    fp16 *outDiff = mhsa_args->output->diff; // L x E
    fp16 *inDiff = mhsa_args->input->diff; // L x E
    fp16 *attention_map_diff = mhsa_args->attention_map->diff; // F x L
    fp16 *head_buffer_diff = mhsa_args->head_buffer->diff; // L x L

    
    
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
    struct matMul_args_fp16 matMul_args1;
    matMul_args1.A = outDiff; 
    matMul_args1.B = coeffDataWout; 
    matMul_args1.C = attention_map_diff;
    matMul_args1.N = L;
    matMul_args1.K = E;
    matMul_args1.M = F;
    matMul_args1.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args1); // Gradient of attention map: (L x E)*(E x F) - > (L x F)
    #else
    struct mm_manager_args_fp16 man_args1;
    man_args1.mm_args = &matMul_args1;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args1);
    #endif

    //Transpose map gradients (copy required because transpose can't be done inplace)
    struct transp_args_fp16 transp_args1;
    transp_args1.matrix = attention_map_diff;
    transp_args1.transp_matrix = temp;
    transp_args1.N = L;
    transp_args1.M = F;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args1);

    struct copy_args_fp16 copy_args1;
    copy_args1.from = temp;
    copy_args1.to = attention_map_diff;
    copy_args1.size = F*L;

    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args1); // Transposed gradient of attention map: (L x F) - > (F x L)

    // Output Projection Weights


    // matmul setup 2
    struct matMul_args_fp16 matMul_args2;
    matMul_args2.A = attention_map; 
    matMul_args2.B = outDiff; 
    matMul_args2.C = coeffDiffWout;
    matMul_args2.N = F;
    matMul_args2.K = L;
    matMul_args2.M = E;
    matMul_args2.trans_B = 0;


    #ifdef DEBUG
    printf("\ngrad\n");
    for (int i=0; i<total_dim; i++){
        if(!(i%E)) printf("\n");
        printf("%4.2e  ", outDiff[i]);
    }
    printf("\n");

    printf("\nTransposed Attention map\n");
    for (int i=0; i<F*L; i++){
        if(!(i%L)) printf("\n");
        printf("%4.2e  ", attention_map[i]);
    }
    printf("\n");
    #endif

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args2); // Output weight gradient: (F x L)*(L x E) - > (F x E)
    #else
    struct mm_manager_args_fp16 man_args2;
    man_args2.mm_args = &matMul_args2;
    man_args2.layer_type = LAYER_LINEAR;
    man_args2.step_type = STEP_FW;
    man_args2.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args2);
    #endif


    // Transpose Linear Output Weight gradients
    struct transp_args_fp16 transp_args9;
    transp_args9.matrix = coeffDiffWout;
    transp_args9.transp_matrix = temp;
    transp_args9.N = F;
    transp_args9.M = E;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args9);

    struct copy_args_fp16 copy_args9;
    copy_args9.from = temp;
    copy_args9.to = coeffDiffWout;
    copy_args9.size = E*F;

    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args9); // Transposed output weight gradient: (F x E) - > (E x F)


    #ifdef DEBUG
    printf("\nLinear coeffDiffWout");
    for (int i=0; i<E*F; i++){
      if(!(i%F)) printf("\n");
      printf("%4.2e (i=%d)", coeffDiffWout[i], i);
    }
    printf("\n");
    #endif

    // Cycle on the heads
    for(int i=0; i<n_heads; i++){
        // I-th head Value Gradient

        // matmul setup 3
        struct matMul_args_fp16 matMul_args3;
        matMul_args3.A = attention_map_diff + i*L*H; 
        matMul_args3.B = softmax_buffer + i*L*L; 
        matMul_args3.C = v_diff + i*L*H;
        matMul_args3.N = H;
        matMul_args3.K = L;
        matMul_args3.M = L;
        matMul_args3.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES,  mm_fp16, &matMul_args3); // i-th head Value gradient: (H x L)*(L x L) - > (H x L)
        #else
        struct mm_manager_args_fp16 man_args3;
        man_args3.mm_args = &matMul_args3;
        man_args3.layer_type = LAYER_LINEAR;
        man_args3.step_type = STEP_FW;
        man_args3.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args3);
        #endif


        // I-th head Buffer Gradient

        // Transpose output gradients
        struct transp_args_fp16 transp_args3;
        transp_args3.matrix = attention_map_diff + i*L*H;
        transp_args3.transp_matrix = temp;
        transp_args3.N = H;
        transp_args3.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args3); // Transpose i-th head attention map gradient: (H x L) - > (L x H) 


        // matmul setup 4
        struct matMul_args_fp16 matMul_args4;
        matMul_args4.A = temp; 
        matMul_args4.B = v + i*L*H; 
        matMul_args4.C = head_buffer_diff + i*L*L;
        matMul_args4.N = L;
        matMul_args4.K = H;
        matMul_args4.M = L;
        matMul_args4.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args4); // i-th head Buffer gradient: (L x H)*(H x L) - > (L x L)
        #else
        struct mm_manager_args_fp16 man_args4;
        man_args4.mm_args = &matMul_args4;
        man_args4.layer_type = LAYER_LINEAR;
        man_args4.step_type = STEP_FW;
        man_args4.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args4);
        #endif


        struct act_args_fp16 softmax_arg;
        struct blob_fp16 input;
        struct blob_fp16 output;
        input.diff = grad;
        input.dim = L*L;
        output.data = softmax_buffer + i*L*L;
        output.diff = head_buffer_diff + i*L*L;
        output.dim = i;
        softmax_arg.input = &input;
        softmax_arg.output = &output;
        // Back propagation of i-th head Buffer gradient through the softmax operation

        
        pi_cl_team_fork(1, pulp_softmax_fp16_bw_cl, &softmax_arg);




        // I-th head Query Gradient
        
        // Transpose softmax gradients
        struct transp_args_fp16 transp_args4;
        transp_args4.matrix = grad;
        transp_args4.transp_matrix = temp;
        transp_args4.N = L;
        transp_args4.M = L;

        pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args4); 

        struct copy_args_fp16 copy_args3;
        copy_args3.from = temp;
        copy_args3.to = grad;
        copy_args3.size = L*L;

        pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args3); // Still (L x L). Unsure if it is needed.

        // matmul setup 5
        struct matMul_args_fp16 matMul_args5;
        matMul_args5.A = k + i*L*H; 
        matMul_args5.B = grad; 
        matMul_args5.C = q_diff + i*L*H;
        matMul_args5.N = H;
        matMul_args5.K = L;
        matMul_args5.M = L;
        matMul_args5.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args5); // i-th head Query gradient: (H x L)*(L x L) - > (H x L)
        #else
        struct mm_manager_args_fp16 man_args5;
        man_args5.mm_args = &matMul_args5;
        man_args5.layer_type = LAYER_LINEAR;
        man_args5.step_type = STEP_FW;
        man_args5.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args5);
        #endif


        // I-th head Key Gradients

        // matmul setup 6
        struct matMul_args_fp16 matMul_args6;
        matMul_args6.A = q + i*L*H; 
        matMul_args6.B = grad; 
        matMul_args6.C = k_diff + i*L*H;
        matMul_args6.N = H;
        matMul_args6.K = L;
        matMul_args6.M = L;
        matMul_args6.trans_B = 0;

        #ifndef OPTIMIZE
        pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args6); // i-th head Key gradient: (H x L)*(L x L) - > (H x L)
        #else
        struct mm_manager_args_fp16 man_args6;
        man_args6.mm_args = &matMul_args6;
        man_args6.layer_type = LAYER_LINEAR;
        man_args6.step_type = STEP_FW;
        man_args6.matmul_type = opt_matmul_type; //MATMUL_TYPE
        pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args6);
        #endif
    }

    // Input projection Gradients
    
    // matmul setup 7
    struct matMul_args_fp16 matMul_args7;
    matMul_args7.A = q_diff; 
    matMul_args7.B = inData; 
    matMul_args7.C = coeffDiffWin;
    matMul_args7.N = 3*F;
    matMul_args7.K = L;
    matMul_args7.M = E;
    matMul_args7.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args7); // Input weight gradient: (3F x L)*(L x E) - > (3F x E)
    #else
    struct mm_manager_args_fp16 man_args7;
    man_args7.mm_args = &matMul_args7;
    man_args7.layer_type = LAYER_LINEAR;
    man_args7.step_type = STEP_FW;
    man_args7.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args7);
    #endif

    // Transpose input weight gradients
    struct transp_args_fp16 transp_args5;
    transp_args5.matrix = coeffDiffWin;
    transp_args5.transp_matrix = temp;
    transp_args5.N = 3*F;
    transp_args5.M = E;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args5); 

    struct copy_args_fp16 copy_args4;
    copy_args4.from = temp;
    copy_args4.to = coeffDiffWin;
    copy_args4.size = E*3*F;

    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args4); // Transpose input weight gradient: (3F x E) - > (E x 3F)




    // Input Gradients

    // matmul setup 8
    struct matMul_args_fp16 matMul_args8;
    matMul_args8.A = coeffDataWin; 
    matMul_args8.B = q_diff; 
    matMul_args8.C = inDiff;
    matMul_args8.N = E;
    matMul_args8.K = 3*F;
    matMul_args8.M = L;
    matMul_args8.trans_B = 0;

    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args8); // Input gradients: (E x 3F)*(3F x L) - > (E x L)
    #else
    struct mm_manager_args_fp16 man_args8;
    man_args8.mm_args = &matMul_args8;
    man_args8.layer_type = LAYER_LINEAR;
    man_args8.step_type = STEP_FW;
    man_args8.matmul_type = opt_matmul_type; //MATMUL_TYPE
    pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args8);
    #endif

    // Transpose input weight gradients
    struct transp_args_fp16 transp_args6;
    transp_args6.matrix = inDiff;
    transp_args6.transp_matrix = temp;
    transp_args6.N = E;
    transp_args6.M = L;

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args6); 

    struct copy_args_fp16 copy_args5;
    copy_args5.from = temp;
    copy_args5.to = inDiff;
    copy_args5.size = L*E;

    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args5); // Input gradients transpose: (E x L) - > (L x E)
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
  



