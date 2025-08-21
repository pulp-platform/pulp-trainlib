/*
 * Copyright (C) 2024 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Alberto Dequino (alberto.dequino@unibo.it)
 */

 #include "pulp_train_utils_fp16.h"
 #include "pulp_embedding_fp16.h"
 #include "pulp_train_defines.h"
 #include "pmsis.h"

 // FORWARD, TILED
 
 void embedding_fw_tiled_fp16(void *embedding_args){
    struct Embedding_args_fp16 *args = (struct Embedding_args_fp16*) embedding_args;

    fp16 *BUFF = args->BUFF;

    int dim = args->dim;
    int embed_dim = args->embed_dim;

    pi_cl_dma_cmd_t * cmd_store;
    pi_cl_dma_cmd_t * cmd_load;

    const int blockSize=(dim+NUM_CORES-1)/NUM_CORES;
    const int start = pi_core_id()*blockSize;
    const int stop = start + blockSize > dim ? dim : start+blockSize;

    for(int i = start; i < stop; i++){
        int id = (args->ids)[i];
        pi_cl_dma_cmd((uint32_t) (args->embeds + id * embed_dim), (uint32_t) (BUFF + (int) (pi_core_id()) * embed_dim), 2 * embed_dim, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
        pi_cl_dma_cmd_wait(cmd_load);
        pi_cl_dma_cmd((uint32_t) (args->out + i * embed_dim), (uint32_t) (BUFF + (int) (pi_core_id()) * embed_dim), 2 * embed_dim, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
        pi_cl_dma_cmd_wait(cmd_store);
    }
 }