#include "pulp_train.h"

#include "stats.h"
#include "net.h"

// Tensors
#if BITS == 32
PI_L2 float tensor_L2[TENSOR_SIZE];
PI_L1 float tensor_L1[TENSOR_SIZE];
PI_L2 float tensor_L2_back[TENSOR_SIZE];
#elif BITS == 16
PI_L2 fp16 tensor_L2[TENSOR_SIZE];
PI_L1 fp16 tensor_L1[TENSOR_SIZE];
PI_L2 fp16 tensor_L2_back[TENSOR_SIZE];
#endif


// Other functions
static inline void tensor_init(){
    for (int i=0; i<TENSOR_SIZE; i++)   tensor_L2[i] = i;
}


// Transfer data
static inline void transfer_data() 
{
    // DMA Copy structures
    pi_cl_dma_copy_t dma_data;

    // Load first data into L1A
    dma_data.dir = PI_CL_DMA_DIR_EXT2LOC;
    dma_data.merge = 0; 
    dma_data.size = (BITS/8)*TENSOR_SIZE;
    dma_data.id = pi_core_id();
    dma_data.ext = (uint32_t) tensor_L2;
    dma_data.loc = (uint32_t) &tensor_L1;
    pi_cl_dma_memcpy(&dma_data);    

    pi_cl_dma_wait(&dma_data);          
}

// Transfer back data
static inline void retransfer_data() 
{
    // DMA Copy structures
    pi_cl_dma_copy_t dma_data;

    // Load first data into L1A
    dma_data.dir = PI_CL_DMA_DIR_LOC2EXT;
    dma_data.merge = 0;
    dma_data.size = (BITS/8)*TENSOR_SIZE;
    dma_data.id = pi_core_id();
    dma_data.ext = (uint32_t) tensor_L2_back;
    dma_data.loc = (uint32_t) &tensor_L1;
    pi_cl_dma_memcpy(&dma_data);    

    pi_cl_dma_wait(&dma_data);          
}



// Transfer data
static inline void transfer_data_parallel() 
{
    // Internal parallelization scheme on sizes
    int blockSize = (TENSOR_SIZE+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > TENSOR_SIZE ? TENSOR_SIZE : start+blockSize;  

    #ifdef DEBUG_APP
    printf("blockSize=%d, start=%d, stop=%d\n", blockSize, start, stop);
    #endif

    // DMA Copy structures
    pi_cl_dma_copy_t dma_data;

    // Load first data into L1A
    dma_data.dir = PI_CL_DMA_DIR_EXT2LOC;
    dma_data.merge = 1;
    dma_data.size = (BITS/8)*blockSize;
    dma_data.id = pi_core_id();
    dma_data.ext = (uint32_t) (tensor_L2 + start);
    dma_data.loc = (uint32_t) &tensor_L1[start];
    pi_cl_dma_memcpy(&dma_data);    

    pi_cl_dma_wait(&dma_data);          
}

// Transfer back data
static inline void retransfer_data_parallel() 
{
    // Internal parallelization scheme on sizes
    int blockSize = (TENSOR_SIZE+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > TENSOR_SIZE ? TENSOR_SIZE : start+blockSize;  

    #ifdef DEBUG_APP
    printf("blockSize=%d, start=%d, stop=%d\n", blockSize, start, stop);
    #endif

    // DMA Copy structures
    pi_cl_dma_copy_t dma_data;

    // Load first data into L1A
    dma_data.dir = PI_CL_DMA_DIR_LOC2EXT;
    dma_data.merge = 1;
    dma_data.size = (BITS/8)*blockSize;
    dma_data.id = pi_core_id();
    dma_data.ext = (uint32_t) (tensor_L2_back + start);
    dma_data.loc = (uint32_t) &tensor_L1[start];
    pi_cl_dma_memcpy(&dma_data);    

    pi_cl_dma_wait(&dma_data);          
}


// Print tensors
static inline void print_data() 
{
    printf("\nL2 original tensor: \n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%HEIGHT*WIDTH))  printf("\n");
        printf("%f ", tensor_L2[i]);
    }
    printf("\n");

    printf("\nL1 transfered tensor: \n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%HEIGHT*WIDTH))  printf("\n");
        printf("%f ", tensor_L1[i]);
    }
    printf("\n");

    printf("\nL2 retransfered tensor: \n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%HEIGHT*WIDTH))  printf("\n");
        printf("%f ", tensor_L2_back[i]);
    }
    printf("\n");
}


// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    printf("\nHello, beginning L2 to L1 transfer (FP%d data)!\n", BITS);

    tensor_init();

    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifndef MERGE_PARALLEL
    transfer_data();
    #else
    pi_cl_team_fork(NUM_CORES, transfer_data_parallel, 0);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nHello, beginning L1 to L2 transfer (FP%d data)!\n", BITS);

    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifndef MERGE_PARALLEL
    retransfer_data();
    #else
    pi_cl_team_fork(NUM_CORES, retransfer_data_parallel, 0);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_OUTPUT
    print_data();
    #endif

    return;
}