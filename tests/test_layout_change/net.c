#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#define TENSOR_SIZE (C_t*H_t*W_t)

PI_L1 float data_fp32[TENSOR_SIZE];
PI_L1 float grad_fp32[TENSOR_SIZE];

PI_L1 fp16 data_fp16[TENSOR_SIZE];
PI_L1 fp16 grad_fp16[TENSOR_SIZE];

PI_L1 struct blob tensor;
PI_L1 struct blob_fp16 tensor_fp16;

// Initialize matrices
void init_matrices () 
{
    for (int i=0; i<TENSOR_SIZE; i++) 
    {
        data_fp32[i] = i;
        grad_fp32[i] = i/10;

        data_fp16[i] = i;
        grad_fp16[i] = i/10;
    }
}


// MAIN FUNCTION
void change_layout () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif   

    init_matrices();

    printf("Transposing data and grad from CHW to HWC (C=%d, H=%d, W=%d)\n", C_t, H_t, W_t);

    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("Transposing data and grad from HWC to CHW (C=%d, H=%d, W=%d)\n", C_t, H_t, W_t);

    #ifdef PROF_NET
    START_STATS();
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

}