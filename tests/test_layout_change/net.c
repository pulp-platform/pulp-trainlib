#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#define TENSOR_SIZE (T_C*T_H*T_W)

PI_L1 float data_fp32[TENSOR_SIZE];
PI_L1 float grad_fp32[TENSOR_SIZE];

PI_L1 fp16 data_fp16[TENSOR_SIZE];
PI_L1 fp16 grad_fp16[TENSOR_SIZE];

PI_L1 float transp_buffer[TENSOR_SIZE];

PI_L1 struct blob tensor;
PI_L1 struct blob_fp16 tensor_fp16;

PI_L1 struct layout_args args;
PI_L1 struct layout_args_fp16 args_fp16;


// Initialize matrices
void init_data () 
{
    tensor.data = data_fp32;
    tensor.diff = grad_fp32;
    tensor.C = T_C;
    tensor.H = T_H;
    tensor.W = T_W;

    tensor_fp16.data = data_fp16;
    tensor_fp16.diff = grad_fp16;
    tensor_fp16.C = T_C;
    tensor_fp16.H = T_H;
    tensor_fp16.W = T_W;

    for (int i=0; i<TENSOR_SIZE; i++) 
    {
        data_fp32[i] = i;
        grad_fp32[i] = (float) i/10;

        data_fp16[i] = i;
        grad_fp16[i] = (float) i/10;
    }

    args.tensor = &tensor;
    args.transp_buffer = transp_buffer;
    args.transpose_data = 1;
    args.transpose_data = 1;

    args_fp16.tensor = &tensor_fp16;
    args_fp16.transp_buffer = (fp16 *) transp_buffer;
    args_fp16.transpose_data = 1;
    args_fp16.transpose_grad = 1;
}


// Print CHW data
void print_CHW () {
    // FP32 data
    printf("\nCHW FP32 DATA:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_H*T_W)) printf("\n");
        printf("%f ", data_fp32[i]);
    }
    printf("\n");
    // FP32 grad
    printf("\nCHW FP32 GRAD:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_H*T_W)) printf("\n");
        printf("%f ", grad_fp32[i]);
    }
    printf("\n");
    // FP16 data
    printf("\nCHW FP16 DATA:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_H*T_W)) printf("\n");
        printf("%f ", data_fp16[i]);
    }
    printf("\n");
    // FP16 grad
    printf("\nCHW FP16 GRAD:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_H*T_W)) printf("\n");
        printf("%f ", grad_fp16[i]);        
    }
    printf("\n");
}


// Print HWC data
void print_HWC () {
    // FP32 data
    printf("\nHWC FP32 DATA:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_C)) printf("\n");
        printf("%f ", data_fp32[i]);
    }
    printf("\n");
    // FP32 grad
    printf("\nHWC FP32 GRAD:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_C)) printf("\n");
        printf("%f ", grad_fp32[i]);
    }
    printf("\n");
    // FP16 data
    printf("\nHWC FP16 DATA:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_C)) printf("\n");
        printf("%f ", data_fp16[i]);
    }
    printf("\n");
    // FP16 grad
    printf("\nHWC FP16 GRAD:\n");
    for (int i=0; i<TENSOR_SIZE; i++) {
        if (!(i%T_C)) printf("\n");
        printf("%f ", grad_fp16[i]);        
    }
    printf("\n");
}


// MAIN FUNCTION
void change_layout () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif   

    init_data();

    printf("Original data:\n");
    print_CHW();

    printf("\n\n<<<<< Transposing data and grad from CHW to HWC (C=%d, H=%d, W=%d) >>>>>\n", T_C, T_H, T_W);
    printf("FP32 stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    CHW_to_HWC(&args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nFP16 stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    CHW_to_HWC_fp16(&args_fp16);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    print_HWC();




    printf("\n\n<<<<< Transposing data and grad from HWC to CHW (C=%d, H=%d, W=%d) >>>>>\n", T_C, T_H, T_W);
    printf("FP32 stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    HWC_to_CHW(&args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("\nFP16 stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    HWC_to_CHW_fp16(&args_fp16);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    print_CHW();

}