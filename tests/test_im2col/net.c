#include "pulp_train.h"

#include "stats.h"
#include "net.h"

// Blobs for im2col
#if DATA_BITS == 32
PI_L1 struct blob layer1_in, layer1_wgt, layer1_out;
#elif DATA_BITS == 16
PI_L1 struct blob_fp16 layer1_in, layer1_wgt, layer1_out;
#endif


// IM2COL data
#if DATA_BITS == 32 
#if DMA_ENABLE == 1
PI_L2 float l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
#else
PI_L1 float l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
#endif
#if MOD==0
PI_L1 float im2col_buffer[i2c_b_size*2];
#else
PI_L1 float im2col_buffer_bw[i2c_b_size_bw*2];
#endif
PI_L1 float l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
#if DMA_ENABLE == 1
PI_L2 float l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#else 
PI_L1 float l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif

#elif DATA_BITS == 16
#if DMA_ENABLE == 1
PI_L2 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
#else
PI_L1 fp16 l1_in[Tin_H_l1*Tin_W_l1*Tin_C_l1];
#endif
#if MOD==0
PI_L1 fp16 im2col_buffer[i2c_b_size*2];
#else
PI_L1 fp16 im2col_buffer_bw[i2c_b_size_bw*2];
#endif
PI_L1 fp16 l1_ker[Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1];
#if DMA_ENABLE == 1
PI_L2 fp16 l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#else 
PI_L1 fp16 l1_out[Tout_H_l1*Tout_W_l1*Tout_C_l1];
#endif
#endif


// Extra variables
#if DATA_BITS == 32
float temp_val = 0.1f;
#elif DATA_BITS == 16
fp16 temp_val = 0.1f;
#endif


// Other functions
static inline void tensor_init(){
  for (int i=0; i<Tin_H_l1*Tin_W_l1*Tin_C_l1; i++)                             {l1_in[i] = temp_val; temp_val+=0.1;}
  for (int i=0; i<Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1; i++)                 l1_ker[i] = weight_init;
  #if MOD==0
  for (int i=0; i<i2c_b_size*2; i++)                                           im2col_buffer[i] = 0.0f;
  #else 
  for (int i=0; i<i2c_b_size_bw*2; i++)                                        im2col_buffer_bw[i] = 0.0f;
  #endif
  temp_val = 0.1f;
  for (int i=0; i<Tout_H_l1*Tout_W_l1*Tout_C_l1; i++)                          {l1_out[i] = temp_val; temp_val+=0.1;} //l1_out[i] =  0.0f;
}

static inline void connect_blobs(){

  // ********** LAYER SEPARABLE CONV **************
  layer1_in.data = l1_in;
  layer1_in.dim = Tin_H_l1*Tin_W_l1*Tin_C_l1;
  layer1_in.W = Tin_W_l1;
  layer1_in.H = Tin_H_l1;
  layer1_in.C = Tin_C_l1;

  #if MOD==0
  layer1_out.data = l1_out;
  layer1_out.dim = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  layer1_out.W = Tout_W_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.C = Tout_C_l1;
  #else 
  layer1_out.diff = l1_out;
  layer1_out.dim = Tout_H_l1*Tout_W_l1*Tout_C_l1;
  layer1_out.W = Tout_W_l1;
  layer1_out.H = Tout_H_l1;
  layer1_out.C = Tout_C_l1;
  #endif

  layer1_wgt.data = l1_ker;
  layer1_wgt.dim = Tker_H_l1*Tker_W_l1*Tin_C_l1*Tout_C_l1;
  layer1_wgt.W = Tker_W_l1;
  layer1_wgt.H = Tker_H_l1;
  layer1_wgt.C = Tin_C_l1;
}


// Launcher
static inline void train () 
{
    #if DATA_BITS == 32
    struct im2col_args im2col_args;
    #elif DATA_BITS == 16
    struct im2col_args_fp16 im2col_args;
    #endif

    im2col_args.input = &layer1_in;
    im2col_args.c = &layer1_wgt;
    im2col_args.output = &layer1_out;
    im2col_args.Lpad = 0;
    im2col_args.Rpad = 0;
    im2col_args.Upad = 0;
    im2col_args.Dpad = 0;
    im2col_args.stride_h = HSTR;
    im2col_args.stride_w = WSTR;
    im2col_args.mod = MOD;
    im2col_args.tile_start = 0;
    im2col_args.tile_h = Tin_H_l1;
    im2col_args.USE_DMA = DMA_ENABLE;

    #if MOD==0
        im2col_args.Lpad = LPAD;
        im2col_args.Rpad = RPAD;
        im2col_args.Upad = UPAD;
        im2col_args.Dpad = DPAD;
        im2col_args.pBuffer = im2col_buffer;
    #else
        im2col_args.Lpad = PAD_BW;
        im2col_args.Rpad = PAD_BW;
        im2col_args.Upad = PAD_BW;
        im2col_args.Dpad = PAD_BW; 
        im2col_args.pBuffer = im2col_buffer_bw;
    #endif

    #ifdef PROF_NET
    START_STATS();
    #endif
    
    #if DATA_BITS == 32
    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp32, &im2col_args);
    #elif DATA_BITS == 16
    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_OUTPUT
    // FORWARD
    #if MOD==0
    printf("\n\nInput:\n");
    for (int idx=0; idx<Tin_H_l1*Tin_W_l1*Tin_C_l1; idx++)
    {
        if (!(idx%Tin_H_l1)) printf("\n");
        if (!(idx%(Tin_H_l1*Tin_W_l1))) printf("\n");
        printf("%f ", l1_in[idx]);
    }
    printf("\n\n");
    
    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_b_size*2; idx++)
    {
        //if (!(idx%Tker_H_l1)) printf("\n");
        if (!(idx%(Tker_H_l1*Tker_W_l1))) printf("\n");
        printf("%f ", im2col_buffer[idx]);

        if (idx==i2c_b_size-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");
    

    // BACKWARD
    #else 
    printf("\n\nOuput:\n");
    for (int idx=0; idx<Tout_H_l1*Tout_W_l1*Tout_C_l1; idx++)
    {
        if (!(idx%Tout_H_l1)) printf("\n");
        if (!(idx%(Tout_H_l1*Tout_W_l1))) printf("\n");
        printf("%f ", l1_out[idx]);
    }
    printf("\n\n");

    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_b_size_bw*2; idx++)
    {
        //if (!(idx%Tker_H_l1)) printf("\n");
        if (!(idx%((Tker_H_l1)*(Tker_W_l1)))) printf("\n");
        printf("%f ", im2col_buffer_bw[idx]);

        if (idx==i2c_b_size_bw-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");
    #endif
    #endif

}


// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    printf("\nHello, starting im2col FP%d!\n", DATA_BITS);
    if (MOD==0) {
        printf("Performing IM2COL for forward and weight gradient (DMA=%d).\n", DMA_ENABLE);
    }
    else if (MOD==1) {
        printf("Performing IM2COL for input gradient (DMA=%d).\n", DMA_ENABLE);
    }
    else {
        printf("[net.c:182]: INVALID MOD PARAMETER!!");
    }

    if (MOD == 0) printf("IM2COL size: %d\n", i2c_b_size);
    if (MOD == 1) printf("IM2COL size: %d\n", i2c_b_size_bw); 

    tensor_init();

    connect_blobs();

    train();

    return;
}