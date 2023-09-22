#include "pulp_train.h"

#include "stats.h"
#include "net.h"

// ----------------- FP32 data ----------------------
PI_L1 float IN_DATA_FP32[Tin_C*Tin_H*Tin_W];
PI_L1 float PADDED_DATA_FP32[Tin_C*Tout_H*Tout_W];
#if HWC_LAYOUT == 1
struct blob data;
PI_L1 float TEMP_BUFF[Tin_C*Tout_H*Tout_W];
#endif
// ----------------- FP16 data ----------------------
PI_L1 fp16 IN_DATA_FP16[Tin_C*Tin_H*Tin_W];
PI_L1 fp16 PADDED_DATA_FP16[Tin_C*Tout_H*Tout_W];
#if HWC_LAYOUT == 1
struct blob_fp16 data_fp16;
PI_L1 fp16 TEMP_BUFF_FP16[Tin_C*Tout_H*Tout_W];
#endif


static void init_matrix ()
{
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++)
    {
        IN_DATA_FP32[i] = 0.1 + (float) i/10;
        IN_DATA_FP16[i] = 0.1 + (fp16) i/10;
    }
}


// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    init_matrix();

    printf("Padding a matrix with sizes C = %d, H = %d, W = %d.\n", Tin_C, Tin_H, Tin_W);
    printf("Executing on %d cores.\n", NUM_CORES);
    #if HWC_LAYOUT == 0
    printf("Input/Output data is in CHW layout.\n");
    #else
    printf("Input/Output data is in HWC layout.\n");
    #endif

// ---------FP32-------------------------------

    // INITIALIZE STRUCTURE
    struct pad_args args;
    args.source = IN_DATA_FP32;
    args.dest = PADDED_DATA_FP32;
    args.C = Tin_C;
    args.H = Tin_H;
    args.W = Tin_W;
    args.T_LPAD = LPAD;
    args.T_RPAD = RPAD;
    args.T_UPAD = UPAD;
    args.T_DPAD = DPAD;
    #if HWC_LAYOUT == 0
    args.HWC_lay = 0;
    #else 
    args.HWC_lay = 1;
    // Structure to visualize data
    data.data = IN_DATA_FP32;
    data.dim = Tin_C*Tin_H*Tin_W;
    data.C = Tin_C;
    data.H = Tin_H;
    data.W = Tin_W;
    struct layout_args lay_args;
    lay_args.tensor = &data;
    lay_args.transp_buffer = TEMP_BUFF;
    lay_args.transpose_data = 1;
    lay_args.transpose_grad = 0;
    #endif

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nHello, starting to pad in FP32!\n");
    printf("\nINPUT CHW MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_W))) printf("\n");
        if (!(i%(Tin_H*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP32[i]);
    }
    #else
    printf("\nHello, starting to pad in FP32!\n");
    printf("\nINPUT HWC MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_W))) printf("\n");
        if (!(i%(Tin_H*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP32[i]);
    }    
    #endif
    #endif
    printf("\n\nFP32 Stats:\n");
    
    #if HWC_LAYOUT == 1
    CHW_to_HWC(&lay_args);
    #endif 

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pad_tensor, &args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #if HWC_LAYOUT == 1
    data.data = PADDED_DATA_FP32;
    data.dim = Tin_C*Tout_H*Tout_W;
    data.C = Tin_C;
    data.H = Tout_H;
    data.W = Tout_W;
    HWC_to_CHW(&lay_args);
    #endif    

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nPADDED CHW MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tout_W))) printf("\n");
        if (!(i%(Tout_H*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP32[i]);
    }
    printf("\n\n");
    #else
    printf("\nPADDED HWC MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tout_W))) printf("\n");
        if (!(i%(Tout_H*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP32[i]);
    }
    printf("\n\n");    
    #endif
    #endif



// ---------FP16-------------------------------

    // INITIALIZE STRUCTURE
    struct pad_args_fp16 args_fp16;
    args_fp16.source = IN_DATA_FP16;
    args_fp16.dest = PADDED_DATA_FP16;
    args_fp16.C = Tin_C;
    args_fp16.H = Tin_H;
    args_fp16.W = Tin_W;
    args_fp16.T_LPAD = LPAD;
    args_fp16.T_RPAD = RPAD;
    args_fp16.T_UPAD = UPAD;
    args_fp16.T_DPAD = DPAD;
    #if HWC_LAYOUT == 0
    args_fp16.HWC_lay = 0;
    #else 
    args_fp16.HWC_lay = 1;
    // Structure to visualize data
    data_fp16.data = IN_DATA_FP16;
    data_fp16.dim = Tin_C*Tin_H*Tin_W;
    data_fp16.C = Tin_C;
    data_fp16.H = Tin_H;
    data_fp16.W = Tin_W;
    struct layout_args_fp16 lay_args_fp16;
    lay_args_fp16.tensor = &data_fp16;
    lay_args_fp16.transp_buffer = TEMP_BUFF_FP16;
    lay_args_fp16.transpose_data = 1;
    lay_args_fp16.transpose_grad = 0;
    #endif

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nHello, starting to pad in FP16!\n");
    printf("\nINPUT CHW MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_W))) printf("\n");
        if (!(i%(Tin_H*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP16[i]);
    }
    #else
    printf("\nHello, starting to pad in FP16!\n");
    printf("\nINPUT HWC MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_W))) printf("\n");
        if (!(i%(Tin_H*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP16[i]);
    }    
    #endif
    #endif
    printf("\n\nFP16 Stats:\n");
    
    #if HWC_LAYOUT == 1
    CHW_to_HWC_fp16(&lay_args_fp16);
    #endif 

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pad_tensor_fp16, &args_fp16);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #if HWC_LAYOUT == 1
    data_fp16.data = PADDED_DATA_FP16;
    data_fp16.dim = Tin_C*Tout_H*Tout_W;
    data_fp16.C = Tin_C;
    data_fp16.H = Tout_H;
    data_fp16.W = Tout_W;
    HWC_to_CHW_fp16(&lay_args_fp16);
    #endif    

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nPADDED CHW MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tout_W))) printf("\n");
        if (!(i%(Tout_H*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP16[i]);
    }
    printf("\n\n");
    #else
    printf("\nPADDED HWC MATRIX (CHW visualization):\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tout_W))) printf("\n");
        if (!(i%(Tout_H*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP16[i]);
    }
    printf("\n\n");    
    #endif
    #endif



    return;
}
