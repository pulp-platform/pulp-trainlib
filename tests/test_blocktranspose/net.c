#include "pulp_train.h"

#include "stats.h"
#include "net.h"

PI_L1 float IN_MATRIX_FP32[Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk];
PI_L1 fp16 IN_MATRIX_FP16[Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk];

PI_L1 float BT_MATRIX_FP32[Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk];
PI_L1 fp16 BT_MATRIX_FP16[Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk];

static void init_matrix ()
{
    for (int i=0; i<Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk; i++)
    {
        IN_MATRIX_FP32[i] = i;
        IN_MATRIX_FP16[i] = i;
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

    printf("Block-transposing matrix with sizes Cout=%d, Cin=%d, Hk=%d, Wk=%d.\n", Tin_Cout, Tin_Cin, Tin_Hk, Tin_Wk);
    printf("Executing on %d cores.\n", NUM_CORES);

// ---------FP32-------------------------------

    struct blocktransp_args args;
    args.weights = IN_MATRIX_FP32;
    args.bt_weights = BT_MATRIX_FP32;
    args.Cout = Tin_Cout;
    args.Cin = Tin_Cin;
    args.Hk = Tin_Hk;
    args.Wk = Tin_Wk;

    #ifdef PRINT_MATS
    printf("\nHello, starting block transposition in FP32!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk; i++) 
    {
        if (!(i%(Tin_Hk*Tin_Wk))) printf("  ");
        if (!(i%(Tin_Hk*Tin_Wk*Tin_Cin))) printf("\n");
        printf("%f ", IN_MATRIX_FP32[i]);
    }
    #endif
    printf("\n\nFP32 Stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pulp_blocktransp_fp32, &args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_MATS
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk; i++) 
    {
        if (!(i%(Tin_Hk*Tin_Wk))) printf("  ");
        if (!(i%(Tin_Hk*Tin_Wk*Tin_Cout))) printf("\n");
        printf("%f ", BT_MATRIX_FP32[i]);
    }
    printf("\n\n");
    #endif



// ---------FP16-------------------------------

    struct blocktransp_args_fp16 args_fp16;
    args_fp16.weights = IN_MATRIX_FP16;
    args_fp16.bt_weights = BT_MATRIX_FP16;
    args_fp16.Cout = Tin_Cout;
    args_fp16.Cin = Tin_Cin;
    args_fp16.Hk = Tin_Hk;
    args_fp16.Wk = Tin_Wk;

    #ifdef PRINT_MATS
    printf("\nHello, starting transposition in FP16!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk; i++) 
    {
        if (!(i%(Tin_Hk*Tin_Wk))) printf("  ");
        if (!(i%(Tin_Hk*Tin_Wk*Tin_Cin))) printf("\n");
        printf("%f ", IN_MATRIX_FP16[i]);
    }
    #endif
    printf("\n\nFP16 Stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pulp_blocktransp_fp16, &args_fp16);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_MATS
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_Cout*Tin_Cin*Tin_Hk*Tin_Wk; i++) 
    {
        if (!(i%(Tin_Hk*Tin_Wk))) printf("  ");
        if (!(i%(Tin_Hk*Tin_Wk*Tin_Cout))) printf("\n");
        printf("%f ", BT_MATRIX_FP16[i]);
    }
    printf("\n\n");
    #endif    
    
    return;
}
