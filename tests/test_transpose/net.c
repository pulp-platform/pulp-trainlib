#include "pulp_train.h"

#include "stats.h"
#include "net.h"

PI_L1 float IN_MATRIX_FP32[Tin_N*Tin_M];
PI_L1 fp16 IN_MATRIX_FP16[Tin_N*Tin_M];

PI_L1 float temp[Tin_M*Tin_N];
PI_L1 fp16 temp_fp16[Tin_M*Tin_N];

static void init_matrix ()
{
    for (int i=0; i<Tin_N*Tin_M; i++)
    {
        IN_MATRIX_FP32[i] = i;
        IN_MATRIX_FP16[i] = i;
    }
}

// Main function
void transpose_matricex_fp () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    init_matrix();

    printf("Transposing matrix with size N=%d, M=%d.\n", Tin_N, Tin_M);
    printf("Executing on %d cores.\n", NUM_CORES);

// ---------FP32-------------------------------

    struct transp_args args;
    args.matrix = IN_MATRIX_FP32;
    args.transp_matrix = temp;
    args.N = Tin_N;
    args.M = Tin_M;

    struct copy_args copy_args;
    copy_args.from = temp;
    copy_args.to = IN_MATRIX_FP32;
    copy_args.size = Tin_N*Tin_M;

    #ifdef PRINT_MATS
    printf("\nHello, starting transposition in FP32!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_N*Tin_M; i++) 
    {
        if (!(i%Tin_M)) printf("\n");
        printf("%f ", IN_MATRIX_FP32[i]);
    }
    #endif
    printf("\n\nFP32 Stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, transpose, &args);
    pi_cl_team_fork(NUM_CORES, copy, &copy_args);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_MATS
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_N*Tin_M; i++) 
    {
        if (!(i%Tin_N)) printf("\n");
        printf("%f ", IN_MATRIX_FP32[i]);
    }
    printf("\n\n");
    #endif



// ---------FP16-------------------------------

    struct transp_args_fp16 args2;
    args2.matrix = IN_MATRIX_FP16;
    args2.transp_matrix = temp_fp16;
    args2.N = Tin_N;
    args2.M = Tin_M;

    struct copy_args_fp16 copy_args2;
    copy_args2.from = temp_fp16;
    copy_args2.to = IN_MATRIX_FP16;
    copy_args2.size = Tin_N*Tin_M;

    #ifdef PRINT_MATS
    printf("\nHello, starting transposition in FP16!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_N*Tin_M; i++) 
    {
        if (!(i%Tin_M)) printf("\n");
        printf("%f ", IN_MATRIX_FP16[i]);
    }
    #endif
    printf("\n\nFP16 Stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, transpose_fp16, &args2);
    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args2);
    
    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_MATS
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_N*Tin_M; i++) 
    {
        if (!(i%Tin_N)) printf("\n");
        printf("%f ", IN_MATRIX_FP16[i]);
    }
    printf("\n\n");
    #endif    
    
    return;
}
