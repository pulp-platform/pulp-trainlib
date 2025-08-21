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
#define i2c_check_size (i2c_b_size*2)
PI_L1 float im2col_buffer[i2c_check_size];
#else
#define i2c_check_size (i2c_b_size_bw*2)
PI_L1 float im2col_buffer_bw[i2c_check_size];
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
#define i2c_check_size (i2c_b_size*2)
PI_L1 fp16 im2col_buffer[i2c_check_size];
#else
#define i2c_check_size (i2c_b_size_bw*2)
PI_L1 fp16 im2col_buffer_bw[i2c_check_size];
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
float zero_val = 0.0f;
#elif DATA_BITS == 16
fp16 temp_val = 0.1f;
fp16 zero_val = 0.0f;
#endif


// Other functions
static inline void tensor_init() {
    for (int i = 0; i < Tin_H_l1 * Tin_W_l1 * Tin_C_l1; i++) {
        l1_in[i] = temp_val;
        temp_val += 0.1;
    }
    for (int i = 0; i < Tker_H_l1 * Tker_W_l1 * Tin_C_l1 * Tout_C_l1; i++) l1_ker[i] = weight_init;
#if MOD == 0
    for (int i = 0; i < i2c_check_size; i++) im2col_buffer[i] = zero_val;
#else
    for (int i=0; i<i2c_check_size; i++)                                         im2col_buffer_bw[i] = zero_val;
#endif
    temp_val = 0.1f;
    for (int i = 0; i < Tout_H_l1 * Tout_W_l1 * Tout_C_l1; i++) {
        l1_out[i] = temp_val;
        temp_val += 0.1;
    } //l1_out[i] =  0.0f;
}

static inline void connect_blobs() {

    layer1_in.data = l1_in;
    layer1_in.dim = Tin_H_l1 * Tin_W_l1 * Tin_C_l1;
    layer1_in.W = Tin_W_l1;
    layer1_in.H = Tin_H_l1;
    layer1_in.C = Tin_C_l1;

#if MOD == 0
    layer1_out.data = l1_out;
    layer1_out.dim = Tout_H_l1 * Tout_W_l1 * Tout_C_l1;
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
    layer1_wgt.dim = Tker_H_l1 * Tker_W_l1 * Tin_C_l1 * Tout_C_l1;
    layer1_wgt.W = Tker_W_l1;
    layer1_wgt.H = Tker_H_l1;
    layer1_wgt.C = Tin_C_l1;
}


// Launcher
static inline void train() {
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
    im2col_args.USE_DMA = DMA_ENABLE;
    im2col_args.HWC = HWC_format;

#if MOD == 0
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

#if HWC_format == 1
    // Transpose input matrix to HWC
#if DATA_BITS == 32
    struct transp_args transp_args;
#if MOD == 0
    int dims[] = {Tin_C_l1, Tin_H_l1 * Tin_W_l1};
    // int t_axes = {1, 0};

    float transp_buffer[Tin_H_l1 * Tin_W_l1 * Tin_C_l1];

    transp_args.in_matrix = l1_in;
    transp_args.out_matrix = transp_buffer;
    transp_args.dim = dims;
    transp_args.N = Tin_C_l1;
    transp_args.M = Tin_H_l1 * Tin_W_l1;
    // transp_args.transposed_axes = t_axes;
    // transp_args.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_matrix, &transp_args);

    struct copy_args copy_args;
    copy_args.from = transp_buffer;
    copy_args.to = l1_in;
    copy_args.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
    pi_cl_team_fork(NUM_CORES, copy, &copy_args);
#else
    int dims[] = {Tout_C_l1, Tout_H_l1 * Tout_W_l1};
    // int t_axes = {1, 0};

    float transp_buffer[Tout_H_l1 * Tout_W_l1 * Tout_C_l1];

    transp_args.in_matrix = l1_out;
    transp_args.out_matrix = transp_buffer;
    transp_args.dim = dims;
    transp_args.N = Tout_C_l1;
    transp_args.M = Tout_H_l1 * Tout_W_l1;
    // transp_args.transposed_axes = t_axes;
    // transp_args.n_dim = 2;

    pi_cl_team_fork(NUM_CORES, transpose_matrix, &transp_args);

    struct copy_args copy_args;
    copy_args.from = transp_buffer;
    copy_args.to = l1_out;
    copy_args.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
    pi_cl_team_fork(NUM_CORES, copy, &copy_args);
#endif


#elif DATA_BITS == 16
    struct transp_args_fp16 transp_args;
#if MOD == 0
    int dims = {Tin_C_l1, Tin_H_l1 * Tin_W_l1};
    // int t_axes = {1, 0};

    fp16 transp_buffer[Tin_H_l1 * Tin_W_l1 * Tin_C_l1];

    transp_args.in_matrix = l1_in;
    transp_args.out_matrix = transp_buffer;
    // transp_args.dim = dims;
    // transp_args.transposed_axes = t_axes;
    // transp_args.n_dim = 2;
    transp_args.N = Tin_C_l1;
    transp_args.M = Tin_H_l1 * Tin_W_l1;

    pi_cl_team_fork(NUM_CORES, transpose_matrix_fp16, &transp_args);

    struct copy_args_fp16 copy_args;
    copy_args.from = transp_buffer;
    copy_args.to = l1_in;
    copy_args.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args);
#else
    int dims = {Tout_C_l1, Tout_H_l1 * Tout_W_l1};
    int t_axes = {1, 0};

    fp16 transp_buffer[Tout_H_l1 * Tout_W_l1 * Tout_C_l1];

    transp_args.transp_matrix = transp_buffer;
    transp_args.matrix = l1_out;
    // transp_args.dim = dims;
    // transp_args.transposed_axes = t_axes;
    // transp_args.n_dim = 2;
    transp_args.N = Tout_C_l1;
    transp_args.M = Tout_H_l1 * Tout_W_l1;

    pi_cl_team_fork(NUM_CORES, transpose_matrix_fp16, &transp_args);

    struct copy_args_fp16 copy_args;
    copy_args.from = transp_buffer;
    copy_args.to = l1_out;
    copy_args.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args);
#endif

#endif
#endif

#ifdef PROF_NET
    START_STATS();
#endif

#if DATA_BITS == 32
#if IM2ROW == 0
    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp32, &im2col_args);
#else
    pi_cl_team_fork(NUM_CORES, pulp_im2row_fp32, &im2col_args);
#endif
#elif DATA_BITS == 16
#if IM2ROW == 0
    pi_cl_team_fork(NUM_CORES, pulp_im2col_fp16, &im2col_args);
#else
    pi_cl_team_fork(NUM_CORES, pulp_im2row_fp16, &im2col_args);
#endif
#endif

#ifdef PROF_NET
    STOP_STATS();
#endif

#ifdef PRINT_OUTPUT

#if HWC_format == 0
    // FORWARD
#if MOD==0
    printf("\n\nCHW Reference Input:\n");
    for (int idx=0; idx<Tin_H_l1*Tin_W_l1*Tin_C_l1; idx++)
    {
        if (!(idx%Tin_H_l1)) printf("\n");
        if (!(idx%(Tin_H_l1*Tin_W_l1))) printf("\n");
        printf("%f ", l1_in[idx]);
    }
    printf("\n\n");

    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_check_size; idx++)
    {
        //if (!(idx%Tker_H_l1)) printf("\n");
#if IM2ROW == 0
        if (!(idx%(Tout_H_l1*Tout_W_l1))) printf("\n");
#else
        if (!(idx%(Tker_H_l1*Tker_W_l1*Tin_C_l1))) printf("\n");
#endif
        printf("%f ", im2col_buffer[idx]);

        if (idx==i2c_b_size-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");


    // BACKWARD
#else
    printf("\n\nCHW Reference Ouput:\n");
    for (int idx=0; idx<Tout_H_l1*Tout_W_l1*Tout_C_l1; idx++)
    {
        if (!(idx%Tout_H_l1)) printf("\n");
        if (!(idx%(Tout_H_l1*Tout_W_l1))) printf("\n");
        printf("%f ", l1_out[idx]);
    }
    printf("\n\n");

    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_check_size; idx++)
    {
        //if (!(idx%Tker_H_l1)) printf("\n");
#if IM2ROW == 0
        if (!(idx%((Tin_H_l1)*(Tin_W_l1)))) printf("\n");
#else
        if (!(idx%((Tker_H_l1)*(Tker_W_l1)*Tout_C_l1))) printf("\n");
#endif
        printf("%f ", im2col_buffer_bw[idx]);

        if (idx==i2c_b_size_bw-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");
#endif
#endif



#if HWC_format == 1
    // FORWARD
#if MOD==0
    printf("\n\nCHW Reference Input:\n");
    // Transpose again to CHW to better visualize
    transp_args.in_matrix = l1_in;
    transp_args.out_matrix = transp_buffer;
    transp_args.M = Tin_C_l1;
    transp_args.N = Tin_H_l1*Tin_W_l1;
#if DATA_BITS == 32
    pi_cl_team_fork(NUM_CORES, transpose_matrix, &transp_args);
#elif DATA_BITS == 16
    pi_cl_team_fork(NUM_CORES, transpose_matrix_fp16, &transp_args);
#endif
    copy_args.from = transp_buffer;
    copy_args.to = l1_in;
    copy_args.size = Tin_H_l1*Tin_W_l1*Tin_C_l1;
#if DATA_BITS == 32
    pi_cl_team_fork(NUM_CORES, copy, &copy_args);
#elif DATA_BITS == 16
    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args);
    printf("\n>>> POSSIBLE VISUALIZATION BUGS IN THE INPUT DATA, DOUBLE CHECK WITH FP32 RESULTS, IM2COL/IM2ROW MAY BE CORRECT <<<\n");
#endif

    for (int idx=0; idx<Tin_H_l1*Tin_W_l1*Tin_C_l1; idx++)
    {
        if (!(idx%Tin_H_l1)) printf("\n");
        if (!(idx%(Tin_H_l1*Tin_W_l1))) printf("\n");
        printf("%f ", l1_in[idx]);
    }
    printf("\n\n");

    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_check_size; idx++)
    {
#if IM2ROW == 0
        if (!(idx%(Tin_C_l1*Tker_H_l1))) printf("\n");
#else
        if (!(idx%(Tin_C_l1*Tker_H_l1*Tker_W_l1))) printf("\n");
#endif
        printf("%f ", im2col_buffer[idx]);

        if (idx==i2c_b_size-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");


    // BACKWARD
#else
    printf("\n\nCHW Reference Ouput:\n");
    // Transpose again to CHW to better visualize
    transp_args.in_matrix = l1_out;
    transp_args.out_matrix = transp_buffer;
    transp_args.M = Tout_C_l1;
    transp_args.N = Tout_H_l1*Tout_W_l1;
#if DATA_BITS == 32
    pi_cl_team_fork(NUM_CORES, transpose, &transp_args);
#elif DATA_BITS == 16
    pi_cl_team_fork(NUM_CORES, transpose_fp16, &transp_args);
#endif
    copy_args.from = transp_buffer;
    copy_args.to = l1_out;
    copy_args.size = Tout_H_l1*Tout_W_l1*Tout_C_l1;
#if DATA_BITS == 32
    pi_cl_team_fork(NUM_CORES, copy, &copy_args);
#elif DATA_BITS == 16
    pi_cl_team_fork(NUM_CORES, copy_fp16, &copy_args);
    printf("\n>>> POSSIBLE VISUALIZATION BUGS IN THE INPUT DATA, DOUBLE CHECK WITH FP32 RESULTS, IM2COL/IM2ROW MAY BE CORRECT <<<\n");
#endif

    for (int idx=0; idx<Tout_H_l1*Tout_W_l1*Tout_C_l1; idx++)
    {
        if (!(idx%Tout_H_l1)) printf("\n");
        if (!(idx%(Tout_H_l1*Tout_W_l1))) printf("\n");
        printf("%f ", l1_out[idx]);
    }
    printf("\n\n");

    printf("\n\nIm2col buffer:\n");
    for (int idx=0; idx<i2c_check_size; idx++)
    {
#if IM2ROW == 0
        if (!(idx%((Tout_C_l1)*(Tker_H_l1)))) printf("\n");
#else
        if (!(idx%((Tker_H_l1)*(Tker_W_l1)*Tout_C_l1))) printf("\n");
#endif
        printf("%f ", im2col_buffer_bw[idx]);

        if (idx==i2c_b_size_bw-1) printf("\n\nError: Leftovers (Overflowing elements):\n\n");
    }
    printf("\n\n");
#endif
#endif


#endif

}


// Main function
void net_step() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

    printf("\nHello, starting im2col FP%d!\n", DATA_BITS);
    if (MOD == 0) {
        printf("Performing IM2COL for forward and weight gradient (DMA=%d).\n", DMA_ENABLE);
    } else if (MOD == 1) {
        printf("Performing IM2COL for input gradient (DMA=%d).\n", DMA_ENABLE);
    } else {
        printf("[net.c:182]: INVALID MOD PARAMETER!!");
    }

    if (MOD == 0) printf("IM2COL size: %d\n", i2c_b_size);
    if (MOD == 1) printf("IM2COL size: %d\n", i2c_b_size_bw);

    tensor_init();

    connect_blobs();

    train();

    return;
}
