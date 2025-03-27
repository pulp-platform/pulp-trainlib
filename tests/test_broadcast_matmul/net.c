#include "pulp_train.h"

#include "net.h"
#include "stats.h"

#include "test_data.h"

#include "tensor_checkers.h"


void broadcast_matmul_test() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif
    printf("Executing on %d cores.\n", NUM_CORES);

#if DATA_TYPE == 32
    struct broadcastMatMul_args_fp32 args;
    printf("WORKING ON FP32\n");
#elif DATA_TYPE == 16
    struct broadcastMatMul_args_fp16 args;
    printf("WORKING ON FP16\n");
#endif

    // Get arguments
    args.A = IN_MATRIX_1;
    args.B = IN_MATRIX_2;
    args.C = OUT_MATRIX;

    args.A_dims = DIMS_1;
    args.B_dims = DIMS_2;

    args.A_dims_len = N_DIMS_1;
    args.B_dims_len = N_DIMS_2;

#ifdef PROF_NET
    START_STATS();
#endif

    // Perform transposition
#if DATA_TYPE == 32
    mm_broadcast_fp32(&args);
#elif DATA_TYPE == 16
    mm_broadcast_fp16(&args);
#endif

    // Stop stats
#ifdef PROF_NET
    STOP_STATS();
#endif

    mean_error_checker(args.C, TEST_OUT, TOTAL_SIZE_OUT);
    elementwise_checker(args.C, TEST_OUT, TOTAL_SIZE_OUT);

    return;
}
