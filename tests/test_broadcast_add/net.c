#include "pulp_train.h"

#include "net.h"
#include "stats.h"

#include "test_data.h"

#include "tensor_checkers.h"


void broadcast_add_test() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif
    printf("Executing on %d cores.\n", NUM_CORES);

#if DATA_TYPE == 32
    struct array_broadcast_sum_fp32_args args;
    printf("WORKING ON FP32\n");
#elif DATA_TYPE == 16
    struct array_broadcast_sum_fp16_args args;
    printf("WORKING ON FP16\n");
#endif

    // Get arguments
    args.op_1 = IN_MATRIX_1;
    args.op_2 = IN_MATRIX_2;
    args.dest = OUT_MATRIX;

    args.op_1_dims = DIMS_1;
    args.op_2_dims = DIMS_2;
    
    args.op_1_dims_len = N_DIMS_1;
    args.op_2_dims_len = N_DIMS_2;

#ifdef PROF_NET
    START_STATS();
#endif

    // Perform transposition
#if DATA_TYPE == 32
    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &args);
#elif DATA_TYPE == 16
    pi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp16, &args);
#endif

    // Stop stats
#ifdef PROF_NET
    STOP_STATS();
#endif

    mean_error_checker(args.dest, TEST_OUT, TOTAL_SIZE_OUT);
    elementwise_checker(args.dest, TEST_OUT, TOTAL_SIZE_OUT);

    return;
}
