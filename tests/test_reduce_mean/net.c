#include "pulp_train.h"

#include "net.h"
#include "stats.h"

#include "test_data.h"

#include "tensor_checkers.h"


void reduce_mean_test() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif
    printf("Executing on %d cores.\n", NUM_CORES);

#if DATA_TYPE == 32
    struct reduce_mean_args_fp32 args;
    printf("WORKING ON FP32\n");
#elif DATA_TYPE == 16
    struct reduce_mean_args_fp16 args;
    printf("WORKING ON FP16\n");
#endif

    // Get arguments
    args.input = IN_MATRIX;
    args.output = OUT_MATRIX;

    args.dims = DIMS;
    args.dims_len = N_DIMS;
    args.reduce_axis = REDUCE_AXIS;

#ifdef PROF_NET
    START_STATS();
#endif

    // Perform operation
#if DATA_TYPE == 32
    pi_cl_team_fork(NUM_CORES, reduce_mean_fp32, &args);
#elif DATA_TYPE == 16
    pi_cl_team_fork(NUM_CORES, reduce_mean_fp16, &args);
#endif

    // Stop stats
#ifdef PROF_NET
    STOP_STATS();
#endif

    mean_error_checker(args.output, TEST_OUT, TOTAL_SIZE_OUT);
    elementwise_checker(args.output, TEST_OUT, TOTAL_SIZE_OUT);

    return;
}
