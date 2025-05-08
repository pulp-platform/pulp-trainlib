#include "pulp_train.h"

#include "net.h"
#include "stats.h"

#include "test_data.h"

#include "tensor_checkers.h"


void transpose_matrices_fp32() {
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif
    printf("Executing on %d cores.\n", NUM_CORES);

#if DATA_TYPE == 32
    struct transp_args args;
    printf("WORKING ON FP32\n");
#elif DATA_TYPE == 16
    struct transp_args_fp16 args;
    printf("WORKING ON FP16\n");
#endif

    // Get arguments
    args.in_matrix = IN_M;
    args.out_matrix = OUT_M;
    args.dim = DIMS;
    args.transposed_axes = TRANSPOSED_AXES;
    args.n_dim = N_DIMS;

#ifdef PROF_NET
    START_STATS();
#endif

    // Perform transposition
#if DATA_TYPE == 32
    pi_cl_team_fork(NUM_CORES, transpose, &args);
#elif DATA_TYPE == 16
    pi_cl_team_fork(NUM_CORES, transpose_fp16, &args);
#endif

    // Stop stats
#ifdef PROF_NET
    STOP_STATS();
#endif

    mean_error_checker(OUT_M, TEST_TRANSPOSE_OUT, TOTAL_SIZE);
    elementwise_checker(OUT_M, TEST_TRANSPOSE_OUT, TOTAL_SIZE);

    return;
}
