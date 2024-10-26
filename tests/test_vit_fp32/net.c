// ~~~~~~~~~~ INCLUDES ~~~~~~~~~~
#include "pulp_train.h"

#include "model_components.h"
#include "tensor_checkers.h"

#include "stats.h"
#include "net.h"

// ~~~~~~~~~~ FORWARD AND MAIN FUNCS ~~~~~~~~~~
// Define forward step function
void forward() {
    // Shape: b, c, h, w
    // patch_embedding fw pass
    pulp_conv2d_fp32_fw_cl(&patch_embedding_conv2d_args);

    // Shape: b, dim, nph, npw (number of patches - height and width)
    // flatten and transpose
    pi_cl_team_fork(NUM_CORES, transpose, &flatten_and_transpose_transpose_args);

    // Shape: b, nph * npw, dim
    // concat
    pi_cl_team_fork(NUM_CORES, copy, &concat_copy_0_args);
    pi_cl_team_fork(NUM_CORES, copy, &concat_copy_1_args);

    // Shape: b, nph * npw + 1, dim
    // vect_sum
    pi_cl_team_fork(NUM_CORES, vect_sum, &positional_embedding_vect_sum_args);

    return;
}

// Main function
void net_step() {
    // Initialize performance counters
#ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
#endif

    // Initialize model components
    printf("ViT test:\n");
    printf("Initializing components...\n");
    init_and_connect_blobs();

    // Forward pass
    printf("Forward pass...\n");
#ifdef PROF_NET
    START_STATS();
#endif
    forward();
#ifdef PROF_NET
    STOP_STATS();
#endif

    // Perform forward check
    printf("\nChecking forward step results: \n");
    mean_error_checker(concat_output_data, OUTPUT, OUTPUT_SIZE);
    elementwise_checker(concat_output_data, OUTPUT, OUTPUT_SIZE);

    return;
}
