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

    // TODO: Loop over transformer blocks
    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_norm1
    pulp_layerNorm_fp32_fw_cl(&transformer_blocks_0_norm1_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_attn
    pulp_mhsa_fp32_fw_cl(&transformer_blocks_0_mhsa_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_proj
    pulp_linear_fp32_fw_cl(&transformer_blocks_0_proj_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_add_1
    pi_cl_team_fork(NUM_CORES, vect_sum, &transformer_blocks_0_add_1_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_norm2
    pulp_layerNorm_fp32_fw_cl(&transformer_blocks_0_norm2_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_pwff_fc1
    pulp_linear_fp32_fw_cl(&transformer_blocks_0_pwff_fc1_args);

    // Shape: b, dim, nph * npw + 1
    // transformer_blocks_0_pwff_gelu
    // TODO: IMPLEMENT

    // Shape: b, dim, nph * npw + 1
    // transformer_blocks_0_pwff_fc2
    pulp_linear_fp32_fw_cl(&transformer_blocks_0_pwff_fc2_args);

    // Shape: b, nph * npw + 1, dim
    // transformer_blocks_0_add_2
    pi_cl_team_fork(NUM_CORES, vect_sum, &transformer_blocks_0_add_2_args);

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
