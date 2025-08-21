// ~~~~~~~~~~ INCLUDES ~~~~~~~~~~
#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "layer_norm_init_defines.h"
#include "layer_norm_input.h"
#include "layer_norm_wb.h"
#include "layer_norm_output.h"

// ~~~~~~~~~~ PREPARE COMPONENTS ~~~~~~~~~~
// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

#include "tensor_checkers.h"

// Constants
PI_L1 float eps[1] = {0.00001};

// Structures
PI_L2 struct LayerNorm_args_fp32 layer_norm_args;

// Data
PI_L2 float output_data[SHAPE];

// Init and connect blobs
void init_and_connect_blobs() {
    for (int i = 0; i < SHAPE; i++) output_data[i] = zero_init;

    layer_norm_args.x = INPUT;
    layer_norm_args.weight = WEIGHT;
    layer_norm_args.bias = BIAS;
    layer_norm_args.output = output_data;
    layer_norm_args.eps = eps;
    layer_norm_args.size = SHAPE;
    layer_norm_args.step_size = STEP_SIZE;
}

// ~~~~~~~~~~ FORWARD AND MAIN FUNCS ~~~~~~~~~~
// Define forward step function
void forward() {
    pi_cl_team_fork(NUM_CORES, pulp_layerNorm_fp32_fw_cl, &layer_norm_args);
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
    printf("LayerNorm test:\n");
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
    mean_error_checker(output_data, OUTPUT, SHAPE);
    elementwise_checker(output_data, OUTPUT, SHAPE);

    return;
}
