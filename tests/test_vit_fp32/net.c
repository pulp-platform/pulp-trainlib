// ~~~~~~~~~~ INCLUDES ~~~~~~~~~~
#include "pulp_train.h"

#include "model_components.h"
#include "tensor_checkers.h"

#include "stats.h"
#include "net.h"


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

    mean_error_checker(fc_output_data, OUTPUT, OUTPUT_SIZE);
    elementwise_checker(fc_output_data, OUTPUT, OUTPUT_SIZE);

    return;
}
