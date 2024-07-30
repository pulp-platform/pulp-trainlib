#include "pulp_train.h"

#include "stats.h"
#include "net.h"

// ----------------- FP32 data ----------------------
PI_L1 float probability;
PI_L1 int output;

// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    struct integer_random_args args_fp32;
    probability = PROBABILITY;
    args_fp32.seed = SEED;
    args_fp32.probability = probability;
    args_fp32.output = &output;

    printf("Bernoulli Random Number Generator (from seed):\n");
    #ifdef PROF_NET
    START_STATS();
    #endif

    pulp_random_bernoulli(&args_fp32);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    printf("First run output: %d\n", output);

    printf("\nTest random number generation stats:");
    float mean = 0;
    int acc = 0;
    float var  = 0;
    for (int i=0; i<1000; i++) {
        pulp_random_bernoulli(&args_fp32);
        acc += output;
    }
    mean = (float) acc / 1000.0;
    printf("Mean: %f\n", mean);

    return;
}
