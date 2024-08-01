#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "net_args.h"
#include "dropout_data.h"


// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    struct dropout_args_fp32 args;
    args.seed = SEED;
    args.probability = PROBABILITY;
    args.input = input;
    args.mask = mask;
    args.use_mask = USE_MASK;
    args.size = IN_SIZE;

    printf("Dropout function:\n");
    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pulp_cl_dropout_fp32, &args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    int count = 0;

    for(int i = 0; i < IN_SIZE; i++){
        //printf("%f\n", input[i]);
        if(input[i]==0.0f)
            count++;
    }

    printf("%d\n", count);
    printf("Percentage of dropped out values: %f\%\n", (count*100.0f/SIZE));

    return;
}
