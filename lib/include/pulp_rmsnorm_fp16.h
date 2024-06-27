#include "pmsis.h"
#include "pulp_train_defines.h"

struct weighted_scaling_args_fp16 {
    fp16* out;
    fp16* in;
    fp16* w;
    fp16 scaling_factor;
    unsigned int size;
};

struct sum_of_squares_args_fp16 {
    fp16* out;
    fp16* in;
    unsigned int size;
};

void weighted_scaling_fp16_cl(void* weighted_scaling_args_fp16);

void sum_of_squares_fp16_cl(void* sum_of_squares_args_fp16);

void rmsnorm_parallelized_fp16(fp16* o, fp16* x, fp16* weight, fp16* buffer_n_cores, int size);
