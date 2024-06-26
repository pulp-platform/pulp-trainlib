#include "pmsis.h"
#include "pulp_train_utils_fp16.h"
#include "pulp_rmsnorm_fp16.h"
#include "math.h"

void rmsnorm_parallelized_fp16(fp16* o, fp16* x, fp16* weight, fp16* buffer_n_cores, int size){
    struct sum_of_squares_args_fp16 ss_args;
    ss_args.in = x;
    ss_args.out = buffer_n_cores;
    ss_args.size = size;
    
    pi_cl_team_fork(NUM_CORES, sum_of_squares_fp16_cl, &ss_args);

    float ss = 0;
    for(int i=0; i<NUM_CORES; i++)
        ss += (float) buffer_n_cores[i];

    ss /= size;
    ss += 1e-5f;

    #ifdef Q_RSQRT
    ss = q_rsqrt_fp16(ss);
    #else
    ss = 1.0f / sqrtf(ss);
    #endif

    struct weighted_scaling_args_fp16 ws_args;
    ws_args.in = x;
    ws_args.out = o;
    ws_args.w = weight;
    ws_args.size = size;
    ws_args.scaling_factor = (fp16) ss;

    pi_cl_team_fork(NUM_CORES, weighted_scaling_fp16_cl, &ws_args);
}

void weighted_scaling_fp16_cl(void* weighted_scaling_args) {
    struct weighted_scaling_args_fp16* args = (struct weighted_scaling_args_fp16* )weighted_scaling_args;
    fp16* out = args->out;
    fp16* in = args->in;
    fp16* w = args->w;
    fp16 sf = args->scaling_factor;
    unsigned int size = args->size;

    const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > size ? size : start+blockSize;

    for (uint32_t i = start; i < stop; i++) {
        out[i] = w[i] * (sf * in[i]);
    }
}

void sum_of_squares_fp16_cl(void* ss_args){
    struct sum_of_squares_args_fp16* args = (struct sum_of_squares_args_fp16*)ss_args;
    fp16* out = args->out;
    fp16* in = args->in;
    unsigned int size = args->size;

    int id = pi_core_id();

    const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = id*blockSize;
    const uint32_t stop = start+blockSize > size ? size : start+blockSize;

    fp16 res = 0;
    for (uint32_t i = start; i < stop; i++) {
        res += in[i] * in[i];
    }
    out[id] = res;
}