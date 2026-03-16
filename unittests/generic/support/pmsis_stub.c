#include "pmsis.h"

static int core_id = 0;

int pi_core_id(void)
{
    return core_id;
}

void pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg)
{
    // execute in time instead of parallel
    for (core_id=0; core_id<nb_cores; core_id++) {
        entry(arg);
    }
}

void pi_cl_team_barrier(void)
{
}

void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *copy)
{
    // TODO: implement!
    (void)copy;
}

void pi_cl_dma_wait(void *copy)
{
    (void)copy;
}

void pi_perf_conf(unsigned int events)
{
}

void pi_perf_start(void)
{
}

void pi_perf_stop(void)
{
}

void pi_perf_reset(void)
{
}

unsigned int pi_perf_read(int event)
{
    return 0;
}
