#ifndef PMSIS_H
#define PMSIS_H

#include <stdint.h>

/* 
 * typedefs
 */
typedef uint16_t float16alt;

typedef enum {
  PI_CL_DMA_DIR_LOC2EXT = 0,
  PI_CL_DMA_DIR_EXT2LOC = 1
} pi_cl_dma_dir_e;

typedef struct
{
    uint32_t ext;
    uint32_t loc;
    uint32_t id; 
    uint16_t size;
    pi_cl_dma_dir_e dir;                        \
    uint8_t merge;
    // 2d transfers args
    uint32_t stride;
    uint32_t length;
} pi_cl_dma_copy_2d_t;

/*
 * functions
 */
int pi_core_id(void);
void pi_cl_team_fork(int nb_cores, void (*entry)(void *), void *arg);
void pi_cl_team_barrier(void);

void pi_cl_dma_memcpy_2d(pi_cl_dma_copy_2d_t *copy);
void pi_cl_dma_wait(void *copy);

void pi_perf_conf(unsigned int events);
void pi_perf_start(void);
void pi_perf_stop(void);
void pi_perf_reset(void);
unsigned int pi_perf_read(int event);

#endif /* PMSIS_H */
