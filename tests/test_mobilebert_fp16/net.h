// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0
#define MAX_SIZE        33824
#define MAX_SIZE_L2     180736

#include "pulp_train_defines.h"
#include "pmsis.h"
#include <bsp/bsp.h>
#include "bsp/ram/hyperram.h"
#include "bsp/ram/spiram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/flash/spiflash.h"
#include <bsp/fs/readfs.h>
#include "bsp/fs.h"

// Tensor checksum definition
#define CHECK_TOLERANCE 0.001
#define ERROR_TOLERANCE 0.001

static inline void compare_tensors(fp16 *A, fp16 *B, int length);
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size);

// Netowork functions
void DNN_init();
void forward();
void net_step();
//void tiled_matmul(void* matmul_args, int flash_input);
void tiled_matmul(void* matmul_args);
void tiled_norm(void* nonorm_args);
// void tiled_skip(void* residual_args, int flash_lout);
void tiled_skip(void* residual_args);
void tiled_relu(void* Relu_args);

// DMA managment functions
void reset_arguments();
void update_blob();
void reset_dim();

//utility struct and functions for reading from a file inside GAP9

typedef struct{
  struct pi_device fs;
  struct pi_device flash;
  pi_fs_file_t *file;
} AT_FLASH_FS_T;

static inline void __at_flash_fs_open(AT_FLASH_FS_T *file, int is_write, struct pi_readfs_conf *conf, const char *filename, int *err);
static inline void __at_default_flash_fs_open(AT_FLASH_FS_T *file, int is_write, struct pi_readfs_conf *conf, const char *filename, int *err);
static inline void __at_flash_fs_close(AT_FLASH_FS_T *file);
static inline void __at_default_flash_file_open(AT_FLASH_FS_T *file, int is_write, const char *filename, int *err);

