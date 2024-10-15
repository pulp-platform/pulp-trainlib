// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0
#define MAX_SIZE        33824

#include "pulp_train_defines.h"

// Tensor checksum definition
#define CHECK_TOLERANCE 0.001
#define ERROR_TOLERANCE 0.001

static inline void compare_tensors(fp16 *A, fp16 *B, int length);
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size);

// Netowork functions
void DNN_init();
void forward();
void net_step();
void tiled_matmul(void* matmul_args);
void tiled_norm(void* nonorm_args);
void tiled_skip(void* residual_args);
void tiled_relu(void* Relu_args);

// DMA managment functions
void reset_arguments();
void update_blob();
void reset_dim();