// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0
#define MAX_SIZE        25104

#include "pulp_train_defines.h"

// Tensor checksum definition
#define CHECK_TOLERANCE 0.001
#define ERROR_TOLERANCE 0.001

static inline void compare_tensors(float *A, float *B, int length);
int check_tensor(float * tensor_out, float * tensor_ref, int size);

// Netowork functions
void DNN_init();
void forward();
void net_step();
void tiled_matmul(void* matmul_args);
void tiled_norm(void* nonorm_args);
void tiled_skip(void* residual_args);
void tiled_relu(void* Relu_args);

// DMA managment functions
void load_input(void * src_blob, uint8_t data_diff_both);
void load_output(void * src_blob, uint8_t data_diff_both);
void load_coeff(void * src_blob, uint8_t data_diff_both);
void store_output(void * dest_blob, uint8_t data_diff_both);
void store_input(void * dest_blob, uint8_t data_diff_both);
void store_coeff(void * dest_blob, uint8_t data_diff_both);
void copy_struct_param(unsigned int from, unsigned int to, int size);
void get_input_dim(void * b);
void get_output_dim(void * b);
void get_weight_dim(void * b);
void reset_arguments();
void update_blob();
void reset_dim();