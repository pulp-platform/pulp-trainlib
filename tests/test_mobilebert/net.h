// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#include "pulp_train_defines.h"

// Tensor checksum definition
#define CHECK_TOLERANCE 0.001
#define ERROR_TOLERANCE 0.001

static inline void forward();
static inline void compare_tensors(float *A, float *B, int length);
int check_tensor(float * tensor_out, float * tensor_ref, int size);

void net_step ();