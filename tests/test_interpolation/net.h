// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#include "pulp_train_defines.h"
#include "net_args.h"

// Tensor checksum definition
#define CHECK_TOLERANCE 1e-6
#define ERROR_TOLERANCE 1e-6

#ifdef FLOAT32
static inline void compare_tensors(float *A, float *B, int length);
int check_tensor(float * tensor_out, float * tensor_ref, int size);
#elif defined(BFLOAT16)
static inline void compare_tensors(fp16 *A, fp16 *B, int length);
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size);
#endif
void net_step ();
