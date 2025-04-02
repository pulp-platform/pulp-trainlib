#ifndef NET_H
#define NET_H

// PULP DEFINES
#define STACK_SIZE      40960
#define MOUNT           1
#define UNMOUNT         0
#define CID             0
#define MAX_SIZE        25104

#include "pulp_train_defines.h"

// net functions
void forward();
void net_step();

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
#endif
