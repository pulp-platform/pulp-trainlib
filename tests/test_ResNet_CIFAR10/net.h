// PULP Defines
#define STACK_SIZE      4096

// Tolerance to check updated output
#define TOLERANCE 1e-12

// Training functions
void DNN_init();
void compute_loss(int idx);
void update_weights();
void forward();
void backward();
void net_step();

// Print and check functions
void print_output();
void check_post_training_output();

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
void PrintBlob(void * b, int step);
void reset_dim();
#define MAX_IN_SIZE 6400
#define MAX_WGT_SIZE 6400
#define MAX_SIZE 27664
