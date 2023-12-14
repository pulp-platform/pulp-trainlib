// PULP Defines
#define STACK_SIZE      4096

// Tolerance to check updated output
#define TOLERANCE 1e-6

// Training functions
void DNN_init();
void compute_loss();
void update_weights();
void forward();
void backward();
void net_step();

// Print and check functions
void print_output();
void check_post_training_output();
