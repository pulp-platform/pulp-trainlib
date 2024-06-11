/**
 * @brief Structure for weighted scaling in FP32
 * @param out                   weight * (scalinf_factor * input)            
 * @param in                    input vector
 * @param w                     weights
 * @param scaling_factor        scaling factor  
 * @param size                  size of input vector
 */
struct weighted_scaling_args {
    float* out;
    float* in;
    float* w;
    float scaling_factor;
    unsigned int size;
};

/**
 * @brief Structure for summing the squared values in FP32
 * @param out       sum(input ^ 2)         
 * @param in        input vector              
 * @param size      size of input vector      
 */
struct sum_of_squares_args {
    float* out;
    float* in;
    unsigned int size;
};

/**
 * @brief weight * (scalinf_factor * input). Set up the arguments by using a "struct weighted_scaling_args" structure. Use pi_cl_team_fork(NUM_CORES, weighted_scaling_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct weighted_scaling_args void_args)    
 */
void weighted_scaling_fp32_cl(void* weighted_scaling_args);

/**
 * @brief sum(input ^ 2). Set up the arguments by using a "struct sum_of_squares_args" structure. Use pi_cl_team_fork(NUM_CORES, sum_of_squares_fp32_cl, &args) to parallelize.
 * @param (void *)  (struct sum_of_squares_args void_args)    
 */
void sum_of_squares_fp32_cl(void* sum_of_squares_args);

/**
 * @brief Rmsnorm kernel
 * @param o                     output vector for eighter scaling operation
 * @param x                     input vector
 * @param weight                rmsnorm weights
 * @param buffer_n_cores        support vector to save the sum of squares (one for each core)
 * @param size                  size of input vector
 */
void rmsnorm_parallelized_fp32(float* o, float* x, float* weight, float* buffer_n_cores, int size);