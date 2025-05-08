#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "math.h"

#include "net-args.h"
#include "attention-defines.h"
#include "input-sequence.h"
#include "output-defines.h"
#include "output-sequence.h"

// ~~~~~~~~~~ VARIABLE DEFINITION ~~~~~~~~~~

// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

// ~~~~~~~~~~ L1 DATA ~~~~~~~~~~

PI_L1 float * IN_DATA, * W_DATA, * OUT_DATA, * BIAS_DATA;
PI_L1 float  BUFF[MAX_SIZE];
PI_L1 struct blob input_blob;
PI_L1 struct blob weight_blob;
PI_L1 struct blob output_blob;
PI_L1 struct blob bias_blob;
PI_L1 struct matMul_args mm_args;
PI_L1 struct act_args relu_args;
PI_L1 struct mm_manager_args man_args;
PI_L1 struct Tiled_Matmul_Mhsa_args tiled_matmul_mhsa_args;
PI_L1 pi_cl_dma_cmd_t * cmd_store;
PI_L1 pi_cl_dma_cmd_t * cmd_load;


// ~~~~~~~~~~ L2 DATA ~~~~~~~~~~

// Define DNN blobs

// MHSA
PI_L2 struct blob mhsa_input, mhsa_input_bn, mhsa_output, mhsa_output_wgt, mhsa_output_bias, mhsa_q, mhsa_q_wgt, mhsa_q_bias, mhsa_k, mhsa_k_wgt, mhsa_k_bias, mhsa_v, mhsa_v_wgt, mhsa_v_bias, mhsa_softmax_buffer, mhsa_att_map;

// Define DNN layer structures

PI_L2 struct Mhsa_args mhsa_args;                               // MHSA

// Define L2 I/O tensors
#if EMBED_SIZE > HIDDEN_SIZE
PI_L2 float buff_a[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_b[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_c[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_d[SEQ_LEN * EMBED_SIZE * 4];
#else
PI_L2 float buff_a[SEQ_LEN * HIDDEN_SIZE];
PI_L2 float buff_b[SEQ_LEN * HIDDEN_SIZE];
PI_L2 float buff_c[SEQ_LEN * HIDDEN_SIZE];
PI_L2 float buff_d[SEQ_LEN * HIDDEN_SIZE * 4];
#endif


// Other Data for MHSA

PI_L1 float mhsa_maxes[SEQ_LEN];
PI_L1 float mhsa_sums[SEQ_LEN];
PI_L2 float mhsa_softmax_buffer_v[SEQ_LEN * SEQ_LEN * N_HEADS];


// ~~~~~~~~~~ DNN BACKEND FUNCTIONS ~~~~~~~~~~

// DNN initialization function
void DNN_init()
{
    // ~~~~~~~~~~ ASSIGN POINTERS IN L1 ~~~~~~~~~~
    IN_DATA = BUFF;
    W_DATA = BUFF;
    OUT_DATA = BUFF;
    BIAS_DATA = BUFF;
    update_blob();
    reset_arguments();

    // ~~~~~~~~~~ INITIALIZING BUFFER DATA TO ZERO ~~~~~~~~~~

    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_a[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_b[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_c[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*4*SEQ_LEN; i++)	        buff_d[i] = zero_init;
    for(int i=0; i<SEQ_LEN; i++)		                mhsa_maxes[i] = min_float;
    for(int i=0; i<SEQ_LEN; i++)		                mhsa_sums[i] = zero_init;
    for(int i=0; i<SEQ_LEN * SEQ_LEN * N_HEADS; i++)	mhsa_softmax_buffer_v[i] = zero_init;
    for(int i = 0; i < MAX_SIZE; i++)                   BUFF[i] = zero_init;

    // ~~~~~~~~~~ CONNECT TENSOR TO BLOBS ~~~~~~~~~~

    // Connecting MHSA 
    mhsa_input.data = INPUT;
    mhsa_input.dim = INPUT_SIZE;
    mhsa_input.H = SEQ_LEN; 
    mhsa_input.W = HIDDEN_SIZE;
    mhsa_input.C = 1;

    mhsa_input_bn.data = INPUT;
    mhsa_input_bn.dim = SEQ_LEN*HIDDEN_SIZE;
    mhsa_input_bn.H = SEQ_LEN;
    mhsa_input_bn.W = HIDDEN_SIZE;
    mhsa_input_bn.C = 1;

    mhsa_output.data = buff_a;
    mhsa_output.dim = SEQ_LEN*HIDDEN_SIZE;
    mhsa_output.H = SEQ_LEN;
    mhsa_output.W = HIDDEN_SIZE;
    mhsa_output.C = 1;

    mhsa_output_wgt.data = ATTENTION_OUTPUT_WEIGHTS;
    mhsa_output_wgt.dim = HIDDEN_SIZE*EMBED_SIZE;
    mhsa_output_wgt.H = HIDDEN_SIZE;
    mhsa_output_wgt.W = EMBED_SIZE;
    mhsa_output_wgt.C = 1;

    mhsa_output_bias.data = ATTENTION_OUTPUT_BIASES;
    mhsa_output_bias.dim = HIDDEN_SIZE;
    mhsa_output_bias.H = 1;
    mhsa_output_bias.W = HIDDEN_SIZE;
    mhsa_output_bias.C = 1;

    mhsa_q.data = buff_b; 
    mhsa_q.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_q.H = EMBED_SIZE; //Transposed
    mhsa_q.W = SEQ_LEN;
    mhsa_q.C = 1;

    mhsa_q_wgt.data = INPUT_WEIGHTS_Q;
    mhsa_q_wgt.dim = EMBED_SIZE * HIDDEN_SIZE;
    mhsa_q_wgt.H = EMBED_SIZE;
    mhsa_q_wgt.W = HIDDEN_SIZE;
    mhsa_q_wgt.C = 1;

    mhsa_q_bias.data = INPUT_BIASES_Q;
    mhsa_q_bias.dim = EMBED_SIZE;
    mhsa_q_bias.H = 1;
    mhsa_q_bias.W = EMBED_SIZE;
    mhsa_q_bias.C = 1; 

    mhsa_k.data = buff_c; 
    mhsa_k.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_k.H = EMBED_SIZE; //Transposed
    mhsa_k.W = SEQ_LEN;
    mhsa_k.C = 1;

    mhsa_k_wgt.data = INPUT_WEIGHTS_K;
    mhsa_k_wgt.dim = EMBED_SIZE * EMBED_SIZE;
    mhsa_k_wgt.H = EMBED_SIZE;
    mhsa_k_wgt.W = HIDDEN_SIZE;
    mhsa_k_wgt.C = 1;

    mhsa_k_bias.data = INPUT_BIASES_K;
    mhsa_k_bias.dim = EMBED_SIZE;
    mhsa_k_bias.H = 1;
    mhsa_k_bias.W = EMBED_SIZE;
    mhsa_k_bias.C = 1; 

    mhsa_v.data = buff_a;
    mhsa_v.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_v.H = EMBED_SIZE;
    mhsa_v.W = SEQ_LEN;
    mhsa_v.C = 1;

    mhsa_v_wgt.data = INPUT_WEIGHTS_V;
    mhsa_v_wgt.dim = HIDDEN_SIZE * EMBED_SIZE;
    mhsa_v_wgt.H = EMBED_SIZE;
    mhsa_v_wgt.W = HIDDEN_SIZE; 
    mhsa_v_wgt.C = 1;

    mhsa_v_bias.data = INPUT_BIASES_V;
    mhsa_v_bias.dim = EMBED_SIZE;
    mhsa_v_bias.H = 1;
    mhsa_v_bias.W = EMBED_SIZE;
    mhsa_v_bias.C = 1; 

    mhsa_softmax_buffer.data = mhsa_softmax_buffer_v;
    mhsa_softmax_buffer.dim = SEQ_LEN * SEQ_LEN * N_HEADS;
    mhsa_softmax_buffer.H = SEQ_LEN * N_HEADS;
    mhsa_softmax_buffer.W = SEQ_LEN;
    mhsa_softmax_buffer.C = 1;

    mhsa_att_map.data = buff_b;
    mhsa_att_map.dim = SEQ_LEN * EMBED_SIZE;
    mhsa_att_map.H = EMBED_SIZE;
    mhsa_att_map.W = SEQ_LEN;
    mhsa_att_map.C = 1;


    // ~~~~~~~~~~ CONFIGURE LAYER STRUCTURES ~~~~~~~~~~

    // MHSA
    mhsa_args.input = &mhsa_input;
    mhsa_args.input_bn = &mhsa_input_bn;
    mhsa_args.n_heads = N_HEADS;
    mhsa_args.q = &mhsa_q;
    mhsa_args.k = &mhsa_k;
    mhsa_args.v = &mhsa_v;
    mhsa_args.output = &mhsa_output;

    mhsa_args.coeff_in_q = &mhsa_q_wgt;
    mhsa_args.coeff_in_k = &mhsa_k_wgt;
    mhsa_args.coeff_in_v = &mhsa_v_wgt;

    mhsa_args.bias_in_q = &mhsa_q_bias;
    mhsa_args.bias_in_k = &mhsa_k_bias;
    mhsa_args.bias_in_v = &mhsa_v_bias;

    mhsa_args.coeff_out = &mhsa_output_wgt;
    mhsa_args.bias_out = &mhsa_output_bias;
    mhsa_args.attention_map = &mhsa_att_map;
    mhsa_args.softmax_buffer = &mhsa_softmax_buffer;
    mhsa_args.temp_buffer = buff_d;
    mhsa_args.sums = mhsa_sums;
    mhsa_args.maxes = mhsa_maxes;
    mhsa_args.opt_matmul_type_fw = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_wg = MATMUL_TYPE;
    mhsa_args.opt_matmul_type_ig = MATMUL_TYPE;

    return;
}


void forward(){
    // MHSA HERE
    tiled_mhsa_fp32((void*) &mhsa_args, (void*) &tiled_matmul_mhsa_args);
    return;
}

// ~~~~~~~~~~~~~~~~~~~~ UTILITY FUNCTIONS ~~~~~~~~~~~~~~~~~~~~
// Mean error checker
static inline void compare_tensors(float *A, float *B, int length) {
    float mean_err_rel = zero_init;
    float diff;
    float mean_abs_value = zero_init;
    double err_variance = zero_init;
    double abs_value_variance = zero_init;

    for (int i = 0; i < length; i++) {
        diff = A[i] - B[i];
        if (diff > 0) diff = diff;
        else diff = -diff;
        mean_err_rel = mean_err_rel + diff;
        if (B[i] > 0)
            mean_abs_value = mean_abs_value + B[i] / length;
        else
            mean_abs_value = mean_abs_value - B[i] / length;
    }

    mean_err_rel = mean_err_rel / length;


    for (int i = 0; i < length; i++) {
        diff = A[i] - B[i];
        if (diff > 0) diff = diff;
        else diff = -diff;

        err_variance = err_variance + pow((diff - mean_err_rel),2) / length;

        if (B[i] > 0)
            abs_value_variance = abs_value_variance + pow((B[i] - mean_abs_value),2) / length;
        else
            abs_value_variance = abs_value_variance + pow(((-B[i]) - mean_abs_value),2) / length;
    }

    float std_err = sqrt(err_variance);
    float std_abs = sqrt(abs_value_variance);

    if (mean_err_rel < ERROR_TOLERANCE) printf("\n>>>TENSOR MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
    else printf("\n>>>TENSOR NOT MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);

    printf("\n>>>MEAN ERROR:%f MEAN GM ABS OUTPUT:%f\n", mean_err_rel, mean_abs_value);
    printf("\n>>>MEAN ERROR / MEAN GM OUTPUT ABS VALUE:%f\n",  mean_err_rel / mean_abs_value);
    printf("\n>>>ERROR VARIANCE:%f ABS GM OUTPUT VARIANCE:%f\n", err_variance, abs_value_variance);
    printf("\n>>>STD DEVIATIONS: ERROR->%f  ABS ->%f\n", std_err, std_abs);
}


// Elementwise checker
int check_tensor(float *tensor_out, float *tensor_ref, int size) {
    int error_flag = 0;

    for (int i = 0; i < size; i++) {
        if (ABS(tensor_out[i] - tensor_ref[i]) > CHECK_TOLERANCE) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i,
                   tensor_ref[i], *(unsigned int *) &tensor_ref[i], tensor_out[i], *(unsigned int *) &tensor_out[i]);
            error_flag = 1;
        }
    }

    return error_flag;
}

// A thing
void print_stuff(){
    //printf("\n%f %f %f\n", buff_a[0], buff_b[0], buff_c[0]);

    /*
    for(int i = 0; i < (EMBED_SIZE*SEQ_LEN); i++){
        if(!(i % EMBED_SIZE))
            printf("\n");
        printf("%f ", buff_c[i]);
    }
    printf("\n");
    */

    for(int i = 0; i < (HIDDEN_SIZE*SEQ_LEN); i++){
        if(!(i % HIDDEN_SIZE))
            printf("\n");
        printf("%f ", buff_a[i]);
    }
    printf("\n");
}

// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    printf("MHSA test:\n");
    DNN_init();
    #ifdef PROF_NET
    START_STATS();
    #endif
    forward();
    #ifdef PROF_NET
    STOP_STATS();
    #endif
    //print_stuff();

    printf("\nFORWARD CHECK: \n");
    compare_tensors(buff_a, OUTPUT, OUTPUT_SIZE);
    //check_tensor(buff_a, OUTPUT, OUTPUT_SIZE);

    return;
}

// ~~~~~~~~~~~~~~~~~~~~ DMA MANAGEMENT FUNCTIONS ~~~~~~~~~~~~~~~~~~~~

void reset_dim(){
	input_blob.dim = 0;
	weight_blob.dim = 0;
	output_blob.dim = 0;
}

void update_blob(){
	input_blob.data = IN_DATA;
	output_blob.data = OUT_DATA;
	weight_blob.data = W_DATA;
	bias_blob.data = BIAS_DATA;
}

void reset_arguments(){
    man_args.layer_type = LAYER_LINEAR;
    man_args.step_type = STEP_FW;
    man_args.matmul_type = MATMUL_TYPE;
    tiled_matmul_mhsa_args.mm_args = &mm_args;
    tiled_matmul_mhsa_args.BUFF = BUFF;
    tiled_matmul_mhsa_args.tile_h = TILE_H;
    tiled_matmul_mhsa_args.tile_w = TILE_W;
    tiled_matmul_mhsa_args.tile_dim = TILE_DIM;
    tiled_matmul_mhsa_args.tile_h_p = TILE_H_P;
    tiled_matmul_mhsa_args.tile_w_p = TILE_W_P;
    tiled_matmul_mhsa_args.tile_dim_p = TILE_DIM_P;
    tiled_matmul_mhsa_args.tile_h_sm = TILE_H_SM;
    tiled_matmul_mhsa_args.tile_w_sm = TILE_W_SM;
    tiled_matmul_mhsa_args.tile_dim_sm = TILE_DIM_SM;
    tiled_matmul_mhsa_args.tile_h_tr = TILE_H_TR;
    tiled_matmul_mhsa_args.tile_w_tr = TILE_W_TR;
    tiled_matmul_mhsa_args.tile_dim_tr = TILE_DIM_TR;
    tiled_matmul_mhsa_args.tile_h_attv = TILE_H_ATTV;
    tiled_matmul_mhsa_args.tile_w_attv = TILE_W_ATTV;
    tiled_matmul_mhsa_args.tile_dim_attv = TILE_DIM_ATTV;
    tiled_matmul_mhsa_args.tile_h_out_tr = TILE_H_OUT_TR;
    tiled_matmul_mhsa_args.tile_w_out_tr = TILE_W_OUT_TR;
    tiled_matmul_mhsa_args.tile_dim_out_tr = TILE_DIM_OUT_TR;
    tiled_matmul_mhsa_args.man_args = &man_args;
    tiled_matmul_mhsa_args.cmd_load = cmd_load;
    tiled_matmul_mhsa_args.cmd_store = cmd_store;
}
