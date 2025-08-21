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
PI_L1 fp16 zero_init = 0.0f;
PI_L1 fp16 min_float = -65504.0f;

// ~~~~~~~~~~ L2 DATA ~~~~~~~~~~

// Define DNN blobs

// MHSA
PI_L2 struct blob_fp16 mhsa_input, mhsa_input_bn, mhsa_output, mhsa_output_wgt, mhsa_output_bias, mhsa_q, mhsa_q_wgt, mhsa_q_bias, mhsa_k, mhsa_k_wgt, mhsa_k_bias, mhsa_v, mhsa_v_wgt, mhsa_v_bias, mhsa_softmax_buffer, mhsa_att_map;

// Define DNN layer structures

PI_L2 struct Mhsa_args_fp16 mhsa_args;                               // MHSA

// Define I/O tensors

PI_L2 fp16 buff_a[SEQ_LEN * EMBED_SIZE];
PI_L2 fp16 buff_b[SEQ_LEN * EMBED_SIZE];
PI_L2 fp16 buff_c[SEQ_LEN * EMBED_SIZE];
PI_L2 fp16 buff_d[SEQ_LEN * EMBED_SIZE * 4];

// Other Data

PI_L2 fp16 mhsa_maxes[SEQ_LEN];
PI_L2 fp16 mhsa_sums[SEQ_LEN];
PI_L2 fp16 mhsa_softmax_buffer_v[SEQ_LEN * SEQ_LEN * N_HEADS];


// ~~~~~~~~~~ DNN BACKEND FUNCTIONS ~~~~~~~~~~

// DNN initialization function
void DNN_init()
{
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		buff_a[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		buff_b[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		buff_c[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*4*SEQ_LEN; i++)	buff_d[i] = zero_init;
    // ~~~~~~~~~~ CONNECT TENSOR TO BLOBS ~~~~~~~~~~

    // Connecting MHSA 
    mhsa_input.data = INPUT;
    mhsa_input.dim = INPUT_SIZE;
    mhsa_input.H = SEQ_LEN; 
    mhsa_input.W = HIDDEN_SIZE;
    mhsa_input.C = 1;

    mhsa_input_bn.data = INPUT;
    mhsa_input_bn.dim = SEQ_LEN*EMBED_SIZE;
    mhsa_input_bn.H = SEQ_LEN;
    mhsa_input_bn.W = EMBED_SIZE;
    mhsa_input_bn.C = 1;

    mhsa_output.data = buff_a;
    mhsa_output.dim = SEQ_LEN*EMBED_SIZE;
    mhsa_output.H = SEQ_LEN;
    mhsa_output.W = EMBED_SIZE;
    mhsa_output.C = 1;

    mhsa_output_wgt.data = ATTENTION_OUTPUT_WEIGHTS;
    mhsa_output_wgt.dim = EMBED_SIZE*EMBED_SIZE;
    mhsa_output_wgt.H = EMBED_SIZE;
    mhsa_output_wgt.W = EMBED_SIZE;
    mhsa_output_wgt.C = 1;

    mhsa_output_bias.data = ATTENTION_OUTPUT_BIASES;
    mhsa_output_bias.dim = EMBED_SIZE;
    mhsa_output_bias.H = 1;
    mhsa_output_bias.W = EMBED_SIZE;
    mhsa_output_bias.C = 1;

    mhsa_q.data = buff_b; 
    mhsa_q.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_q.H = EMBED_SIZE; //Transposed
    mhsa_q.W = SEQ_LEN;
    mhsa_q.C = 1;

    mhsa_q_wgt.data = INPUT_WEIGHTS_Q;
    mhsa_q_wgt.dim = EMBED_SIZE * EMBED_SIZE;
    mhsa_q_wgt.H = EMBED_SIZE;
    mhsa_q_wgt.W = EMBED_SIZE;
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
    mhsa_k_wgt.W = EMBED_SIZE;
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
    pulp_mhsa_mobilebert_inference_fp16_fw_cl((void*) &mhsa_args);
    return;
}

// ~~~~~~~~~~~~~~~~~~~~ UTILITY FUNCTIONS ~~~~~~~~~~~~~~~~~~~~
// Mean error checker
static inline void compare_tensors(fp16 *A, fp16 *B, int length) {
    fp16 mean_err_rel = zero_init;
    fp16 diff;
    fp16 mean_abs_value = zero_init;
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

    fp16 std_err = sqrt(err_variance);
    fp16 std_abs = sqrt(abs_value_variance);

    if (mean_err_rel < ERROR_TOLERANCE) printf("\n>>>TENSOR MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
    else printf("\n>>>TENSOR NOT MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);

    printf("\n>>>MEAN ERROR:%f MEAN GM ABS OUTPUT:%f\n", mean_err_rel, mean_abs_value);
    printf("\n>>>MEAN ERROR / MEAN GM OUTPUT ABS VALUE:%f\n",  mean_err_rel / mean_abs_value);
    printf("\n>>>ERROR VARIANCE:%f ABS GM OUTPUT VARIANCE:%f\n", err_variance, abs_value_variance);
    printf("\n>>>STD DEVIATIONS: ERROR->%f  ABS ->%f\n", std_err, std_abs);
}


// Elementwise checker
int check_tensor(fp16 *tensor_out, fp16 *tensor_ref, int size) {
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
