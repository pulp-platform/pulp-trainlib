#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "math.h"

#include "net-args.h"
#include "attention-defines.h"
#include "input-sequence.h"
#include "output-defines.h"
#include "output-sequence.h"
#include "attention_scores.h"
#include "mhsa-grads.h"

// ~~~~~~~~~~ VARIABLE DEFINITION ~~~~~~~~~~

// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

// ~~~~~~~~~~ L2 DATA ~~~~~~~~~~

// Define DNN blobs

// MHSA
PI_L2 struct blob mhsa_input, mhsa_input_bn, mhsa_output, mhsa_output_wgt, mhsa_output_bias, mhsa_q, mhsa_q_wgt, mhsa_q_bias, mhsa_k, mhsa_k_wgt, mhsa_k_bias, mhsa_v, mhsa_v_wgt, mhsa_v_bias, mhsa_softmax_buffer, mhsa_att_map;

// Define DNN layer structures

PI_L2 struct Mhsa_args mhsa_args;                               // MHSA

// Define I/O tensors

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

// Other Data

PI_L2 float mhsa_maxes[SEQ_LEN];
PI_L2 float mhsa_sums[SEQ_LEN];
PI_L2 float mhsa_softmax_buffer_v[SEQ_LEN * SEQ_LEN * N_HEADS];

#ifdef BACKWARD
PI_L2 float input_diff[SEQ_LEN * HIDDEN_SIZE];
PI_L2 float output[SEQ_LEN * HIDDEN_SIZE];
PI_L2 float output_weights_diff[EMBED_SIZE * EMBED_SIZE];
PI_L2 float q[EMBED_SIZE * SEQ_LEN];
PI_L2 float q_diff[EMBED_SIZE * SEQ_LEN];
PI_L2 float q_weight_diff[EMBED_SIZE * EMBED_SIZE];
PI_L2 float k[EMBED_SIZE * SEQ_LEN];
PI_L2 float k_diff[EMBED_SIZE * SEQ_LEN];
PI_L2 float k_weight_diff[EMBED_SIZE * EMBED_SIZE];
PI_L2 float v[EMBED_SIZE * SEQ_LEN];
PI_L2 float v_diff[EMBED_SIZE * SEQ_LEN];
PI_L2 float v_weight_diff[EMBED_SIZE * EMBED_SIZE];
PI_L2 float mhsa_softmax_buffer_diff[SEQ_LEN * SEQ_LEN * N_HEADS];
PI_L2 float att_map[SEQ_LEN * EMBED_SIZE];
PI_L2 float att_map_diff[SEQ_LEN * EMBED_SIZE];
#endif


// ~~~~~~~~~~ DNN BACKEND FUNCTIONS ~~~~~~~~~~

// DNN initialization function
void DNN_init_forward()
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

#ifdef BACKWARD
void DNN_init_backward()
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
    mhsa_input.diff = input_diff;

    mhsa_input_bn.data = INPUT;
    mhsa_input_bn.dim = SEQ_LEN*EMBED_SIZE;
    mhsa_input_bn.H = SEQ_LEN;
    mhsa_input_bn.W = EMBED_SIZE;
    mhsa_input_bn.C = 1;

    mhsa_output.data = output;
    mhsa_output.dim = SEQ_LEN*EMBED_SIZE;
    mhsa_output.H = SEQ_LEN;
    mhsa_output.W = EMBED_SIZE;
    mhsa_output.C = 1;
    mhsa_output.diff = OUTPUT_GRAD;

    mhsa_output_wgt.data = ATTENTION_OUTPUT_WEIGHTS;
    mhsa_output_wgt.dim = EMBED_SIZE*EMBED_SIZE;
    mhsa_output_wgt.H = EMBED_SIZE;
    mhsa_output_wgt.W = EMBED_SIZE;
    mhsa_output_wgt.C = 1;
    mhsa_output_wgt.diff = output_weights_diff;

    mhsa_output_bias.data = ATTENTION_OUTPUT_BIASES;
    mhsa_output_bias.dim = EMBED_SIZE;
    mhsa_output_bias.H = 1;
    mhsa_output_bias.W = EMBED_SIZE;
    mhsa_output_bias.C = 1;

    mhsa_q.data = q; 
    mhsa_q.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_q.H = EMBED_SIZE; //Transposed
    mhsa_q.W = SEQ_LEN;
    mhsa_q.C = 1;
    mhsa_q.diff = q_diff;

    mhsa_q_wgt.data = INPUT_WEIGHTS_Q;
    mhsa_q_wgt.dim = EMBED_SIZE * EMBED_SIZE;
    mhsa_q_wgt.H = EMBED_SIZE;
    mhsa_q_wgt.W = EMBED_SIZE;
    mhsa_q_wgt.C = 1;
    mhsa_q_wgt.diff = q_weight_diff;

    mhsa_q_bias.data = INPUT_BIASES_Q;
    mhsa_q_bias.dim = EMBED_SIZE;
    mhsa_q_bias.H = 1;
    mhsa_q_bias.W = EMBED_SIZE;
    mhsa_q_bias.C = 1; 

    mhsa_k.data = k; 
    mhsa_k.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_k.H = EMBED_SIZE; //Transposed
    mhsa_k.W = SEQ_LEN;
    mhsa_k.C = 1;
    mhsa_k.diff = k_diff;

    mhsa_k_wgt.data = INPUT_WEIGHTS_K;
    mhsa_k_wgt.dim = EMBED_SIZE * EMBED_SIZE;
    mhsa_k_wgt.H = EMBED_SIZE;
    mhsa_k_wgt.W = EMBED_SIZE;
    mhsa_k_wgt.C = 1;
    mhsa_k_wgt.diff = k_weight_diff;

    mhsa_k_bias.data = INPUT_BIASES_K;
    mhsa_k_bias.dim = EMBED_SIZE;
    mhsa_k_bias.H = 1;
    mhsa_k_bias.W = EMBED_SIZE;
    mhsa_k_bias.C = 1; 

    mhsa_v.data = v;
    mhsa_v.dim = EMBED_SIZE*SEQ_LEN;
    mhsa_v.H = EMBED_SIZE;
    mhsa_v.W = SEQ_LEN;
    mhsa_v.C = 1;
    mhsa_v.diff = v_diff;

    mhsa_v_wgt.data = INPUT_WEIGHTS_V;
    mhsa_v_wgt.dim = HIDDEN_SIZE * EMBED_SIZE;
    mhsa_v_wgt.H = EMBED_SIZE;
    mhsa_v_wgt.W = HIDDEN_SIZE; 
    mhsa_v_wgt.C = 1;
    mhsa_v_wgt.diff = v_weight_diff;

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
    mhsa_softmax_buffer.diff = mhsa_softmax_buffer_diff;

    mhsa_att_map.data = att_map;
    mhsa_att_map.dim = SEQ_LEN * EMBED_SIZE;
    mhsa_att_map.H = EMBED_SIZE;
    mhsa_att_map.W = SEQ_LEN;
    mhsa_att_map.C = 1;
    mhsa_att_map.diff = att_map_diff;


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
#endif


void forward(){
    #ifdef PROF_NET
    #ifdef FORWARD
    START_STATS();
    #endif
    #endif
    // MHSA FORWARD HERE
    pulp_mhsa_mobilebert_inference_fp32_fw_cl((void*) &mhsa_args);

    #ifdef BACKWARD
    printf("\nFORWARD CHECK: \n");
    compare_tensors(output, OUTPUT, OUTPUT_SIZE);
    printf("\nATTENTION SCORE CHECK: \n");
    compare_tensors(att_map, ATTENTION_SCORES, ATTENTION_S_LENGTH);

    #ifdef PROF_NET
    START_STATS();
    #endif
    //MHSA BACKWARD HERE
    pulp_mhsa_mobilebert_inference_fp32_bw_cl((void*) &mhsa_args);
    #endif

    #ifdef PROF_NET
    STOP_STATS();
    #endif
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
    #ifdef FORWARD
    DNN_init_forward();
    #endif
    #ifdef BACKWARD
    DNN_init_backward();
    #endif
    
    forward();
    
    //print_stuff();
    #ifdef FORWARD
    printf("\nFORWARD CHECK: \n");
    compare_tensors(buff_a, OUTPUT, OUTPUT_SIZE);
    //check_tensor(buff_a, OUTPUT, OUTPUT_SIZE);
    #endif

    #ifdef BACKWARD
    printf("\nBACKWARD CHECK: \n");
    printf("\nINPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(q_weight_diff, INPUT_WGT_GRAD_Q, G_INPUT_WGT_SIZE);
    compare_tensors(k_weight_diff, INPUT_WGT_GRAD_K, G_INPUT_WGT_SIZE);
    compare_tensors(v_weight_diff, INPUT_WGT_GRAD_V, G_INPUT_WGT_SIZE);

    printf("\nOUTPUT WEIGHTS GRADIENT CHECK: \n");
    compare_tensors(output_weights_diff, OUTPUT_WGT_GRAD, G_OUTPUT_WGT_SIZE);

    printf("\nINPUT GRADIENT CHECK: \n");
    compare_tensors(input_diff, INPUT_GRAD, G_IN_SIZE);
    #endif

    return;
}
