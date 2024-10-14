#include "pulp_train.h"

#include "stats.h"
#include "net.h"

#include "math.h"

#include "net_args.h"
#include "attention-defines.h"
#include "bottleneck-defines.h"
#include "ffn-defines.h"
#include "input-sequence.h"
#include "intermediate-defines.h"
#include "output-defines.h"
#include "output-sequence.h"
//#include "vocabulary.h"

// ~~~~~~~~~~ VARIABLE DEFINITION ~~~~~~~~~~

// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

// ~~~~~~~~~~ L2 DATA ~~~~~~~~~~

// Define DNN blobs

// Bottleneck for attention values
PI_L2 struct blob bneck_norm_att_in, bneck_norm_att_out, bneck_norm_att_wgt, bneck_norm_att_bias;
// Bottleneck for input values
PI_L2 struct blob bneck_norm_inp_in, bneck_norm_inp_out, bneck_norm_inp_wgt, bneck_norm_inp_bias;
// MHSA
PI_L2 struct blob mhsa_input, mhsa_input_bn, mhsa_output, mhsa_output_wgt, mhsa_output_bias, mhsa_q, mhsa_q_wgt, mhsa_q_bias, mhsa_k, mhsa_k_wgt, mhsa_k_bias, mhsa_v, mhsa_v_wgt, mhsa_v_bias, mhsa_softmax_buffer, mhsa_att_map;
// Residual connection MHSA output + input bottleneck
PI_L2 struct blob residual_1_skip, residual_1_lout, residual_1_output;
PI_L2 struct blob attention_output_norm_bias, attention_output_norm_in, attention_output_norm_out, attention_output_norm_wgt;
// FFN 0
PI_L2 struct blob ffn_0_relu_input, ffn_0_relu_output, ffn_0_residual_skip, ffn_0_residual_lout, ffn_0_residual_output;
PI_L2 struct blob ffn_0_norm_in, ffn_0_norm_wgt, ffn_0_norm_bias, ffn_0_norm_out;
// FFN 1
PI_L2 struct blob ffn_1_relu_input, ffn_1_relu_output, ffn_1_residual_skip, ffn_1_residual_lout, ffn_1_residual_output;
PI_L2 struct blob ffn_1_norm_in, ffn_1_norm_wgt, ffn_1_norm_bias, ffn_1_norm_out;
// FFN 2
PI_L2 struct blob ffn_2_relu_input, ffn_2_relu_output, ffn_2_residual_skip, ffn_2_residual_lout, ffn_2_residual_output;
PI_L2 struct blob ffn_2_norm_in, ffn_2_norm_wgt, ffn_2_norm_bias, ffn_2_norm_out;
// Intermediate
PI_L2 struct blob intermediate_relu_input, intermediate_relu_output;
// Output
PI_L2 struct blob output_residual_skip, output_residual_lout, output_residual_output;
PI_L2 struct blob output_norm_in, output_norm_wgt, output_norm_bias, output_norm_out;
// Output Bottleneck
PI_L2 struct blob output_bottleneck_residual_skip, output_bottleneck_residual_lout, output_bottleneck_residual_output;
PI_L2 struct blob output_bottleneck_norm_in, output_bottleneck_norm_wgt, output_bottleneck_norm_bias, output_bottleneck_norm_out;

// Define DNN layer structures

PI_L2 struct matMul_args bneck_dense_att_args;                  // Bottleneck for attention values
PI_L2 struct Nonorm_args bneck_norm_att_args;       
PI_L2 struct matMul_args bneck_dense_inp_args;                  // Bottleneck for input values
PI_L2 struct Nonorm_args bneck_norm_inp_args;       
PI_L2 struct Mhsa_args mhsa_args;                               // MHSA
PI_L2 struct SkipConn_args residual_1_args;                     // Residual connection MHSA output + input bottleneck
PI_L2 struct Nonorm_args attention_output_norm_args;            
PI_L2 struct matMul_args ffn_0_intermediate_args;               // FFN 0
PI_L2 struct act_args ffn_0_relu_args;      
PI_L2 struct matMul_args ffn_0_output_args;     
PI_L2 struct SkipConn_args ffn_0_residual_args;     
PI_L2 struct Nonorm_args ffn_0_norm_args;       
PI_L2 struct matMul_args ffn_1_intermediate_args;               // FFN 1
PI_L2 struct act_args ffn_1_relu_args;      
PI_L2 struct matMul_args ffn_1_output_args;     
PI_L2 struct SkipConn_args ffn_1_residual_args;     
PI_L2 struct Nonorm_args ffn_1_norm_args;       
PI_L2 struct matMul_args ffn_2_intermediate_args;               // FFN 2
PI_L2 struct act_args ffn_2_relu_args;      
PI_L2 struct matMul_args ffn_2_output_args;     
PI_L2 struct SkipConn_args ffn_2_residual_args;     
PI_L2 struct Nonorm_args ffn_2_norm_args;       
PI_L2 struct matMul_args intermediate_args;                     // Intermediate
PI_L2 struct act_args intermediate_relu_args;       
PI_L2 struct matMul_args output_dense_args;                     // Output
PI_L2 struct SkipConn_args output_residual_args;        
PI_L2 struct Nonorm_args output_norm_args;      
PI_L2 struct matMul_args output_bottleneck_dense_args;          // Output Bottleneck
PI_L2 struct SkipConn_args output_bottleneck_residual_args;     
PI_L2 struct Nonorm_args output_bottleneck_norm_args;


// Define I/O tensors

PI_L2 float buff_a[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_b[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_c[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_d[SEQ_LEN * HIDDEN_SIZE];

// Other Data

PI_L2 float mhsa_maxes[SEQ_LEN];
PI_L2 float mhsa_sums[SEQ_LEN];
PI_L2 float mhsa_softmax_buffer_v[SEQ_LEN * SEQ_LEN * N_HEADS];


// ~~~~~~~~~~ DNN BACKEND FUNCTIONS ~~~~~~~~~~

// DNN initialization function
void DNN_init()
{
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		buff_a[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		buff_b[i] = zero_init;
    // ~~~~~~~~~~ CONNECT TENSOR TO BLOBS ~~~~~~~~~~


    // Connecting Bottleneck for attention values
    bneck_norm_att_in.data = buff_a;
    bneck_norm_att_in.dim = SEQ_LEN * EMBED_SIZE;
    bneck_norm_att_in.H = SEQ_LEN;
    bneck_norm_att_in.W = EMBED_SIZE;
    bneck_norm_att_in.C = 1;

    bneck_norm_att_out.data = buff_a;
    bneck_norm_att_out.dim = SEQ_LEN * EMBED_SIZE;
    bneck_norm_att_out.H = SEQ_LEN;
    bneck_norm_att_out.W = EMBED_SIZE;
    bneck_norm_att_out.C = 1;

    bneck_norm_att_wgt.data = BOTTLENECK_ATTENTION_NORM_WEIGHTS;
    bneck_norm_att_wgt.dim = EMBED_SIZE;
    bneck_norm_att_wgt.H = 1;
    bneck_norm_att_wgt.W = EMBED_SIZE;
    bneck_norm_att_wgt.C = 1;

    bneck_norm_att_bias.data = BOTTLENECK_ATTENTION_NORM_BIASES;
    bneck_norm_att_bias.dim = EMBED_SIZE;
    bneck_norm_att_bias.H = 1;
    bneck_norm_att_bias.W = EMBED_SIZE;
    bneck_norm_att_bias.C = 1;

    // Connecting Bottleneck for input values
    bneck_norm_inp_in.data = buff_b;
    bneck_norm_inp_in.dim = SEQ_LEN * EMBED_SIZE;
    bneck_norm_inp_in.H = SEQ_LEN;
    bneck_norm_inp_in.W = EMBED_SIZE;
    bneck_norm_inp_in.C = 1;

    bneck_norm_inp_out.data = buff_b;
    bneck_norm_inp_out.dim = SEQ_LEN * EMBED_SIZE;
    bneck_norm_inp_out.H = SEQ_LEN;
    bneck_norm_inp_out.W = EMBED_SIZE;
    bneck_norm_inp_out.C = 1;

    bneck_norm_inp_wgt.data = BOTTLENECK_INPUT_NORM_WEIGHTS;
    bneck_norm_inp_wgt.dim = EMBED_SIZE;
    bneck_norm_inp_wgt.H = 1;
    bneck_norm_inp_wgt.W = EMBED_SIZE;
    bneck_norm_inp_wgt.C = 1;

    bneck_norm_inp_bias.data = BOTTLENECK_INPUT_NORM_BIASES;
    bneck_norm_inp_bias.dim = EMBED_SIZE;
    bneck_norm_inp_bias.H = 1;
    bneck_norm_inp_bias.W = EMBED_SIZE;
    bneck_norm_inp_bias.C = 1;

    // Connecting MHSA 
    mhsa_input.data = INPUT;
    mhsa_input.dim = INPUT_SIZE;
    mhsa_input.H = SEQ_LEN; 
    mhsa_input.W = HIDDEN_SIZE;
    mhsa_input.C = 1;

    mhsa_input_bn.data = buff_a;
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

    // Connecting residual connection MHSA output + input bottleneck
    residual_1_skip.data = buff_a;
    residual_1_skip.dim = SEQ_LEN*EMBED_SIZE;
    residual_1_skip.H = SEQ_LEN;
    residual_1_skip.W = EMBED_SIZE;
    residual_1_skip.C = 1;

    residual_1_lout.data = buff_b;
    residual_1_lout.dim = SEQ_LEN*EMBED_SIZE;
    residual_1_lout.H = SEQ_LEN;
    residual_1_lout.W = EMBED_SIZE;
    residual_1_lout.C = 1;

    residual_1_output.data = buff_c;
    residual_1_output.dim = SEQ_LEN*EMBED_SIZE;
    residual_1_output.H = SEQ_LEN;
    residual_1_output.W = EMBED_SIZE;
    residual_1_output.C = 1;

    attention_output_norm_in.data = buff_c;
    attention_output_norm_in.dim = SEQ_LEN*EMBED_SIZE;
    attention_output_norm_in.H = SEQ_LEN;
    attention_output_norm_in.W = EMBED_SIZE;
    attention_output_norm_in.C = 1;

    attention_output_norm_wgt.data = ATTENTION_OUTPUT_NORM_WEIGHTS;
    attention_output_norm_wgt.dim = EMBED_SIZE;
    attention_output_norm_wgt.H = 1;
    attention_output_norm_wgt.W = EMBED_SIZE;
    attention_output_norm_wgt.C = 1;

    attention_output_norm_bias.data = ATTENTION_OUTPUT_NORM_BIASES;
    attention_output_norm_bias.dim = EMBED_SIZE;
    attention_output_norm_bias.H = 1;
    attention_output_norm_bias.W = EMBED_SIZE;
    attention_output_norm_bias.C = 1;

    attention_output_norm_out.data = buff_c;
    attention_output_norm_out.dim = SEQ_LEN*EMBED_SIZE;
    attention_output_norm_out.H = SEQ_LEN;
    attention_output_norm_out.W = EMBED_SIZE;
    attention_output_norm_out.C = 1;

    // Connecting FFN 0
    ffn_0_relu_input.data = buff_d;
    ffn_0_relu_input.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_0_relu_input.H = SEQ_LEN;
    ffn_0_relu_input.W = HIDDEN_SIZE;
    ffn_0_relu_input.C = 1;

    ffn_0_relu_output.data = buff_d;
    ffn_0_relu_output.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_0_relu_output.H = SEQ_LEN;
    ffn_0_relu_output.W = HIDDEN_SIZE;
    ffn_0_relu_output.C = 1;

    ffn_0_residual_skip.data = buff_b;
    ffn_0_residual_skip.dim = SEQ_LEN*EMBED_SIZE;
    ffn_0_residual_skip.H = SEQ_LEN;
    ffn_0_residual_skip.W = EMBED_SIZE;
    ffn_0_residual_skip.C = 1;

    ffn_0_residual_lout.data = buff_c;
    ffn_0_residual_lout.dim = SEQ_LEN*EMBED_SIZE;
    ffn_0_residual_lout.H = SEQ_LEN;
    ffn_0_residual_lout.W = EMBED_SIZE;
    ffn_0_residual_lout.C = 1;

    ffn_0_residual_output.data = buff_a;
    ffn_0_residual_output.dim = SEQ_LEN*EMBED_SIZE;
    ffn_0_residual_output.H = SEQ_LEN;
    ffn_0_residual_output.W = EMBED_SIZE;
    ffn_0_residual_output.C = 1;

    ffn_0_norm_in.data = buff_a;
    ffn_0_norm_in.dim = SEQ_LEN*EMBED_SIZE;
    ffn_0_norm_in.H = SEQ_LEN;
    ffn_0_norm_in.W = EMBED_SIZE;
    ffn_0_norm_in.C = 1;

    ffn_0_norm_wgt.data = FFN0_OUTPUT_NORM_WEIGHTS;
    ffn_0_norm_wgt.dim = EMBED_SIZE;
    ffn_0_norm_wgt.H = 1;
    ffn_0_norm_wgt.W = EMBED_SIZE;
    ffn_0_norm_wgt.C = 1;

    ffn_0_norm_bias.data = FFN0_OUTPUT_NORM_BIASES;
    ffn_0_norm_bias.dim = EMBED_SIZE;
    ffn_0_norm_bias.H = 1;
    ffn_0_norm_bias.W = EMBED_SIZE;
    ffn_0_norm_bias.C = 1;

    ffn_0_norm_out.data = buff_a;
    ffn_0_norm_out.dim = SEQ_LEN*EMBED_SIZE;
    ffn_0_norm_out.H = SEQ_LEN;
    ffn_0_norm_out.W = EMBED_SIZE;
    ffn_0_norm_out.C = 1;

    // Connecting FFN 1
    ffn_1_relu_input.data = buff_d;
    ffn_1_relu_input.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_1_relu_input.H = SEQ_LEN;
    ffn_1_relu_input.W = HIDDEN_SIZE;
    ffn_1_relu_input.C = 1;

    ffn_1_relu_output.data = buff_d;
    ffn_1_relu_output.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_1_relu_output.H = SEQ_LEN;
    ffn_1_relu_output.W = HIDDEN_SIZE;
    ffn_1_relu_output.C = 1;

    ffn_1_residual_skip.data = buff_b;
    ffn_1_residual_skip.dim = SEQ_LEN*EMBED_SIZE;
    ffn_1_residual_skip.H = SEQ_LEN;
    ffn_1_residual_skip.W = EMBED_SIZE;
    ffn_1_residual_skip.C = 1;

    ffn_1_residual_lout.data = buff_a;
    ffn_1_residual_lout.dim = SEQ_LEN*EMBED_SIZE;
    ffn_1_residual_lout.H = SEQ_LEN;
    ffn_1_residual_lout.W = EMBED_SIZE;
    ffn_1_residual_lout.C = 1;

    ffn_1_residual_output.data = buff_c;
    ffn_1_residual_output.dim = SEQ_LEN*EMBED_SIZE;
    ffn_1_residual_output.H = SEQ_LEN;
    ffn_1_residual_output.W = EMBED_SIZE;
    ffn_1_residual_output.C = 1;

    ffn_1_norm_in.data = buff_c;
    ffn_1_norm_in.dim = SEQ_LEN*EMBED_SIZE;
    ffn_1_norm_in.H = SEQ_LEN;
    ffn_1_norm_in.W = EMBED_SIZE;
    ffn_1_norm_in.C = 1;

    ffn_1_norm_wgt.data = FFN1_OUTPUT_NORM_WEIGHTS;
    ffn_1_norm_wgt.dim = EMBED_SIZE;
    ffn_1_norm_wgt.H = 1;
    ffn_1_norm_wgt.W = EMBED_SIZE;
    ffn_1_norm_wgt.C = 1;

    ffn_1_norm_bias.data = FFN1_OUTPUT_NORM_BIASES;
    ffn_1_norm_bias.dim = EMBED_SIZE;
    ffn_1_norm_bias.H = 1;
    ffn_1_norm_bias.W = EMBED_SIZE;
    ffn_1_norm_bias.C = 1;

    ffn_1_norm_out.data = buff_c;
    ffn_1_norm_out.dim = SEQ_LEN*EMBED_SIZE;
    ffn_1_norm_out.H = SEQ_LEN;
    ffn_1_norm_out.W = EMBED_SIZE;
    ffn_1_norm_out.C = 1;

    // Connecting FFN 2
    ffn_2_relu_input.data = buff_d;
    ffn_2_relu_input.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_2_relu_input.H = SEQ_LEN;
    ffn_2_relu_input.W = HIDDEN_SIZE;
    ffn_2_relu_input.C = 1;

    ffn_2_relu_output.data = buff_d;
    ffn_2_relu_output.dim = SEQ_LEN * HIDDEN_SIZE;
    ffn_2_relu_output.H = SEQ_LEN;
    ffn_2_relu_output.W = HIDDEN_SIZE;
    ffn_2_relu_output.C = 1;

    ffn_2_residual_skip.data = buff_b;
    ffn_2_residual_skip.dim = SEQ_LEN*EMBED_SIZE;
    ffn_2_residual_skip.H = SEQ_LEN;
    ffn_2_residual_skip.W = EMBED_SIZE;
    ffn_2_residual_skip.C = 1;

    ffn_2_residual_lout.data = buff_c;
    ffn_2_residual_lout.dim = SEQ_LEN*EMBED_SIZE;
    ffn_2_residual_lout.H = SEQ_LEN;
    ffn_2_residual_lout.W = EMBED_SIZE;
    ffn_2_residual_lout.C = 1;

    ffn_2_residual_output.data = buff_a;
    ffn_2_residual_output.dim = SEQ_LEN*EMBED_SIZE;
    ffn_2_residual_output.H = SEQ_LEN;
    ffn_2_residual_output.W = EMBED_SIZE;
    ffn_2_residual_output.C = 1;

    ffn_2_norm_in.data = buff_a;
    ffn_2_norm_in.dim = SEQ_LEN*EMBED_SIZE;
    ffn_2_norm_in.H = SEQ_LEN;
    ffn_2_norm_in.W = EMBED_SIZE;
    ffn_2_norm_in.C = 1;

    ffn_2_norm_wgt.data = FFN2_OUTPUT_NORM_WEIGHTS;
    ffn_2_norm_wgt.dim = EMBED_SIZE;
    ffn_2_norm_wgt.H = 1;
    ffn_2_norm_wgt.W = EMBED_SIZE;
    ffn_2_norm_wgt.C = 1;

    ffn_2_norm_bias.data = FFN2_OUTPUT_NORM_BIASES;
    ffn_2_norm_bias.dim = EMBED_SIZE;
    ffn_2_norm_bias.H = 1;
    ffn_2_norm_bias.W = EMBED_SIZE;
    ffn_2_norm_bias.C = 1;

    ffn_2_norm_out.data = buff_a;
    ffn_2_norm_out.dim = SEQ_LEN*EMBED_SIZE;
    ffn_2_norm_out.H = SEQ_LEN;
    ffn_2_norm_out.W = EMBED_SIZE;
    ffn_2_norm_out.C = 1;

    // Connecting Intermediate
    intermediate_relu_input.data = buff_d;
    intermediate_relu_input.dim = SEQ_LEN * HIDDEN_SIZE;
    intermediate_relu_input.H = SEQ_LEN;
    intermediate_relu_input.W = HIDDEN_SIZE;
    intermediate_relu_input.C = 1;

    intermediate_relu_output.data = buff_d;
    intermediate_relu_output.dim = SEQ_LEN * HIDDEN_SIZE;
    intermediate_relu_output.H = SEQ_LEN;
    intermediate_relu_output.W = HIDDEN_SIZE;
    intermediate_relu_output.C = 1;

    // Connecting Output
    output_residual_skip.data = buff_b;
    output_residual_skip.dim = SEQ_LEN*EMBED_SIZE;
    output_residual_skip.H = SEQ_LEN;
    output_residual_skip.W = EMBED_SIZE;
    output_residual_skip.C = 1;

    output_residual_lout.data = buff_a;
    output_residual_lout.dim = SEQ_LEN*EMBED_SIZE;
    output_residual_lout.H = SEQ_LEN;
    output_residual_lout.W = EMBED_SIZE;
    output_residual_lout.C = 1;

    output_residual_output.data = buff_c;
    output_residual_output.dim = SEQ_LEN*EMBED_SIZE;
    output_residual_output.H = SEQ_LEN;
    output_residual_output.W = EMBED_SIZE;
    output_residual_output.C = 1;

    output_norm_in.data = buff_c;
    output_norm_in.dim = SEQ_LEN*EMBED_SIZE;
    output_norm_in.H = SEQ_LEN;
    output_norm_in.W = EMBED_SIZE;
    output_norm_in.C = 1;

    output_norm_wgt.data = OUTPUT_NORM_WEIGHTS;
    output_norm_wgt.dim = EMBED_SIZE;
    output_norm_wgt.H = 1;
    output_norm_wgt.W = EMBED_SIZE;
    output_norm_wgt.C = 1;

    output_norm_bias.data = OUTPUT_NORM_BIASES;
    output_norm_bias.dim = EMBED_SIZE;
    output_norm_bias.H = 1;
    output_norm_bias.W = EMBED_SIZE;
    output_norm_bias.C = 1;

    output_norm_out.data = buff_c;
    output_norm_out.dim = SEQ_LEN*EMBED_SIZE;
    output_norm_out.H = SEQ_LEN;
    output_norm_out.W = EMBED_SIZE;
    output_norm_out.C = 1;

    // Connecting Output Bottleneck
    output_bottleneck_residual_skip.data = buff_d;
    output_bottleneck_residual_skip.dim = SEQ_LEN*HIDDEN_SIZE;
    output_bottleneck_residual_skip.H = SEQ_LEN;
    output_bottleneck_residual_skip.W = HIDDEN_SIZE;
    output_bottleneck_residual_skip.C = 1;

    output_bottleneck_residual_lout.data = INPUT;
    output_bottleneck_residual_lout.dim = SEQ_LEN*HIDDEN_SIZE;
    output_bottleneck_residual_lout.H = SEQ_LEN;
    output_bottleneck_residual_lout.W = HIDDEN_SIZE;
    output_bottleneck_residual_lout.C = 1;

    output_bottleneck_residual_output.data = buff_d;
    output_bottleneck_residual_output.dim = SEQ_LEN*HIDDEN_SIZE;
    output_bottleneck_residual_output.H = SEQ_LEN;
    output_bottleneck_residual_output.W = HIDDEN_SIZE;
    output_bottleneck_residual_output.C = 1;

    output_bottleneck_norm_in.data = buff_d;
    output_bottleneck_norm_in.dim = SEQ_LEN*HIDDEN_SIZE;
    output_bottleneck_norm_in.H = SEQ_LEN;
    output_bottleneck_norm_in.W = HIDDEN_SIZE;
    output_bottleneck_norm_in.C = 1;

    output_bottleneck_norm_wgt.data = OUTPUT_BOTTLENECK_NORM_WEIGHTS;
    output_bottleneck_norm_wgt.dim = HIDDEN_SIZE;
    output_bottleneck_norm_wgt.H = 1;
    output_bottleneck_norm_wgt.W = HIDDEN_SIZE;
    output_bottleneck_norm_wgt.C = 1;

    output_bottleneck_norm_bias.data = OUTPUT_BOTTLENECK_NORM_BIASES;
    output_bottleneck_norm_bias.dim = HIDDEN_SIZE;
    output_bottleneck_norm_bias.H = 1;
    output_bottleneck_norm_bias.W = HIDDEN_SIZE;
    output_bottleneck_norm_bias.C = 1;

    output_bottleneck_norm_out.data = buff_d;
    output_bottleneck_norm_out.dim = SEQ_LEN*HIDDEN_SIZE;
    output_bottleneck_norm_out.H = SEQ_LEN;
    output_bottleneck_norm_out.W = HIDDEN_SIZE;
    output_bottleneck_norm_out.C = 1;


    // ~~~~~~~~~~ CONFIGURE LAYER STRUCTURES ~~~~~~~~~~

    // Bottleneck for attention values
    bneck_dense_att_args.A = INPUT;
    bneck_dense_att_args.B = BOTTLENECK_ATTENTION_WEIGHTS;
    bneck_dense_att_args.C = buff_a;
    bneck_dense_att_args.N = SEQ_LEN;
    bneck_dense_att_args.K = HIDDEN_SIZE;
    bneck_dense_att_args.M = EMBED_SIZE;
    bneck_dense_att_args.trans_B = 0;
    bneck_dense_att_args.bias = BOTTLENECK_ATTENTION_BIASES;
    bneck_dense_att_args.bias_dim = EMBED_SIZE;
    bneck_dense_att_args.USE_BIASES = 1;
    bneck_dense_att_args.bias_transposed = 0;

    bneck_norm_att_args.input = &bneck_norm_att_in;
    bneck_norm_att_args.coeff = &bneck_norm_att_wgt;
    bneck_norm_att_args.bias = &bneck_norm_att_bias;
    bneck_norm_att_args.output = &bneck_norm_att_out;

    // Bottleneck for input values
    bneck_dense_inp_args.A = INPUT;
    bneck_dense_inp_args.B = BOTTLENECK_INPUT_WEIGHTS;
    bneck_dense_inp_args.C = buff_b;
    bneck_dense_inp_args.N = SEQ_LEN;
    bneck_dense_inp_args.K = HIDDEN_SIZE;
    bneck_dense_inp_args.M = EMBED_SIZE;
    bneck_dense_inp_args.trans_B = 0;
    bneck_dense_inp_args.bias = BOTTLENECK_INPUT_BIASES;
    bneck_dense_inp_args.bias_dim = EMBED_SIZE;
    bneck_dense_inp_args.USE_BIASES = 1;
    bneck_dense_inp_args.bias_transposed = 0;

    bneck_norm_inp_args.input = &bneck_norm_inp_in;
    bneck_norm_inp_args.output = &bneck_norm_inp_out;
    bneck_norm_inp_args.bias = &bneck_norm_inp_bias;
    bneck_norm_inp_args.coeff = &bneck_norm_inp_wgt;

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

    // Residual connection MHSA output + input bottleneck
    residual_1_args.skip = &residual_1_skip;
    residual_1_args.lout = &residual_1_lout;
    residual_1_args.output = &residual_1_output;

    attention_output_norm_args.input = &attention_output_norm_in;
    attention_output_norm_args.coeff = &attention_output_norm_wgt;
    attention_output_norm_args.bias = &attention_output_norm_bias;
    attention_output_norm_args.output = &attention_output_norm_out;

    // FFN 0
    ffn_0_intermediate_args.A = buff_c;
    ffn_0_intermediate_args.B = FFN0_INTERMEDIATE_WEIGHTS;
    ffn_0_intermediate_args.C = buff_d;
    ffn_0_intermediate_args.N = SEQ_LEN;
    ffn_0_intermediate_args.K = EMBED_SIZE;
    ffn_0_intermediate_args.M = HIDDEN_SIZE;
    ffn_0_intermediate_args.trans_B = 0;
    ffn_0_intermediate_args.bias = FFN0_INTERMEDIATE_BIASES;
    ffn_0_intermediate_args.bias_dim = HIDDEN_SIZE;
    ffn_0_intermediate_args.USE_BIASES = 1;
    ffn_0_intermediate_args.bias_transposed = 0;

    ffn_0_relu_args.input = &ffn_0_relu_input;
    ffn_0_relu_args.output = &ffn_0_relu_output;
    ffn_0_relu_args.H = SEQ_LEN;
    ffn_0_relu_args.W = HIDDEN_SIZE;

    ffn_0_output_args.A = buff_d;
    ffn_0_output_args.B = FFN0_OUTPUT_WEIGHTS;
    ffn_0_output_args.C = buff_b;
    ffn_0_output_args.N = SEQ_LEN;
    ffn_0_output_args.K = HIDDEN_SIZE;
    ffn_0_output_args.M = EMBED_SIZE;
    ffn_0_output_args.trans_B = 0;
    ffn_0_output_args.bias = FFN0_OUTPUT_BIASES;
    ffn_0_output_args.bias_dim = EMBED_SIZE;
    ffn_0_output_args.USE_BIASES = 1;
    ffn_0_output_args.bias_transposed = 0;

    ffn_0_residual_args.skip = &ffn_0_residual_skip;
    ffn_0_residual_args.lout = &ffn_0_residual_lout;
    ffn_0_residual_args.output = &ffn_0_residual_output;

    ffn_0_norm_args.input = &ffn_0_norm_in;
    ffn_0_norm_args.coeff = &ffn_0_norm_wgt;
    ffn_0_norm_args.bias = &ffn_0_norm_bias;
    ffn_0_norm_args.output = &ffn_0_norm_out;

    // FFN 1
    ffn_1_intermediate_args.A = buff_a;
    ffn_1_intermediate_args.B = FFN1_INTERMEDIATE_WEIGHTS;
    ffn_1_intermediate_args.C = buff_d;
    ffn_1_intermediate_args.N = SEQ_LEN;
    ffn_1_intermediate_args.K = EMBED_SIZE;
    ffn_1_intermediate_args.M = HIDDEN_SIZE;
    ffn_1_intermediate_args.trans_B = 0;
    ffn_1_intermediate_args.bias = FFN1_INTERMEDIATE_BIASES;
    ffn_1_intermediate_args.bias_dim = HIDDEN_SIZE;
    ffn_1_intermediate_args.USE_BIASES = 1;
    ffn_1_intermediate_args.bias_transposed = 0;

    ffn_1_relu_args.input = &ffn_1_relu_input;
    ffn_1_relu_args.output = &ffn_1_relu_output;
    ffn_1_relu_args.H = SEQ_LEN;
    ffn_1_relu_args.W = HIDDEN_SIZE;

    ffn_1_output_args.A = buff_d;
    ffn_1_output_args.B = FFN1_OUTPUT_WEIGHTS;
    ffn_1_output_args.C = buff_b;
    ffn_1_output_args.N = SEQ_LEN;
    ffn_1_output_args.K = HIDDEN_SIZE;
    ffn_1_output_args.M = EMBED_SIZE;
    ffn_1_output_args.trans_B = 0;
    ffn_1_output_args.bias = FFN1_OUTPUT_BIASES;
    ffn_1_output_args.bias_dim = EMBED_SIZE;
    ffn_1_output_args.USE_BIASES = 1;
    ffn_1_output_args.bias_transposed = 0;

    ffn_1_residual_args.skip = &ffn_1_residual_skip;
    ffn_1_residual_args.lout = &ffn_1_residual_lout;
    ffn_1_residual_args.output = &ffn_1_residual_output;

    ffn_1_norm_args.input = &ffn_1_norm_in;
    ffn_1_norm_args.coeff = &ffn_1_norm_wgt;
    ffn_1_norm_args.bias = &ffn_1_norm_bias;
    ffn_1_norm_args.output = &ffn_1_norm_out;

    // FFN 2
    ffn_2_intermediate_args.A = buff_c;
    ffn_2_intermediate_args.B = FFN2_INTERMEDIATE_WEIGHTS;
    ffn_2_intermediate_args.C = buff_d;
    ffn_2_intermediate_args.N = SEQ_LEN;
    ffn_2_intermediate_args.K = EMBED_SIZE;
    ffn_2_intermediate_args.M = HIDDEN_SIZE;
    ffn_2_intermediate_args.trans_B = 0;
    ffn_2_intermediate_args.bias = FFN2_INTERMEDIATE_BIASES;
    ffn_2_intermediate_args.bias_dim = HIDDEN_SIZE;
    ffn_2_intermediate_args.USE_BIASES = 1;
    ffn_2_intermediate_args.bias_transposed = 0;

    ffn_2_relu_args.input = &ffn_2_relu_input;
    ffn_2_relu_args.output = &ffn_2_relu_output;
    ffn_2_relu_args.H = SEQ_LEN;
    ffn_2_relu_args.W = HIDDEN_SIZE;

    ffn_2_output_args.A = buff_d;
    ffn_2_output_args.B = FFN2_OUTPUT_WEIGHTS;
    ffn_2_output_args.C = buff_b;
    ffn_2_output_args.N = SEQ_LEN;
    ffn_2_output_args.K = HIDDEN_SIZE;
    ffn_2_output_args.M = EMBED_SIZE;
    ffn_2_output_args.trans_B = 0;
    ffn_2_output_args.bias = FFN2_OUTPUT_BIASES;
    ffn_2_output_args.bias_dim = EMBED_SIZE;
    ffn_2_output_args.USE_BIASES = 1;
    ffn_2_output_args.bias_transposed = 0;

    ffn_2_residual_args.skip = &ffn_2_residual_skip;
    ffn_2_residual_args.lout = &ffn_2_residual_lout;
    ffn_2_residual_args.output = &ffn_2_residual_output;

    ffn_2_norm_args.input = &ffn_2_norm_in;
    ffn_2_norm_args.coeff = &ffn_2_norm_wgt;
    ffn_2_norm_args.bias = &ffn_2_norm_bias;
    ffn_2_norm_args.output = &ffn_2_norm_out;

    // Intermediate
    intermediate_args.A = buff_a;
    intermediate_args.B = INTERMEDIATE_WEIGHTS;
    intermediate_args.C = buff_d;
    intermediate_args.N = SEQ_LEN;
    intermediate_args.K = EMBED_SIZE;
    intermediate_args.M = HIDDEN_SIZE;
    intermediate_args.trans_B = 0;
    intermediate_args.bias = INTERMEDIATE_BIASES;
    intermediate_args.bias_dim = HIDDEN_SIZE;
    intermediate_args.USE_BIASES = 1;
    intermediate_args.bias_transposed = 0;

    intermediate_relu_args.input = &intermediate_relu_input;
    intermediate_relu_args.output = &intermediate_relu_output;
    intermediate_relu_args.H = SEQ_LEN;
    intermediate_relu_args.W = HIDDEN_SIZE;

    // Output
    output_dense_args.A = buff_d;
    output_dense_args.B = OUTPUT_WEIGHTS;
    output_dense_args.C = buff_b;
    output_dense_args.N = SEQ_LEN;
    output_dense_args.K = HIDDEN_SIZE;
    output_dense_args.M = EMBED_SIZE;
    output_dense_args.trans_B = 0;
    output_dense_args.bias = OUTPUT_BIASES;
    output_dense_args.bias_dim = EMBED_SIZE;
    output_dense_args.USE_BIASES = 1;
    output_dense_args.bias_transposed = 0;

    output_residual_args.skip = &output_residual_skip;
    output_residual_args.lout = &output_residual_lout;
    output_residual_args.output = &output_residual_output;

    output_norm_args.input = &output_norm_in;
    output_norm_args.coeff = &output_norm_wgt;
    output_norm_args.bias = &output_norm_bias;
    output_norm_args.output = &output_norm_out;

    // Output Bottleneck
    output_bottleneck_dense_args.A = buff_c;
    output_bottleneck_dense_args.B = OUTPUT_BOTTLENECK_WEIGHTS;
    output_bottleneck_dense_args.C = buff_d;
    output_bottleneck_dense_args.N = SEQ_LEN;
    output_bottleneck_dense_args.K = EMBED_SIZE;
    output_bottleneck_dense_args.M = HIDDEN_SIZE;
    output_bottleneck_dense_args.trans_B = 0;
    output_bottleneck_dense_args.bias = OUTPUT_BOTTLENECK_BIASES;
    output_bottleneck_dense_args.bias_dim = HIDDEN_SIZE;
    output_bottleneck_dense_args.USE_BIASES = 1;
    output_bottleneck_dense_args.bias_transposed = 0;

    output_bottleneck_residual_args.skip = &output_bottleneck_residual_skip;
    output_bottleneck_residual_args.lout = &output_bottleneck_residual_lout;
    output_bottleneck_residual_args.output = &output_bottleneck_residual_output;

    output_bottleneck_norm_args.input = &output_bottleneck_norm_in;
    output_bottleneck_norm_args.coeff = &output_bottleneck_norm_wgt;
    output_bottleneck_norm_args.bias = &output_bottleneck_norm_bias;
    output_bottleneck_norm_args.output = &output_bottleneck_norm_out;

    return;

}


void forward(){

    
    // Bottleneck for attention values
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &bneck_dense_att_args);
    #else
    struct mm_manager_args man_args1;
    man_args1.mm_args = &bneck_dense_att_args;
    man_args1.layer_type = LAYER_LINEAR;
    man_args1.step_type = STEP_FW;
    man_args1.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args1);
    #endif
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &bneck_norm_att_args);
    
    
    
    // MHSA HERE
    pulp_mhsa_mobilebert_inference_fp32_fw_cl((void*) &mhsa_args);

    
    // Bottleneck for input values
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &bneck_dense_inp_args);
    #else
    struct mm_manager_args man_args2;
    man_args2.mm_args = &bneck_dense_inp_args;
    man_args2.layer_type = LAYER_LINEAR;
    man_args2.step_type = STEP_FW;
    man_args2.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args2);
    #endif
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &bneck_norm_inp_args);


    // Residual connection MHSA output + input bottleneck
    pulp_residualconn_fp32_fw((void*) &residual_1_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &attention_output_norm_args);
    
    // FFN 0
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_0_intermediate_args);
    #else
    struct mm_manager_args man_args3;
    man_args3.mm_args = &ffn_0_intermediate_args;
    man_args3.layer_type = LAYER_LINEAR;
    man_args3.step_type = STEP_FW;
    man_args3.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args3);
    #endif
    pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &ffn_0_relu_args);
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_0_output_args);
    #else
    struct mm_manager_args man_args4;
    man_args4.mm_args = &ffn_0_output_args;
    man_args4.layer_type = LAYER_LINEAR;
    man_args4.step_type = STEP_FW;
    man_args4.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args4);
    #endif
    pulp_residualconn_fp32_fw((void*) &ffn_0_residual_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &ffn_0_norm_args);

    // FFN 1
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_1_intermediate_args);
    #else
    struct mm_manager_args man_args5;
    man_args5.mm_args = &ffn_1_intermediate_args;
    man_args5.layer_type = LAYER_LINEAR;
    man_args5.step_type = STEP_FW;
    man_args5.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args5);
    #endif
    pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &ffn_1_relu_args);
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_1_output_args);
    #else
    struct mm_manager_args man_args6;
    man_args6.mm_args = &ffn_1_output_args;
    man_args6.layer_type = LAYER_LINEAR;
    man_args6.step_type = STEP_FW;
    man_args6.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args6);
    #endif
    pulp_residualconn_fp32_fw((void*) &ffn_1_residual_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &ffn_1_norm_args);

    // FFN 2
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_2_intermediate_args);
    #else
    struct mm_manager_args man_args7;
    man_args7.mm_args = &ffn_2_intermediate_args;
    man_args7.layer_type = LAYER_LINEAR;
    man_args7.step_type = STEP_FW;
    man_args7.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args7);
    #endif
    pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &ffn_2_relu_args);
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &ffn_2_output_args);
    #else
    struct mm_manager_args man_args8;
    man_args8.mm_args = &ffn_2_output_args;
    man_args8.layer_type = LAYER_LINEAR;
    man_args8.step_type = STEP_FW;
    man_args8.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args8);
    #endif
    pulp_residualconn_fp32_fw((void*) &ffn_2_residual_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &ffn_2_norm_args);

    // Intermediate
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &intermediate_args);
    #else
    struct mm_manager_args man_args9;
    man_args9.mm_args = &intermediate_args;
    man_args9.layer_type = LAYER_LINEAR;
    man_args9.step_type = STEP_FW;
    man_args9.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args9);
    #endif
    pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &intermediate_relu_args);

    // Output
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &output_dense_args);
    #else
    struct mm_manager_args man_args10;
    man_args10.mm_args = &output_dense_args;
    man_args10.layer_type = LAYER_LINEAR;
    man_args10.step_type = STEP_FW;
    man_args10.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args10);
    #endif
    pulp_residualconn_fp32_fw((void*) &output_residual_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &output_norm_args);

    // Output Bottleneck
    #ifndef OPTIMIZE
    pi_cl_team_fork(NUM_CORES, mm, &output_bottleneck_dense_args);
    #else
    struct mm_manager_args man_args11;
    man_args11.mm_args = &output_bottleneck_dense_args;
    man_args11.layer_type = LAYER_LINEAR;
    man_args11.step_type = STEP_FW;
    man_args11.matmul_type = MATMUL_TYPE;
    pi_cl_team_fork(NUM_CORES, mm_manager, &man_args11);
    #endif
    pulp_residualconn_fp32_fw((void*) &output_bottleneck_residual_args);
    pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &output_bottleneck_norm_args);

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
        printf("%f ", buff_d[i]);
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

    printf("Mobilebert test:\n");
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
    compare_tensors(buff_d, OUTPUT, OUTPUT_SIZE);
    //check_tensor(buff_d, OUTPUT, OUTPUT_SIZE);

    return;
}
