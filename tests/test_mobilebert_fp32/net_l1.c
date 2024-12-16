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

// ~~~~~~~~~~ L1 DATA ~~~~~~~~~~

// Constants definition
PI_L1 float zero_init = 0.0f;
PI_L1 float min_float = -340282346638528859811704183484516925440.0f;

// Define structures and pointers to data in L1 memory
PI_L1 float * IN_DATA, * W_DATA, * OUT_DATA, * BIAS_DATA;
PI_L1 float  BUFF[MAX_SIZE];
PI_L1 struct blob input_blob;
PI_L1 struct blob weight_blob;
PI_L1 struct blob output_blob;
PI_L1 struct blob bias_blob;
PI_L1 struct matMul_args mm_args;
PI_L1 struct Nonorm_args nn_args;
PI_L1 struct SkipConn_args skip_args;
PI_L1 struct act_args relu_args;
PI_L1 struct mm_manager_args man_args;
PI_L1 struct Tiled_Matmul_Mhsa_args tiled_matmul_mhsa_args;
PI_L1 pi_cl_dma_cmd_t * cmd_store;
PI_L1 pi_cl_dma_cmd_t * cmd_load;






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

// Define L2 I/O tensors

PI_L2 float buff_a[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_b[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_c[SEQ_LEN * EMBED_SIZE];
PI_L2 float buff_d[SEQ_LEN * HIDDEN_SIZE];

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


    // ~~~~~~~~~~ INITIALIZING L2 BUFFER DATA TO ZERO ~~~~~~~~~~

    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_a[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_b[i] = zero_init;
    for(int i=0; i<EMBED_SIZE*SEQ_LEN; i++)		        buff_c[i] = zero_init;
    for(int i=0; i<HIDDEN_SIZE*SEQ_LEN; i++)	        buff_d[i] = zero_init;
    for(int i=0; i<SEQ_LEN; i++)		                mhsa_maxes[i] = zero_init;
    for(int i=0; i<SEQ_LEN; i++)		                mhsa_sums[i] = zero_init;
    for(int i=0; i<SEQ_LEN * SEQ_LEN * N_HEADS; i++)	mhsa_softmax_buffer_v[i] = zero_init;
    for(int i = 0; i < MAX_SIZE; i++)                   BUFF[i] = zero_init;

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

void tiled_matmul(void* matmul_args){
    struct matMul_args * args = (struct matMul_args *)matmul_args;
    
    int n_tiles_i = (args->N) / TILE_H;
    int n_tiles_j = (args->M) / TILE_W;
    int K = args->K;
    
    IN_DATA = BUFF;
    W_DATA = BUFF + K * TILE_H;
    OUT_DATA = W_DATA + K * TILE_W;
    BIAS_DATA = OUT_DATA + TILE_DIM;
    
    mm_args.A = IN_DATA;
    mm_args.B = W_DATA;
    mm_args.C = OUT_DATA;
    mm_args.N = TILE_H;
    mm_args.K = K;
    mm_args.M = TILE_W;
    mm_args.trans_B = args->trans_B;
    mm_args.bias = BIAS_DATA;
    mm_args.bias_dim = TILE_W;
    mm_args.USE_BIASES = args->USE_BIASES;
    mm_args.bias_transposed = args->bias_transposed;
    
    for(int j = 0; j < n_tiles_j; j++){
        pi_cl_dma_cmd_2d((uint32_t) (args->B + j * TILE_W), (uint32_t) (W_DATA), 4 * TILE_W * K, 4 * (args->M), 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
        pi_cl_dma_cmd_wait(cmd_load);
        pi_cl_dma_cmd((uint32_t) (args->bias + j * TILE_W), (uint32_t) (BIAS_DATA), 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
        pi_cl_dma_cmd_wait(cmd_load);
        for(int i = 0; i < n_tiles_i; i++){
            pi_cl_dma_cmd_2d((uint32_t) (args->A + i * K * TILE_H), (uint32_t) (IN_DATA), 4 * K * TILE_H, 4 * K, 4 * K, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            man_args.mm_args = &mm_args;
            pi_cl_team_fork(NUM_CORES, mm_manager, &man_args);
            pi_cl_dma_cmd_2d((uint32_t) (args->C + i * args->M * TILE_H + TILE_W * j), (uint32_t) (OUT_DATA), 4 * TILE_DIM, 4 * (args->M), 4 * TILE_W, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
        }
    }
}

void tiled_norm(void *nonorm_args){
    struct Nonorm_args * args = (struct Nonorm_args*) nonorm_args;

    int W = args->input->W;
    int n_tiles_i = args->input->H / TILE_H;
    int n_tiles_j = W / TILE_W;

    input_blob.data = BUFF;
    input_blob.dim = TILE_DIM;
    input_blob.W = TILE_W;
    input_blob.H = TILE_H;

    weight_blob.data = BUFF + TILE_DIM;
    weight_blob.dim = TILE_W;
    weight_blob.W = TILE_W;
    weight_blob.H = TILE_W;

    output_blob.data = BUFF + TILE_DIM + TILE_W;
    output_blob.dim = TILE_DIM;
    output_blob.W = TILE_W;
    output_blob.H = TILE_H;

    bias_blob.data = BUFF + TILE_DIM + TILE_W + TILE_DIM;
    bias_blob.dim = TILE_W;
    bias_blob.W = TILE_W;
    bias_blob.H = TILE_H;

    nn_args.input = &input_blob;
    nn_args.coeff = &weight_blob;
    nn_args.bias = &bias_blob;
    nn_args.output = &output_blob;

    for(int j=0; j < n_tiles_j; j++){
       pi_cl_dma_cmd((uint32_t) (args->coeff->data + j * TILE_W), (uint32_t) (nn_args.coeff->data), 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
       pi_cl_dma_cmd_wait(cmd_load);
       pi_cl_dma_cmd((uint32_t) (args->bias->data + j * TILE_W), (uint32_t) (nn_args.bias->data), 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
       pi_cl_dma_cmd_wait(cmd_load);
       for(int i=0; i < n_tiles_i; i++){
            pi_cl_dma_cmd_2d((uint32_t) (args->input->data + i * TILE_H * W + j * TILE_W), (uint32_t) (nn_args.input->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            pi_cl_team_fork(NUM_CORES, pulp_nonorm_fp32_fw_cl, &nn_args);
            pi_cl_dma_cmd_2d((uint32_t) (args->output->data + i * W * TILE_H + TILE_W * j), (uint32_t) (nn_args.output->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
       } 
    }
}

void tiled_skip(void *residual_args){
    struct SkipConn_args * args = (struct SkipConn_args*) residual_args;

    int W = args->skip->W;
    int n_tiles_i = args->skip->H / TILE_H;
    int n_tiles_j = W / TILE_W; 

    input_blob.data = BUFF;
    input_blob.dim = TILE_DIM;
    input_blob.H = TILE_H;
    input_blob.W = TILE_W;

    weight_blob.data = BUFF + TILE_DIM;
    weight_blob.dim = TILE_DIM;
    weight_blob.H = TILE_H;
    weight_blob.W = TILE_W;
    
    output_blob.data = BUFF + TILE_DIM + TILE_DIM;
    output_blob.dim = TILE_DIM;
    output_blob.H = TILE_H;
    output_blob.W = TILE_W;

    skip_args.skip = &input_blob;
    skip_args.lout = &weight_blob;
    skip_args.output = &output_blob;

    for(int i = 0; i < n_tiles_i; i++){
        for(int j = 0; j < n_tiles_j; j++){
            pi_cl_dma_cmd_2d((uint32_t) (args->skip->data + i * TILE_H * W + j * TILE_W), (uint32_t) (skip_args.skip->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            pi_cl_dma_cmd_2d((uint32_t) (args->lout->data + i * TILE_H * W + j * TILE_W), (uint32_t) (skip_args.lout->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            pulp_residualconn_fp32_fw((void*) &skip_args);
            pi_cl_dma_cmd_2d((uint32_t) (args->output->data + i * W * TILE_H + TILE_W * j), (uint32_t) (skip_args.output->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
        }
    }
}

void tiled_relu(void* Relu_args){
    struct act_args * args = (struct act_args*) Relu_args;

    int W = args->input->W;
    int H = args->input->H;
    int n_tiles_i = H / TILE_H;
    int n_tiles_j = W / TILE_W;

    input_blob.data = BUFF;
    input_blob.dim = TILE_DIM;
    input_blob.H = TILE_H;
    input_blob.W = TILE_W;

    output_blob.data = BUFF + TILE_DIM;
    output_blob.dim = TILE_DIM;
    output_blob.H = TILE_H;
    output_blob.W = TILE_W;

    relu_args.input = &input_blob;
    relu_args.output = &output_blob;
    relu_args.H = H;
    relu_args.W = W;

    for(int i = 0; i < n_tiles_i; i++){
        for(int j=0; j<n_tiles_j; j++){
            pi_cl_dma_cmd_2d((uint32_t) (args->input->data + i * TILE_H * W + j * TILE_W), (uint32_t) (relu_args.input->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_EXT2LOC, cmd_load);
            pi_cl_dma_cmd_wait(cmd_load);
            pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &relu_args);
            pi_cl_dma_cmd_2d((uint32_t) (args->output->data + i * W * TILE_H + TILE_W * j), (uint32_t) (relu_args.output->data), 4 * TILE_DIM, 4 * W, 4 * TILE_W, PI_CL_DMA_DIR_LOC2EXT, cmd_store);
            pi_cl_dma_cmd_wait(cmd_store);
        }
    }
}


void forward(){
    // Bottleneck for attention values
    tiled_matmul(&bneck_dense_att_args);
    tiled_norm(&bneck_norm_att_args);
    
    // MHSA HERE
    tiled_mhsa_fp32((void*) &mhsa_args, (void*) &tiled_matmul_mhsa_args);
    
    // Bottleneck for input values
    tiled_matmul(&bneck_dense_inp_args);
    tiled_norm(&bneck_norm_inp_args);

    // Residual connection MHSA output + input bottleneck
    tiled_skip((void*) &residual_1_args);
    tiled_norm(&attention_output_norm_args);
    
    // FFN 0
    tiled_matmul(&ffn_0_intermediate_args);
    tiled_relu(&ffn_0_relu_args);
    tiled_matmul(&ffn_0_output_args);
    tiled_skip((void*) &ffn_0_residual_args);
    tiled_norm(&ffn_0_norm_args);

    // FFN 1
    tiled_matmul(&ffn_1_intermediate_args);
    tiled_relu(&ffn_1_relu_args);
    tiled_matmul(&ffn_1_output_args);
    tiled_skip((void*) &ffn_1_residual_args);
    tiled_norm(&ffn_1_norm_args);

    // FFN 2
    tiled_matmul(&ffn_2_intermediate_args);
    tiled_relu(&ffn_2_relu_args);
    tiled_matmul(&ffn_2_output_args);
    tiled_skip((void*) &ffn_2_residual_args);
    tiled_norm(&ffn_2_norm_args);

    // Intermediate
    tiled_matmul(&intermediate_args);
    tiled_relu(&intermediate_relu_args);

    // Output
    tiled_matmul(&output_dense_args);
    tiled_skip((void*) &output_residual_args);
    tiled_norm(&output_norm_args);

    // Output Bottleneck
    tiled_matmul(&output_bottleneck_dense_args);
    tiled_skip((void*) &output_bottleneck_residual_args);
    tiled_norm(&output_bottleneck_norm_args);

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
    /*
    for(int i = 0; i < (HIDDEN_SIZE*SEQ_LEN); i++){
        if(!(i % HIDDEN_SIZE))
            printf("\n");
        printf("%f ", buff_d[i]);
    }
    printf("\n");*/
    for(int i = 0; i < (EMBED_SIZE*SEQ_LEN); i++){
        if(!(i % EMBED_SIZE))
            printf("\n");
        printf("%f ", buff_a[i]);
    }
    printf("\n");
}



// ~~~~~~~~~~~~~~~~~~~~ MAIN FUNCTION ~~~~~~~~~~~~~~~~~~~~
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
	nn_args.input = &input_blob;
    nn_args.coeff = &weight_blob;
    nn_args.bias = &bias_blob;
    nn_args.output = &output_blob;
    skip_args.skip = &input_blob;
    skip_args.lout = &weight_blob;
    skip_args.output = &output_blob;
    relu_args.input = &input_blob;
    relu_args.output = &output_blob;
    man_args.layer_type = LAYER_LINEAR;
    man_args.step_type = STEP_FW;
    man_args.matmul_type = MATMUL_TYPE;
    tiled_matmul_mhsa_args.mm_args = &mm_args;
    tiled_matmul_mhsa_args.BUFF = BUFF;
    tiled_matmul_mhsa_args.tile_h = TILE_H;
    tiled_matmul_mhsa_args.tile_w = TILE_W;
    tiled_matmul_mhsa_args.tile_dim = TILE_DIM;
    tiled_matmul_mhsa_args.man_args = &man_args;
    tiled_matmul_mhsa_args.cmd_load = cmd_load;
    tiled_matmul_mhsa_args.cmd_store = cmd_store;
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
}



