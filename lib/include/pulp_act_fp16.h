/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Authors: Davide Nadalini, Leonardo Ravaglia, Calin Diaconu
 *
 * Activation functions configuration structure
 */

#include "pulp_train_defines.h"

/**
 * @brief Structure for activation functions
 * @param input blob structure for the input data of the activation layer
 * @param output blob structure for the output data of the activation layer
 */
struct act_args_fp16 {
    struct blob_fp16 * input;
    struct blob_fp16 * output;
    int H;
    int W;
};


/**
 * @brief Structure for leaky relu activation functions
 * @param input blob structure for the input data of the activation layer
 * @param output blob structure for the output data of the activation layer
 */
struct leakyrelu_args_fp16 {
    struct blob_fp16 * input;
    struct blob_fp16 * output;
    fp16 negative_slope;
};

/**
 * @brief Arguments for exponential and softmax in parallel
 * @param input   pointer to input vector
 * @param dim     dimension vector
 * @param output  pointer to output vector
 * @param sum     final sum value of all exponentials
*/
struct softmax_args_fp16 {
    fp16 *input_data;
    fp16 *input_diff;
    fp16 *output_data;
    fp16 *output_diff;
    int H;
    int W;
    int L;
    int n_heads;
    fp16 *global_max;
    fp16 *partial_exp_sum;
    fp16 *maxes;
    fp16 *sums;
};


/**
 * Activation functions, both FW and BW
 **/


/**
 * @brief Forward pass function. Configure and pass a act_args_fp16 structure pointer as argument.
 * @param input Input for sigmoid.
 * @param output Output of sigmoid.
*/
void pulp_sigmoid_fp16_fw_cl( void * act_args );


/**
 * @brief Backward pass function.
 * @param input Input for sigmoid.
 * @param output Output of sigmoid.
*/
void pulp_sigmoid_fp16_bw_cl( void * act_args );


/**
 * @brief Core function to implement the forward of sigmoid (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, sigmoid_core_fw_fp16, &args)).
 * @param act_args Input and output data (data only will be used)
*/
void sigmoid_core_fw_fp16( void * act_args );


/**
 * @brief Core function to implement the backward of sigmoid (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, sigmoid_core_bw_fp16, &args)).
 * @param act_args Input and output data (gradients only will be used)
*/
void sigmoid_core_bw_fp16( void * act_args );


/**
 * @brief Forward pass function. Configure and pass a act_args structure pointer as argument.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void pulp_relu_fp16_fw_cl( void * act_args_fp16 );


/**
 * @brief Backward pass function.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void pulp_relu_fp16_bw_cl( void * act_args_fp16 );


/**
 * @brief Core function to implement the forward of ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, relu_core_fw_fp16, &args)).
 * @param act_args Input and output data (data only will be used)
*/
void relu_core_fw_fp16( void * act_args_fp16 );


/**
 * @brief Core function to implement the backward of ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, relu_core_bw_fp16, &args)).
 * @param act_args Input and output data (gradients only will be used)
*/
void relu_core_bw_fp16( void * act_args_fp16 );


/**
 * @brief Forward pass function. Configure and pass a leakyrelu_args structure pointer as argument.
 * @param input Input for leaky relu.
 * @param output Output of leaky relu.
*/
void pulp_leakyrelu_fp16_fw_cl( void * leakyrelu_args_fp16 );

/**
 * @brief Backward pass function.
 * @param input Input for leaky relu.
 * @param output Output of leaky relu.
*/
void pulp_leakyrelu_fp16_bw_cl( void * leakyrelu_args_fp16 );

/**
 * @brief Core function to implement the forward of Leaky ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, leakyrelu_core_fw_fp16, &leakyrelu_args)).
 * @param leakyrelu_args_fp16 Input and output data (data only will be used)
*/
void leakyrelu_core_fw_fp16( void * leakyrelu_args_fp16 );

/**
 * @brief Core function to implement the backward of Leaky ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, leakyrelu_core_bw_fp16, &leakyrelu_args)).
 * @param leakyrelu_args_fp16 Input and output data (gradients only will be used)
*/
void leakyrelu_core_bw_fp16( void * leakyrelu_args_fp16 );





/**
 * @brief Forward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_softmax_fp16_fw_cl( void * act_args_fp16 );

/**
 * @brief Forward pass function tiled in L1.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_softmax_fp16_fw_cl_tiled( void * act_args , void * Tiled_matmul_mhsa_args);


/**
 * @brief Backward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_softmax_fp16_bw_cl( void * act_args_fp16 );


/**
 * @brief Forward pass function. Configure and pass a act_args structure pointer as argument.
 * @param input Input for gelu.
 * @param output Output of gelu.
*/
void pulp_gelu_fp16_fw_cl( void* act_args_fp16);

struct swiglu_args_fp16{
    fp16* in1;
    fp16* in2;
    fp16* out;
    int dim;
};

void pulp_vector_softmax_fp16(fp16* out, fp16* in, fp16* buffer_n_cores, unsigned int size);

void pulp_swiglu_fp16_cl(void *swiglu_args);
