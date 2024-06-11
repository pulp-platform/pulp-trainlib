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
 */


/**
 * Recurrent layer training functions, grouped into FW and BW
*/

/**
 * Authors: Francesco Conoscenti, Alberto Dequino
*/ 


/**
 * Recursive Neural Network layer configuration structure
 */

/**
 * @brief Structure for RNN Training in FP32
 * @param input             Input vector for the RNN layer.
 * @param state             Vector of current hidden state.
 * @param output            Output vector & next current state.
 * @param coeff_x           Weight for input vector.
 * @param coeff_s           Weight for state vector.
 * @param temp_buffer       Temporary vector for BW transpose operations.
 * @param grad_buffer       Buffer used for saving the output gradient in the BW step.
 */

struct Rnn_args {
    struct blob * input;
    struct blob * state; 
    struct blob * output;
    struct blob * coeff_x;
    struct blob * coeff_s;
    float * temp_buffer;
    float * grad_buffer; 
};




/**
 * RNN layer training functions, grouped into FW and BW
 */

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input     Input vector.
 * @param state     Vector of current hidden state.
 * @param output    Output vector & next current state.
 * @param coeff_x   Weight for input vector.
 * @param coeff_s   Weight for state vector.
 */
void pulp_rnn_fp32_fw_cl(void * Rnn_args);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calculate both weight gradient and input gradient.
 * @param input     Input vector.
 * @param state     Vector current hidden state.
 * @param output    Output vector & next current state.
 * @param coeff_x   weight input matrix 
 * @param coeff_s   weight state matrix
 */
void pulp_rnn_fp32_bw_cl(void * Rnn_args);
