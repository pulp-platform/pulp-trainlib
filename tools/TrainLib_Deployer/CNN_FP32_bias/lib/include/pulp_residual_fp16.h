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
 * Authors: Davide Nadalini, Giacomo Saporetti
*/ 

/**
 * Forward and Backward primitives for the tensor summation node (useful for skip and residual connections).
 * Forward primitive needs to be put where the summation of forward propagations occurs (after the skipped layers, to accumulate activations).
 * Backward primitive is used to accumulate two gradients in a single one (return node from a skip connection)
 */

/**
 * 
 * Skip / Residual Connection layer configuration structure
 * 
 * How to make a skip connection:
 * 
 *   Input        
 *     |______    - pulp_sumnode_fp16_bw();
 *     |     |
 *  LAYERS   |
 *     |_____|     
 *     +          - FW: pulp_residualconn_fp16_fw(), BW: pulp_residualconn_fp16_bw()
 *     |
 *   Output
 * 
 */

/**
 * @brief Structure to configure a residual connection or summation node
 * @param skip activation to be forwarded at the output 
 * @param lout second activation to be summed
 * @param output result from the sum of Input (@param skip) and LAYERS output (@param lout) (forward: activation data, backward: activation gradient)
 */
struct SkipConn_args_fp16 {
    struct blob_fp16 * skip;
    struct blob_fp16 * lout;
    struct blob_fp16 * output;
    int skip_in_grad;
};


// FORWARD FUNCTIONS

/**
 * @brief Sums the input activations to the output  
 * 
 * @param skip: activation to be forwarded at the output 
 * @param lout: layers output, second activation to be summed
 * @param output: result of the sum between Input (@param skip) and LAYERS output (@param lout) (forward: activation data, backward: activation gradient)
 */
void pulp_residualconn_fp16_fw( void * SkipConn_args );



// BACKWARD FUNCTIONS

/**
 * @brief Accumulates the output gradients on an input gradient 
 * 
 * @param skip: activation to be forwarded at the output 
 * @param lout: layers output, second activation to be summed
 * @param output: result of the sum between Input (@param skip) and LAYERS output (@param lout) (forward: activation data, backward: activation gradient)
 */
void pulp_sumnode_fp16_bw( void * SkipConn_args );

/**
 * @brief Dispatches the gradient coming from the output of a residual connection sum node to the layers and their input
 * 
 * @param skip: activation to be forwarded at the output 
 * @param lout: layers output, second activation to be summed
 * @param output: result of the sum between Input (@param skip) and LAYERS output (@param lout) (forward: activation data, backward: activation gradient)
 */
void pulp_residualconn_fp16_bw( void * SkipConn_args );


