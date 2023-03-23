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
 * Authors: Davide Nadalini
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
 *     |______    - pulp_sumnode_fp32_bw();
 *     |     |
 *  LAYERS   |
 *     |_____|     
 *     +          - FW: pulp_residualconn_fp32_fw(), BW: pulp_residualconn_fp32_bw()
 *     |
 *   Output
 * 
 */

/**
 * @brief Structure to configure a residual connection or summation node
 * @param activation0 first activation to be summed 
 * @param activation1 second activation to be summed
 * @param activation2 output sum activation (forward: activation data, backward: activation gradient)
 */
struct SkipConn_args {
    struct blob * activation0;
    struct blob * activation1;
    struct blob * activation2;
};


// FORWARD FUNCTIONS

/**
 * @brief Accumulates the input activations on an output activation 
 * 
 * @param activation0 first activation to be summed
 * @param activation1 second activation to be summed
 * @param activation2 output sum activation (forward: activation data, backward: activation gradient)
 */
void pulp_sumnode_fp32_fw( void * SkipConn_args );

/**
 * @brief Accumulates the input 
 * 
 * @param activation0 activation coming from the layers
 * @param activation1 input activation of the skipped layer
 * @param activation2 output activation data sum
 */
void pulp_residualconn_fp32_fw( void * SkipConn_args );



// BACKWARD FUNCTIONS

/**
 * @brief Accumulates the output gradients on an input gradient 
 * 
 * @param activation0 first activation to be summed
 * @param activation1 second activation to be summed
 * @param activation2 output sum activation (forward: activation data, backward: activation gradient)
 */
void pulp_sumnode_fp32_bw( void * SkipConn_args );

/**
 * @brief Dispatches the gradient coming from the output of a residual connection sum node to the layers and their input
 * 
 * @param activation0 gradient going to the skipped layers
 * @param activation1 gradient going to the input of the skipped layers
 * @param activation2 output gradient to be dispatched to the input grads
 */
void pulp_residualconn_fp32_bw( void * SkipConn_args );
