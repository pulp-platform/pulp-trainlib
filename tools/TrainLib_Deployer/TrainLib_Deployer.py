'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Davide Nadalini
'''

"""
TrainLib Deployer: a tool to deploy DNN on-device training on MCUs

Available DNN layer names:
'linear'    -> fully-connected layer
'conv2d'    -> 2d convolution layer
'PW'        -> pointwise convolution
'DW'        -> depthwise convolution
'ReLU'      -> ReLU activation
'LeakyReLU' -> LeakyReLU activation
'Sigmoid'   -> Sigmoid activation
'MaxPool'   -> max pooling layer
'AvgPool'   -> average pooling layer
'Skipnode'  -> node at which data is taken and passes forward, to add an additional layer after the skip derivation simply substitute 'Skipnode' with any kind of layer
'Sumnode'   -> node at which data from Skipnode is summed 
'InstNorm'  -> instance Normalization layer

Available losses:
'MSELoss'           -> Mean Square Error loss
'CrossEntropyLoss'  -> CrossEntropy loss

Available optimizers:
'SGD'       -> Stochastic Gradient Descent
"""

import deployer_utils.DNN_Reader     as reader
import deployer_utils.DNN_Composer   as composer

# ---------------------
# --- USER SETTINGS ---
# ---------------------

# GENERAL PROPERTIES
project_name    = 'Test_CNN'
project_path    = './' 
proj_folder     = project_path + project_name + '/'


# TRAINING PROPERTIES
epochs          = 20
batch_size      = 1                   # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.01
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "MSELoss"            # Name of PyTorch's loss function

NET = 0

if NET == 0:
    # ------- NETWORK GRAPH --------
    # Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
    layer_list          = [ 'conv2d', 'ReLU', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'Sigmoid', 'DW', 'PW', 'LeakyReLU', 'linear']
    # Layer properties
    sumnode_connections = [ 0,         0,      0,    0,    0,      0,    0,     0,        0,    0,     0,          0 ]            # For Skipnode and Sumnode only, for each Skipnode-Sumnode couple choose a value and assign it to both, all other layer MUST HAVE 0

    in_ch_list          = [ 3,         8,      8,    8,    8,      8,    8,     8,        8,    8,     8,          8*8*8 ]         # Linear: size of input vector
    out_ch_list         = [ 8,         8,      8,    8,    8,      8,    8,     8,        8,    8,     8,          2 ]            # Linear: size of output vector
    hk_list             = [ 3,         1,      3,    1,    1,      3,    1,     1,        5,    1,     1,          1 ]            # Linear: = 1
    wk_list             = [ 3,         1,      3,    1,    1,      3,    1,     1,        5,    1,     1,          1 ]            # Linear: = 1
    # Input activations' properties
    hin_list            = [ 32,        16,     16,   14,   14,     14,   12,    12,       12,   8,     8,          1 ]            # Linear: = 1
    win_list            = [ 32,        16,     16,   14,   14,     14,   12,    12,       12,   8,     8,          1 ]            # Linear: = 1
    # Convolutional strides
    h_str_list          = [ 2,         1,      1,    1,    1,      1,    1,     1,        1,    1,     1,          1 ]            # Only for conv2d, maxpool, avgpool 
    w_str_list          = [ 2,         1,      1,    1,    1,      1,    1,     1,        1,    1,     1,          1 ]            # Only for conv2d, maxpool, avgpool 
    # Padding (bilateral, adds the specified padding to both image sides)
    h_pad_list          = [ 1,         0,      0,    0,    0,      0,    0,     0,        0,    0,     0,          0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
    w_pad_list          = [ 1,         0,      0,    0,    0,      0,    0,     0,        0,    0,     0,          0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
    # Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
    opt_mm_fw_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    opt_mm_wg_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    opt_mm_ig_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # Data type list for layer-by-layer deployment (mixed precision)
    #data_type_list      = ['FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16']
    data_type_list     = ['FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32']
    # Data layout list (CHW or HWC) 
    data_layout_list    = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
    # Bias
    bias_list           = [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1 ]
    # Sparse Update
    update_layer_list   = [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1 ]             # Set to 1 for each layer you want to update, 0 if you want to skip weight update
    # ----- END OF NETWORK GRAPH -----
elif NET == 1:
    # ------- NETWORK GRAPH --------
    # Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
    layer_list          = [ 'conv2d', 'Skipnode', 'ReLU', 'Sumnode', 'DW', 'PW', 'ReLU', 'DW', 'PW', 'Sigmoid', 'DW', 'PW', 'LeakyReLU', 'linear']
    # Layer properties
    sumnode_connections = [ 0,         1,          0,      1,         0,    0,    0,      0,    0,     0,        0,    0,     0,      0 ]            # For Skipnode and Sumnode only, for each Skipnode-Sumnode couple choose a value and assign it to both, all other layer MUST HAVE 0

    in_ch_list          = [ 3,         8,          8,      8,         8,    8,    8,      8,    8,     8,        8,    8,     8,      8*8*8 ]         # Linear: size of input vector
    out_ch_list         = [ 8,         8,          8,      8,         8,    8,    8,      8,    8,     8,        8,    8,     8,      2 ]            # Linear: size of output vector
    hk_list             = [ 3,         1,          1,      1,         3,    1,    1,      3,    1,     1,        5,    1,     1,      1 ]            # Linear: = 1
    wk_list             = [ 3,         1,          1,      1,         3,    1,    1,      3,    1,     1,        5,    1,     1,      1 ]            # Linear: = 1
    # Input activations' properties
    hin_list            = [ 32,        16,         16,     16,        16,   14,   14,     14,   12,    12,       12,   8,     8,      1 ]            # Linear: = 1
    win_list            = [ 32,        16,         16,     16,        16,   14,   14,     14,   12,    12,       12,   8,     8,      1 ]            # Linear: = 1
    # Convolutional strides
    h_str_list          = [ 2,         1,          1,      1,         1,    1,    1,      1,    1,     1,        1,    1,     1,      1 ]            # Only for conv2d, maxpool, avgpool 
    w_str_list          = [ 2,         1,          1,      1,         1,    1,    1,      1,    1,     1,        1,    1,     1,      1 ]            # Only for conv2d, maxpool, avgpool 
    # Padding (bilateral, adds the specified padding to both image sides)
    h_pad_list          = [ 1,         0,          0,      0,         0,    0,    0,      0,    0,     0,        0,    0,     0,      0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
    w_pad_list          = [ 1,         0,          0,      0,         0,    0,    0,      0,    0,     0,        0,    0,     0,      0 ]                            # Implemented for conv2d (naive kernel), DW TO DO
    # Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
    opt_mm_fw_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    opt_mm_wg_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    opt_mm_ig_list      = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # Data type list for layer-by-layer deployment (mixed precision)
    #data_type_list      = ['FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16', 'FP16']
    data_type_list     = ['FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32', 'FP32']
    # Data layout list (CHW or HWC) 
    data_layout_list    = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
    # Bias
    bias_list           = [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1 ]
    # Sparse Update
    update_layer_list   = [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 ]             # Set to 1 for each layer you want to update, 0 if you want to skip weight update
    # ----- END OF NETWORK GRAPH -----




# EXECUTION PROPERTIES
NUM_CORES       = 8
L1_SIZE_BYTES   = 128*(2**10)
USE_DMA = 'SB'                          # choose whether to load all structures in L1 ('NO') or in L2 and use Single Buffer mode ('SB') or Double Buffer mode ('DB', CURRENTLY UNAVAILABLE) 
# BACKWARD SETTINGS
SEPARATE_BACKWARD_STEPS = True          # If True, the tool writes separate weight and input gradient in the backward step
# PROFILING OPTIONS
PROFILE_SINGLE_LAYERS = False            # If True, profiles forward and backward layer-by-layer
# CONV2D SETUPS
CONV2D_USE_IM2COL = False                # Choose if the Conv2D layers should use Im2Col or not (computatational optimization)
# PRINT TRAIN LOSS
PRINT_TRAIN_LOSS = True                 # Set to true if you want to print the train loss for each epoch
# OTHER PROPERTIES
# Select if to read the network from an external source
READ_MODEL_ARCH = False                # NOT IMPLEMENTED!!

# ---------------------------
# --- END OF USER SETTING ---
# ---------------------------



"""
BACKEND
"""

# Call the DNN Reader and then the DNN Composer 
if READ_MODEL_ARCH :
    pass


else:

    print("Generating project at location "+proj_folder)

    # Check if Residual Connections are valid
    
    sumnode_connections = composer.AdjustResConnList(sumnode_connections)

    composer.CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections, update_layer_list)

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, bias_list, update_layer_list,
                                L1_SIZE_BYTES, USE_DMA, CONV2D_USE_IM2COL)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, bias_list, update_layer_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list,
                            sumnode_connections, USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)

    print("PULP project generation successful!")

    pass
