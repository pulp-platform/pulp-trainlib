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
'MaxPool'   -> max pooling layer
'AvgPool'   -> average pooling layer
"""

import utils.DNN_Reader     as reader
import utils.DNN_Composer   as composer

# ---------------------
# --- USER SETTINGS ---
# ---------------------

# GENERAL PROPERTIES
project_name    = 'Test_CNN'
project_path    = '../../../DNN_Tests/'
proj_folder     = project_path + project_name + '/'

# TRAINING PROPERTIES
epochs          = 1
batch_size      = 1                    # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.5
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "MSELoss"            # Name of PyTorch's loss function

# ------- NETWORK GRAPH --------
# Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
layer_list      = [ 'conv2d', 'ReLU', 'DW', 'PW', 'ReLU', 'linear' ]
# Layer properties
in_ch_list      = [ 1, 4, 4, 4, 4, 4*2*2 ]          # Linear: size of input vector
out_ch_list     = [ 4, 4, 4, 4, 4, 2 ]            # Linear: size of output vector
hk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
wk_list         = [ 3, 1, 3, 1, 1, 1 ]             # Linear: = 1
# Input activations' properties
hin_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
win_list        = [ 6, 4, 4, 2, 2, 1 ]             # Linear: = 1
# Convolutional strides
h_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
w_str_list      = [ 1, 1, 1, 1, 1, 1 ]             # Only for conv2d, maxpool, avgpool
# Padding (bilateral, adds the specified padding to both image sides)
h_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
w_pad_list      = [ 0, 0, 0, 0, 0, 0 ]             # Only for conv2d, DW
# Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
opt_mm_fw_list  = [ 0, 0, 0, 0, 0, 0 ]
opt_mm_wg_list  = [ 0, 0, 0, 0, 0, 0 ]
opt_mm_ig_list  = [ 0, 0, 0, 0, 0, 0 ]
# Data type list for layer-by-layer deployment (mixed precision)
data_type_list   = ['FP32', 'FP32', 'FP32', 'FP16', 'FP16', 'FP16']
# Data layout list (CHW or HWC) 
data_layout_list = ['CHW', 'CHW', 'CHW', 'CHW', 'CHW', 'CHW']   # TO DO
# ----- END OF NETWORK GRAPH -----

# EXECUTION PROPERTIES
NUM_CORES       = 8
L1_SIZE_BYTES   = 60*(2**10)

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

    # Check if the network training fits L1
    memocc = composer.DNN_Size_Checker(layer_list, in_ch_list, out_ch_list, hk_list, wk_list, hin_list, win_list, 
                                h_str_list, w_str_list, h_pad_list, w_pad_list,
                                data_type_list, L1_SIZE_BYTES)

    print("DNN memory occupation: {} bytes of {} available L1 bytes ({}%).".format(memocc, L1_SIZE_BYTES, (memocc/L1_SIZE_BYTES)*100))

    # Call DNN Composer on the user-provided graph
    composer.DNN_Composer(proj_folder, project_name, 
                            layer_list, in_ch_list, out_ch_list, hk_list, wk_list, 
                            hin_list, win_list, h_str_list, w_str_list, h_pad_list, w_pad_list,
                            epochs, batch_size, learning_rate, optimizer, loss_fn,
                            NUM_CORES, data_type_list, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list)

    print("PULP project generation successful!")

    pass

