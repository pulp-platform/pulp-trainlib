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
Authors: Davide Nadalini, Giacomo Saporetti
'''

import os
import shutil
import math

import torch 
from torch import mm
import deployer_utils.GM_templates as Gtemp
import deployer_utils.net_templates as ntemp


"""
DNN Size Checker backend functions
"""

def compute_wgt_act_memocc_bytes(layer_number, layer_type, chin, chout, hk, wk, hin, win, h_pad, w_pad, h_str, w_str, DATA_TYPE, use_bias, update_layer, compute_ingrad, is_last_layer):

    memocc_bytes = 0
    bias_present = use_bias

    # First layer does not have in grad
    in_grad_present = 1
    if layer_number == 0 or compute_ingrad == 0:
        in_grad_present = 0
    
    # Last layer occupies output memory (other activations overlap)
    output_separate_occupation = 0
    if is_last_layer:
        output_separate_occupation = 1

    # If the layer is an activation, no weights or biases!
    wgt_present = 1
    if layer_type in ['ReLU', 'LeakyReLU', 'Sigmoid', 'Skipnode', 'Sumnode'] or update_layer == 0:
        wgt_present = 0
        bias_present = 0

    byte_size = 4
    if DATA_TYPE == 'FP32':
        byte_size = 4
    elif DATA_TYPE == 'FP16':
        byte_size = 2
    else:
        print("[deployment_utils.compute_wgt_act_memocc_bytes]: Invalid data type!!")
        exit()

    # Output H and W
    hout = math.floor( (hin-hk+2*h_pad+h_str)/h_str )
    wout = math.floor( (win-wk+2*w_pad+w_str)/w_str )

    # FORWARD
    # Input act
    memocc_bytes += chin * hin * win * byte_size
    # Weights
    if  layer_type == 'InstNorm':
        memocc_bytes += 2 * chin * byte_size    # beta, gamma
        memocc_bytes += 3 * chin * byte_size    # running mean, var, stdev
    else:    
        memocc_bytes += chin * chout * hk * wk * byte_size * wgt_present
    # Biases
    memocc_bytes += chout * bias_present
    # Out act
    memocc_bytes += chout * hout * wout * byte_size * output_separate_occupation

    # BACKWARD
    # Input act grad
    memocc_bytes += chin * hin * win * byte_size * in_grad_present
    # Weight grad
    memocc_bytes += chin * chout * hk * wk * byte_size * wgt_present
    # Bias grad
    memocc_bytes += chout * bias_present
    # Output grad
    memocc_bytes += chout * hout * wout * byte_size * output_separate_occupation


    return memocc_bytes


def compute_im2col_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_pad_l, w_pad_l, h_str_l, w_str_l, data_type_l, update_layer_l, CONV2D_USE_IM2COL):

    memocc_bytes = 0

    # Find last layer to be updated
    last_updated_idx = len(layers_l) - 1
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 1:
            last_updated_idx = layer
            break

    max_im2col_size = 0
    max_im2col_index = 0
    for layer in range(len(layers_l)):
        is_first_layer = 0
        if layer == 0:
            is_first_layer = 1
        if layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True:
            # Check layer data type
            byte_size = 4
            if data_type_l[layer] == 'FP32':
                byte_size = 4
            elif data_type_l[layer] == 'FP16':
                byte_size = 2
            else:
                print("[deployment_utils.compute_im2col_memocc_bytes]: Invalid data type @Layer{}!!".format(layer))
                exit()
            # Output H and W
            hout = math.floor( (hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer] )
            wout = math.floor( (win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer] )
            # Find max im2col size
            if layers_l[layer] == 'conv2d': # or layers_l[layer] == 'DW':
                im2col_size = 0
                size_FW = hk_l[layer] * wk_l[layer] * in_ch_l[layer] * hout * wout * byte_size
                size_BW = 0
                if is_first_layer == 0 and layer > last_updated_idx:
                    size_BW = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * hin_l[layer] * win_l[layer] * byte_size
                if size_FW > size_BW:
                    im2col_size = size_FW
                else:
                    im2col_size = size_BW
                if im2col_size > max_im2col_size:
                    max_im2col_size = im2col_size
                    max_im2col_index = layer
    
    #print("Max im2col size (@layer {}): {}".format(max_im2col_index, max_im2col_size))
    memocc_bytes += max_im2col_size

    return memocc_bytes, max_im2col_index


def compute_cast_buffer_memocc_bytes (layers_l, chin_l, chout_l, hk_l, wk_l, hin_l, win_l, h_pad_l, w_pad_l, h_str_l, w_str_l, data_type_l):

    memocc_bytes = 0

    # Find the largest activation size (buffer for temporary casts)
    act_inout = 'Input'
    previous_type = data_type_l[0]
    curr_in_act_size = 0
    curr_out_act_size = 0
    curr_max_act_size = 0
    max_act_size = 0
    max_act_index = 0
    for layer in range(len(layers_l)):
        # Check layer data type
        byte_size = 4
        if data_type_l[layer] == 'FP32':
            byte_size = 4
        elif data_type_l[layer] == 'FP16':
            byte_size = 2
        else:
            print("[deployment_utils.compute_im2col_memocc_bytes]: Invalid data type @Layer{}!!".format(layer))
            exit()    
        # Output H and W
        hout = math.floor( (hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer] )
        wout = math.floor( (win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer] )
        # Find mixed precision
        if data_type_l[layer] != previous_type:
            # Find current sizes
            curr_in_act_size = chin_l[layer] * hin_l[layer] * win_l[layer] * byte_size
            curr_out_act_size = chout_l[layer] * hout * wout * byte_size
            if curr_in_act_size > curr_out_act_size:
                curr_max_act_size = curr_in_act_size
                act_inout = 'Input'
            else:
                curr_max_act_size = curr_out_act_size
                act_inout = 'Output'
            if curr_max_act_size > max_act_size:
                max_act_size = curr_max_act_size
                max_act_index = layer

    memocc_bytes = max_act_size

    return memocc_bytes, max_act_index, act_inout


def compute_bt_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, data_type_l, update_layer_l, CONV2D_USE_IM2COL):

    memocc_bytes = 0

    # Find last layer to be updated
    last_updated_idx = len(layers_l) - 1
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 1:
            last_updated_idx = layer
            break

    max_bt_size = 0
    max_bt_index = 0
    for layer in range(len(layers_l)):
        # Check layer data layout
        data_layout = 'CHW'     # Change to input list of data layouts
        # Check layer data type
        byte_size = 4
        if data_type_l[layer] == 'FP32':
            byte_size = 4
        elif data_type_l[layer] == 'FP16':
            byte_size = 2
        else:
            print("[deployment_utils.compute_bt_memocc_bytes]: Invalid data type @Layer{}!!".format(layer))
            exit()
        # Find max blocktransp size
        if (((layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True) or layers_l[layer] == 'PW') and (layer > last_updated_idx)):
            bt_size = hk_l[layer] * wk_l[layer] * in_ch_l[layer] * out_ch_l[layer] * byte_size
            if bt_size > max_bt_size:
                max_bt_size = bt_size
                max_bt_index = layer

        # Check special conditions in case of HWC
        if (data_layout == 'HWC' and layers_l[layer] == 'PW'):
            # Special allocation for weight grad in HWC
            bt_size = in_ch_l[layer] * hin_l[layer] * win_l[layer] * byte_size
            if data_type_l[layer] == 'FP16':
                hout = hin_l[layer]; wout = win_l[layer]
                bt_size += out_ch_l[layer] * hout * wout * byte_size
            if bt_size > max_bt_size:
                max_bt_size = bt_size
                max_bt_index = layer

    memocc_bytes += max_bt_size            

    return memocc_bytes, max_bt_index


"""
DNN Composer backend functions
"""

# Initializes the project folder with basic files
def InitProject(proj_folder_path):

    trainlib_src_folder = '../../lib/'
    proj_folder = proj_folder_path
    utils_folder = proj_folder + 'utils/'
    trainlib_dest_folder = proj_folder + 'lib/' 
    
    os.mkdir(proj_folder)
    os.mkdir(utils_folder)

    shutil.copy2('./deployer_utils/srcfiles/main.c', proj_folder)
    shutil.copy2('./deployer_utils/srcfiles/stats.h', proj_folder)
    shutil.copy2('./deployer_utils/srcfiles/dump_utils.py', utils_folder)
    shutil.copytree(trainlib_src_folder, trainlib_dest_folder)

    f = open(proj_folder+'readme.txt', 'w')
    f.write('To compile the application, run "make clean get_golden all run > log.txt".\nIf running on a board (not GVSoC), add "APP_CFLAGS += -DBOARD" to the user section of the Makefile (profiling of cycles only).\n')
    f.write('To modify the hyperparameters (learning rate, epochs, batch size still not implemented), \nedit the variables inside "utils/GM.py".\n')
    f.close()

    return





# Generates the Makefile
def GenerateMakefile(proj_folder_path, project_name, layers_l, NUM_CORES, data_type_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list):

    proj_folder = proj_folder_path
    makefile_name = proj_folder + 'Makefile'

    f = open(makefile_name, 'w')

    f.write('APP = ' + project_name + '\n\n')

    f.write('# User settings\n')
    f.write('NUM_CORES?=' + str(NUM_CORES) + '\n')
    f.write('#APP_CFLAGS += -DDEBUG' + '\n')
    f.write('#APP_CFLAGS += -DOPTIMIZE' + '     # Selects nth matmul to optimize execution\n')
    for layer in range(len(layers_l)):
        f.write('MATMUL_TYPE_FW_L'+str(layer)+'?='+str(opt_mm_fw_list[layer])+'         # Selects which optimized matmul to be used in FW (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
        f.write('MATMUL_TYPE_WG_L'+str(layer)+'?='+str(opt_mm_wg_list[layer])+'         # Selects which optimized matmul to be used in WEIGHT GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
        f.write('MATMUL_TYPE_IG_L'+str(layer)+'?='+str(opt_mm_ig_list[layer])+'         # Selects which optimized matmul to be used in IN GRAD (see mm_manager_list.txt or "MM_manager()" body to verify which one is called)' + '\n')
    f.write('# End of user settings\n\n')

    f.write('NUM_MATMULS?=24		# Available standard matmuls in the library' + '\n')
    f.write('TRAIN_LIB=./lib\n')
    f.write('TRAIN_LIB_SRCS=$(TRAIN_LIB)/sources\n')
    f.write('APP_SRCS = main.c net.c\n\n')

    f.write('APP_CFLAGS += -I. -I$(TRAIN_LIB)/include\n')
    f.write('APP_CFLAGS += -O3 -g3\n')
    f.write('APP_CFLAGS += -DFABRIC\n')
    f.write('APP_CFLAGS += -DCLUSTER\n')
    f.write('APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)\n')
    f.write('APP_CFLAGS += -DPROF_NET\n')
    f.write('APP_CFLAGS += -mhwloopalign\n')
    for layer in range(len(layers_l)):
        f.write('APP_CFLAGS += -DMATMUL_TYPE_FW_L'+str(layer)+'=$'+str('{MATMUL_TYPE_FW_L')+str(layer)+str('}')+'\n')
        f.write('APP_CFLAGS += -DMATMUL_TYPE_WG_L'+str(layer)+'=$'+str('{MATMUL_TYPE_WG_L')+str(layer)+str('}')+'\n')
        f.write('APP_CFLAGS += -DMATMUL_TYPE_IG_L'+str(layer)+'=$'+str('{MATMUL_TYPE_IG_L')+str(layer)+str('}')+'\n')
    f.write('APP_LDFLAGS += -lm\n\n')

    f.write('# STATISTICS\n')
    f.write('APP_CFLAGS += -DSTATS\n\n')

    check_FP32 = False
    check_FP16 = False
    for layer in range(len(layers_l)):
        if data_type_l[layer] == 'FP32':
            check_FP32 = True
        elif data_type_l[layer] == 'FP16':
            check_FP16 = True

    f.write('# SOURCES\n')
    if check_FP32 == True:
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_naive_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_instnorm_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c\n\n')
    if check_FP16 == True:
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_naive_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_residual_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_instnorm_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c\n\n')
    # if (check_FP16 and check_FP32) == False:
    #     print("[deployment_utils.GenerateMakefile] Data format not implemented!!\n")
    #     exit()

    f.write('# RULES\n')
    f.write('get_golden:\n')
    f.write('\tpython ./utils/GM.py\n')
    f.write('\n')

    f.write('include $(RULES_DIR)/pmsis_rules.mk\n')

    f.close()

    return


# Generates the Golden Model
def GenerateGM(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l, bias_l, update_layer_l, sumnode_connections, USE_DMA):

    # Check if GPU is available, else keep fake FP16
    cuda_is_on = torch.cuda.is_available()
    
    # Print DNN structure
    print("---------- DNN ARCHITECTURE ----------")
    for layer in range(len(layers_l)):
        h_out = math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer])
        w_out = math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer])
        print("Layer {}: {} {}, in=[{}, {}, {}], wgt=[{}, {}, {}, {}], out=[{}, {}, {}]".format(layer, data_type_l[layer], layers_l[layer], in_ch_l[layer], hin_l[layer], win_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], in_ch_l[layer], out_ch_l[layer], h_out, w_out))
    print("--------------------------------------")

    f = open(proj_folder_path+'utils/GM.py', 'w')

    f.write("import torch\n")
    f.write("import torch.nn as nn\n")
    f.write("import torch.optim as optim\n")
    f.write("import dump_utils as dump\n")
    f.write("import math\n")
    f.write("\n")

    f.write("# Set device\n")
    f.write("if torch.cuda.is_available():\n")
    f.write("\tdevice = torch.device('cuda')\n")
    f.write("else:\n")
    f.write("\tdevice = torch.device('cpu')\n")  

    # Define hyperparameters
    f.write("# Define hyperparameters\n")
    f.write("learning_rate = "+str(learning_rate)+"\n")
    f.write("batch_size = "+str(batch_size)+"\n")
    f.write("epochs = "+str(epochs)+"\n")
    f.write("\n")

    # Write sizes
    for layer in range(len(layers_l)):
        f.write("# LAYER "+str(layer)+" SIZES\n")
        f.write("l"+str(layer)+"_in_ch = "+str(in_ch_l[layer])+"\n")
        f.write("l"+str(layer)+"_out_ch = "+str(out_ch_l[layer])+"\n")
        f.write("l"+str(layer)+"_hk = "+str(hk_l[layer])+"\n")
        f.write("l"+str(layer)+"_wk = "+str(wk_l[layer])+"\n")
        f.write("l"+str(layer)+"_hin = "+str(hin_l[layer])+"\n")
        f.write("l"+str(layer)+"_win = "+str(win_l[layer])+"\n")
        # Padding and stride
        f.write("l"+str(layer)+"_hstr = "+str(h_str_l[layer])+"\n")
        f.write("l"+str(layer)+"_wstr = "+str(w_str_l[layer])+"\n")
        f.write("l"+str(layer)+"_hpad = "+str(h_pad_l[layer])+"\n")
        f.write("l"+str(layer)+"_wpad = "+str(w_pad_l[layer])+"\n")
    f.write("\n")

    # Write sizes to the header files 
    f.write("f = open('init-defines.h', 'w')\n")
    for layer in range(len(layers_l)):
        f.write("f.write('// Layer"+str(layer)+"\\n')\n")
        f.write("f.write('#define Tin_C_l"+str(layer)+" '+str(l"+str(layer)+"_in_ch)+'\\n')\n")
        f.write("f.write('#define Tout_C_l"+str(layer)+" '+str(l"+str(layer)+"_out_ch)+'\\n')\n")
        if layers_l[layer]  != 'Skipnode' and layers_l[layer]  != 'Sumnode':
            f.write("f.write('#define Tker_H_l"+str(layer)+" '+str(l"+str(layer)+"_hk)+'\\n')\n")
            f.write("f.write('#define Tker_W_l"+str(layer)+" '+str(l"+str(layer)+"_wk)+'\\n')\n")
        f.write("f.write('#define Tin_H_l"+str(layer)+" '+str(l"+str(layer)+"_hin)+'\\n')\n")
        f.write("f.write('#define Tin_W_l"+str(layer)+" '+str(l"+str(layer)+"_win)+'\\n')\n")
        f.write("f.write('#define Tout_H_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_hin-l"+str(layer)+"_hk+2*l"+str(layer)+"_hpad+l"+str(layer)+"_hstr)/l"+str(layer)+"_hstr))+'\\n')\n")
        f.write("f.write('#define Tout_W_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_win-l"+str(layer)+"_wk+2*l"+str(layer)+"_wpad+l"+str(layer)+"_wstr)/l"+str(layer)+"_wstr))+'\\n')\n")
        # Padding and stride
        if layers_l[layer]  != 'Skipnode' and layers_l[layer]  != 'Sumnode':
            f.write("f.write('#define Tstr_H_l"+str(layer)+" '+str(l"+str(layer)+"_hstr)+'\\n')\n")
            f.write("f.write('#define Tstr_W_l"+str(layer)+" '+str(l"+str(layer)+"_wstr)+'\\n')\n")
            f.write("f.write('#define Tpad_H_l"+str(layer)+" '+str(l"+str(layer)+"_hpad)+'\\n')\n")
            f.write("f.write('#define Tpad_W_l"+str(layer)+" '+str(l"+str(layer)+"_wpad)+'\\n')\n")
    f.write("f.close()\n\n")

    # Write hyperparameters to header
    f.write("f = open('init-defines.h', 'a')\n")
    f.write("f.write('\\n// HYPERPARAMETERS\\n')\n")
    f.write("f.write('#define LEARNING_RATE '+str(learning_rate)+'\\n')\n")
    f.write("f.write('#define EPOCHS '+str(epochs)+'\\n')\n")
    f.write("f.write('#define BATCH_SIZE '+str(batch_size)+'\\n')\n")
    f.write("f.close()\n\n")

    # Create input data and label
    f.write("\n# Simple input data \n")
    if (layers_l[0] == 'linear'):
        f.write("inp = torch.div(torch.ones(l0_in_ch), 1e6).to(device)\n")
    elif (layers_l[0] in ['conv2d', 'DW', 'PW', 'Skipnode', 'InstNorm']):
        f.write("inp = torch.torch.div(torch.rand(batch_size, l0_in_ch, l0_hin, l0_win), 1e6).to(device)\n")
    # Throw error
    else:
        print("[deployment_utils.GenerateGM]: Input layer not valid!\n")
        exit()

    # Set input layer to half() in case of FP16
    if data_type_l[0] == 'FP16':
        f.write("inp = inp.half()\n")


    #Sumnode and Skipnode class generation 
    f.write("\nclass Sumnode():\n") 
    f.write("\tdef __init__(self, ls):\n") 
    f.write("\t\tself.MySkipNode = ls\n\n") 

    f.write("class Skipnode():\n") 
    f.write("\tdef __init__(self):\n") 
    f.write("\t\tself.data = 0\n\n") 
    f.write("\tdef __call__(self, x):\n") 
    f.write("\t\tself.data = x\n") 
    f.write("\t\treturn self.data\n\n")

    # Generate DNN model
    f.write("class DNN(nn.Module):\n")
    f.write("\tdef __init__(self):\n")
    f.write("\t\tsuper().__init__()\n")
    # Create neural network model
    for layer in range(len(layers_l)):
        if cuda_is_on:
            current_type = data_type_l[layer]
        else:
            current_type = 'FP32'
        # Layers
        if layers_l[layer] == "linear":
            f.write(Gtemp.linear_template(layer, in_ch_l[layer], out_ch_l[layer], bias_l[layer], current_type)) # "False"
        elif layers_l[layer] == "conv2d":
            f.write(Gtemp.conv2d_template(layer, in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], bias_l[layer], current_type))
        elif layers_l[layer] == "DW":
            f.write(Gtemp.DW_template(layer, in_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False", current_type))
        elif layers_l[layer] == "PW":
            f.write(Gtemp.PW_template(layer, in_ch_l[layer], out_ch_l[layer], "False", current_type))
        # Activations
        elif layers_l[layer] == "ReLU":
            f.write(Gtemp.ReLU_template(layer, current_type))
        elif layers_l[layer] == "LeakyReLU":
            f.write(Gtemp.LeakyReLU_template(layer, current_type))
        elif layers_l[layer] == "Sigmoid":
            f.write(Gtemp.Sigmoid_template(layer, current_type))
        # Pooling
        elif layers_l[layer] == "MaxPool":
            f.write(Gtemp.MaxPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], current_type))
        elif layers_l[layer] == "AvgPool":
            f.write(Gtemp.AvgPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], current_type))
        #Skipconn
        elif layers_l[layer] == "Skipnode": 
            f.write(Gtemp.Skipnode_template(layer)) 
        elif layers_l[layer] == "Sumnode": 
            f.write(Gtemp.Sumnode_template(layer, sumnode_connections[layer])) 
        #Normalization
        elif layers_l[layer] == "InstNorm":
            f.write(Gtemp.InstNorm_template(layer, in_ch_l[layer], current_type))
        # Throw error
        else:
            print("[deployment_utils.GenerateGM]: Layer {} not recognized!!\n".format(layer))
            exit()
    # Create Forward
    f.write("\n")
    f.write("\tdef forward(self, x):")
    for layer in range(len(layers_l)):

        variable = 'x'
        if sumnode_connections[layer] != -1:
            variable = f'y{sumnode_connections[layer]}' # Create a temporary variable for skip connections

        # Vectorize inputs in case of linear layer
        if layers_l[layer] == 'linear':
            f.write(f"\n\t\t{variable} = torch.reshape(x, (-1,))")
        # Set data format for each layer
        if layer == 0 and data_type_l[layer] == 'FP16':
            if cuda_is_on: 
                f.write(f"\n\t\tx = x")
            else:    
                f.write(f"\n\t\tx = x.float()")
        elif data_type_l[layer] == 'FP32' and data_type_l[layer-1] != data_type_l[layer]:
            f.write(f"\n\t\tx = x.float()")
        elif data_type_l[layer] == 'FP16' and data_type_l[layer-1] != data_type_l[layer]:
            if cuda_is_on:
                f.write(f"\n\t\tx = x.half()")
            else:    
                f.write(f"\n\t\tx = x.float()")
        # Forward layers 
        # (ReLU works with FP32 only)
        if layers_l[layer] in ['ReLU', 'LeakyReLU', 'Sigmoid']: # and data_type_l[layer-1] == 'FP32' and data_type_l[layer] == 'FP16':
            if cuda_is_on:
                f.write(f"\n\t\t{variable} = self.l"+str(layer)+f"({variable})")
            else:
                f.write(f"\n\t\t{variable} = self.l"+str(layer)+f"({variable}.float())")
        #Skipconn
        elif sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            f.write(f"\n\t\t{variable} = self.l{layer}(x)")
            f.write(f"\n\t\tx = {variable}")
        elif layers_l[layer] == "Sumnode": 
            f.write(f"\n\t\tx = y{layer} + x\t# Sumnode") 
        # Last layer
        elif layer == len(layers_l)-1:
            if cuda_is_on:
                f.write(f"\n\t\t{variable} = self.l"+str(layer)+"(x)")
            else:
                f.write(f"\n\t\t{variable} = self.l"+str(layer)+"(x).float()")
        else:
            f.write("\n\t\tx = self.l"+str(layer)+"(x)")
        
    f.write("\n\t\treturn x\n")
    print("[deployment_utils.GenerateNet]: Setting last layer's output to float for PyTorch compatibility with loss function backward (future fix).")


    last_layer = len(layers_l) - 1

    # Initialize network
    f.write("\n# Initialize network\n")
    f.write("net = DNN().to(device)\n")
    f.write("for p in net.parameters():\n")
    f.write("\tnn.init.normal_(p, mean=0.0, std=0.01)\n")
    f.write("net.zero_grad()\n\n")

    # Freeze layers excluded from sparse update
    # Find last layer to be updated
    last_updated_idx = len(layers_l) - 1
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 1:
            last_updated_idx = layer
            break
    write_sparse_update = False
    sparse_comment_written = False
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 0:
            write_sparse_update = True
        if sparse_comment_written == False:
            f.write("# Freeze weights for sparse update\n")
            sparse_comment_written = True
        if write_sparse_update and update_layer_l[layer] == 0 and layers_l[layer] not in ['ReLU', 'LeakyReLU', 'Sigmoid', 'AvgPool', 'MaxPool', 'Sumnode', 'Skipnode']:
            f.write("net.l"+str(layer)+".weight.requires_grad = False\n")
    f.write("\n")

    # Write all-ones sample label
    f.write("\n# All-ones fake label \n")
    f.write("output_test = net(inp).to(device)\n")
    f.write("label = torch.ones_like(output_test).to(device)\n")

    # Write init weights to header file
    f.write("f = open('io_data.h', 'w')\n")
    f.write("f.write('// Init weights\\n')\n")
    for layer in range(len(layers_l)):
        if (layers_l[layer] not in ['ReLU', 'LeakyReLU', 'Sigmoid', 'MaxPool',  'AvgPool', 'Skipnode', 'Sumnode']):
            dump = f"+dump.tensor_to_string(net.l{layer}.weight.data)+"
            bias_dump = f"+dump.tensor_to_string(net.l{layer}.bias.data)+"
            if layers_l[layer] != 'InstNorm':   # Generic layer's weights
                f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
                if bias_l[layer] == 1:
                    f.write("f.write('#define BIAS_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_out_ch)+'\\n')\n")
            else:                               # InstanceNorm weights
                f.write("f.write(f'#define WGT_SIZE_L" + f"{layer}" + "  2*{" + f"l{layer}_in_ch" + "}\\n')\n")
                dump = f"+dump.tensor_to_string(net.l{layer}.weight.data)+dump.tensor_to_string(net.l{layer}.bias.data)+"
            if data_type_l[layer] == 'FP32':
                f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"] = {'"+dump+"'};\\n')\n")
                if bias_l[layer] == 1:
                    f.write("f.write('PI_L2 float init_BIAS_l"+str(layer)+"[BIAS_SIZE_L"+str(layer)+"] = {'"+bias_dump+"'};\\n')\n")
            elif data_type_l[layer] == 'FP16':
                f.write("f.write('PI_L2 fp16 init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"] = {'"+dump+"'};\\n')\n")
                if bias_l[layer] == 1:
                    f.write("f.write('PI_L2 fp16 init_BIAS_l"+str(layer)+"[BIAS_SIZE_L"+str(layer)+"] = {'"+bias_dump+"'};\\n')\n")
            else:
                print("[deployment_utils.GenerateGM] Error in data type definition! (weight init)")
                exit()
        else:
            f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
            if data_type_l[layer] == 'FP32':
                f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"];\\n')\n")
            elif data_type_l[layer] == 'FP16':
                f.write("f.write('PI_L2 fp16 init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"];\\n')\n")
            else:
                print("[deployment_utils.GenerateGM] Error in data type definition! (weight init - empty ones)")
                exit()
    f.write("f.close()\n\n")

    # Define optimizer
    if optimizer == 'SGD':
        f.write("optimizer = optim."+str(optimizer)+"(net.parameters(), lr=learning_rate, momentum=0)\n")
    else:
        print("[deployment_utils.GenerateGM]: Invalid optimizer!!\n!")
        exit()
    f.write("loss_fn = nn."+str(loss_fn)+"()\n")
    f.write("\n")

    # Perform training
    f.write("train_loss_list = []\n")
    f.write("# Train the DNN\n")
    f.write("for batch in range(epochs):\n")
    f.write("\toptimizer.zero_grad()\n")
    f.write("\tout = net(inp)\n")
    f.write("\tloss = loss_fn(out, label)\n")
    f.write("\ttrain_loss_list.append(loss)\n")
    f.write("\tloss.backward()\n")
    f.write("\toptimizer.step()\n")
    f.write("\ntrain_loss = torch.tensor(train_loss_list)\n")

    # Inference after training
    f.write("\n# Inference once after training\n")
    f.write("out = net(inp)\n")
    f.write("\n")

    # Dump input and output of the network to the header file for the MCU
    f.write("f = open('io_data.h', 'a')\n")
    f.write("f.write('// Input and Output data\\n')\n")
    f.write("f.write('#define IN_SIZE "+str(in_ch_l[0]*win_l[0]*hin_l[0])+"\\n')\n")
    # Fake input data definition
    memory_loc = 'L1'
    if USE_DMA == 'SB' or USE_DMA == 'DB':
        memory_loc = 'L2'
    if data_type_l[0] == 'FP32':
        f.write(f"f.write('PI_{memory_loc} float INPUT[IN_SIZE] ="+" {'+dump.tensor_to_string(inp)+'};\\n')\n")
    elif data_type_l[0] == 'FP16':
        f.write(f"f.write('PI_{memory_loc} fp16 INPUT[IN_SIZE] ="+" {'+dump.tensor_to_string(inp)+'};\\n')\n")
    else:
        print("[deployment_utils.GenerateGM] Invalid input data size!")
    f.write("out_size = (int(math.floor(l"+str(last_layer)+"_hin-l"+str(last_layer)+"_hk+2*l"+str(last_layer)+"_hpad+l"+str(last_layer)+"_hstr)/l"+str(last_layer)+"_hstr)) * (int(math.floor(l"+str(last_layer)+"_win-l"+str(last_layer)+"_wk+2*l"+str(last_layer)+"_wpad+l"+str(last_layer)+"_wstr)/l"+str(last_layer)+"_wstr)) * l"+str(last_layer)+"_out_ch\n") 
    f.write("f.write('#define OUT_SIZE '+str(out_size)+'\\n')\n")
    # Fake output data and label definition
    if data_type_l[-1] == 'FP32':
        f.write("f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\\n')\n")
        f.write(f"f.write('PI_{memory_loc} float LABEL[OUT_SIZE] = "+"{'+dump.tensor_to_string(label)+'};\\n')\n")
        f.write("f.write('PI_L2 float TRAIN_LOSS['+str(epochs)+'] = {'+dump.tensor_to_string(train_loss)+'};\\n')\n")
    elif data_type_l[-1] == 'FP16':
        f.write("f.write('PI_L2 fp16 REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\\n')\n")
        f.write(f"f.write('PI_{memory_loc} fp16 LABEL[OUT_SIZE] = "+"{'+dump.tensor_to_string(label)+'};\\n')\n")
        f.write("f.write('PI_L2 fp16 TRAIN_LOSS['+str(epochs)+'] = {'+dump.tensor_to_string(train_loss)+'};\\n')\n")
    else:
        print("[deployment_utils.GenerateGM] Invalid output data size!")
    f.write("f.close()\n")

    f.close()

    return





# Generate the net.c and net.h files for the execution on PULP
def GenerateNet(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l, bias_l, update_layer_l, sumnode_connections,
                PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS):

    # Generate net.h
    f = open(proj_folder_path+'net.h', 'w')

    f.write("// PULP Defines\n")
    f.write("#define STACK_SIZE      4096\n")
    f.write("\n")

    f.write("// Tolerance to check updated output\n")
    f.write("#define TOLERANCE 1e-12\n\n")

    f.write("// Training functions\n")
    f.write("void DNN_init();\n")
    f.write("void compute_loss();\n")
    f.write("void update_weights();\n")
    f.write("void forward();\n")
    f.write("void backward();\n")
    f.write("void net_step();\n")

    f.write("\n// Print and check functions\n")
    f.write("void print_output();\n")
    f.write("void check_post_training_output();\n")

    f.close()    


    # Generate net.c
    f = open(proj_folder_path+'net.c', 'w')

    f.write("/**\n * INCLUDES\n**/\n\n")

    f.write("#include \"pulp_train.h\"\n")
    f.write("#include \"net.h\"\n")
    f.write("#include \"stats.h\"\n\n")
    f.write("#include \"init-defines.h\"\n")
    f.write("#include \"io_data.h\"\n")



    f.write("\n\n\n/**\n * DATA\n**/\n")

    f.write("\n// Define loss\n")
    if data_type_l[-1] == 'FP32':
        f.write("PI_L1 float loss = 0;\n")
    elif data_type_l[-1] == 'FP16':
        f.write("PI_L1 fp16 loss = 0;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid last layer data type!")
        exit()


    f.write("\n// Define DNN blobs\n")
    for layer in range(len(layers_l)):
        bias_content = "layer" + str(layer) + "_bias, " if bias_l[layer] == 1 else ""
        if data_type_l[layer] == 'FP32':
            f.write("PI_L1 struct blob layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, " + bias_content + "layer"+str(layer)+"_out;\n")
        elif data_type_l[layer] == 'FP16':
            f.write("PI_L1 struct blob_fp16 layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, " + bias_content + "layer"+str(layer)+"_out;\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for blob definition @Layer{}!".format(layer))
            exit()


    f.write("\n// Define DNN layer structures\n")
    f.write("PI_L1 struct vect_sum_args vect_sum_args;\n")
    f.write("PI_L1 struct vect_sum_args_fp16 vect_sum_args_fp16;\n")
    for layer in range(len(layers_l)):
        # Define FP32 structure
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'linear':
                f.write("PI_L1 struct Linear_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L1 struct Conv2D_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L1 struct PointWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L1 struct DepthWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L1 struct act_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'LeakyReLU':
                f.write("PI_L1 struct leakyrelu_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'Sigmoid':
                f.write("PI_L1 struct act_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass 
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L1 struct SkipConn_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'InstNorm':
                f.write(f"PI_L1 struct InstNorm_args l{layer}_args;\n")
            else:
                print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")
        # Define FP16 structure
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'linear':
                f.write("PI_L1 struct Linear_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L1 struct Conv2D_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L1 struct PointWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L1 struct DepthWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L1 struct act_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'LeakyReLU':
                f.write("PI_L1 struct leakyrelu_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'Sigmoid':
                f.write("PI_L1 struct act_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L1 struct SkipConn_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'InstNorm':
                f.write(f"PI_L1 struct InstNorm_args_fp16 l{layer}_args;\n")
            else:
                print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")
        # Invalid data type
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for structure initialization @Layer{}!".format(layer))


    pooling_exist = False
    for layer in range(len(layers_l)):
        if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
            pooling_exist = True
    if pooling_exist:
        f.write("\n// Define Pooling Structures\n")
        for layer in range(len(layers_l)):
            if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
                if data_type_l[layer] == 'FP32':
                    f.write("PI_L1 struct pool_args l"+str(layer)+"_pool_args;\n")
                elif data_type_l[layer] == 'FP16':
                    f.write("PI_L1 struct pool_args_fp16 l"+str(layer)+"_pool_args;\n")
                else:
                    print("[deployment_utils.GenerateNet] Invalid data type for pooling initialization @Layer{}!".format(layer))
                    exit()


    f.write("\n// Define kernel tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] in ['MaxPool', 'AvgPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("PI_L1 float l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L1 float l"+str(layer)+f"_ker[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L1 float l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                if bias_l[layer] == 1:
                    f.write("PI_L1 float l"+str(layer)+"_bias[Tout_C_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] in ['MaxPool', 'AvgPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("PI_L1 fp16 l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L1 fp16 l"+str(layer)+f"_ker[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L1 fp16 l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                if bias_l[layer] == 1:
                    f.write("PI_L1 fp16 l"+str(layer)+"_bias[Tout_C_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for kernel definition @Layer{}!".format(layer))
            exit()

    f.write("\n// Define kernel grad tensors\n")
    for layer in range(len(layers_l)):
        # Define tensor only if layer is updated
        if update_layer_l[layer] == 1:
            # Define FP32 tensors
            if data_type_l[layer] == 'FP32':
                if layers_l[layer] in ['MaxPool', 'AvgPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                    f.write("PI_L1 float l"+str(layer)+"_ker_diff[1];\n")
                elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                    pass
                elif layers_l[layer] == 'InstNorm':
                    f.write("PI_L1 float l"+str(layer)+f"_ker_diff[2*Tin_C_l{layer}];\n")
                else:
                    f.write("PI_L1 float l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                    if bias_l[layer] == 1:
                        f.write("PI_L1 float l"+str(layer)+"_bias_diff[Tout_C_l"+str(layer)+"];\n")
            # Define FP16 tensors
            elif data_type_l[layer] == 'FP16':
                if layers_l[layer] in ['MaxPool', 'AvgPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                    f.write("PI_L1 fp16 l"+str(layer)+"_ker_diff[1];\n")
                elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                    pass
                elif layers_l[layer] == 'InstNorm':
                    f.write("PI_L1 fp16 l"+str(layer)+f"_ker_diff[2*Tin_C_l{layer}];\n")
                else:
                    f.write("PI_L1 fp16 l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                    if bias_l[layer] == 1:
                        f.write("PI_L1 fp16 l"+str(layer)+"_bias_diff[Tout_C_l"+str(layer)+"];\n")
            # Data type error
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for kernel grad definition @Layer{}!".format(layer))
                exit()
        elif update_layer_l[layer] == 0:
            pass
        else:
            print("[deployment_utils.GenerateNet] Invalid sparse update variable for layer {}!!".format(layer))


    # Identify last updated layer
    last_updated_idx = len(layers_l) - 1
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 1:
            last_updated_idx = layer
            break


    f.write("\n// Define I/O tensors\n")

    max_nonsaved_activation = 0     # Determine the size for the transient activation buffer
    use_activation_buffer = False   # In case of sparse update, save the buffer which will be pointed to all layers that do not save activations
    activation_buffer_bytes = 4
    previous_was_skip = False 
    for layer in range(len(layers_l)):
        # Determine bytes of the current activation
        data_size = 4
        if data_type_l[layer] == 'FP16':
            data_size = 2
        # Find if the layer needs to store in grad for weight grad computation (next layer needs it)
        save_activation = True
        if layer > 0 and layer < (len(layers_l)-1) and update_layer_l[layer] == 0 and layers_l[layer] not in ['Skipnode', 'ReLU', 'LeakyReLU', 'Sigmoid']:
            in_size = in_ch_l[layer] * hin_l[layer] * win_l[layer] 
            save_activation = False
            use_activation_buffer = True
            if max_nonsaved_activation * activation_buffer_bytes < in_size * data_size:
                max_nonsaved_activation = in_size
                activation_buffer_bytes = data_size
        # Define FP32 tensors
        if not previous_was_skip and save_activation: # If the previous layer was a Skipnode, then do not generate layer in and diff
            if data_type_l[layer] == 'FP32':
                f.write("PI_L1 float l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L1 float l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Define FP16 tensors
            elif data_type_l[layer] == 'FP16':
                f.write("PI_L1 fp16 l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L1 fp16 l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Data type error
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for I/O definition @Layer{}!".format(layer))
                exit()

        if layers_l[layer] == 'Skipnode':
            previous_was_skip = True
        else:
            previous_was_skip = False

    # Save shared activation buffer for layers to be updated
    if use_activation_buffer == True:
        f.write("// Store shared activation buffer for sparse update\n")
        if activation_buffer_bytes == 4:
            f.write("PI_L1 float act_shared_buffer["+str(max_nonsaved_activation)+"];\n")
        elif activation_buffer_bytes == 2:
            f.write("PI_L1 fp16 act_shared_buffer["+str(max_nonsaved_activation)+"];\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid shared activation buffer for sparse update type!!")
            exit()

    # Write IM2COL buffers
    im2col_flag = False
    im2col_type = 'FW'  # 'FW' or 'BW'
    im2col_max_memocc = 0
    im2col_layer_index = 0
    im2col_byte_length = 0
    im2col_max_data_type = 'FP32'
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True: # or layers_l[layer] == 'DW':
            if data_type_l[layer] == 'FP32':
                im2col_byte_length = 4
            elif data_type_l[layer] == 'FP16':
                im2col_byte_length = 2
            im2col_flag = True
            i2c_mem = 0
            i2c_FW = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer]) * math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer]) * im2col_byte_length
            i2c_BW = 0
            if layer > 0 and layer > last_updated_idx:
                i2c_BW = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * hin_l[layer] * win_l[layer] * im2col_byte_length
            if i2c_FW > i2c_BW:
                i2c_mem = i2c_FW
                im2col_type = 'FW'
            else:
                i2c_mem = i2c_BW
                im2col_type = 'BW'
            if i2c_mem > im2col_max_memocc:
                im2col_max_memocc = i2c_mem
                im2col_layer_index = layer
                im2col_max_data_type = data_type_l[layer]
    if im2col_flag == True:
        if im2col_type == 'FW':
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            if im2col_max_data_type == 'FP32':
                f.write("PI_L1 float im2col_buffer[Tin_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tout_H_l"+str(im2col_layer_index)+"*Tout_W_l"+str(im2col_layer_index)+"];\n")
            elif im2col_max_data_type == 'FP16':
                f.write("PI_L1 fp16 im2col_buffer[Tin_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tout_H_l"+str(im2col_layer_index)+"*Tout_W_l"+str(im2col_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for im2col!!")
                exit()
        else:
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            if im2col_max_data_type == 'FP32':
                f.write("PI_L1 float im2col_buffer[Tout_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tin_H_l"+str(im2col_layer_index)+"*Tin_W_l"+str(im2col_layer_index)+"];\n")
            elif im2col_max_data_type == 'FP16':
                f.write("PI_L1 fp16 im2col_buffer[Tout_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tin_H_l"+str(im2col_layer_index)+"*Tin_W_l"+str(im2col_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for im2col!!")
                exit()
    # No im2col buffer
    allocate_no_im2col = False
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == False:
            allocate_no_im2col = True
    if allocate_no_im2col == True:
        f.write("\n// Fake IM2COL buffer for all the convolutions\n")
        f.write("PI_L1 float im2col_buffer[1];\n")

    # Write in grad transposition / blocktranspose buffer
    bt_flag = False
    bt_max_memocc = 0
    bt_layer_index = 0
    wgt_grad_pw = False
    bt_max_data_type = 'FP32'
    for layer in range(len(layers_l)):
        # Check layer data layout
        data_layout = 'CHW'     # Change to input list of data layouts
        if ((layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True) or layers_l[layer] == 'PW') and layer == last_updated_idx: #layer == 0:
            bt_flag = True
            bt_layer_index = 0
        elif ((layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True) or layers_l[layer] == 'PW') and layer > last_updated_idx: #layer > 0:
            bt_flag = True
            bt_mem = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * out_ch_l[layer]
            if bt_mem > bt_max_memocc:
                bt_max_memocc = bt_mem
                bt_layer_index = layer
                bt_max_data_type = data_type_l[layer]
        # Special conditions in case of HWC
        if (data_layout == 'HWC' and layers_l[layer] == 'PW'):
            # Special allocation for weight grad in HWC
            bt_flag = True
            bt_mem = in_ch_l[layer] * hin_l[layer] * win_l[layer]
            if data_type_l[layer] == 'FP16':
                hout = hin_l[layer]; wout = win_l[layer]
                bt_mem += out_ch_l[layer] * hout * wout
            if bt_mem > bt_max_memocc:
                bt_max_memocc = bt_mem
                bt_layer_index = layer
                bt_max_data_type = data_type_l[layer]
                wgt_grad_pw = True
    if (bt_flag == True) and (wgt_grad_pw == False):
        f.write("\n// Define transposition / block transposition buffer for all conv2d and PW layers\n")
        if bt_layer_index == 0:
            f.write("PI_L1 float bt_buffer[1];")
        elif bt_layer_index > 0:
            if bt_max_data_type == 'FP32':
                f.write("PI_L1 float bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tout_C_l"+str(bt_layer_index)+"*Tker_H_l"+str(bt_layer_index)+"*Tker_W_l"+str(bt_layer_index)+"];\n")
            elif bt_max_data_type == 'FP16':
                f.write("PI_L1 fp16 bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tout_C_l"+str(bt_layer_index)+"*Tker_H_l"+str(bt_layer_index)+"*Tker_W_l"+str(bt_layer_index)+"];\n")
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for blocktranspose!")
                exit()
    elif (bt_flag == True) and (wgt_grad_pw == True):
        f.write("\n// Define transposition / block transposition buffer for all conv2d and PW layers\n")
        if bt_max_data_type == 'FP32':
            f.write("PI_L1 float bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tin_H_l"+str(bt_layer_index)+"*Tin_W_l"+str(bt_layer_index)+"];\n")
        elif bt_max_data_type == 'FP16':
            f.write("PI_L1 fp16 bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tin_H_l"+str(bt_layer_index)+"*Tin_W_l"+str(bt_layer_index)+"+Tout_C_l"+str(bt_layer_index)+"*Tout_H_l"+str(bt_layer_index)+"*Tout_W_l"+str(bt_layer_index)+"];\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for pw transp buffer definition!\n")
            exit()
    # No blocktranspose buffer
    if (bt_flag == False):
        print("No blockstranspose buffer detected\n")
        f.write("PI_L1 float bt_buffer[1];\n")



    # Define tensors to backpropagate the output error
    f.write("\n// Define error propagation tensors\n")
    previous_was_skip = False
    for layer in range(len(layers_l)):
        if (not previous_was_skip) and (layer >= last_updated_idx):
            # Define FP32 tensors
            if data_type_l[layer] == 'FP32':
                if layer > 0:
                    f.write("PI_L1 float l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L1 float l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Define FP16 tensors
            elif data_type_l[layer] == 'FP16':
                if layer > 0:
                    f.write("PI_L1 fp16 l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L1 fp16 l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Data type error
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for input grad definition @Layer{}!".format(layer))
                exit()  
        if layers_l[layer] == 'Skipnode':
            previous_was_skip = True
        else:
            previous_was_skip = False   


    # Normalization layer running stats
    f.write("\n// Define running parameters for normalization layers\n")
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'InstNorm':
            if data_type_l[layer] == 'FP32':
                f.write("PI_L1 float l"+str(layer)+"_running_mean[Tin_C_l"+str(layer)+"];\n")
                f.write("PI_L1 float l"+str(layer)+"_running_var[Tin_C_l"+str(layer)+"];\n")
                f.write("PI_L1 float l"+str(layer)+"_running_stdev[Tin_C_l"+str(layer)+"];\n")
            elif data_type_l[layer] == 'FP16':
                f.write("PI_L1 fp16 l"+str(layer)+"_running_mean[Tin_C_l"+str(layer)+"];\n")
                f.write("PI_L1 fp16 l"+str(layer)+"_running_var[Tin_C_l"+str(layer)+"];\n")
                f.write("PI_L1 fp16 l"+str(layer)+"_running_stdev[Tin_C_l"+str(layer)+"];\n")


    # Define buffer for mixed precision propagation
    # TODO: When extending bias to FP16, adapt this section accordingly
    previous_type = data_type_l[0]
    is_mixed_precision = False
    curr_cast_in_size = 0
    curr_cast_out_size = 0
    curr_max_size = 0
    is_max_input = False
    max_cast_buffer_index = 0
    max_cast_buffer_size = 0
    max_cast_buffer_type = 'FP32'
    for layer in range(len(layers_l)):
        # Output size for current layer
        h_out = math.floor((hin_l[layer]-hk_l[layer]+2*h_pad_l[layer]+h_str_l[layer])/h_str_l[layer])
        w_out = math.floor((win_l[layer]-wk_l[layer]+2*w_pad_l[layer]+w_str_l[layer])/w_str_l[layer])
        # Find if there are mixed types
        if data_type_l[layer] != previous_type:
            is_mixed_precision = True
            # Find biggest size
            curr_cast_in_size = in_ch_l[layer] * hin_l[layer] * win_l[layer]
            curr_cast_out_size = out_ch_l[layer] * h_out * w_out
            if curr_cast_in_size > curr_cast_out_size:
                curr_max_size = curr_cast_in_size
                is_max_input = True
            else:
                curr_max_size = curr_cast_out_size
                is_max_input = False
            if curr_max_size > max_cast_buffer_size:
                max_cast_buffer_size = curr_max_size
                max_cast_buffer_type = data_type_l[layer-1]
                max_cast_buffer_index = layer
        previous_type = data_type_l[layer]

    # Allocate buffer
    if is_mixed_precision:
        f.write("\n// Define cast buffer to manage mixed precision (size="+str(max_cast_buffer_size)+")\n")
        if max_cast_buffer_type == 'FP32':
            if is_max_input:
                f.write("PI_L1 float cast_buffer[Tin_C_l"+str(max_cast_buffer_index)+" * Tin_H_l"+str(max_cast_buffer_index)+" * Tin_W_l"+str(max_cast_buffer_index)+"];\n")
            else:
                f.write("PI_L1 float cast_buffer[Tout_C_l"+str(max_cast_buffer_index)+" * Tout_H_l"+str(max_cast_buffer_index)+" * Tout_W_l"+str(max_cast_buffer_index)+"];\n")
        elif max_cast_buffer_type == 'FP16':
            if is_max_input:
                f.write("PI_L1 fp16 cast_buffer[Tin_C_l"+str(max_cast_buffer_index)+" * Tin_H_l"+str(max_cast_buffer_index)+" * Tin_W_l"+str(max_cast_buffer_index)+"];\n")
            else:
                f.write("PI_L1 fp16 cast_buffer[Tout_C_l"+str(max_cast_buffer_index)+" * Tout_H_l"+str(max_cast_buffer_index)+" * Tout_W_l"+str(max_cast_buffer_index)+"];\n")
        else:
            print("[deployment_utils.GenerateNet]: Invalid data type for mixed precision buffer!")
            exit() 



    f.write("\n// Loss function configuration structure\n")
    if data_type_l[-1] == 'FP32':
        f.write("PI_L1 struct loss_args loss_args;\n")
    elif data_type_l[-1] == 'FP16':
        f.write("PI_L1 struct loss_args_fp16 loss_args;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid data type for loss definition!")
        exit()
        

    f.write("\n\n\n/**\n * DNN BACKEND FUNCTIONS\n**/\n")

    f.write("\n// DNN initialization function\n")
    f.write("void DNN_init()\n{\n")
    for layer in range(len(layers_l)):
        if layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)\t\t\tl0_in[i] = INPUT[i];\n")
            if layers_l[layer] not in ['Skipnode', 'Sumnode', 'InstNorm', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)\t\tl0_ker[i] = init_WGT_l0[i];\n")
                if bias_l[layer] == 1:
                    f.write("  for(int i=0; i<Tout_C_l0; i++)\t\tl0_bias[i] = init_BIAS_l0[i];\n")
            elif layers_l[layer] == 'InstNorm':
                f.write("  for(int i=0; i<2*Tin_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            if layers_l[layer] == 'DW':
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
            elif layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool':
                f.write("  //   Pooling kernel (no parameters)\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                f.write("  //   Resconn layer (no parameters)\n")
            elif layers_l[layer] in ['ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  //   Activation layer (no parameters)\n")
            elif layers_l[layer] == 'InstNorm':
                f.write("  for(int i=0; i<2*Tin_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"; i++) {\n")
                f.write("  \t\tl"+str(layer)+"_running_mean[i] = 0;")
                f.write("  \t\tl"+str(layer)+"_running_var[i] = 1;")
                f.write("  \t\tl"+str(layer)+"_running_stdev[i] = 1;\n  }")
            else:
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
                if bias_l[layer] == 1:
                    f.write("  for(int i=0; i<Tout_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_bias[i] = init_BIAS_l"+str(layer)+"[i];\n")
        elif layer == len(layers_l)-1:
            if layers_l[layer] not in  ['Skipnode', 'Sumnode', 'InstNorm', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  // Layer "+str(layer)+"\n")
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
                if bias_l[layer] == 1:
                    f.write("  for(int i=0; i<Tout_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_bias[i] = init_BIAS_l"+str(layer)+"[i];\n")
            elif layers_l[layer] == 'InstNorm':
                f.write("  for(int i=0; i<2*Tin_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()



    # Mixed precision check
    # TODO: When extending bias to FP16, adapt this section accordingly
    C_data_type = 'float'
    f.write("\n  // Connect tensors to blobs\n")
    previous_was_skip_data = 0
    previous_was_skip_diff = 0
    for layer in range(len(layers_l)):
        # Find data type for each layer
        if data_type_l[layer] == 'FP32':
            C_data_type = 'float'
        elif data_type_l[layer] == 'FP16':
            C_data_type = 'fp16'
        else:
            print("[deployment_utils.GenerateNet]: Invalid data type for structure assignment @layer{}!".format(layer))
            exit()
        f.write(f"\n\n//Connecting {layers_l[layer]}\n")

        # INPUT BLOB
        if len(layers_l) == 1:                          # DNN is 1 layer long
            f.write("  layer"+str(layer)+"_in.data = l0_in;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l0;\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l0;\n")
        elif layer == 0:                                # First layer
            f.write("  // Layer "+str(layer)+"\n")
            if layer > 0 and update_layer_l[layer] == 0 and layers_l[layer] not in ['Skipnode', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  layer"+str(layer)+"_in.data = ("+C_data_type+"*) act_shared_buffer;\n")
            else:
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
        elif layer > 0 and layer < len(layers_l)-1:     # Hidden layers
            f.write("  // Layer "+str(layer)+"\n")
            if layer > 0 and update_layer_l[layer] == 0 and layers_l[layer] not in ['Skipnode', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  layer"+str(layer)+"_in.data = ("+C_data_type+"*) act_shared_buffer;\n")
            else:
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip_data)+"_in;\n")
            if (layer - previous_was_skip) > last_updated_idx: #0: # Avoid assignement of l0_in_diff
                f.write("  layer"+str(layer)+"_in.diff = l"+str(layer - previous_was_skip_diff)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
        elif layer == len(layers_l)-1:                  # Last layer
            f.write("  // Layer "+str(layer)+"\n")
            if layer > 0 and update_layer_l[layer] == 0 and layers_l[layer] not in ['Skipnode', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  layer"+str(layer)+"_in.data = ("+C_data_type+"*) act_shared_buffer;\n")
            else:
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip_data)+"_in;\n")
            if (layer) > last_updated_idx:
                f.write("  layer"+str(layer)+"_in.diff = l"+str(layer - previous_was_skip_diff)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

        # WEIGHT BLOB
        if len(layers_l) == 1:                          # DNN is 1 layer long
            f.write("  layer"+str(layer)+"_wgt.data = l0_ker;\n")
            if update_layer_l[layer] == 1:    # Sparse Update
                f.write("  layer"+str(layer)+"_wgt.diff = l0_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tker_H_l0*Tker_W_l0;\n")
            elif layers_l[layer] == 'InstNorm':
                f.write("  layer"+str(layer)+"_wgt.dim = 2*Tin_C_l0;\n")
            elif layers_l[layer] in  ['Skipnode', 'Sumnode', 'AvgPool', 'MaxPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                f.write("  layer"+str(layer)+"_wgt.dim = 1;\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l0;\n")
            if bias_l[layer] == 1:
                f.write("  layer"+str(layer)+"_bias.data = l"+str(layer)+"_bias;\n")
                if update_layer_l[layer] == 1:
                    f.write("  layer"+str(layer)+"_bias.diff = l"+str(layer)+"_bias_diff;\n")
                f.write("  layer"+str(layer)+"_bias.dim = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_bias.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_bias.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_bias.W = Tout_W_l"+str(layer)+";\n")
        # First layer connection
        elif layer == 0:
            if layers_l[0] != 'Skipnode': # Avoid weight assignment for Skip Connections
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                if update_layer_l[layer] == 1:    # Sparse Update
                    f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                elif layers_l[layer] == 'InstNorm':
                    f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                elif layers_l[layer] in  ['Skipnode', 'Sumnode', 'AvgPool', 'MaxPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                    f.write("  layer"+str(layer)+"_wgt.dim = 1;\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")

                if bias_l[layer] == 1:
                    f.write("  layer"+str(layer)+"_bias.data = l"+str(layer)+"_bias;\n")
                    if update_layer_l[layer] == 1:
                        f.write("  layer"+str(layer)+"_bias.diff = l"+str(layer)+"_bias_diff;\n")
                    f.write("  layer"+str(layer)+"_bias.dim = Tout_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.C = Tout_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.H = Tout_H_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.W = Tout_W_l"+str(layer)+";\n")
        elif layer > 0 and layer < len(layers_l)-1:     # Hidden layers
            if layers_l[layer] != 'Skipnode':   # Avoid weight assignment for Skipnodes and out data assignement
                if layers_l[layer]  != 'Sumnode':    # Avoid ONLY weight assignment for Sumnodes
                    f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                    if update_layer_l[layer] == 1:    # Sparse Update
                        f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                    if layers_l[layer] == 'DW':
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    elif layers_l[layer] == 'InstNorm':
                        f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                    elif layers_l[layer] in  ['Skipnode', 'Sumnode', 'AvgPool', 'MaxPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                        f.write("  layer"+str(layer)+"_wgt.dim = 1;\n")
                    else:
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                    if bias_l[layer] == 1:
                        f.write("  layer"+str(layer)+"_bias.data = l"+str(layer)+"_bias;\n")
                        if update_layer_l[layer] == 1:
                            f.write("  layer"+str(layer)+"_bias.diff = l"+str(layer)+"_bias_diff;\n")
                        f.write("  layer"+str(layer)+"_bias.dim = Tout_C_l"+str(layer)+";\n")
                        f.write("  layer"+str(layer)+"_bias.C = Tout_C_l"+str(layer)+";\n")
                        f.write("  layer"+str(layer)+"_bias.H = Tout_H_l"+str(layer)+";\n")
                        f.write("  layer"+str(layer)+"_bias.W = Tout_W_l"+str(layer)+";\n")
        elif layer == len(layers_l)-1:                  # Last layer
            if layers_l[layer] !=  'Sumnode':
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                if update_layer_l[layer] == 1:
                    f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                elif layers_l[layer] == 'InstNorm':
                        f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                elif layers_l[layer] in  ['Skipnode', 'Sumnode', 'AvgPool', 'MaxPool', 'ReLU', 'LeakyReLU', 'Sigmoid']:
                    f.write("  layer"+str(layer)+"_wgt.dim = 1;\n")            
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                if bias_l[layer] == 1:
                    f.write("  layer"+str(layer)+"_bias.data = l"+str(layer)+"_bias;\n")
                    if update_layer_l[layer] == 1:
                        f.write("  layer"+str(layer)+"_bias.diff = l"+str(layer)+"_bias_diff;\n")
                    f.write("  layer"+str(layer)+"_bias.dim = Tout_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.C = Tout_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.H = Tout_H_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_bias.W = Tout_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

        # OUTPUT BLOB
        if len(layers_l) == 1:                          # DNN is 1 layer long
            f.write("  layer"+str(layer)+"_out.data = l0_out;\n")
            if layer >= last_updated_idx:
                f.write("  layer"+str(layer)+"_out.diff = l0_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l0;\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l0;\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l0;\n")
        elif layer == 0:                                # First layer
            if layers_l[0] != 'Skipnode': # Avoid weight assignment for Skip Connections
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    if layer >= last_updated_idx:
                        f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    if layer < (len(layers_l)-1) and update_layer_l[layer+1] == 0 and layers_l[layer+1] not in ['ReLU', 'LeakyReLU', 'Sigmoid']:
                        f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) act_shared_buffer;\n")
                    else:
                        f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if layer >= last_updated_idx:
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1)+"_in_diff;\n")     
                # End of assignment       
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        elif layer > 0 and layer < len(layers_l)-1:     # Hidden layers
            if layers_l[layer] != 'Skipnode':   # Avoid weight assignment for Skipnodes and out data assignement
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    if layer >= last_updated_idx:
                        f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    if layer < (len(layers_l)-1) and update_layer_l[layer+1] == 0 and layers_l[layer+1] not in ['ReLU', 'LeakyReLU', 'Sigmoid']:
                        f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) act_shared_buffer;\n")
                    else:
                        f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if sumnode_connections[layer] == -1:
                        if layer >= last_updated_idx:
                            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1)+"_in_diff;\n")
                    elif layers_l[layer] == 'Sumnode':
                        if layer >= last_updated_idx:
                            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1)+"_in_diff;\n")
                    else:
                        if layer >= last_updated_idx:
                            f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")
                # End of assignment     
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        elif layer == len(layers_l)-1:                  # Last layer
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_out;\n")
            if layer >= last_updated_idx:
                f.write("  layer"+str(layer)+"_out.diff = l"+str(layer)+"_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

        if sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            if layers_l[layer] == 'Skipnode':
                previous_was_skip_data += 1
                previous_was_skip_diff += 1
            else: 
                previous_was_skip_diff = 0
        else:
            previous_was_skip_data = 0
            previous_was_skip_diff = 0



    f.write("\n  // Configure layer structures\n")
    first_is_skip = False # Avoid calculation of gradient if the first Layer is a skipnode
    if sumnode_connections[0] != -1:
        first_is_skip = True
    previous_was_skip = 0
    for layer in range(len(layers_l)):
        f.write("  // Layer "+str(layer)+"\n")
        if layer == 0 or layer <= last_updated_idx:
            skip_inputgrad = 1
        elif layer - previous_was_skip <= 0: # If the 0 layer is a Skipnode, then layer1's diff is the input gradient
            skip_inputgrad = 1
        else: 
            skip_inputgrad = 0
        # Check if to perform in grad
        #if layer > 0:
        #    if update_layer_l[layer-1] == 0:
        #        skip_inputgrad = 1
        # Write configuration templates
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_config_template(layer, skip_inputgrad, data_type_l[layer], bias_l[layer], update_layer_l[layer]))
        elif layers_l[layer] == 'conv2d':
            IM2COL_USEIT = 1
            if CONV2D_USE_IM2COL == False:
                IM2COL_USEIT = 0
            f.write(ntemp.conv2d_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer], bias_l[layer], IM2COL_USEIT, update_layer_l[layer]))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_config_template(layer, skip_inputgrad, data_type_l[layer], update_layer_l[layer]))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer], update_layer_l[layer]))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_config_template(layer, data_type_l[layer]))
        elif layers_l[layer] == 'LeakyReLU':
            f.write(ntemp.LeakyReLU_config_template(layer, data_type_l[layer]))
        elif layers_l[layer] == 'Sigmoid':
            f.write(ntemp.Sigmoid_config_template(layer, data_type_l[layer]))
        elif layers_l[layer] == 'MaxPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'AvgPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'Sumnode':
            f.write(ntemp.resconn_config_template(layer, sumnode_connections[layer], first_is_skip, layers_l[sumnode_connections[layer]]))
            first_is_skip = False
        elif layers_l[layer] == 'Skipnode':
            pass
        elif layers_l[layer] == 'InstNorm':
            f.write(ntemp.InstNorm_config_template(layer, skip_inputgrad, update_layer_l[layer]))
        else:
            print("[deployment_utils.GenerateNet] Undefined layer "+str(layer)+" (unable to write configuration structure)!!")
        if sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            previous_was_skip += 1
        else:
            previous_was_skip = 0

    pooling_exist = False
    for layer in range(len(layers_l)):
        if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
            pooling_exist = True
    if pooling_exist:
        f.write("\n  // Connect blobs to pooling structures\n")
        for layer in range(len(layers_l)):
            if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
                f.write("  // Layer "+str(layer)+"\n")
                f.write("  l"+str(layer)+"_pool_args.input = &layer"+str(layer)+"_in;\n")
                f.write("  l"+str(layer)+"_pool_args.output = &layer"+str(layer)+"_out;\n")
                f.write("  l"+str(layer)+"_pool_args.Hker = Tker_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Wker = Tker_W_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Hstride = Tstr_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_pool_args.Wstride = Tstr_W_l"+str(layer)+";\n")
    f.write("}\n\n")



    f.write("\n// Forward pass function\n")
    f.write("void forward()\n{\n")

    # Profiling options: single layer or all
    if PROFILE_SINGLE_LAYERS == True:
        f.write("  printf(\"\\nFORWARD PROFILING:\\n\");\n")

    for layer in range(len(layers_l)):

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  printf(\"\\nLayer "+str(layer)+"\\n\");\n")
            f.write("  #ifdef PROF_NET\n")
            f.write("  START_STATS();\n")
            f.write("  #endif\n")      

        # Generate layer template
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'conv2d':
            f.write(ntemp.conv2d_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'LeakyReLU':
            f.write(ntemp.LeakyReLU_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'Sigmoid':
            f.write(ntemp.Sigmoid_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'AvgPool':
            f.write(ntemp.AvgPool_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'MaxPool':
            f.write(ntemp.MaxPool_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer] == 'Skipnode':
            pass
        elif layers_l[layer] == 'Sumnode':
            f.write(ntemp.residualconn_template_FW(layer, data_type_l[layer]))
        elif layers_l[layer]  == 'InstNorm':
            f.write(ntemp.InstNorm_template_FW(layer, data_type_l[layer]))
        else:
            print("[deployment_utils.GenerateNet FW]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
        # Insert casting operator for data type variation
        if layer < len(layers_l)-1 and data_type_l[layer] != data_type_l[layer+1]:
            if data_type_l[layer] == 'FP32' and data_type_l[layer+1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(layer, "FW", data_type_l[layer]))
            elif data_type_l[layer] == 'FP16' and data_type_l[layer+1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(layer, "FW", data_type_l[layer]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[layer], data_type_l[layer+1], layer))
        
        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  #ifdef PROF_NET\n")
            f.write("  STOP_STATS();\n")
            f.write("  #endif\n")  
    f.write("}\n")



    f.write("\n// Backward pass function\n")
    f.write("void backward()\n{\n")

    # Compute loss
    if loss_fn == "MSELoss":
        f.write("  loss_args.output = &layer"+str(len(layers_l)-1)+"_out;\n")
        f.write("  loss_args.target = LABEL;\n")
        f.write("  loss_args.wr_loss = &loss;\n") 
        if data_type_l[-1] == 'FP32':
            f.write("  pulp_MSELoss_backward(&loss_args);\n")   
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_MSELoss_backward_fp16(&loss_args);\n") 
    elif loss_fn == 'CrossEntropyLoss':
        f.write("  loss_args.output = &layer"+str(len(layers_l)-1)+"_out;\n")
        f.write("  loss_args.target = LABEL;\n")
        f.write("  loss_args.wr_loss = &loss;\n")
        if data_type_l[-1] == 'FP32':
            f.write("  pulp_CrossEntropyLoss_backward(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_CrossEntropyLoss_backward_fp16(&loss_args);\n")
    else:
        print("[deployment_utils.GenerateNet]: invalid loss function for backward!!")

    # Profiling options: single layer or all
    if PROFILE_SINGLE_LAYERS == True:
        f.write("  printf(\"\\nBACKWARD PROFILING:\\n\");\n")

    prev_sumnode = 0 #For Skip Connections
    for layer in range(len(layers_l)):
        lay = len(layers_l) - layer - 1

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  printf(\"\\nLayer "+str(lay)+"\\n\");\n")
            f.write("  #ifdef PROF_NET\n")
            f.write("  START_STATS();\n")
            f.write("  #endif\n")    

        # Determine if backprop is needed
        stop_backprop = False
        if lay <= last_updated_idx:
            stop_backprop = True

        # Generate backward layer template
        skip_in_grad = 0
        FIRST_LAYER = False
        if lay == 0 or stop_backprop:
            skip_in_grad = 1
            FIRST_LAYER = True
        print(f"Layer {lay}: stop_backprop = {FIRST_LAYER}, update_layer = {update_layer_l[lay]} (have last_updated_layer = {last_updated_idx})")
        if layers_l[lay] == 'linear':
            f.write(ntemp.linear_template_BW(lay, data_type_l[lay], SEPARATE_BACKWARD_STEPS, FIRST_LAYER, update_layer_l[lay]))
        elif layers_l[lay] == 'conv2d':
            f.write(ntemp.conv2d_template_BW(lay, data_type_l[lay], SEPARATE_BACKWARD_STEPS, FIRST_LAYER, update_layer_l[lay]))
        elif layers_l[lay] == 'DW':
            f.write(ntemp.DW_template_BW(lay, data_type_l[lay], SEPARATE_BACKWARD_STEPS, FIRST_LAYER, update_layer_l[lay]))
        elif layers_l[lay] == 'PW':
            f.write(ntemp.PW_template_BW(lay, data_type_l[lay], SEPARATE_BACKWARD_STEPS, FIRST_LAYER, update_layer_l[lay]))
        elif layers_l[lay] == 'ReLU':
            f.write(ntemp.ReLU_template_BW(lay, data_type_l[lay], FIRST_LAYER))
        elif layers_l[lay] == 'LeakyReLU':
            f.write(ntemp.LeakyReLU_template_BW(lay, data_type_l[lay], FIRST_LAYER))
        elif layers_l[lay] == 'Sigmoid':
            f.write(ntemp.Sigmoid_template_BW(lay, data_type_l[lay], FIRST_LAYER))
        elif layers_l[lay] == 'AvgPool':
            f.write(ntemp.AvgPool_template_BW(lay, data_type_l[lay], FIRST_LAYER))
        elif layers_l[lay] == 'MaxPool':
            f.write(ntemp.MaxPool_template_BW(lay, data_type_l[lay], FIRST_LAYER))
        elif layers_l[lay] == 'Skipnode':
            f.write(ntemp.residualconn_template_sum_BW(sumnode_connections[lay], data_type_l[lay], last_updated_idx))
        elif layers_l[lay] == 'Sumnode':
            f.write(ntemp.residualconn_template_copy_BW(lay, data_type_l[lay], last_updated_idx))
            prev_sumnode = lay
        elif layers_l[lay]  == 'InstNorm':
            f.write(ntemp.InstNorm_template_BW(lay, data_type_l[lay], SEPARATE_BACKWARD_STEPS, FIRST_LAYER, update_layer_l[lay]))
        else:
            print("[deployment_utils.GenerateNet BW]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
        # Insert casting operator for data type variation
        if lay < len(layers_l)-1 and lay > 0 and data_type_l[lay] != data_type_l[lay-1]:
            if data_type_l[lay] == 'FP32' and data_type_l[lay-1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(lay, "BW", data_type_l[lay]))
            elif data_type_l[lay] == 'FP16' and data_type_l[lay-1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(lay, "BW", data_type_l[lay]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[lay], data_type_l[lay-1], lay))
        if sumnode_connections[lay] != -1 and layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and skip_in_grad==0:
            f.write(ntemp.sum(lay, data_type_l[lay]))

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  #ifdef PROF_NET\n")
            f.write("  STOP_STATS();\n")
            f.write("  #endif\n")  
    f.write("}\n")


    f.write("\n// Compute loss and output gradient\n")
    f.write("void compute_loss()\n{\n")

    if loss_fn == "MSELoss":
        f.write("  loss_args.output = &layer"+str(len(layers_l)-1)+"_out;\n")
        f.write("  loss_args.target = LABEL;\n")
        f.write("  loss_args.wr_loss = &loss;\n")
        if data_type_l[-1] == 'FP32':
            f.write("  pulp_MSELoss(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_MSELoss_fp16(&loss_args);\n")
        else:
            print("[deplyment_utils.GenerateNet]: Invalid loss type!")
            exit()
    elif loss_fn == "CrossEntropyLoss":
        f.write("  loss_args.output = &layer"+str(len(layers_l)-1)+"_out;\n")
        f.write("  loss_args.target = LABEL;\n")
        f.write("  loss_args.wr_loss = &loss;\n")
        if data_type_l[-1] == 'FP32':
            f.write("  pulp_CrossEntropyLoss(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_CrossEntropyLoss_fp16(&loss_args);\n")
        else:
            print("[deplyment_utils.GenerateNet]: Invalid loss type!")
            exit()
    else:
        print("[deployment_utils.GenerateNet]: Loss function not valid for PULP deployment!!")
        exit()

    f.write("}\n")


    f.write("\n// Function to update the network\n")
    f.write("void update_weights()\n{\n")

    for layer in range(len(layers_l)):
        if layers_l[layer] in ['linear', 'conv2d', 'DW', 'PW', 'InstNorm'] and update_layer_l[layer] == 1:
            if data_type_l[layer] == 'FP32':
                f.write("  struct optim_args opt_l"+str(layer)+";\n")
            elif data_type_l[layer] == 'FP16':
                f.write("  struct optim_args_fp16 opt_l"+str(layer)+";\n")
            else:
                print("[deployment_utils.GenerateNet]: Invalid data type for optimizer structure generation @layer{}!".format(layer))  
            f.write("  opt_l"+str(layer)+".weights = &layer"+str(layer)+"_wgt;\n")
            if bias_l[layer] == 1:
                f.write("  opt_l"+str(layer)+".biases = &layer"+str(layer)+"_bias;\n")
                f.write("  opt_l"+str(layer)+".use_biases = 1;\n")
            else:
                f.write("  opt_l"+str(layer)+".use_biases = 0;\n")
            f.write("  opt_l"+str(layer)+".learning_rate = LEARNING_RATE;\n")
            if optimizer == "SGD":
                if data_type_l[layer] == 'FP32':
                    f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l"+str(layer)+");\n")
                elif data_type_l[layer] == 'FP16':
                    f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp16, &opt_l"+str(layer)+");\n")
                else:
                    print("[deployment_utils.GenerateNet]: Invalid data type for gradient descent @Layer{}!".format(layer))
            else:
                print("[deployment_utils.GenerateNet]: Invalid optimizer for PULP deployment!!")
                exit()
    f.write("}\n")


    f.write("\n\n\n/**\n * DATA VISUALIZATION AND CHECK TOOLS\n**/\n")

    f.write("\n// Function to print FW output\n")
    f.write("void print_output()\n{\n")
    output_index = len(layers_l) - 1
    f.write("  printf(\"\\nLayer "+str(output_index)+" output:\\n\");\n\n")
    f.write("  for (int i=0; i<Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+"; i++)\n  {\n")
    f.write("    printf(\"%f \", l"+str(output_index)+"_out[i]);\n")
    f.write("    // Newline when an output row ends\n")
    f.write("    // if(!(i%Tout_W_l"+str(output_index)+")) printf(\"\\n\");\n")
    f.write("    // Newline when an output channel ends\n")
    f.write("    if(!(i%Tout_W_l"+str(output_index)+"*Tout_H_l"+str(output_index)+")) printf(\"\\n\");\n")
    f.write("  }\n")
    f.write("  printf(\"\\n\");\n")
    f.write("}\n")


    f.write("\n// Function to check post-training output wrt Golden Model (GM)\n")
    f.write("void check_post_training_output()\n{\n")

    output_index = len(layers_l) - 1
    f.write("  int integrity_check = 0;\n")
    if data_type_l[output_index] == 'FP32':
        f.write("  integrity_check = verify_tensor(l"+str(output_index)+"_out, REFERENCE_OUTPUT, Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+", TOLERANCE);\n")
    elif data_type_l[output_index] == 'FP16':
        f.write("  integrity_check = verify_tensor_fp16(l"+str(output_index)+"_out, REFERENCE_OUTPUT, Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+", TOLERANCE);\n")
    else:
        print("[deployment_utils.GenerateNet]: Invalid inference verification data type!!")
        exit()
    f.write("  if (integrity_check > 0)\n")
    f.write("    printf(\"\\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\\n\");\n")

    f.write("}\n")



    f.write("\n\n\n/**\n * DNN MODEL TRAINING\n**/\n")

    f.write("\n// Call for a complete training step\n")
    f.write("void net_step()\n{\n")

    f.write("\tprintf(\"Initializing network..\\n\");\n")
    f.write("\tDNN_init();\n")

    f.write("\tprintf(\"Testing DNN initialization forward..\");\n")
    f.write("\tforward();\n")
    f.write("\tprint_output();\n\n")

    # Profile layer by layer?
    if PROFILE_SINGLE_LAYERS == False:
        f.write("\t#ifdef PROF_NET\n")
        f.write("\tINIT_STATS();\n\tPRE_START_STATS();\n\tSTART_STATS();\n")
        f.write("\t#endif\n\n")

    f.write("\tfor (int epoch=0; epoch<EPOCHS; epoch++)\n\t{\n")
    f.write("\t\tforward();\n")
    f.write("\t\tcompute_loss();\n")
    if PRINT_TRAIN_LOSS == True:
        f.write("\t\t/* Stop profiling */ pi_perf_stop();\n")
        f.write("\t\tif (epoch == 0) printf(\"\\n\");\n")
        f.write("\t\tprintf(\">>> EPOCH %d: train_loss = %f (GM: %f)\\n\", epoch, loss, TRAIN_LOSS[epoch]);\n")
        f.write("\t\t/* Continue profiling */ pi_perf_start();\n")
    f.write("\t\tbackward();\n")
    f.write("\t\tupdate_weights();\n")
    f.write("\t}\n\n")

    # Profile layer by layer?
    if PROFILE_SINGLE_LAYERS == False:
        f.write("\t#ifdef PROF_NET\n")
        f.write("\tSTOP_STATS();\n")
        f.write("\t#endif\n\n")


    f.write("\t// Check and print updated output\n")
    f.write("\tforward();\n")
    f.write("\tprintf(\"Checking updated output..\\n\");\n")
    f.write("\tcheck_post_training_output();\n")
    f.write("\tprint_output();\n")

    f.write("}\n")



    f.close()

    return

