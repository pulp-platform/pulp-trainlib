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

import os
import shutil

from torch import mm
import utils.GM_templates as Gtemp
import utils.net_templates as ntemp


"""
DNN Size Checker backend functions
"""

def compute_wgt_act_memocc_bytes(layer_number, layer_type, chin, chout, hk, wk, hin, win, DATA_TYPE, is_last_layer):

    memocc_bytes = 0

    # First layer does not have in grad
    in_grad_present = 1
    if layer_number == 0:
        in_grad_present = 0
    
    # Last layer occupies output memory (other activations overlap)
    output_separate_occupation = 0
    if is_last_layer:
        output_separate_occupation = 1

    # If the layer is an activation, no weights!
    wgt_present = 1
    if layer_type == 'ReLu':
        wgt_present = 0

    byte_size = 4
    if DATA_TYPE == 'FP32':
        byte_size = 4
    elif DATA_TYPE == 'FP16':
        byte_size = 2
    else:
        print("[deployment_utils.compute_wgt_act_memocc_bytes]: Invalid data type!!")
        exit()

    # FORWARD
    # Input act
    memocc_bytes += chin * hin * win * byte_size
    # Weights
    memocc_bytes += chin * chout * hk * wk * byte_size * wgt_present
    # Out act
    memocc_bytes += chout * (hin-hk+1) * (win-wk+1) * byte_size * output_separate_occupation

    # BACKWARD
    # Input act grad
    memocc_bytes += chin * hin * win * byte_size * in_grad_present
    # Weight grad
    memocc_bytes += chin * chout * hk * wk * byte_size * wgt_present
    # Output grad
    memocc_bytes += chout * (hin-hk+1) * (win-wk+1) * byte_size * output_separate_occupation


    return memocc_bytes


def compute_im2col_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, data_type_l):

    memocc_bytes = 0

    max_im2col_size = 0
    max_im2col_index = 0
    for layer in range(len(layers_l)):
        # Check layer data type
        byte_size = 4
        if data_type_l[layer] == 'FP32':
            byte_size = 4
        elif data_type_l[layer] == 'FP16':
            byte_size = 2
        else:
            print("[deployment_utils.compute_im2col_memocc_bytes]: Invalid data type @Layer"+str{layer}+"!!")
            exit()        
        # Find max im2col size
        if layers_l[layer] == 'conv2d' or layers_l[layer] == 'DW':
            im2col_size = 0
            size_FW = hk_l[layer] * wk_l[layer] * in_ch_l[layer] * (hin_l[layer]-hk_l[layer]+1) * (win_l[layer]-wk_l[layer]+1) * byte_size
            size_BW = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * hin_l[layer] * win_l[layer] * byte_size
            if size_FW > size_BW:
                im2col_size = size_FW
            else:
                im2col_size = size_BW
            #im2col_size = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * in_ch_l[layer] * (hin_l[layer]-hk_l[layer]+1) * (win_l[layer]-wk_l[layer]+1) * byte_size
            if im2col_size > max_im2col_size:
                max_im2col_size = im2col_size
                max_im2col_index = layer
    
    #print("Max im2col size (@layer {}): {}".format(max_im2col_index, max_im2col_size))
    memocc_bytes += max_im2col_size

    return memocc_bytes, max_im2col_index


def compute_bt_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, data_type_l):

    memocc_bytes = 0
    
    byte_size = 4
    if data_type_l[layer] == 'FP32':
        byte_size = 4
    elif data_type_l[layer] == 'FP16':
        byte_size = 2
    else:
        print("[deployment_utils.compute_bt_memocc_bytes]: Invalid data type!!")
        exit()

    max_bt_size = 0
    max_bt_index = 0
    for layer in range(len(layers_l)):
        # Check layer data type
        byte_size = 4
        if data_type_l[layer] == 'FP32':
            byte_size = 4
        elif data_type_l[layer] == 'FP16':
            byte_size = 2
        else:
            print("[deployment_utils.compute_bt_memocc_bytes]: Invalid data type @Layer"+str(layer)+"!!")
            exit()
        # Find max blocktransp size
        if layers_l[layer] == 'conv2d' and layer > 0:
            bt_size = hk_l[layer] * wk_l[layer] * in_ch_l[layer] * out_ch_l[layer] * byte_size
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

    shutil.copy2('./utils/srcfiles/main.c', proj_folder)
    shutil.copy2('./utils/srcfiles/stats.h', proj_folder)
    shutil.copy2('./utils/srcfiles/dump_utils.py', utils_folder)
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
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp32.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp32.c\n\n')
    elif check_FP16 == True:
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_matmul_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_im2col_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv2d_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_linear_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_pw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_conv_dw_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_train_utils_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_losses_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_optimizers_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_act_fp16.c\n')
        f.write('APP_SRCS += $(TRAIN_LIB_SRCS)/pulp_pooling_fp16.c\n\n')        
    else:
        print("[deployment_utils.GenerateMakefile] Data format not implemented!!\n")
        exit()

    f.write('# RULES\n')
    f.write('get_golden:\n')
    f.write('\tpython3.6 ./utils/GM.py\n')
    f.write('\n')

    f.write('include $(RULES_DIR)/pmsis_rules.mk\n')

    f.close()

    return






# Generates the Golden Model
def GenerateGM(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l):

    f = open(proj_folder_path+'utils/GM.py', 'w')

    f.write("import torch\n")
    f.write("import torch.nn as nn\n")
    f.write("import torch.optim as optim\n")
    f.write("import dump_utils as dump\n")
    f.write("import math\n")
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
        # Padging and stride
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
        f.write("f.write('#define Tker_H_l"+str(layer)+" '+str(l"+str(layer)+"_hk)+'\\n')\n")
        f.write("f.write('#define Tker_W_l"+str(layer)+" '+str(l"+str(layer)+"_wk)+'\\n')\n")
        f.write("f.write('#define Tin_H_l"+str(layer)+" '+str(l"+str(layer)+"_hin)+'\\n')\n")
        f.write("f.write('#define Tin_W_l"+str(layer)+" '+str(l"+str(layer)+"_win)+'\\n')\n")
        f.write("f.write('#define Tout_H_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_hin-l"+str(layer)+"_hk+2*l"+str(layer)+"_hpad+l"+str(layer)+"_hstr)/l"+str(layer)+"_hstr))+'\\n')\n")
        f.write("f.write('#define Tout_W_l"+str(layer)+" '+str(math.floor((l"+str(layer)+"_win-l"+str(layer)+"_wk+2*l"+str(layer)+"_wpad+l"+str(layer)+"_wstr)/l"+str(layer)+"_wstr))+'\\n')\n")
        # Padding and stride
        f.write("f.write('#define Tstr_H_l"+str(layer)+" '+str(l"+str(layer)+"_hstr)+'\\n')\n")
        f.write("f.write('#define Tstr_W_l"+str(layer)+" '+str(l"+str(layer)+"_wstr)+'\\n')\n")
        f.write("f.write('#define Tpad_H_l"+str(layer)+" '+str(l"+str(layer)+"_hpad)+'\\n')\n")
        f.write("f.write('#define Tpad_W_l"+str(layer)+" '+str(l"+str(layer)+"_wpad)+'\\n')\n")
    f.write("f.close()\n\n")

    # Define hyperparameters
    f.write("# Define hyperparameters\n")
    f.write("learning_rate = "+str(learning_rate)+"\n")
    f.write("batch_size = "+str(batch_size)+"\n")
    f.write("epochs = "+str(epochs)+"\n")
    f.write("\n")

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
        f.write("inp = torch.div(torch.ones(l0_in_ch), 100000)\n")
    elif (layers_l[0] == 'conv2d' or layers_l[0] == 'DW' or layers_l[0] == 'PW'):
        f.write("inp = torch.torch.div(torch.ones(batch_size, l0_in_ch, l0_hin, l0_win), 1000)\n")
    # Throw error
    else:
        print("[deployment_utils.GenerateGM]: Input layer not valid!\n")
        exit()

    # Generate DNN model
    f.write("class DNN(nn.Module):\n")
    f.write("\tdef __init__(self):\n")
    f.write("\t\tsuper().__init__()\n")
    # Create neural network model
    for layer in range(len(layers_l)):
        # Layers
        if layers_l[layer] == "linear":
            f.write(Gtemp.linear_template(layer, in_ch_l[layer], out_ch_l[layer], "False"))
        elif layers_l[layer] == "conv2d":
            f.write(Gtemp.conv2d_template(layer, in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False"))
        elif layers_l[layer] == "DW":
            f.write(Gtemp.DW_template(layer, in_ch_l[layer], hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer], h_pad_l[layer], w_pad_l[layer], "False"))
        elif layers_l[layer] == "PW":
            f.write(Gtemp.PW_template(layer, in_ch_l[layer], out_ch_l[layer], "False"))
        # Activations
        elif layers_l[layer] == "ReLU":
            f.write(Gtemp.ReLU_template(layer))
        # Pooling
        elif layers_l[layer] == "MaxPool":
            f.write(Gtemp.MaxPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer]))
        elif layers_l[layer] == "AvgPool":
            f.write(Gtemp.AvgPool_template(layer, hk_l[layer], wk_l[layer], h_str_l[layer], w_str_l[layer]))
        # Throw error
        else:
            print("[deployment_utils.GenerateGM]: Layer {} not recognized!!\n".format(layer))
            exit()
    # Create Forward
    f.write("\n")
    f.write("\tdef forward(self, x):")
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'linear':
            f.write("\n\t\tx = torch.reshape(x, (-1,))")
        f.write("\n\t\tx = self.l"+str(layer)+"(x)")
    f.write("\n\t\treturn x\n")

    # f.write("\n# All-ones fake label \n")
    last_layer = len(layers_l) - 1
    # if (layers_l[-1] == 'linear' or layers_l[-1] == 'ReLU'):
    #     f.write("label = torch.ones(l"+str(last_layer)+"_out_ch)\n")
    # elif (layers_l[0] == 'conv2d' or layers_l[0] == 'DW' or layers_l[0] == 'PW'):
    #     f.write("label = torch.ones(batch_size, l"+str(last_layer)+"_out_ch, math.floor((l"+str(last_layer)+"_hin-l"+str(last_layer)+"_hk+2*l"+str(last_layer)+"_hpad+l"+str(last_layer)+"_hstr)/l"+str(last_layer)+"_hstr), math.floor((l"+str(last_layer)+"_win-l"+str(last_layer)+"_wk+2*l"+str(last_layer)+"_wpad+l"+str(last_layer)+"_wstr)/l"+str(last_layer)+"_wstr))\n")
    # # Throw error
    # else: 
    #     print("[deployment_utils.GenerateGM]: Output layer not valid\n!")
    #     exit()

    # Initialize network
    f.write("\n# Initialize network\n")
    f.write("net = DNN()\n")
    f.write("for p in net.parameters():\n")
    f.write("\tnn.init.normal_(p, mean=0.0, std=1.0)\n")
    f.write("net.zero_grad()\n\n")

    # Write all-ones sample label
    f.write("\n# All-ones fake label \n")
    f.write("output_test = net(inp)\n")
    f.write("label = torch.ones_like(output_test)\n")

    # Write init weights to header file
    f.write("f = open('io_data.h', 'w')\n")
    f.write("f.write('// Init weights\\n')\n")
    for layer in range(len(layers_l)):
        if (layers_l[layer] != 'ReLU' and layers_l[layer] != 'MaxPool' and layers_l[layer] != 'AvgPool'):
            f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
            f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"] = {'+dump.tensor_to_string(net.l"+str(layer)+".weight.data)+'};\\n')\n")
        else:
            f.write("f.write('#define WGT_SIZE_L"+str(layer)+" '+str(l"+str(layer)+"_in_ch*l"+str(layer)+"_out_ch*l"+str(layer)+"_hk*l"+str(layer)+"_wk)+'\\n')\n")
            f.write("f.write('PI_L2 float init_WGT_l"+str(layer)+"[WGT_SIZE_L"+str(layer)+"];\\n')\n")
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
    f.write("# Train the DNN\n")
    f.write("for batch in range(epochs):\n")
    f.write("\toptimizer.zero_grad()\n")
    f.write("\tout = net(inp)\n")
    f.write("\tloss = loss_fn(out, label)\n")
    f.write("\tloss.backward()\n")
    f.write("\toptimizer.step()\n")
    
    # Inference after training
    f.write("\n# Inference once after training\n")
    f.write("out = net(inp)\n")
    f.write("\n")

    # Dump input and output of the network to the header file for the MCU
    f.write("f = open('io_data.h', 'a')\n")
    f.write("f.write('// Input and Output data\\n')\n")
    f.write("f.write('#define IN_SIZE "+str(in_ch_l[0]*win_l[0]*hin_l[0])+"\\n')\n")
    f.write("f.write('PI_L1 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\\n')\n")
    f.write("out_size = (int(math.floor(l"+str(last_layer)+"_hin-l"+str(last_layer)+"_hk+2*l"+str(last_layer)+"_hpad+l"+str(last_layer)+"_hstr)/l"+str(last_layer)+"_hstr)) * (int(math.floor(l"+str(last_layer)+"_win-l"+str(last_layer)+"_wk+2*l"+str(last_layer)+"_wpad+l"+str(last_layer)+"_wstr)/l"+str(last_layer)+"_wstr)) * l"+str(last_layer)+"_out_ch\n") 
    f.write("f.write('#define OUT_SIZE '+str(out_size)+'\\n')\n")
    f.write("f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\\n')\n")
    f.write("f.write('PI_L1 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\\n')\n")
    f.write("f.close()\n")

    f.close()

    return





# Generate the net.c and net.h files for the execution on PULP
def GenerateNet(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l):

    # Generate net.h
    f = open(proj_folder_path+'net.h', 'w')

    f.write("// PULP Defines\n")
    f.write("#define STACK_SIZE      4096\n")
    #f.write("#define MOUNT           1\n")
    #f.write("#define UNMOUNT         0\n")
    #f.write("#define CID             0\n")
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
    f.write("PI_L1 float loss = 0;\n")

    f.write("\n// Define DNN blobs\n")
    for layer in range(len(layers_l)):
        f.write("PI_L1 struct blob layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, layer"+str(layer)+"_out;\n")

    f.write("\n// Define DNN layer structures\n")
    for layer in range(len(layers_l)):
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
        elif layers_l[layer] == 'MaxPool':
            pass
        elif layers_l[layer] == 'AvgPool':
            pass
        else:
            print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")

    pooling_exist = False
    for layer in range(len(layers_l)):
        if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
            pooling_exist = True
    if pooling_exist:
        f.write("\n// Define Pooling Structures\n")
        for layer in range(len(layers_l)):
            if (layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool'):
                f.write("PI_L1 struct pool_args l"+str(layer)+"_pool_args;\n")


    f.write("\n// Define kernel tensors\n")
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
            f.write("PI_L1 float l"+str(layer)+"_ker[1];\n")
        else:    
            f.write("PI_L1 float l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")

    f.write("\n// Define kernel grad tensors\n")
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
            f.write("PI_L1 float l"+str(layer)+"_ker_diff[1];\n")
        else:    
            f.write("PI_L1 float l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")

    f.write("\n// Define I/O tensors\n")
    for layer in range(len(layers_l)):
        f.write("PI_L1 float l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
        if (layer == len(layers_l)-1):
            f.write("PI_L1 float l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")


    # Write IM2COL buffers
    im2col_flag = False
    im2col_type = 'FW'  # 'FW' or 'BW'
    im2col_max_memocc = 0
    im2col_layer_index = 0
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' or layers_l[layer] == 'DW':
            im2col_flag = True
            i2c_mem = 0
            i2c_FW = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * (hin_l[layer]-hk_l[layer]+1) * (win_l[layer]-wk_l[layer]+1)
            i2c_BW = out_ch_l[layer] * hk_l[layer] * wk_l[layer] * hin_l[layer] * win_l[layer]
            if i2c_FW > i2c_BW:
                i2c_mem = i2c_FW
                im2col_type = 'FW'
            else:
                i2c_mem = i2c_BW
                im2col_type = 'BW'
            #i2c_mem = in_ch_l[layer] * out_ch_l[layer] * hk_l[layer] * wk_l[layer] * (hin_l[layer]-hk_l[layer]+1) * (win_l[layer]-wk_l[layer]+1)
            if i2c_mem > im2col_max_memocc:
                im2col_max_memocc = i2c_mem
                im2col_layer_index = layer
    if im2col_flag == True:
        if im2col_type == 'FW':
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            f.write("PI_L1 float im2col_buffer[Tin_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tout_H_l"+str(im2col_layer_index)+"*Tout_W_l"+str(im2col_layer_index)+"];\n")
        else:
            f.write("\n// Define IM2COL buffer for all the convolutions\n")
            f.write("PI_L1 float im2col_buffer[Tout_C_l"+str(im2col_layer_index)+"*Tker_H_l"+str(im2col_layer_index)+"*Tker_W_l"+str(im2col_layer_index)+"*Tin_H_l"+str(im2col_layer_index)+"*Tin_W_l"+str(im2col_layer_index)+"];\n")

    # Write conv2d in grad blocktranspose buffer
    bt_flag = False
    bt_max_memocc = 0
    bt_layer_index = 0
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' and layer == 0:
            bt_flag = True
            bt_layer_index = 0
        elif layers_l[layer] == 'conv2d' and layer > 0:
            bt_flag = True
            bt_mem = in_ch_l[layer] * hk_l[layer] * wk_l[layer] * out_ch_l[layer]
            if bt_mem > bt_max_memocc:
                bt_max_memocc = bt_mem
                bt_layer_index = layer
    if bt_flag == True:
        f.write("\n// Define block transposition buffer for all conv2d layers\n")
        if bt_layer_index == 0:
            f.write("PI_L1 float bt_buffer[1];")
        elif bt_layer_index > 0:
            f.write("PI_L1 float bt_buffer[Tin_C_l"+str(bt_layer_index)+"*Tout_C_l"+str(bt_layer_index)+"*Tker_H_l"+str(bt_layer_index)+"*Tker_W_l"+str(bt_layer_index)+"];\n")


    f.write("\n// Define error propagation tensors\n")
    for layer in range(len(layers_l)):
        if layer > 0:
            f.write("PI_L1 float l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
        if (layer == len(layers_l)-1):
            f.write("PI_L1 float l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")


    f.write("\n// Loss function configuration structure\n")
    f.write("PI_L1 struct loss_args loss_args;\n")



    f.write("\n\n\n/**\n * DNN BACKEND FUNCTIONS\n**/\n")

    f.write("\n// DNN initialization function\n")
    f.write("void DNN_init()\n{\n")
    for layer in range(len(layers_l)):
        if layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)\t\t\tl0_in[i] = INPUT[i];\n")
            f.write("  for(int i=0; i<Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0; i++)\t\tl0_ker[i] = init_WGT_l0[i];\n")
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            if layers_l[layer] == 'DW':
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
            elif layers_l[layer] == 'AvgPool' or layers_l[layer] == 'MaxPool':
                f.write("  //   Pooling kernel (no parameters)\n")
            else:
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        elif layer == len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

    f.write("\n  // Connect tensors to blobs\n")
    for layer in range(len(layers_l)):
        if len(layers_l) == 1:
            f.write("  layer"+str(layer)+"_in.data = l0_in;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l0*Tin_H_l0*Tin_W_l0;\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l0;\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.data = l0_ker;\n")
            f.write("  layer"+str(layer)+"_wgt.diff = l0_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tker_H_l0*Tker_W_l0;\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.data = l0_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l0_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l0;\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l0;\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l0;\n")
        elif layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
            f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tin_C_l"+str(layer+1)+"*Tin_H_l"+str(layer+1)+"*Tin_W_l"+str(layer+1)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.diff = l"+str(layer)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
            f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tin_C_l"+str(layer+1)+"*Tin_H_l"+str(layer+1)+"*Tin_W_l"+str(layer+1)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        elif layer == len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.diff = l"+str(layer)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
            f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
            if layers_l[layer] == 'DW':
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer)+"_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

    f.write("\n  // Configure layer structures\n")
    for layer in range(len(layers_l)):
        f.write("  // Layer "+str(layer)+"\n")
        if layer == 0:
            skip_inputgrad = 1
        else: 
            skip_inputgrad = 0
        # Write configuration templates
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_config_template(layer, skip_inputgrad))
        elif layers_l[layer] == 'conv2d':
            f.write(ntemp.conv2d_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_config_template(layer, skip_inputgrad))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_config_template(layer))
        elif layers_l[layer] == 'MaxPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'AvgPool':
            f.write("  //   Pooling layer (see next section)\n")
        else:
            print("[deployment_utils.GenerateNet] Undefined layer "+str(layer)+" (unable to write configuration structure)!!")


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
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_template_FW(layer))
        elif layers_l[layer] == 'conv2d':
            f.write(ntemp.conv2d_template_FW(layer))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_template_FW(layer))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_template_FW(layer))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_template_FW(layer))
        elif layers_l[layer] == 'AvgPool':
            f.write(ntemp.AvgPool_template_FW(layer))
        elif layers_l[layer] == 'MaxPool':
            f.write(ntemp.MaxPool_template_FW(layer))
        else:
            print("[deployment_utils.GenerateNet]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
    f.write("}\n")


    f.write("\n// Backward pass function\n")
    f.write("void backward()\n{\n")
    for layer in range(len(layers_l)):
        lay = len(layers_l) - layer - 1
        skip_in_grad = 0
        if lay == 0:
            skip_in_grad = 1
        if layers_l[lay] == 'linear':
            f.write(ntemp.linear_template_BW(lay))
        elif layers_l[lay] == 'conv2d':
            f.write(ntemp.conv2d_template_BW(lay))
        elif layers_l[lay] == 'DW':
            f.write(ntemp.DW_template_BW(lay))
        elif layers_l[lay] == 'PW':
            f.write(ntemp.PW_template_BW(lay))
        elif layers_l[lay] == 'ReLU':
            f.write(ntemp.ReLU_template_BW(lay))
        elif layers_l[lay] == 'AvgPool':
            f.write(ntemp.AvgPool_template_BW(lay))
        elif layers_l[lay] == 'MaxPool':
            f.write(ntemp.MaxPool_template_BW(lay))
        else:
            print("[deployment_utils.GenerateNet]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()
    f.write("}\n")


    f.write("\n// Compute loss and output gradient\n")
    f.write("void compute_loss()\n{\n")

    if loss_fn == "MSELoss":
        f.write("  loss_args.output = &layer"+str(len(layers_l)-1)+"_out;\n")
        f.write("  loss_args.target = LABEL;\n")
        f.write("  loss_args.wr_loss = &loss;\n")
        f.write("  pulp_MSELoss(&loss_args);\n")
        #f.write("  pulp_MSELoss(&layer"+str(len(layers_l)-1)+"_out, LABEL, &loss);\n")
    else:
        print("[deployment_utils.GenerateNet]: Loss function not valid for PULP deployment!!")
        exit()

    f.write("}\n")


    f.write("\n// Function to update the network\n")
    f.write("void update_weights()\n{\n")

    for layer in range(len(layers_l)):
        if layers_l[layer] == 'linear' or layers_l[layer] == 'conv2d' or layers_l[layer] == 'DW' or layers_l[layer] == 'PW':
            f.write("  struct optim_args opt_l"+str(layer)+";\n")
            f.write("  opt_l"+str(layer)+".weights = &layer"+str(layer)+"_wgt;\n")
            f.write("  opt_l"+str(layer)+".learning_rate = LEARNING_RATE;\n")
            if optimizer == "SGD":
                f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l"+str(layer)+");\n")
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
    f.write("}\n")


    f.write("\n// Function to check post-training output wrt Golden Model (GM)\n")
    f.write("void check_post_training_output()\n{\n")

    output_index = len(layers_l) - 1
    f.write("  int integrity_check = 0;\n")
    f.write("  integrity_check = verify_tensor(l"+str(output_index)+"_out, REFERENCE_OUTPUT, Tout_C_l"+str(output_index)+"*Tout_H_l"+str(output_index)+"*Tout_W_l"+str(output_index)+", TOLERANCE);\n")
    f.write("  if (integrity_check > 0)\n")
    f.write("    printf(\"\\n*** UPDATED OUTPUT NOT MATCHING GOLDEN MODEL ***\\n\");\n")

    f.write("}\n")



    f.write("\n\n\n/**\n * DNN MODEL TRAINING\n**/\n")

    f.write("\n// Call for a complete training step\n")
    f.write("void net_step()\n{\n")

    f.write("  printf(\"Initializing network..\\n\");\n")
    f.write("  DNN_init();\n")

    f.write("  printf(\"Testing DNN initialization forward..\");\n")
    f.write("  forward();\n")
    f.write("  print_output();\n\n")

    f.write("  #ifdef PROF_NET\n")
    f.write("  INIT_STATS();\n  PRE_START_STATS();\n  START_STATS();\n")
    f.write("  #endif\n\n")

    f.write("  for (int epoch=0; epoch<EPOCHS; epoch++)\n  {\n")
    f.write("    forward();\n")
    f.write("    compute_loss();\n")
    f.write("    backward();\n")
    f.write("    update_weights();\n")
    f.write("  }\n\n")

    f.write("  #ifdef PROF_NET\n")
    f.write("  STOP_STATS();\n")
    f.write("  #endif\n\n")

    f.write("  // Check and print updated output\n")
    f.write("  forward();\n")
    f.write("  printf(\"Checking updated output..\\n\");\n")
    f.write("  check_post_training_output();\n")
    f.write("  print_output();\n")

    f.write("}\n")



    f.close()

    return

