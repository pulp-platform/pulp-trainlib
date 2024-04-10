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
Authors: Giacomo Saporetti, Davide Nadalini
'''

import os
import shutil
import math

from torch import mm
import deployer_utils.net_templates_double_buffer as ntemp


"""
DNN Size Checker backend functions
"""

def max_layer_dim (layers_l, cin_l, hin_l, win_l, cout_l, hk_l, wk_l, data, h_str, w_str, h_pad, w_pad, bias):
    RES = 0
    temp1 = 0  # input
    temp2 = 0  # wgt
    temp3 = 0  # bias
    tot = 0
    max_layer =  0
    for layer in range(len(layers_l)):
        if layers_l[layer] == 'conv2d' : 
            temp2 = hk_l[layer]*wk_l[layer]*cin_l[layer]*cout_l[layer]
            if bias[layer] == 1:
                temp3 = cout_l[layer]
        if layers_l[layer] == 'PW':
            temp2 = cin_l[layer]*cout_l[layer]
        if   layers_l[layer] == 'DW':
            temp2 = hk_l[layer]*wk_l[layer]*cin_l[layer]
        if layers_l[layer] == 'linear' :
            temp2 = cin_l[layer]*cout_l[layer]
        if layers_l[layer] == 'Sumnode':
            temp2 = cin_l[layer]*hin_l[layer]*win_l[layer]
        if layers_l[layer] == 'InstNorm':
            temp2 = 2*cin_l[layer]
        if layers_l[layer] in ['ReLU', 'Skipnode']:
            temp2 = 0

        temp1 = cin_l[layer]*hin_l[layer]*win_l[layer]

        tot = temp1 + temp2 + temp3
        print(f"Layer {layer} ({layers_l[layer]}):  Input: {temp1}, Coefficients: {temp2}, Biases: {temp3}, Total: {tot}")
        if tot > RES:
            RES = tot
            max_layer = layer

    RES = 2 * 2 * RES  #Considering 2 Buffers for both data and diff
    multiplier = 2
    if data == 'FP32':
        multiplier = 4
    print(f"Max Layer size (considering only input data, coefficients, and biases): {multiplier * RES} bytes @layer {max_layer}")
    return multiplier*RES


"""
DNN Composer backend functions
"""


# Generate the net.c and net.h files for the execution on PULP
def GenerateNet(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l, bias_l, update_layer_l, sumnode_connections, MAX_LAYER_DIM,
                PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS):


    data_type = data_type_l[0]
    data_size = 0
    suffix = ""
    if data_type == "FP32":
        data_size = 4
        suffix = ""
    else:
        data_size = 2
        suffix = "_fp16"

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

    f.write("\n// DMA managment functions\n")
    f.write("void load_input(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void load_output(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void load_coeff(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void load_bias(void * src_blob, uint8_t data_diff_both);\n")
    f.write("void swap_in_out();\n")
    f.write("void store_output(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void store_input(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void store_coeff(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void store_bias(void * dest_blob, uint8_t data_diff_both);\n")
    f.write("void copy_struct_param(unsigned int from, unsigned int to, int size);\n")
    f.write("void get_input_dim(void * b);\n")
    f.write("void get_output_dim(void * b);\n")
    f.write("void get_weight_dim(void * b);\n")
    f.write("void get_bias_dim(void * b);\n")
    f.write("void reset_arguments();\n")
    f.write("void update_blob();\n")
    f.write("void reset_dim();\n")
    f.write("void dma_handler(uint8_t do_store, uint8_t do_load, void * src_store, void * dst_store, void * src_load, void * dst_load);\n")
    f.write("void load(uint32_t src, uint32_t dst, int dim);\n")
    f.write("void store(uint32_t src, uint32_t dst, int dim);\n")
    f.write("void update();\n")
    f.write("void get_dim(void * src_blob, void * dst_blob);\n")
    f.write(f"#define MAX_SIZE {MAX_LAYER_DIM}\n")
    f.close()    


    # Generate net.c
    f = open(proj_folder_path+'net.c', 'w')

    f.write("/**\n * INCLUDES\n**/\n\n")

    f.write("#include \"pulp_train.h\"\n")
    f.write("#include \"net.h\"\n")
    f.write("#include \"stats.h\"\n\n")
    f.write("#include \"init-defines.h\"\n")
    f.write("#include \"io_data.h\"\n")


    f.write("\n// Define structures and pointers to data in L1 memory\n")
    if data_type == 'FP32':
        #f.write("PI_L1 float * D0, * d0, * W0, * w0, * D1, * d1, * W1, *w1;\n")
        f.write("PI_L1 float BUFF[MAX_SIZE];\n")
        f.write("PI_L1 struct blob d1_blob;\n")
        f.write("PI_L1 struct blob w1_blob;\n")
        f.write("PI_L1 struct blob b1_blob;\n")
        f.write("PI_L1 struct blob d0_blob;\n")
        f.write("PI_L1 struct blob w0_blob;\n")
        f.write("PI_L1 struct blob b0_blob;\n")
        f.write("PI_L1 struct blob in;\n")
        f.write("PI_L1 struct blob wgt;\n")
        f.write("PI_L1 struct blob bias;\n")
        f.write("PI_L1 struct blob out;\n")
        f.write("PI_L1 struct Linear_args linear_args;\n")
        f.write("PI_L1 struct Conv2D_args conv2d_args;\n")
        f.write("PI_L1 struct PointWise_Conv_args PW_args;\n")
        f.write("PI_L1 struct DepthWise_Conv_args DW_args;\n")
        f.write("PI_L1 struct act_args act_args;\n")
        f.write("PI_L1 struct InstNorm_args InstNorm_args;\n")
        f.write("PI_L1 struct SkipConn_args resconn_args;\n")
        f.write("PI_L1 struct pool_args MaxPool_args;\n")
        f.write("PI_L1 struct pool_args AvgPool_args;\n")
        #f.write("PI_L1 float * t;\n")
    elif data_type == 'FP16':
        f.write("PI_L1 fp16 * D1, * d1, * W1, * w1, * D0, * d0, * W0, *w0;\n")
        f.write("PI_L1 fp16 BUFF[MAX_SIZE];\n")
        f.write("PI_L1 struct blob_fp16 d1_blob;\n")
        f.write("PI_L1 struct blob_fp16 w1_blob;\n")
        f.write("PI_L1 struct blob_fp16 d0_blob;\n")
        f.write("PI_L1 struct blob_fp16 w0_blob;\n")
        f.write("PI_L1 struct blob_fp16 in;\n")
        f.write("PI_L1 struct blob_fp16 wgt;\n")
        f.write("PI_L1 struct blob_fp16 out;\n")
        f.write("PI_L1 struct Linear_args_fp16 linear_args;\n")
        f.write("PI_L1 struct Conv2D_args_fp16 conv2d_args;\n")
        f.write("PI_L1 struct PointWise_Conv_args_fp16 PW_args;\n")
        f.write("PI_L1 struct DepthWise_Conv_args_fp16 DW_args;\n")
        f.write("PI_L1 struct act_args_fp16 act_args;\n")
        f.write("PI_L1 struct InstNorm_args_fp16 InstNorm_args;\n")
        f.write("PI_L1 struct SkipConn_args_fp16 resconn_args;\n")
        f.write("PI_L1 struct pool_args_fp16 MaxPool_args;\n")
        f.write("PI_L1 struct pool_args_fp16 AvgPool_args;\n")
        #f.write("PI_L1 fp16 * t;\n")
    else:
        print("[deployment_utils.GenerateNet] Invalid last layer data type!")
        exit()
    
    f.write("PI_L1 pi_cl_dma_cmd_t * cmd_store;\n")
    f.write("PI_L1 pi_cl_dma_cmd_t * cmd_load;\n")
    f.write("PI_L1 pi_cl_dma_cmd_t * cmd_struct;\n")


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
        if data_type_l[layer] == 'FP32':
            f.write("PI_L2 struct blob layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, layer"+str(layer)+"_out;\n")
        elif data_type_l[layer] == 'FP16':
            f.write("PI_L2 struct blob_fp16 layer"+str(layer)+"_in, layer"+str(layer)+"_wgt, layer"+str(layer)+"_out;\n")
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for blob definition @Layer{}!".format(layer))
            exit()


    f.write("\n// Define DNN layer structures\n")
    if data_type_l[0] == 'FP32':
        f.write("PI_L1 struct vect_sum_args vect_sum_args;\n")
    else:
        f.write("PI_L1 struct vect_sum_args_fp16 vect_sum_args;\n")
    for layer in range(len(layers_l)):
        # Define FP32 structure
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'linear':
                f.write("PI_L2 struct Linear_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L2 struct Conv2D_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L2 struct PointWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L2 struct DepthWise_Conv_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L2 struct act_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass 
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L2 struct SkipConn_args l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'InstNorm':
                f.write(f"PI_L2 struct InstNorm_args l{layer}_args;\n")
            else:
                print("[deployment_utils.GenerateNet] Layer "+str(layer)+" not recognized!!")
        # Define FP16 structure
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'linear':
                f.write("PI_L2 struct Linear_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'conv2d':
                f.write("PI_L2 struct Conv2D_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'PW':
                f.write("PI_L2 struct PointWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'DW':
                f.write("PI_L2 struct DepthWise_Conv_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'ReLU':
                f.write("PI_L2 struct act_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'MaxPool':
                pass
            elif layers_l[layer] == 'AvgPool':
                pass
            elif layers_l[layer] == 'Skipnode': 
                pass
            elif layers_l[layer] == 'Sumnode':
                f.write("PI_L2 struct SkipConn_args_fp16 l"+str(layer)+"_args;\n")
            elif layers_l[layer] == 'InstNorm':
                f.write(f"PI_L2 struct InstNorm_args_fp16 l{layer}_args;\n")
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
                    f.write("PI_L2 struct pool_args l"+str(layer)+"_args;\n")
                elif data_type_l[layer] == 'FP16':
                    f.write("PI_L2 struct pool_args_fp16 l"+str(layer)+"_args;\n")
                else:
                    print("[deployment_utils.GenerateNet] Invalid data type for pooling initialization @Layer{}!".format(layer))
                    exit()


    f.write("\n// Define kernel tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 float l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L2 float l"+str(layer)+f"_ker[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L2 float l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                if bias_l[layer] == 1:
                    f.write("PI_L2 float l"+str(layer)+"_bias[Tout_C_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 fp16 l"+str(layer)+"_ker[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode': 
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L2 fp16 l"+str(layer)+f"_ker[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L2 fp16 l"+str(layer)+"_ker[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for kernel definition @Layer{}!".format(layer))
            exit()

    f.write("\n// Define kernel grad tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 float l"+str(layer)+"_ker_diff[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L2 float l"+str(layer)+f"_ker_diff[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L2 float l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
                if bias_l[layer] == 1:
                    f.write("PI_L2 float l"+str(layer)+"_bias_diff[Tout_C_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layers_l[layer] == 'MaxPool' or layers_l[layer] == 'AvgPool':
                f.write("PI_L2 fp16 l"+str(layer)+"_ker_diff[1];\n")
            elif layers_l[layer] == 'Skipnode' or layers_l[layer] == 'Sumnode':
                pass
            elif layers_l[layer] == 'InstNorm':
                f.write("PI_L2 fp16 l"+str(layer)+f"_ker_diff[2*Tin_C_l{layer}];\n")
            else:    
                f.write("PI_L2 fp16 l"+str(layer)+"_ker_diff[Tin_C_l"+str(layer)+" * Tout_C_l"+str(layer)+" * Tker_H_l"+str(layer)+" * Tker_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for kernel grad definition @Layer{}!".format(layer))
            exit()

    f.write("\n// Define I/O tensors\n")

    previous_was_skip = False 
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if not previous_was_skip: # If the previous layer was a Skipnode, then do not generate layer in and diff
            if data_type_l[layer] == 'FP32':
                f.write("PI_L2 float l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L2 float l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Define FP16 tensors
            elif data_type_l[layer] == 'FP16':
                f.write("PI_L2 fp16 l"+str(layer)+"_in[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
                if (layer == len(layers_l)-1):
                    f.write("PI_L2 fp16 l"+str(layer)+"_out[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
            # Data type error
            else:
                print("[deployment_utils.GenerateNet] Invalid data type for I/O definition @Layer{}!".format(layer))
                exit()

        if layers_l[layer] == 'Skipnode':
            previous_was_skip = True
        else:
            previous_was_skip = False
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
        if ((layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True) or layers_l[layer] == 'PW') and layer == 0:
            bt_flag = True
            bt_layer_index = 0
        elif ((layers_l[layer] == 'conv2d' and CONV2D_USE_IM2COL == True) or layers_l[layer] == 'PW') and layer > 0:
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

    # Define label buffer
    f.write("PI_L1 float label_temp[Tout_C_l"+str(len(layers_l)-1)+"*Tout_H_l"+str(len(layers_l)-1)+"*Tout_W_l"+str(len(layers_l)-1)+"];\n")

    # Define tensors to backpropagate the output error
    f.write("\n// Define error propagation tensors\n")
    for layer in range(len(layers_l)):
        # Define FP32 tensors
        if data_type_l[layer] == 'FP32':
            if layer > 0:
                f.write("PI_L2 float l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
            if (layer == len(layers_l)-1):
                f.write("PI_L2 float l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
        # Define FP16 tensors
        elif data_type_l[layer] == 'FP16':
            if layer > 0:
                f.write("PI_L2 fp16 l"+str(layer)+"_in_diff[Tin_C_l"+str(layer)+" * Tin_H_l"+str(layer)+" * Tin_W_l"+str(layer)+"];\n")
            if (layer == len(layers_l)-1):
                f.write("PI_L2 fp16 l"+str(layer)+"_out_diff[Tout_C_l"+str(layer)+" * Tout_H_l"+str(layer)+" * Tout_W_l"+str(layer)+"];\n")
        # Data type error
        else:
            print("[deployment_utils.GenerateNet] Invalid data type for input grad definition @Layer{}!".format(layer))
            exit()  
        
      

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
    f.write("\n// Assign pointers in L1\n")
    f.write("  d0_blob.data = BUFF;\n")
    f.write("  d0_blob.diff = BUFF;\n")
    f.write("  w0_blob.data = BUFF;\n")
    f.write("  w0_blob.diff = BUFF;\n")
    f.write("  bias0_blob.data = BUFF;\n")
    f.write("  bias0_blob.diff = BUFF;\n")
    f.write("  d1_blob.data = BUFF + MAX_SIZE/2;\n")
    f.write("  d1_blob.diff = BUFF + MAX_SIZE/2;\n")
    f.write("  w1_blob.data = BUFF + MAX_SIZE/2;\n")
    f.write("  w1_blob.diff = BUFF + MAX_SIZE/2;\n")
    f.write("  bias1_blob.data = BUFF + MAX_SIZE/2;\n")
    f.write("  bias1_blob.diff = BUFF + MAX_SIZE/2;\n")
    f.write("  reset_arguments();\n\n")
    for layer in range(len(layers_l)):
        if layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  for(int i=0; i<Tin_C_l0*Tin_H_l0*Tin_W_l0; i++)\t\t\tl0_in[i] = INPUT[i];\n")
            if layers_l[layer] not in ['Skipnode', 'Sumnode', 'InstNorm']:
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
            elif layers_l[layer] == 'InstNorm':
                f.write("  for(int i=0; i<2*Tin_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
            else:
                f.write("  for(int i=0; i<Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_ker[i] = init_WGT_l"+str(layer)+"[i];\n")
                if bias_l[layer] == 1:
                    f.write("  for(int i=0; i<Tout_C_l"+str(layer)+"; i++)\t\tl"+str(layer)+"_bias[i] = init_BIAS_l"+str(layer)+"[i];\n")
        elif layer == len(layers_l)-1:
            if layers_l[layer] not in  ['Skipnode', 'Sumnode', 'InstNorm']:
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
    C_data_type = 'float'
    f.write("\n// Connect tensors to blobs\n")
    previous_was_skip = 0
    
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
        # Verify how many Skipnodes comes after current layer (for diff connections)
        lookahead = 0
        if (layer + 1) <  len(layers_l):
            for l in range(len(layers_l) - layer - 1):
                if sumnode_connections[layer + l + 1] < 0 or layers_l[layer + l + 1] == 'Sumnode':
                    break
                else:
                    lookahead += 1
        # DNN is 1 layer long
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
            elif layers_l[layer] == 'InstNorm':
                f.write("  layer"+str(layer)+"_wgt.dim = 2*Tin_C_l0;\n")
            else:
                f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l0*Tout_C_l0*Tker_H_l0*Tker_W_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l0;\n")
            f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l0;\n")
            if bias_l[layer] == 1:
                f.write("  layer" + str(layer) + "_bias.data = l0_bias;\n")
                f.write("  layer" + str(layer) + "_bias.diff = l0_bias_diff;\n")
                f.write("  layer" + str(layer) + "_bias.dim = Tout_C_l0;\n")
            f.write("  layer"+str(layer)+"_out.data = l0_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l0_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l0*Tout_H_l0*Tout_W_l0;\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l0;\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l0;\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l0;\n")
        # First layer connection
        elif layer == 0:
            f.write("  // Layer "+str(layer)+"\n")
            if layers_l[0] != 'Skipnode': # Avoid weight assignment for Skip Connections
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                elif layers_l[layer] == 'InstNorm':
                    f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                if bias_l[layer] == 1:
                    f.write("  layer" + str(layer) + "_bias.data = l" + str(layer) + "_bias;\n")
                    f.write("  layer" + str(layer) + "_bias.diff = l" + str(layer) + "_bias_diff;\n")
                    f.write("  layer" + str(layer) + "_bias.dim = Tout_C_l" + str(layer) + ";\n")
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if sumnode_connections[layer] < 0 or layers_l[layer] == 'Sumnode':
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(layer + 1 + lookahead)+"_in_diff;\n")   
                    else:
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")
                # End of assignment       
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
            else:
                f.write("  layer"+str(layer)+"_in.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_in;\n")
                f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")             
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        # Hidden layers
        elif layer > 0 and layer < len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip)+"_in;\n")
            if layers_l[layer] != 'Skipnode':
                if (layer - previous_was_skip) > 0: # Avoid assignement of l0_in_diff
                    f.write("  layer"+str(layer)+"_in.diff = l"+str(layer)+"_in_diff;\n")
            else:
                f.write(f"\tlayer{layer}_in.diff = l{sumnode_connections[layer]}_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            if layers_l[layer] != 'Skipnode':   # Avoid weight assignment for Skipnodes and out data assignement
                if layers_l[layer]  != 'Sumnode':    # Different weight assignement for Sumnodes
                    f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                    if layers_l[layer] == 'DW':
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    elif layers_l[layer] == 'InstNorm':
                        f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                    else:
                        f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                    if bias_l[layer] == 1:
                        f.write("  layer" + str(layer) + "_bias.data = l" + str(layer) + "_bias;\n")
                        f.write("  layer" + str(layer) + "_bias.diff = l" + str(layer) + "_bias_diff;\n")
                        f.write("  layer" + str(layer) + "_bias.dim = Tout_C_l" + str(layer) + ";\n")
                    f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                    f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.data = layer"+str(sumnode_connections[layer])+"_out.data;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = layer"+str(sumnode_connections[layer])+"_out.diff;\n")
                    f.write("  layer"+str(layer)+"_wgt.C = layer"+str(sumnode_connections[layer])+"_out.C;\n")
                    f.write("  layer"+str(layer)+"_wgt.H = layer"+str(sumnode_connections[layer])+"_out.H;\n")
                    f.write("  layer"+str(layer)+"_wgt.W = layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    f.write("  layer"+str(layer)+"_wgt.dim = layer"+str(sumnode_connections[layer])+"_out.C*layer"+str(sumnode_connections[layer])+"_out.H*layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    if bias_l[layer] == 1:
                        f.write("  layer" + str(layer) + "_bias.data = layer" + str(sumnode_connections[layer]) + "_bias.data;\n")
                        f.write("  layer" + str(layer) + "_bias.diff = layer" + str(sumnode_connections[layer]) + "_bias.diff;\n")
                        f.write("  layer" + str(layer) + "_bias.dim  = layer" + str(sumnode_connections[layer]) + "_bias.dim;\n")
                # Assign to cast_buffer in case data type changes
                if data_type_l[layer] != data_type_l[layer+1]:
                    f.write("  layer"+str(layer)+"_out.data = ("+C_data_type+"*) cast_buffer;\n")
                    f.write("  layer"+str(layer)+"_out.diff = ("+C_data_type+"*) cast_buffer;\n")
                else:
                    f.write("  layer"+str(layer)+"_out.data = l"+str(layer+1)+"_in;\n")
                    if sumnode_connections[layer] == -1 or layers_l[layer] == 'Sumnode':
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(layer+1+lookahead)+"_in_diff;\n")
                    else:     
                        f.write("  layer"+str(layer)+"_out.diff = l"+str(sumnode_connections[layer])+"_in_diff;\n")
                # End of assignment     
                f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
            else:
                f.write(f"\tlayer{layer}_out = layer{layer}_in;\n")
        # Last layer
        elif layer == len(layers_l)-1:
            f.write("  // Layer "+str(layer)+"\n")
            f.write("  layer"+str(layer)+"_in.data = l"+str(layer - previous_was_skip)+"_in;\n")
            f.write("  layer"+str(layer)+"_in.diff = l"+str(layer + lookahead)+"_in_diff;\n")
            f.write("  layer"+str(layer)+"_in.dim = Tin_C_l"+str(layer)+"*Tin_H_l"+str(layer)+"*Tin_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.C = Tin_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.H = Tin_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_in.W = Tin_W_l"+str(layer)+";\n")
            if layers_l[layer] !=  'Sumnode':
                f.write("  layer"+str(layer)+"_wgt.data = l"+str(layer)+"_ker;\n")
                f.write("  layer"+str(layer)+"_wgt.diff = l"+str(layer)+"_ker_diff;\n")
                if layers_l[layer] == 'DW':
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                elif layers_l[layer] == 'InstNorm':
                    f.write("  layer"+str(layer)+f"_wgt.dim = 2*Tin_C_l{layer};\n")
                else:
                    f.write("  layer"+str(layer)+"_wgt.dim = Tin_C_l"+str(layer)+"*Tout_C_l"+str(layer)+"*Tker_H_l"+str(layer)+"*Tker_W_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.C = Tin_C_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.H = Tker_H_l"+str(layer)+";\n")
                f.write("  layer"+str(layer)+"_wgt.W = Tker_W_l"+str(layer)+";\n")
                if bias_l[layer] == 1:
                    f.write("  layer" + str(layer) + "_bias.data = l" + str(layer) + "_bias;\n")
                    f.write("  layer" + str(layer) + "_bias.diff = l" + str(layer) + "_bias_diff;\n")
                    f.write("  layer" + str(layer) + "_bias.dim = Tout_C_l" + str(layer) + ";\n")
            else:
                    f.write("  layer"+str(layer)+"_wgt.data = layer"+str(sumnode_connections[layer])+"_out.data;\n")
                    f.write("  layer"+str(layer)+"_wgt.diff = layer"+str(sumnode_connections[layer])+"_out.diff;\n")
                    f.write("  layer"+str(layer)+"_wgt.C = layer"+str(sumnode_connections[layer])+"_out.C;\n")
                    f.write("  layer"+str(layer)+"_wgt.H = layer"+str(sumnode_connections[layer])+"_out.H;\n")
                    f.write("  layer"+str(layer)+"_wgt.W = layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    f.write("  layer"+str(layer)+"_wgt.dim = layer"+str(sumnode_connections[layer])+"_out.C*layer"+str(sumnode_connections[layer])+"_out.H*layer"+str(sumnode_connections[layer])+"_out.W;\n")
                    if bias_l[layer] == 1:
                        f.write("  layer" + str(layer) + "_bias.data = layer" + str(sumnode_connections[layer]) + "_bias.data;\n")
                        f.write("  layer" + str(layer) + "_bias.diff = layer" + str(sumnode_connections[layer]) + "_bias.diff;\n")
                        f.write("  layer" + str(layer) + "_bias.dim  = layer" + str(sumnode_connections[layer]) + "_bias.dim;\n")
            f.write("  layer"+str(layer)+"_out.data = l"+str(layer)+"_out;\n")
            f.write("  layer"+str(layer)+"_out.diff = l"+str(layer)+"_out_diff;\n")
            f.write("  layer"+str(layer)+"_out.dim = Tout_C_l"+str(layer)+"*Tout_H_l"+str(layer)+"*Tout_W_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.C = Tout_C_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.H = Tout_H_l"+str(layer)+";\n")
            f.write("  layer"+str(layer)+"_out.W = Tout_W_l"+str(layer)+";\n")
        else:
            print("[deployment_utils.GenerateNet]: Error in PULP layer initialization!")
            exit()

        if sumnode_connections[layer] != -1 and layers_l[layer] != 'Sumnode':
            previous_was_skip += 1
           
        else:
            previous_was_skip = 0
            

    f.write("\n  // Configure layer structures\n")
    first_is_skip = False # Avoid calculation of gradient if the first Layer is a skipnode
    if sumnode_connections[0] != -1:
        first_is_skip = True
    previous_was_skip = 0
    for layer in range(len(layers_l)):
        f.write("  // Layer "+str(layer)+"\n")
        if layer == 0:
            skip_inputgrad = 1
        elif layer - previous_was_skip <= 0: # If the 0 layer is a Skipnode, then layer1's diff is the input gradient
            skip_inputgrad = 1
        else: 
            skip_inputgrad = 0
        # Write configuration templates
        if layers_l[layer] == 'linear':
            f.write(ntemp.linear_config_template(layer, skip_inputgrad, data_type_l[layer], 1))
        elif layers_l[layer] == 'conv2d':
            IM2COL_USEIT = 1
            if CONV2D_USE_IM2COL == False:
                IM2COL_USEIT = 0
            f.write(ntemp.conv2d_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer], bias_l[layer], IM2COL_USEIT, 1))
        elif layers_l[layer] == 'PW':
            f.write(ntemp.PW_config_template(layer, skip_inputgrad, data_type_l[layer], 1))
        elif layers_l[layer] == 'DW':
            f.write(ntemp.DW_config_template(layer, h_pad_l[layer], w_pad_l[layer], h_str_l[layer], w_str_l[layer], skip_inputgrad, data_type_l[layer], 1))
        elif layers_l[layer] == 'ReLU':
            f.write(ntemp.ReLU_config_template(layer, data_type_l[layer]))
        elif layers_l[layer] == 'MaxPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'AvgPool':
            f.write("  //   Pooling layer (see next section)\n")
        elif layers_l[layer] == 'Sumnode':
            f.write(ntemp.resconn_config_template(layer, sumnode_connections[layer], first_is_skip))
            first_is_skip = False
        elif layers_l[layer] == 'Skipnode':
            pass
        elif layers_l[layer] == 'InstNorm':
            f.write(ntemp.InstNorm_config_template(layer, skip_inputgrad))
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
                f.write("  l"+str(layer)+"_args.input = &layer"+str(layer)+"_in;\n")
                f.write("  l"+str(layer)+"_args.output = &layer"+str(layer)+"_out;\n")
                f.write("  l"+str(layer)+"_args.Hker = Tker_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_args.Wker = Tker_W_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_args.Hstride = Tstr_H_l"+str(layer)+";\n")
                f.write("  l"+str(layer)+"_args.Wstride = Tstr_W_l"+str(layer)+";\n")
    f.write("}\n\n")



    f.write("\n// Forward pass function\n")
    f.write("void forward(){\n")

    # Profiling options: single layer or all
    if PROFILE_SINGLE_LAYERS == True:
        f.write("  printf(\"\\nFORWARD PROFILING:\\n\\n\");\n")

    f.write("\tpi_cl_dma_flush();\n")
    f.write("\treset_arguments();\n")
    current_buffer = 0
    output_buffer = 1
    next_buffer = 1
    previous_out = 0
    is_skipder = False
    do_store = True
    bytes_per_data = 2
    if data_type_l[0] == 'FP32':
        bytes_per_data = 4       

    # First Layer load of data and weight 
    f.write("\n\tget_dim(&layer0_in, &d0_blob);\n")
    f.write(f"\tload((uint32_t) layer0_in.data, (uint32_t) d0_blob.data, {bytes_per_data}*layer0_in.dim);\n")
    f.write("\tget_dim(&layer0_wgt, &w0_blob);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tload((uint32_t) layer0_wgt.data, (uint32_t) w0_blob.data, {bytes_per_data}*layer0_wgt.dim);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")

    for layer in range(len(layers_l)):

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  printf(\"\\nLayer "+str(layer)+"\\n\");\n")
            f.write("  #ifdef PROF_NET\n")
            f.write("  START_STATS();\n")
            f.write("  #endif\n")  

        is_skipder = True
        if sumnode_connections[layer] < 0 or layers_l[layer] == 'Sumnode':
            is_skipder = False 

        output_buffer = 1
        if current_buffer == 1:
            output_buffer = 0

        if is_skipder:
            next_buffer = current_buffer
        else:
            next_buffer = 1
            if current_buffer == 1:
                next_buffer = 0

        f.write("\n")
        f.write(f"\t// Layer {layer} ({layers_l[layer]}, {current_buffer}, {next_buffer}, {previous_out})\n")

        bytes_per_data = 2
        if data_type_l[layer] == 'FP32':
            bytes_per_data = 4        

        # Get layers informations 
        if not (layers_l[layer] in ['Skipnode', 'ReLU', 'Sumnode']):
            f.write(f"\tcopy_struct_param((uint32_t) &l{layer}_args, (uint32_t) &{layers_l[layer]}_args, sizeof({layers_l[layer]}_args));\n")
  
        # Get data dimensions to adjust array pointers     
        f.write(f"\tget_dim(&layer{layer}_out, &d{output_buffer}_blob);\n")
        if layer < len(layers_l) - 1:
            if (not is_skipder) or (layers_l[layer] == 'Skipnode'):
                if not (layers_l[layer + 1] in ['Skipnode', 'ReLU']):
                    f.write(f"\tget_dim(&layer{layer+1}_wgt, &w{next_buffer}_blob);\n")

        # Adjust arguments for structures 
        f.write(f"\tin = d{current_buffer}_blob;\n")
        f.write(f"\twgt = w{current_buffer}_blob;\n")
        if bias_l[layer] == 1:
            f.write(f"\tbias = bias{current_buffer}_blob;\n")
        f.write(f"\tout = d{output_buffer}_blob;\n")
    
        # Make sure to have no pending dma transfers
        f.write("\tpi_cl_dma_flush();\n")

        # Store previous calculated output
        if layer > 0 and do_store:
            f.write(f"\tstore((uint32_t) d{previous_out}_blob.data, (uint32_t) layer{layer-1}_out.data, {bytes_per_data}*layer{layer-1}_out.dim);\n")

         # Load coeff for the next layer   
        if (layer + 1 < len(layers_l)) and not (layers_l[layer + 1] in ['Skipnode', 'ReLU']) and not is_skipder:
            f.write(f"\tload((uint32_t) layer{layer+1}_wgt.data, (uint32_t) w{next_buffer}_blob.data, {bytes_per_data}*layer{layer+1}_wgt.dim);\n")
            if bias_l[layer + 1] == 1:
                f.write(f"\tload((uint32_t) layer{layer+1}_bias.data, (uint32_t) bias{next_buffer}_blob.data, {bytes_per_data}*layer{layer+1}_bias.dim);\n")
        if layers_l[layer] == 'Skipnode':
            f.write(f"\tload((uint32_t) layer{layer+1}_wgt.data, (uint32_t) w{next_buffer}_blob.data, {bytes_per_data}*layer{layer+1}_wgt.dim);\n")
            if bias_l[layer + 1] == 1:
                f.write(f"\tload((uint32_t) layer{layer+1}_bias.data, (uint32_t) bias{next_buffer}_blob.data, {bytes_per_data}*layer{layer+1}_bias.dim);\n")

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
            print("[deployment_utils.GenerateNet]: PULP layer not implemented or wrapped in DNN Deployer!")
            exit()

        # Insert casting operator for data type variation
        if layer < len(layers_l)-1 and data_type_l[layer] != data_type_l[layer+1]:
            if data_type_l[layer] == 'FP32' and data_type_l[layer+1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(layer, "FW", data_type_l[layer]))
            elif data_type_l[layer] == 'FP16' and data_type_l[layer+1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(layer, "FW", data_type_l[layer]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[layer], data_type_l[layer+1], layer))

        # Handle last layer store 
        if (layer == len(layers_l) - 1):
            f.write(f"\n\tstore((uint32_t) out.data, (uint32_t) layer{layer}_out.data, {bytes_per_data}*layer{layer}_out.dim);\n")
            f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")

        do_store = True
        if is_skipder:
            if layers_l[layer] != 'Skipnode':
                f.write("\n\t// Save Skipderivation output\n")
                f.write(f"\tstore((uint32_t) out.data, (uint32_t) layer{layer}_out.data, {bytes_per_data}*layer{layer}_out.dim);\n")  
                if not (layers_l[layer + 1] in ['Skipnode', 'ReLU']):
                    f.write(f"\tget_dim(&layer{layer + 1}_wgt, &w{current_buffer}_blob);\n")
                    f.write(f"\tload((uint32_t) layer{layer + 1}_wgt.data, (uint32_t) wgt.data,  {bytes_per_data}*layer{layer + 1}_wgt.dim);\n")
                    if bias_l[layer + 1] == 1:
                        f.write(f"\tload((uint32_t) layer{layer + 1}_bias.data, (uint32_t) bias.data,  {bytes_per_data}*layer{layer + 1}_bias.dim);\n")
                    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
                f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n") 
            do_store = False

        if layers_l[layer] != 'Skipnode':
            previous_out  = output_buffer
        current_buffer = next_buffer

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  #ifdef PROF_NET\n")
            f.write("  STOP_STATS();\n")
            f.write("  #endif\n\n")  
    f.write("}\n")



    f.write("\n// Backward pass function\n")
    f.write("void backward()\n{\n")
    
    # Compute loss
    if loss_fn == "MSELoss":
        if data_type_l[-1] == 'FP32':
            bytes_per_data = 4
        elif data_type_l[-1] == 'FP16':
            bytes_per_data = 2
        f.write("    load((uint32_t) layer"+str(len(layers_l)-1)+"_out.data, (uint32_t) d1_blob.data, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
        f.write("    load((uint32_t) LABEL, (uint32_t) w1_blob.data, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
        f.write("    loss_args.output = &d1_blob;\n") 
        f.write("    loss_args.target = w1_blob.data;\n") 
        f.write("    loss_args.wr_loss = &loss;\n") 
        if data_type_l[-1] == 'FP32':
            f.write("    pulp_MSELoss_backward(&loss_args);\n")   
        elif data_type_l[-1] == 'FP16':
            f.write("    pulp_MSELoss_backward_fp16(&loss_args);\n") 
        f.write("    store((uint32_t) d1_blob.diff, (uint32_t) layer"+str(len(layers_l)-1)+"_out.diff, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
    elif loss_fn == 'CrossEntropyLoss':
        if data_type_l[-1] == 'FP32':
            bytes_per_data = 4
        elif data_type_l[-1] == 'FP16':
            bytes_per_data = 2
        f.write("    load((uint32_t) layer"+str(len(layers_l)-1)+"_out.data, (uint32_t) d1_blob.data, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
        f.write("    load((uint32_t) LABEL, (uint32_t) w1_blob.data, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
        f.write("    loss_args.output = &d1_blob;\n") 
        f.write("    loss_args.target = w1_blob.data;\n") 
        f.write("    loss_args.wr_loss = &loss;\n") 
        if data_type_l[-1] == 'FP32':
            f.write("    pulp_CrossEntropyLoss_backward(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("    pulp_CrossEntropyLoss_backward_fp16(&loss_args);\n")
        f.write("    store((uint32_t) d1_blob.diff, (uint32_t) layer"+str(len(layers_l)-1)+"_out.diff, "+str(bytes_per_data)+"*layer"+str(len(layers_l)-1)+"_out.dim);\n")
    else:
        print("[deployment_utils.GenerateNet]: invalid loss function for backward!!")

    # Profiling options: single layer or all
    if PROFILE_SINGLE_LAYERS == True:
        f.write("  printf(\"\\nBACKWARD PROFILING:\\n\\n\");\n")

    next_input_buffer = 0
    for layer in range(len(layers_l)):
        lay = len(layers_l) - layer - 1

        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  printf(\"\\nLayer "+str(lay)+"\\n\");\n")
            f.write("  #ifdef PROF_NET\n")
            f.write("  START_STATS();\n")
            f.write("  #endif\n")  

        # Bool for Skipnode and layer after Skipnodes detection
        is_skipderivation = False
        next_is_skipderivation  = False
        if layers_l[lay] != 'Sumnode' and sumnode_connections[lay] > -1:
            is_skipderivation = True
        if lay > 0:
            if layers_l[lay - 1] != 'Sumnode' and sumnode_connections[lay - 1] > -1:
                next_is_skipderivation = True

        skip_in_grad = 0
        if lay == 0:
            skip_in_grad = 1

        # Check for target layer's input for diff calculation of Skipnode derivations
        target_layer = lay
        if is_skipderivation: 
            for l in range(len(layers_l)):
                if sumnode_connections[lay + l ] < 0:
                    break
                else:
                    target_layer += 1

        output_buffer = 1
        if current_buffer == 1:
            output_buffer = 0

        if next_is_skipderivation or layers_l[lay] == 'Sumnode':
            next_buffer = current_buffer
            next_input_buffer = output_buffer
        else:
            next_buffer = 1
            next_input_buffer = 0
            if current_buffer == 1:
                next_buffer = 0
                next_input_buffer = 1

        f.write("\n")
        f.write(f"\t// Layer {lay} ({layers_l[lay]}, {current_buffer}, {next_buffer}, {next_input_buffer})\n")

        bytes_per_data = 2
        if data_type_l[layer] == 'FP32':
            bytes_per_data = 4 


        # Copy struct info 
        if not layers_l[lay] in ['Skipnode', 'Sumnode',  'ReLU']:
            f.write(f"\tcopy_struct_param((unsigned int) &l{lay}_args, (unsigned int) &{layers_l[lay]}_args, sizeof(l{lay}_args));\n")
            f.write("\tpi_cl_dma_cmd_wait(cmd_struct);\n")

        # Adjust arguments for structures 
        f.write(f"\tin = d{output_buffer}_blob;\n")
        f.write(f"\twgt = w{output_buffer}_blob;\n")
        if bias_l[lay] == 1:
            f.write(f"\tbias = bias{output_buffer}_blob;\n")
        f.write(f"\tout = d{current_buffer}_blob;\n")
        if is_skipderivation and layers_l[lay] != 'Skipnode':
            f.write(f"\tget_dim(&in, &w{current_buffer}_blob);\n")
            f.write(f"\tin.diff = w{current_buffer}_blob.diff;\n")

        # Store dOUT
        if not is_skipderivation:
            f.write(f"\tstore((uint32_t) d{current_buffer}_blob.diff, (uint32_t) layer{lay}_out.diff, {bytes_per_data}*layer{lay}_out.dim);\n")
        else:
            if not layers_l[lay] in ['ReLU', 'Skipnode']:
                f.write(f"\tload((uint32_t) layer{lay}_wgt.data, (uint32_t) wgt.data, {bytes_per_data}*layer{lay}_wgt.dim);\n")
                if bias_l[lay] == 1:
                    f.write(f"\tload((uint32_t) layer{lay}_bias.data, (uint32_t) bias.data, {bytes_per_data}*layer{lay}_bias.dim);\n")
            f.write(f"\tload((uint32_t) layer{lay}_out.diff, (uint32_t) out.diff, {bytes_per_data}*layer{lay}_out.dim);\n")
            f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")

        # Compute dW if needed
        if not layers_l[lay] in ['Skipnode', 'Sumnode',  'ReLU']:
            if layers_l[lay] == 'linear':
                f.write(ntemp.linear_template_coeff_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'conv2d':
                f.write(ntemp.conv2d_template_coeff_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'DW':
                f.write(ntemp.DW_template_coeff_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'PW':
                f.write(ntemp.PW_template_coeff_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'AvgPool':
                f.write(ntemp.AvgPool_template_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'MaxPool':
                f.write(ntemp.MaxPool_template_BW(lay, data_type_l[lay]))
            elif layers_l[lay]  == 'InstNorm':
                f.write(ntemp.InstNorm_template_BW(lay, data_type_l[lay]))

            # Store dW 
            f.write(f"\tstore((uint32_t) w{output_buffer}_blob.diff, (uint32_t) layer{lay}_wgt.diff, {bytes_per_data}*layer{lay}_wgt.dim);\n")
            if bias_l[lay] == 1:
                f.write(f"\tstore((uint32_t) bias{output_buffer}_blob.diff, (uint32_t) layer{lay}_bias.diff, {bytes_per_data}*layer{lay}_bias.dim);\n")

        # Compute dIN
        if not skip_in_grad:
            if layers_l[lay] == 'linear':
                f.write(ntemp.linear_template_in_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'conv2d':
                f.write(ntemp.conv2d_template_in_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'DW':
                f.write(ntemp.DW_template_in_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'PW':
                f.write(ntemp.PW_template_in_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'ReLU':
                f.write(ntemp.ReLU_template_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'AvgPool':
                f.write(ntemp.AvgPool_template_BW(lay, data_type_l[lay]))
            elif layers_l[lay] == 'MaxPool':
                f.write(ntemp.MaxPool_template_BW(lay, data_type_l[lay]))
            elif layers_l[lay]  == 'InstNorm':
                f.write(ntemp.InstNorm_template_BW(lay, data_type_l[lay]))

            if is_skipderivation:
                f.write(ntemp.sum(lay, layers_l[lay] == 'Skipnode', current_buffer, output_buffer, data_type_l[lay]))

        # Load next layer's input and coefficients
        if lay > 0:
            f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
            if layers_l[lay] == 'Sumnode':
                f.write(f"\tload((uint32_t) layer{lay - 1}_out.data, (uint32_t) out.data, {bytes_per_data}*out.dim);\n")
                f.write("\tpi_cl_dma_flush();\n")
            f.write(f"\tget_dim( &layer{lay-1}_in, &d{next_input_buffer}_blob);\n")
            f.write(f"\tload((uint32_t) layer{lay-1}_in.data, (uint32_t) d{next_input_buffer}_blob.data, {bytes_per_data}*layer{lay-1}_in.dim);\n")
            if not layers_l[lay -1 ] in ['Skipnode', 'Sumnode',  'ReLU']:
                f.write(f"\tget_dim( &layer{lay-1}_wgt, &w{next_input_buffer}_blob);\n")
                f.write(f"\tload((uint32_t) layer{lay-1}_wgt.data, (uint32_t) w{next_input_buffer}_blob.data, {bytes_per_data}*layer{lay-1}_wgt.dim);\n")

                if bias_l[lay - 1] == 1:
                    f.write(f"\tget_dim( &layer{lay-1}_bias, &bias{next_input_buffer}_blob);\n")
                    f.write(f"\tload((uint32_t) layer{lay-1}_bias.data, (uint32_t) bias{next_input_buffer}_blob.data, {bytes_per_data}*layer{lay-1}_bias.dim);\n")
            f.write("\tpi_cl_dma_flush();\n")

        current_buffer = next_buffer

        '''
        # Insert casting operator for data type variation
        if lay < len(layers_l)-1 and lay > 0 and data_type_l[lay] != data_type_l[lay-1]:
            if data_type_l[lay] == 'FP32' and data_type_l[lay-1] == 'FP16':
                f.write(ntemp.cast_fp32_to_fp16_template(lay, "BW", data_type_l[lay]))
            elif data_type_l[lay] == 'FP16' and data_type_l[lay-1] == 'FP32':
                f.write(ntemp.cast_fp16_to_fp32_template(lay, "BW", data_type_l[lay]))
            else:
                print("[deployment_utils.GenerateNet]: Unable to convert {} to {} @layer{}!".format(data_type_l[lay], data_type_l[lay-1], lay))

        if sumnode_connections[lay] != -1 and layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and skip_in_grad==0:
            f.write(f"\tload_output(&layer{target_layer}_in, 0);\n")
            f.write(ntemp.sum(lay, data_type_l[lay]))
        

        if layers_l[lay] != 'Sumnode' and layers_l[lay] != 'Skipnode' and layers_l[lay] != 'ReLU':
            f.write(f"\tstore_coeff(&layer{lay}_wgt, 0);\n")

        if lay > 0 and layers_l[lay] != 'Sumnode':
            f.write(f"\tstore_input(&layer{target_layer}_in, 0);\n")
        '''
        # Profile layer by layer?
        if PROFILE_SINGLE_LAYERS == True:
            f.write("  #ifdef PROF_NET\n")
            f.write("  STOP_STATS();\n")
            f.write("  #endif\n\n")  
    f.write("}\n")


    f.write("\n// Compute loss and output gradient\n")
    f.write("void compute_loss()\n{\n")

    if loss_fn == "MSELoss":
        float_size = 2
        if data_type_l[0] == 'FP32':
            float_size = 4
        f.write("\tloss_args.output = &out;\n")
        f.write("\tloss_args.target = out.diff;\n")
        f.write("\tloss_args.wr_loss = &loss;\n")
        f.write(f"\tload((uint32_t) LABEL, (uint32_t) out.diff, {float_size}*OUT_SIZE);\n")
        f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")

        if data_type_l[-1] == 'FP32':
            f.write("  pulp_MSELoss(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_MSELoss_fp16(&loss_args);\n")
        else:
            print("[deployment_utils.GenerateNet]: Invalid loss type!")
            exit()
        f.write(f"\tstore((uint32_t) out.diff, (uint32_t) layer{len(layers_l)-1}_out.diff, {float_size}*OUT_SIZE);\n")
    elif loss_fn == "CrossEntropyLoss":
        float_size = 2
        if data_type_l[0] == 'FP32':
            float_size = 4
        f.write("\tloss_args.output = &out;\n")
        f.write("\tloss_args.target = out.diff;\n")
        f.write("\tloss_args.wr_loss = &loss;\n")
        f.write(f"\tload((uint32_t) LABEL, (uint32_t) out.diff, {float_size}*OUT_SIZE);\n")
        f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")

        if data_type_l[-1] == 'FP32':
            f.write("  pulp_CrossEntropyLoss(&loss_args);\n")
        elif data_type_l[-1] == 'FP16':
            f.write("  pulp_CrossEntropyLoss_fp16(&loss_args);\n")
        else:
            print("[deployment_utils.GenerateNet]: Invalid loss type!")
            exit()
        f.write(f"\tstore((uint32_t) out.diff, (uint32_t) layer{len(layers_l)-1}_out.diff, {float_size}*OUT_SIZE);\n")
    else:
        print("[deployment_utils.GenerateNet]: Loss function not valid for PULP deployment!!")
        exit()

    f.write("}\n")


    f.write("\n// Function to update the network\n")
    f.write("void update_weights()\n{\n")
    '''
    for layer in range(len(layers_l)):
        if layers_l[layer] in ['conv2d', 'PW', 'DW', 'linear']:

            f.write("\n\tpi_cl_dma_flush();\n")

            bytes_per_data = 2
            if data_type_l[layer] == 'FP32':
                bytes_per_data = 4
                f.write("  struct optim_args opt_l"+str(layer)+";\n")
            elif data_type_l[layer] == 'FP16':
                f.write("  struct optim_args_fp16 opt_l"+str(layer)+";\n")

            f.write(f"\tget_dim(&layer{layer}_wgt, &d0_blob);\n")
            f.write("\topt_l"+str(layer)+f".weights = &d0_blob;\n")
            f.write("  opt_l"+str(layer)+".learning_rate = LEARNING_RATE;\n") 
            f.write(f"  load((uint32_t) layer{layer}_wgt.data, (uint32_t) d0_blob.data, {bytes_per_data}*layer{layer}_wgt.dim);\n")
            f.write(f"  load((uint32_t) layer{layer}_wgt.diff, (uint32_t) d0_blob.diff, {bytes_per_data}*layer{layer}_wgt.dim);\n")
            f.write("\n\tpi_cl_dma_flush();\n")
    
            if data_type_l[layer] == 'FP32':
                f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp32, &opt_l"+str(layer)+");\n")
            elif data_type_l[layer] == 'FP16':
                f.write("  pi_cl_team_fork(NUM_CORES, pulp_gradient_descent_fp16, &opt_l"+str(layer)+");\n")

            f.write(f"\tstore((uint32_t) d0_blob.data, (uint32_t) layer{layer}_wgt.data, {bytes_per_data}*layer{layer}_wgt.dim);\n")
            
    
    '''
    layers_with_weights = []
    
    for layer in range(len(layers_l)):
        if layers_l[layer] in ['linear', 'conv2d', 'DW', 'PW', 'InstNorm']:
            layers_with_weights.append(layer)
    print(layers_with_weights)

    idx = 0
    buff = 0
    next_buff = 1

    f.write("\t// Creates all structures needed\n")

    for layer in layers_with_weights:
        if data_type_l[layer] == 'FP32':
            f.write("\tstruct optim_args opt_l"+str(layer)+";\n")
        elif data_type_l[layer] == 'FP16':
            f.write("\tstruct optim_args_fp16 opt_l"+str(layer)+";\n")


    for layer in layers_with_weights:

        next_buff = 1
        if buff == 1:
            next_buff = 0

        f.write("\n\tpi_cl_dma_flush();\n")
        bytes_per_data = 2

        if idx == 0:
            if data_type_l[layer] == 'FP32':
                bytes_per_data = 4

            f.write(f"\tget_dim(&layer{layer}_wgt, &d{buff}_blob);\n")
            f.write("\topt_l"+str(layer)+f".weights = &d{buff}_blob;\n")
            f.write("\topt_l"+str(layer)+".learning_rate = LEARNING_RATE;\n") 
            f.write(f"\tload((uint32_t) layer{layer}_wgt.data, (uint32_t) d{buff}_blob.data, {bytes_per_data}*layer{layer}_wgt.dim);\n")
            f.write(f"\tload((uint32_t) layer{layer}_wgt.diff, (uint32_t) d{buff}_blob.diff, {bytes_per_data}*layer{layer}_wgt.dim);\n")

            if bias_l[layer] == 1:
                f.write(f"\tget_dim(&layer{layer}_bias, &bias{buff}_blob);\n")
                f.write("\topt_l"+str(layer)+f".bias = &bias{buff}_blob;\n")
                f.write(f"\tload((uint32_t) layer{layer}_bias.data, (uint32_t) bias{buff}_blob.data, {bytes_per_data}*layer{layer}_bias.dim);\n")
                f.write(f"\tload((uint32_t) layer{layer}_bias.diff, (uint32_t) bias{buff}_blob.diff, {bytes_per_data}*layer{layer}_bias.dim);\n")

            f.write("\n\tpi_cl_dma_flush();\n")

        
        # Load next weight
        if (idx + 1) < len(layers_with_weights):
            if data_type_l[layers_with_weights[idx+1]] == 'FP32':
                bytes_per_data = 4

            f.write(f"\tget_dim(&layer{layers_with_weights[idx+1]}_wgt, &d{next_buff}_blob);\n")
            f.write("\topt_l"+str(layers_with_weights[idx+1])+f".weights = &d{next_buff}_blob;\n")
            f.write("\topt_l"+str(layers_with_weights[idx+1])+".learning_rate = LEARNING_RATE;\n")
            f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
            f.write(f"\tload((uint32_t) layer{layers_with_weights[idx+1]}_wgt.data, (uint32_t) d{next_buff}_blob.data, {bytes_per_data}*layer{layers_with_weights[idx+1]}_wgt.dim);\n")
            f.write(f"\tload((uint32_t) layer{layers_with_weights[idx+1]}_wgt.diff, (uint32_t) d{next_buff}_blob.diff, {bytes_per_data}*layer{layers_with_weights[idx+1]}_wgt.dim);\n")

            if bias_l[layers_with_weights[idx+1]] == 1:
                f.write(f"\tget_dim(&layer{layers_with_weights[idx+1]}_bias, &bias{next_buff}_blob);\n")
                f.write("\topt_l"+str(layers_with_weights[idx+1])+f".bias = &bias{next_buff}_blob;\n")
                f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
                f.write(f"\tload((uint32_t) layer{layers_with_weights[idx+1]}_bias.data, (uint32_t) bias{next_buff}_blob.data, {bytes_per_data}*layer{layers_with_weights[idx+1]}_bias.dim);\n")
                f.write(f"\tload((uint32_t) layer{layers_with_weights[idx+1]}_bias.diff, (uint32_t) bias{next_buff}_blob.diff, {bytes_per_data}*layer{layers_with_weights[idx+1]}_bias.dim);\n")
        #f.write("\tpi_cl_dma_flush();\n")

        # Compute gradient descent
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
        # Store updated coefficients
        bytes_per_data = 2
        if data_type_l[layer] == 'FP32':
                bytes_per_data = 4
        f.write(f"\tstore((uint32_t) d{buff}_blob.data, (uint32_t) layer{layer}_wgt.data, {bytes_per_data}*layer{layer}_wgt.dim);\n")
        if bias_l[layer] == 1:
            f.write(f"\tstore((uint32_t) bias{buff}_blob.data, (uint32_t) layer{layer}_bias.data, {bytes_per_data}*layer{layer}_bias.dim);\n")
        f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
        idx += 1
        buff = next_buff
        
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

    f.write("  printf(\"Initializing network..\\n\");\n")
    f.write("  DNN_init();\n")

    f.write("  printf(\"Testing DNN initialization forward..\");\n")
    f.write("  forward();\n")
    f.write("  print_output();\n\n")

    # Profile layer by layer?
    if PROFILE_SINGLE_LAYERS == False:
        f.write("  #ifdef PROF_NET\n")
        f.write("  INIT_STATS();\n  PRE_START_STATS();\n  START_STATS();\n")
        f.write("  #endif\n\n")

    f.write("  for (int epoch=0; epoch<EPOCHS; epoch++)\n  {\n")
    f.write("    forward();\n")
    f.write("    compute_loss();\n")
    if PRINT_TRAIN_LOSS == True:
        f.write("    /* Stop profiling */ pi_perf_stop();\n")
        f.write("    if (epoch == 0) printf(\"\\n\");\n")
        f.write("    printf(\">>> EPOCH %d: train_loss = %f (GM: %f)\\n\", epoch, loss, TRAIN_LOSS[epoch]);\n")
        f.write("    /* Continue profiling */ pi_perf_start();\n")
    f.write("    backward();\n")
    f.write("    update_weights();\n")
    f.write("  }\n\n")

    # Profile layer by layer?
    if PROFILE_SINGLE_LAYERS == False:
        f.write("  #ifdef PROF_NET\n")
        f.write("  STOP_STATS();\n")
        f.write("  #endif\n\n")

    f.write("  // Check and print updated output\n")
    f.write("  forward();\n")
    f.write("  printf(\"Checking updated output..\\n\");\n")
    f.write("  check_post_training_output();\n")
    f.write("  print_output();\n")

    f.write("}\n")

    data_size = 0
    suffix = ""

    if data_type_l[0] == 'FP32':
        data_size = 4
        suffix = ""
    else :
        data_size = 2
        suffix = "_fp16"




    '''
    f.write("\nvoid dma_handler(uint8_t do_store, uint8_t do_load, void * src_store, void * dst_store, void * src_load, void * dst_load){\n")
    f.write("\t\tstruct blob * d;\n")
    f.write("\t\tstruct blob * s;\n")
    f.write("\t\tif (do_store){\n")
    f.write("\t\t\td = (struct blob * ) dst_store;\n")
    f.write("\t\t\ts = (struct blob * ) src_store;\n")
    f.write("\t\t\tstore((uint32_t) s->data, (uint32_t) d->data, s->dim);}\n")
    f.write("\t\tif (do_load){\n")
    f.write("\t\t\td = (struct blob * ) dst_load;\n")
    f.write("\t\t\ts = (struct blob * ) src_load;\n")
    f.write("\t\t\tstore((uint32_t) s->data, (uint32_t) d->data, s->dim);} }\n")
    '''

    f.write("\nvoid load(uint32_t src, uint32_t dst, int dim){\n")
    f.write("\tpi_cl_dma_cmd(src, dst, dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);}\n")

    f.write("\nvoid store(uint32_t src, uint32_t dst, int dim){\n")
    f.write("\tpi_cl_dma_cmd(dst, src, dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);}\n")

    f.write("\nvoid get_dim(void * src_blob, void * dst_blob){\n")
    f.write("\tstruct blob * s = (struct blob * ) src_blob;\n")
    f.write("\tstruct blob * d = (struct blob * ) dst_blob;\n")
    f.write("\td->dim = s->dim;\n")
    f.write("\td->C = s->C;\n")
    f.write("\td->H = s->H;\n")
    f.write("\td->W = s->W;\n")
    f.write("\tupdate();}\n")

    f.write("\nvoid update(){\n")
    f.write("\td0_blob.data = BUFF;\n")
    f.write("\td0_blob.diff = BUFF + d0_blob.dim;\n")
    f.write("\tw0_blob.data = BUFF + 2*d0_blob.dim;\n")
    f.write("\tw0_blob.diff = BUFF + 2*d0_blob.dim + w0_blob.dim;\n")
    f.write("\td1_blob.data = BUFF + MAX_SIZE/2;\n")
    f.write("\td1_blob.diff = BUFF + MAX_SIZE/2 + d1_blob.dim;\n")
    f.write("\tw1_blob.data = BUFF + MAX_SIZE/2 + 2*d1_blob.dim;\n")
    f.write("\tw1_blob.diff = BUFF + MAX_SIZE/2 + 2*d1_blob.dim + w1_blob.dim;}\n")

    f.write("\nvoid reset_arguments(){\n")
    f.write("\td0_blob.dim = 0;\n")
    f.write("\tw0_blob.dim = 0;\n")
    f.write("\td1_blob.dim = 0;\n")
    f.write("\tw1_blob.dim = 0;\n")
    f.write("\tlinear_args.output = &out;\n")
    f.write("\tlinear_args.input = &in;\n")
    f.write("\tlinear_args.coeff = &wgt;\n")

    f.write("\tconv2d_args.output = &out;\n")
    f.write("\tconv2d_args.input = &in;\n")
    f.write("\tconv2d_args.coeff = &wgt;\n")
    f.write("\tconv2d_args.bias = &bias;\n")

    f.write("\tPW_args.output = &out;\n")
    f.write("\tPW_args.input = &in;\n")
    f.write("\tPW_args.coeff = &wgt;\n")

    f.write("\tDW_args.output = &out;\n")
    f.write("\tDW_args.input = &in;\n")
    f.write("\tDW_args.coeff = &wgt;\n")

    f.write("\tact_args.output = &out;\n")
    f.write("\tact_args.input = &in;\n")

    f.write("\tresconn_args.output = &out;\n")
    f.write("\tresconn_args.lout = &in;\n")
    f.write("\tresconn_args.skip = &wgt;\n")
    f.write("}\n\n")

    f.write("\nvoid copy_struct_param(unsigned int from, unsigned int to, int size){\n")
    f.write("\tpi_cl_dma_cmd(from, to, size, PI_CL_DMA_DIR_EXT2LOC , cmd_struct);}\n")




























































    '''
    f.write("\n// Functions for DMA managment\n")
    f.write("\nvoid load_coeff(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_weight_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")


    f.write("\nvoid load_input(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_input_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")

   
    
    f.write("\nvoid load_output(void * src_blob, uint8_t data_diff_both){\n") 
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) src_blob;\n")
    f.write("\tget_output_dim(src_blob);\n")
    f.write("\tif (data_diff_both == 0) // Load only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both == 1) // Load only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tif (data_diff_both > 1) { // Load both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_EXT2LOC , cmd_load);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);} \n")

    f.write("\nvoid store_output(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (OUT_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (OUT_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n") 

    f.write("\nvoid store_coeff(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (W_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (W_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n")

    f.write("\nvoid store_input(void * dest_blob, uint8_t data_diff_both){ \n")
    f.write(f"\tstruct blob{suffix} * b = (struct blob{suffix} *) dest_blob;\n")
    f.write("\tif (data_diff_both == 0) // Store only .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both == 1) // Store only .data\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tif (data_diff_both > 1) { // Store both .data and .diff\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->data), (uint32_t) (IN_DATA), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);\n")
    f.write(f"\tpi_cl_dma_cmd((uint32_t) (b->diff), (uint32_t) (IN_DIFF), {data_size}*b->dim, PI_CL_DMA_DIR_LOC2EXT , cmd_store);"+"}\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_store);} \n")

    f.write("\nvoid get_input_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\tinput_blob.C = src->C;\n")
    f.write("\tinput_blob.H = src->H;\n")
    f.write("\tinput_blob.W = src->W;\n")
    f.write("\tinput_blob.dim = src->dim;\n")
    f.write("\tIN_DIFF = BUFF + input_blob.dim;\n")
    f.write("\tW_DATA = BUFF + 2*input_blob.dim;\n")
    f.write("\tupdate_blob();}\n")

    f.write("\nvoid get_output_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\toutput_blob.C = src->C;\n")
    f.write("\toutput_blob.H = src->H;\n")
    f.write("\toutput_blob.W = src->W;\n")
    f.write("\toutput_blob.dim = src->dim;\n")
    f.write("\tOUT_DIFF = BUFF + 2*weight_blob.dim + 2*input_blob.dim + output_blob.dim;\n")
    f.write("\tupdate_blob();}\n")

    f.write("\nvoid get_weight_dim(void * b){\n")
    f.write(f"\tstruct blob{suffix} * src = (struct blob{suffix} *) b;\n")
    f.write("\tweight_blob.C = src->C;\n")
    f.write("\tweight_blob.H = src->H;\n")
    f.write("\tweight_blob.W = src->W;\n")
    f.write("\tweight_blob.dim = src->dim;\n")
    f.write("\tW_DIFF = BUFF + weight_blob.dim + 2*input_blob.dim;\n")
    f.write("\tOUT_DATA = BUFF + 2*weight_blob.dim + 2*input_blob.dim;\n")
    f.write("\tupdate_blob();}\n")
   


    f.write("\nvoid swap_in_out(){\n")
    f.write("\tt = OUT_DATA;\n")
    f.write("\tOUT_DATA = IN_DATA;\n")
    f.write("\tIN_DATA = t;\n")
    f.write("\tt = OUT_DIFF;\n")
    f.write("\tOUT_DIFF = IN_DIFF;\n")
    f.write("\tIN_DIFF = t; \n")
    f.write("\ttemp_blob = output_blob;\n")
    f.write("\toutput_blob = input_blob;\n")
    f.write("\tinput_blob = temp_blob;}\n")

    f.write("\nvoid copy_struct_param(unsigned int from, unsigned int to, int size){\n")
    f.write("\tpi_cl_dma_cmd(from, to, size, PI_CL_DMA_DIR_EXT2LOC , cmd_load);\n")
    f.write("\tpi_cl_dma_cmd_wait(cmd_load);}\n")

    f.write("\nvoid reset_arguments(){\n")
    f.write("\tlinear_args.output = &output_blob;\n")
    f.write("\tlinear_args.input = &input_blob;\n")
    f.write("\tlinear_args.coeff = &weight_blob;\n")

    f.write("\tconv2d_args.output = &output_blob;\n")
    f.write("\tconv2d_args.input = &input_blob;\n")
    f.write("\tconv2d_args.coeff = &weight_blob;\n")
    f.write("\tconv2d_args.bias = &bias_blob;\n")

    f.write("\tPW_args.output = &output_blob;\n")
    f.write("\tPW_args.input = &input_blob;\n")
    f.write("\tPW_args.coeff = &weight_blob;\n")

    f.write("\tDW_args.output = &output_blob;\n")
    f.write("\tDW_args.input = &input_blob;\n")
    f.write("\tDW_args.coeff = &weight_blob;\n")

    f.write("\tact_args.output = &output_blob;\n")
    f.write("\tact_args.input = &input_blob;\n")

    f.write("\tresconn_args.output = &output_blob;\n")
    f.write("\tresconn_args.lout = &input_blob;\n")
    f.write("\tresconn_args.skip = &weight_blob;\n")
    f.write("}\n\n")

    f.write("\nvoid update_blob(){\n")
    f.write("\tinput_blob.data = IN_DATA;\n")
    f.write("\tinput_blob.diff = IN_DIFF;\n")
    f.write("\toutput_blob.data = OUT_DATA;\n")
    f.write("\toutput_blob.diff = OUT_DIFF;\n")
    f.write("\tweight_blob.data = W_DATA;\n")
    f.write("\tweight_blob.diff = W_DIFF;}\n")

    f.write("\nvoid reset_dim(){\n")
    f.write("\tinput_blob.dim = 0;\n")
    f.write("\tweight_blob.dim = 0;\n")
    f.write("\toutput_blob.dim = 0;}\n")
        '''
    f.close()



    return

