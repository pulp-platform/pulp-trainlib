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

import deployer_utils.deployment_utils_single_buffer as utilsSB
import deployer_utils.deployment_utils_double_buffer as utilsDB
import deployer_utils.deployment_utils as utils
import numpy as np

"""
The DNN Size Checker checks if the DNN fits the available PULP
memory
"""



MAX_LAYER_DIM = 0

def DNN_Size_Checker (layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_str_list, w_str_list, h_pad_list,
                      w_pad_list, data_type_l, bias_l, update_layer_l, avail_mem_bytes, USE_DMA, CONV2D_USE_IM2COL):

    total_memory_occupation_bytes = 0
    l2_occupation = 0
    global MAX_LAYER_DIM 
    l1_structs_mem = 0

    # Find last layer to be updated
    last_updated_idx = len(layers_l) - 1
    for layer in range(len(layers_l)):
        if update_layer_l[layer] == 1:
            last_updated_idx = layer
            break
    print(f"Weights updated up to layer {last_updated_idx}")

    # Compute activation and weight memory occupation
    
    for layer in range(len(layers_l)):
        # Find if the layer needs backprop
        compute_in_grad = 0
        if layer > last_updated_idx:
            compute_in_grad = 1
        # Find other attributes
        is_last_layer = False
        if layer == len(layers_l) - 1:
            is_last_layer = True
        if USE_DMA == 'NO':
            total_memory_occupation_bytes += utils.compute_wgt_act_memocc_bytes(layer, layers_l[layer], in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], hin_l[layer], win_l[layer], h_pad_list[layer], w_pad_list[layer], h_str_list[layer], w_str_list[layer], data_type_l[layer], bias_l[layer], update_layer_l[layer], compute_in_grad, is_last_layer)
        elif USE_DMA in ['SB', 'DB']:
            l2_occupation +=  utils.compute_wgt_act_memocc_bytes(layer, layers_l[layer], in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], hin_l[layer], win_l[layer], h_pad_list[layer], w_pad_list[layer], h_str_list[layer], w_str_list[layer], data_type_l[layer], bias_l[layer], update_layer_l[layer], compute_in_grad, is_last_layer)
    # Compute im2col memory occupation
    mem_im2col = 0
    idx_im2col = 0
    mem_im2col, idx_im2col = utils.compute_im2col_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_pad_list, w_pad_list, h_str_list, w_str_list, data_type_l, update_layer_l, CONV2D_USE_IM2COL)
    total_memory_occupation_bytes += mem_im2col

    if mem_im2col > 0:
        print("Max IM2COL size of {} bytes @layer {}".format(mem_im2col, idx_im2col))

    # Compute transpose and blocktranspose memory occupation 
    mem_blocktransp = 0
    idx_blocktransp = 0
    mem_blocktransp, idx_blocktransp = utils.compute_bt_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, data_type_l, update_layer_l, CONV2D_USE_IM2COL)
    total_memory_occupation_bytes += mem_blocktransp

    if mem_blocktransp > 0:
        print("Max transposition / block transposition buffer size of {} @layer {}".format(mem_blocktransp, idx_blocktransp))

    # Compute additional mixed precision buffer memory occupation
    mem_cast_buffer = 0
    mem_cast_buffer, idx_max_act, max_act_inout = utils.compute_cast_buffer_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_pad_list, w_pad_list, h_str_list, w_str_list, data_type_l)
    total_memory_occupation_bytes += mem_cast_buffer

    #if mem_cast_buffer > 0:
    print("Additional {} bytes allocated for mixed precision management (size @layer {}, {})".format(mem_cast_buffer, idx_max_act, max_act_inout))

    # Buffer memory allocation for Single Buffer mode
    l1_buff_size = 0
    if USE_DMA == 'SB':
        MAX_LAYER_DIM, MAX_LAYER_DIM_BYTES = utilsSB.max_layer_dim(layers_l, in_ch_l, hin_l, win_l, out_ch_l, hk_l, wk_l, data_type_l[0], h_str_list, w_str_list, h_pad_list, w_pad_list, bias_l, data_type_l, update_layer_l)
        #MAX_LAYER_DIM = utilsSB.max_layer_dim(layers_l, in_ch_l, hin_l, win_l, out_ch_l, hk_l, wk_l, data_type_l[0], h_str_list, w_str_list, h_pad_list, w_pad_list, bias_l)
        l1_buff_size = MAX_LAYER_DIM
        

        # TODO: Change hard coding to computation based on their actual definition
        l1_structs_mem = 0
        l1_structs_mem += 6*4 # 6 pointers IN_DATA, IN_DIFF ...
        l1_structs_mem += 4*(6*4) # 4 blobs input_blob, output_blob ..
        # l1_structs_mem += 32 # linear_args
        l1_structs_mem += 44 # linear_args, if add bias: + 8 (blob, data+diff pointers) + 4 (use_biases) = 32+12 = 44
        l1_structs_mem += 84 # conv2d_args
        l1_structs_mem += 40 # PW_args
        l1_structs_mem += 40 # DW_args
        l1_structs_mem += 8 # act_args
        l1_structs_mem += 16 # Skipconn_args
        l1_structs_mem += 20 # InstNorm_args
        l1_structs_mem += 2*4 # 2 pi_cl_dma_cmd_t
        l1_structs_mem += 2 # loss in fp16
        if data_type_l[0] == 'FP32':
            l1_structs_mem += 2 # loss in fp32
        l1_structs_mem += 2*16 # 2 vect_sum_args
        print(f"Size of structures in L1 (Single Buffer Mode): {l1_structs_mem} bytes")
        total_memory_occupation_bytes += l1_buff_size + l1_structs_mem

        # Label storage memory
        # Size of the last layer in bytes
        nbytes_label = 4
        if data_type_l[-1] == 'FP16':
            nbytes_label = 2
        h_out_net = np.floor((hin_l[-1] - hk_l[-1] + h_str_list[-1] + 2*h_pad_list[-1]) / h_str_list[-1])
        w_out_net = np.floor((win_l[-1] - wk_l[-1] + w_str_list[-1] + 2*w_pad_list[-1]) / w_str_list[-1])
        labels_mem = out_ch_l[-1] * h_out_net * w_out_net * nbytes_label
        print(f"Size of allocated memory for labels (Single Buffer Mode): {labels_mem} bytes")
        total_memory_occupation_bytes += labels_mem

    elif USE_DMA == 'DB':
        
        # MAX_LAYER_DIM, MAX_LAYER_DIM_BYTES = utilsDB.max_layer_dim(layers_l, in_ch_l, hin_l, win_l, out_ch_l, hk_l, wk_l, data_type_l[0], h_str_list, w_str_list, h_pad_list, w_pad_list)
        # l1_buff_size = MAX_LAYER_DIM
        
        # l1_structs_mem = 0
        # l1_structs_mem += 7*(6*4) # 4 blobs input_blob, output_blob ..
        # l1_structs_mem += 32 # linear_args
        # l1_structs_mem += 76 # conv2d_args
        # l1_structs_mem += 40 # PW_args
        # l1_structs_mem += 40 # DW_args
        # l1_structs_mem += 8 # act_args
        # l1_structs_mem += 16 # Skipconn_args
        # l1_structs_mem += 3*4 # 3 pi_cl_dma_cmd_t cmd_load, cmd_store and cmd_struct
        # l1_structs_mem += 2 # loss in fp16
        # if data_type_l[0] == 'FP32':
        #     l1_structs_mem += 2 # loss in fp32
        # l1_structs_mem += 16 # vect_sum_args
        # print(f"Size of structures in L1 (Double Buffer Mode): {l1_structs_mem} bytes")
        # total_memory_occupation_bytes += l1_buff_size + l1_structs_mem

        pass

    if total_memory_occupation_bytes > avail_mem_bytes:
        print("[DNN_Size_Checker]: DNN overflows PULP L1 memory!!\nExpected occupation: {} bytes vs {} available L1 ({}%)!".format(total_memory_occupation_bytes, avail_mem_bytes, (total_memory_occupation_bytes/avail_mem_bytes)*100))
        #exit()

    if USE_DMA in ['SB', 'DB']:
        print(f"Total L2 memory occupation: {l2_occupation} bytes")
        MAX_LAYER_DIM = int(MAX_LAYER_DIM / MAX_LAYER_DIM_BYTES)

    return total_memory_occupation_bytes



def AdjustResConnList(sumnode_connections):
    res = []
    for layer in range(len(sumnode_connections)):
        if sumnode_connections[layer] == 0:
            res.append(-1)
        else:
            my_value = sumnode_connections[layer]
            for scanned_layer in range(len(sumnode_connections)):
                if sumnode_connections[scanned_layer] == my_value and layer != scanned_layer:
                    res.append(scanned_layer)
    return res

def CheckResConn(layer_list, in_ch_list, out_ch_list, hin_list, win_list, sumnode_connections, update_layer_l):
    # Check same number of Skipnodes and Sumnodes
    num_skip = 0
    num_sum = 0
    for layer in range(len(layer_list)): 
        if layer_list[layer] == 'Sumnode':
            num_sum += 1
        elif sumnode_connections[layer] != -1:
            num_skip += 1
        else:
            pass
    if num_skip != num_sum:
        print(f"Different number of Skipnode ({num_skip}) and Sumnode ({num_sum})\n")
        exit()


    for layer in range(len(layer_list)):
        if layer_list[layer] == 'Sumnode':
            if in_ch_list[layer] == out_ch_list[layer]:
                param = [in_ch_list[layer], hin_list[layer], win_list[layer]]
                layer_to_test = sumnode_connections[layer]
                if param != [in_ch_list[layer_to_test], hin_list[layer_to_test], win_list[layer_to_test]] and layer_list[layer_to_test] == 'Skipnode':
                    print(f"\nDifferent number of parameters between layers {layer}, {layer_to_test}\n")
                    exit()
            else:
                print(f"\nDifferent number of I/O Channels at layer {layer}\n")
                exit()


        
"""
The DNN Composer takes the lists representing the DNN graph and 
generates the code for PULP
"""
def DNN_Composer (proj_folder_path, project_name,
                  layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                  h_str_l, w_str_l, h_pad_l, w_pad_l,
                  epochs, batch_size, learning_rate, optimizer, loss_fn,
                  NUM_CORES, data_type_l, bias_l, update_layer_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list,
                  sumnode_connections, USE_DMA, PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS):

    # Initialize project (copy the prefab files and create folder)
    utils.InitProject(proj_folder_path)

    # Generate Makefile
    utils.GenerateMakefile(proj_folder_path, project_name, layers_l, NUM_CORES, data_type_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list)

    # Generate Golden Model
    utils.GenerateGM(proj_folder_path, project_name,
                        layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                        h_str_l, w_str_l, h_pad_l, w_pad_l,
                        epochs, batch_size, learning_rate, optimizer, loss_fn,
                        data_type_l, bias_l, update_layer_l, sumnode_connections, USE_DMA)


    global MAX_LAYER_DIM
    # Generate the net.c and net.h files to run the training in L1
    if USE_DMA == 'NO':
        utils.GenerateNet(proj_folder_path, project_name,
                    layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                    h_str_l, w_str_l, h_pad_l, w_pad_l,
                    epochs, batch_size, learning_rate, optimizer, loss_fn,
                    data_type_l, bias_l, update_layer_l, sumnode_connections,
                    PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)
        
    elif USE_DMA == 'SB':
        utilsSB.GenerateNet(proj_folder_path, project_name,
                    layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                    h_str_l, w_str_l, h_pad_l, w_pad_l,
                    epochs, batch_size, learning_rate, optimizer, loss_fn,
                    data_type_l, bias_l, update_layer_l, sumnode_connections, MAX_LAYER_DIM,
                    PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)
        
    elif USE_DMA == 'DB':
        print("[NOT_IMPLEMENTED_ERROR] Double Buffering not available, under revision!")
        exit()
        # utilsDB.GenerateNet(proj_folder_path, project_name,
        #             layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
        #             h_str_l, w_str_l, h_pad_l, w_pad_l,
        #             epochs, batch_size, learning_rate, optimizer, loss_fn,
        #             data_type_l, update_layer_l, sumnode_connections, MAX_LAYER_DIM,
        #             PROFILE_SINGLE_LAYERS, SEPARATE_BACKWARD_STEPS, CONV2D_USE_IM2COL, PRINT_TRAIN_LOSS)

    else:
        print(f"[DNN_Composer]: Not supported argument for USE_DMA: '{USE_DMA}' given")

    return
