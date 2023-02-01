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

import utils.deployment_utils as utils

"""
The DNN Size Checker checks if the DNN fits the available PULP
memory
"""

def DNN_Size_Checker (layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_str_list, w_str_list, h_pad_list, w_pad_list,
                        data_type_l, avail_mem_bytes):

    total_memory_occupation_bytes = 0

    # Compute activation and weight memory occupation
    for layer in range(len(layers_l)):
        is_last_layer = False
        if layer == len(layers_l) - 1:
            is_last_layer = True
        total_memory_occupation_bytes += utils.compute_wgt_act_memocc_bytes(layer, layers_l[layer], in_ch_l[layer], out_ch_l[layer], hk_l[layer], wk_l[layer], hin_l[layer], win_l[layer], h_pad_list[layer], w_pad_list[layer], h_str_list[layer], w_str_list[layer], data_type_l[layer], is_last_layer)

    # Compute im2col memory occupation
    mem_im2col = 0
    idx_im2col = 0
    mem_im2col, idx_im2col = utils.compute_im2col_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_pad_list, w_pad_list, h_str_list, w_str_list, data_type_l)
    total_memory_occupation_bytes += mem_im2col

    if mem_im2col > 0:
        print("Max IM2COL size of {} bytes @layer {}".format(mem_im2col, idx_im2col))

    # Compute blocktranspose memory occupation 
    mem_blocktransp = 0
    idx_blocktransp = 0
    mem_blocktransp, idx_blocktransp = utils.compute_bt_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, data_type_l)
    total_memory_occupation_bytes += mem_blocktransp

    if mem_blocktransp > 0:
        print("Max conv2d block transposition buffer size of {} @layer {}".format(mem_blocktransp, idx_blocktransp))

    # Compute additional mixed precision buffer memory occupation
    mem_cast_buffer = 0
    mem_cast_buffer, idx_max_act, max_act_inout = utils.compute_cast_buffer_memocc_bytes(layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l, h_pad_list, w_pad_list, h_str_list, w_str_list, data_type_l)
    total_memory_occupation_bytes += mem_cast_buffer

    #if mem_cast_buffer > 0:
    print("Additional {} bytes allocated for mixed precision management (size @layer {}, {})".format(mem_cast_buffer, idx_max_act, max_act_inout))

    if total_memory_occupation_bytes > avail_mem_bytes:
        print("[DNN_Size_Checker]: DNN overflows PULP L1 memory!!\nExpected occupation: {} bytes vs {} available L1 ({}%)!".format(total_memory_occupation_bytes, avail_mem_bytes, (total_memory_occupation_bytes/avail_mem_bytes)*100))
        exit()

    return total_memory_occupation_bytes


"""
The DNN Composer takes the lists representing the DNN graph and 
generates the code for PULP
"""
def DNN_Composer (proj_folder_path, project_name,
                  layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                  h_str_l, w_str_l, h_pad_l, w_pad_l,
                  epochs, batch_size, learning_rate, optimizer, loss_fn,
                  NUM_CORES, data_type_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list):

    # Initialize project (copy the prefab files and create folder)
    utils.InitProject(proj_folder_path)

    # Generate Makefile
    utils.GenerateMakefile(proj_folder_path, project_name, layers_l, NUM_CORES, data_type_l, opt_mm_fw_list, opt_mm_wg_list, opt_mm_ig_list)

    # Generate Golden Model
    utils.GenerateGM(proj_folder_path, project_name,
                        layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                        h_str_l, w_str_l, h_pad_l, w_pad_l,
                        epochs, batch_size, learning_rate, optimizer, loss_fn,
                        data_type_l)

    # Generate the net.c and net.h files to run the training in L1 (for now)
    utils.GenerateNet(proj_folder_path, project_name,
                layers_l, in_ch_l, out_ch_l, hk_l, wk_l, hin_l, win_l,
                h_str_l, w_str_l, h_pad_l, w_pad_l,
                epochs, batch_size, learning_rate, optimizer, loss_fn,
                data_type_l)

    return