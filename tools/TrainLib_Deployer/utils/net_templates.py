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
LAYER TEMPLATES
"""

def linear_template_FW(layer_number):
    template = "  pulp_linear_fp32_fw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, MATMUL_TYPE_FW_L"+str(layer_number)+");\n"
    return template

def linear_template_BW(layer_number, skip_in_grad):
    template = "  pulp_linear_fp32_bw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(skip_in_grad)+", MATMUL_TYPE_WG_L"+str(layer_number)+", MATMUL_TYPE_IG_L"+str(layer_number)+");\n"
    return template



def conv2d_template_FW(layer_number, pad, stride_w, stride_h):
    template = "  pulp_conv2d_fp32_fw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", "+str(stride_w)+", "+str(stride_h)+", im2col_buffer, MATMUL_TYPE_FW_L"+str(layer_number)+");\n"
    return template

def conv2d_template_BW(layer_number, pad, stride_w, stride_h, skip_in_grad):
    template = "  pulp_conv2d_fp32_bw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", "+str(stride_w)+", "+str(stride_h)+", im2col_buffer, bt_buffer, "+str(skip_in_grad)+", MATMUL_TYPE_WG_L"+str(layer_number)+", MATMUL_TYPE_IG_L"+str(layer_number)+");\n"
    return template



def DW_template_FW(layer_number, pad):
    template = "  pulp_conv_dw_fp32_fw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", im2col_buffer, MATMUL_TYPE_FW_L"+str(layer_number)+");\n"
    return template

def DW_template_BW(layer_number, pad, skip_in_grad):
    template = "  pulp_conv_dw_fp32_bw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", im2col_buffer, "+str(skip_in_grad)+", MATMUL_TYPE_WG_L"+str(layer_number)+", MATMUL_TYPE_IG_L"+str(layer_number)+");\n"
    return template



def PW_template_FW(layer_number, pad):
    template = "  pulp_conv_pw_fp32_fw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", MATMUL_TYPE_FW_L"+str(layer_number)+");\n"
    return template

def PW_template_BW(layer_number, pad, skip_in_grad):
    template = "  pulp_conv_pw_fp32_bw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", "+str(skip_in_grad)+", MATMUL_TYPE_WG_L"+str(layer_number)+", MATMUL_TYPE_IG_L"+str(layer_number)+");\n"
    return template




"""
ACTIVATIONS TEMPLATES
"""

def ReLU_template_FW(layer_number):
    template = "  pulp_relu_fp32_fw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_out);\n"
    return template

def ReLU_template_BW(layer_number):
    template = "  pulp_relu_fp32_bw_cl(&layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_out);\n"
    return template


"""
POOLING TEMPLATES
"""

def AvgPool_template_FW(layer_number):
    template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    return template

def AvgPool_template_BW(layer_number):
    template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    return template


def MaxPool_template_FW(layer_number):
    template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    return template

def MaxPool_template_BW(layer_number):
    template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    return template
