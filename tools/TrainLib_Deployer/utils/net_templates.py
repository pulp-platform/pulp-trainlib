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

def linear_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_linear_fp32_fw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_linear_fp16_fw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.linear_template_FW]: Invalid data type!")
        exit()
    return template

def linear_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_linear_fp32_bw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_linear_fp16_bw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.linear_template_BW]: Invalid data type!")
        exit()
    return template



def conv2d_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv2d_fp32_fw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv2d_fp16_fw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.conv2d_template_FW]: Invalid data type!")
        exit()    
    return template

def conv2d_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv2d_fp32_bw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv2d_fp16_bw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.conv2d_template_BW]: Invalid data type!")
        exit()  
    return template



def DW_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv_dw_fp32_fw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv_dw_fp16_fw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.DW_template_FW]: Invalid data type!")
        exit()  
    return template

def DW_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv_dw_fp32_bw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv_dw_fp16_bw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.DW_template_BW]: Invalid data type!")
        exit()  
    return template



def PW_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv_pw_fp32_fw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv_pw_fp16_fw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.PW_template_FW]: Invalid data type!")
        exit()  
    return template

def PW_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_conv_pw_fp32_bw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_conv_pw_fp16_bw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.PW_template_BW]: Invalid data type!")
        exit()  
    return template




"""
ACTIVATIONS TEMPLATES
"""

def ReLU_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_relu_fp32_fw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_relu_fp16_fw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.ReLU_template_FW]: Invalid data type!")
        exit()  
    return template

def ReLU_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pulp_relu_fp32_bw_cl(&l"+str(layer_number)+"_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pulp_relu_fp16_bw_cl(&l"+str(layer_number)+"_args);\n"
    else:
        print("[net_templates.ReLU_template_BW]: Invalid data type!")
        exit()  
    return template


"""
POOLING TEMPLATES
"""

def AvgPool_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    else:
        print("[net_templates.AvgPool_template_FW]: Invalid data type!")
        exit()  
    return template

def AvgPool_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    else:
        print("[net_templates.AvgPool_template_BW]: Invalid data type!")
        exit()  
    return template


def MaxPool_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp16_fw_cl, &l"+str(layer_number)+"_pool_args);\n"
    else:
        print("[net_templates.MaxPool_template_FW]: Invalid data type!")
        exit()  
    return template

def MaxPool_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == 'FP32':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    elif DATA_TYPE == 'FP16':
        template = "  pi_cl_team_fork(NUM_CORES, pulp_maxpool_fp16_bw_cl, &l"+str(layer_number)+"_pool_args);\n"
    else:
        print("[net_templates.MaxPool_template_BW]: Invalid data type!")
        exit()  
    return template


"""
TYPE CHANGE TEMPLATES
"""

def cast_fp32_to_fp16_template (layer_number, STEP, DATA_TYPE):
    if STEP == 'FW':
        template =  "  // Propagate FP32 layer "+str(layer_number)+" to FP16\n"
        template += "  struct cast_32t16_args cast_l"+str(layer_number)+"_args;\n"
        template += "  cast_l"+str(layer_number)+"_args.source = (float*) cast_buffer;\n"
        template += "  cast_l"+str(layer_number)+"_args.destination = l"+str(layer_number+1)+"_in;\n"  
        template += "  cast_l"+str(layer_number)+"_args.size = Tout_C_l"+str(layer_number)+" * Tout_H_l"+str(layer_number)+" * Tout_W_l"+str(layer_number)+";\n"
        template += "  pi_cl_team_fork(NUM_CORES, cast_fp32_tensor_to_fp16, &cast_l"+str(layer_number)+"_args);\n"
        template += "  // End of casting\n"
    elif STEP == 'BW':
        template =  "  // Propagate FP32 layer "+str(layer_number)+" back to FP16\n"
        template += "  struct cast_32t16_args cast_l"+str(layer_number)+"_args;\n"
        template += "  cast_l"+str(layer_number)+"_args.source = l"+str(layer_number)+"_in_diff;\n"
        template += "  cast_l"+str(layer_number)+"_args.destination = (fp16*) cast_buffer;\n"  
        template += "  cast_l"+str(layer_number)+"_args.size = Tin_C_l"+str(layer_number)+" * Tin_H_l"+str(layer_number)+" * Tin_W_l"+str(layer_number)+";\n"
        template += "  pi_cl_team_fork(NUM_CORES, cast_fp32_tensor_to_fp16, &cast_l"+str(layer_number)+"_args);\n"
        template += "  // End of casting\n"
    else:
        print("[net_templates.cast_fp32_to_fp16_template]: Invalid training step for template generation @layer{}!".format(layer_number))
    return template

def cast_fp16_to_fp32_template (layer_number, STEP, DATA_TYPE):
    if STEP == 'FW':
        template =  "  // Propagate FP16 layer "+str(layer_number)+" to FP32\n"
        template += "  struct cast_16t32_args cast_l"+str(layer_number)+"_args;\n"
        template += "  cast_l"+str(layer_number)+"_args.source = (fp16*) cast_buffer;\n"
        template += "  cast_l"+str(layer_number)+"_args.destination = l"+str(layer_number+1)+"_in;\n" 
        template += "  cast_l"+str(layer_number)+"_args.size = Tout_C_l"+str(layer_number)+" * Tout_H_l"+str(layer_number)+" * Tout_W_l"+str(layer_number)+";\n" 
        template += "  pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &cast_l"+str(layer_number)+"_args);\n"
        template += "  // End of casting\n"
    elif STEP == 'BW':
        template =  "  // Propagate FP16 layer "+str(layer_number)+" back to FP32\n"
        template += "  struct cast_16t32_args cast_l"+str(layer_number)+"_args;\n"
        template += "  cast_l"+str(layer_number)+"_args.source = l"+str(layer_number)+"_in_diff;\n"
        template += "  cast_l"+str(layer_number)+"_args.destination = (float*) cast_buffer;\n"
        template += "  cast_l"+str(layer_number)+"_args.size = Tin_C_l"+str(layer_number)+" * Tin_H_l"+str(layer_number)+" * Tin_W_l"+str(layer_number)+";\n"  
        template += "  pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &cast_l"+str(layer_number)+"_args);\n"
        template += "  // End of casting\n"
    else:
        print("[net_templates.cast_fp32_to_fp16_template]: Invalid training step for template generation @layer{}!".format(layer_number))
    return template





"""
CONFIGURATION STRUCTURE TEMPLATES
"""

def linear_config_template(layer_number, skip_in_grad, DATA_TYPE):
    template  = "  l"+str(layer_number)+"_args.input = &layer"+str(layer_number)+"_in;\n"
    template += "  l"+str(layer_number)+"_args.coeff = &layer"+str(layer_number)+"_wgt;\n"
    template += "  l"+str(layer_number)+"_args.output = &layer"+str(layer_number)+"_out;\n"
    template += "  l"+str(layer_number)+"_args.skip_in_grad = "+str(skip_in_grad)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"+str(layer_number)+";\n"
    return template

def conv2d_config_template(layer_number, pad_h, pad_w, stride_h, stride_w, skip_in_grad, DATA_TYPE):
    template  = "  l"+str(layer_number)+"_args.input = &layer"+str(layer_number)+"_in;\n"
    template += "  l"+str(layer_number)+"_args.coeff = &layer"+str(layer_number)+"_wgt;\n"
    template += "  l"+str(layer_number)+"_args.output = &layer"+str(layer_number)+"_out;\n"
    template += "  l"+str(layer_number)+"_args.skip_in_grad = "+str(skip_in_grad)+";\n"
    template += "  l"+str(layer_number)+"_args.Lpad = "+str(pad_w)+";\n"
    template += "  l"+str(layer_number)+"_args.Rpad = "+str(pad_w)+";\n"
    template += "  l"+str(layer_number)+"_args.Upad = "+str(pad_h)+";\n"
    template += "  l"+str(layer_number)+"_args.Dpad = "+str(pad_h)+";\n"
    template += "  l"+str(layer_number)+"_args.stride_h = "+str(stride_h)+";\n"
    template += "  l"+str(layer_number)+"_args.stride_w = "+str(stride_w)+";\n"
    if DATA_TYPE == 'FP32':
        template += "  l"+str(layer_number)+"_args.i2c_buffer = (float*) im2col_buffer;\n"
        template += "  l"+str(layer_number)+"_args.bt_buffer = (float*) bt_buffer;\n"
    elif DATA_TYPE == 'FP16':
        template += "  l"+str(layer_number)+"_args.i2c_buffer = (fp16*) im2col_buffer;\n"
        template += "  l"+str(layer_number)+"_args.bt_buffer = (fp16*) bt_buffer;\n"
    else:
        print("[net_templates.conv2d_config_template]: Invalid data type!")
        exit()     
    template += "  l"+str(layer_number)+"_args.HWC = 0;\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.USE_IM2COL = 1;\n"
    template += "  l"+str(layer_number)+"_args.USE_DMA_IM2COL = 0;\n"
    return template

def DW_config_template(layer_number, pad_h, pad_w, stride_h, stride_w, skip_in_grad, DATA_TYPE):
    template  = "  l"+str(layer_number)+"_args.input = &layer"+str(layer_number)+"_in;\n"
    template += "  l"+str(layer_number)+"_args.coeff = &layer"+str(layer_number)+"_wgt;\n"
    template += "  l"+str(layer_number)+"_args.output = &layer"+str(layer_number)+"_out;\n"
    template += "  l"+str(layer_number)+"_args.skip_in_grad = "+str(skip_in_grad)+";\n"
    template += "  l"+str(layer_number)+"_args.Lpad = "+str(pad_w)+";\n"
    template += "  l"+str(layer_number)+"_args.Rpad = "+str(pad_w)+";\n"
    template += "  l"+str(layer_number)+"_args.Upad = "+str(pad_h)+";\n"
    template += "  l"+str(layer_number)+"_args.Dpad = "+str(pad_h)+";\n"
    if DATA_TYPE == 'FP32':
        template += "  l"+str(layer_number)+"_args.i2c_buffer = (float*) im2col_buffer;\n"
    elif DATA_TYPE == 'FP16':
        template += "  l"+str(layer_number)+"_args.i2c_buffer = (fp16*) im2col_buffer;\n"
    else:
        print("[net_templates.DW_config_template]: Invalid data type!")
        exit()
    template += "  l"+str(layer_number)+"_args.HWC = 0;\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"+str(layer_number)+";\n"
    return template

def PW_config_template(layer_number, skip_in_grad, DATA_TYPE):
    # &layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", MATMUL_TYPE_FW_L"+str(layer_number)+"
    template  = "  l"+str(layer_number)+"_args.input = &layer"+str(layer_number)+"_in;\n"
    template += "  l"+str(layer_number)+"_args.coeff = &layer"+str(layer_number)+"_wgt;\n"
    template += "  l"+str(layer_number)+"_args.output = &layer"+str(layer_number)+"_out;\n"
    template += "  l"+str(layer_number)+"_args.skip_in_grad = "+str(skip_in_grad)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"+str(layer_number)+";\n"
    template += "  l"+str(layer_number)+"_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"+str(layer_number)+";\n"
    return template

def ReLU_config_template(layer_number, DATA_TYPE):
    template  = "  l"+str(layer_number)+"_args.input = &layer"+str(layer_number)+"_in;\n"
    template += "  l"+str(layer_number)+"_args.output = &layer"+str(layer_number)+"_out;\n"
    return template

# def MaxPool_config_template(layer_number):
#     template  = "  l"+str(layer_number)+"_args. ;\n"
#     template += "  l"+str(layer_number)+"_args. ;\n"
#     template += "  l"+str(layer_number)+"_args. ;\n"
#     template += "  l"+str(layer_number)+"_args. ;\n"
#     template += "  l"+str(layer_number)+"_args. ;\n"
#     template += "  l"+str(layer_number)+"_args. ;\n"
#     return template

# def AvgPool_config_template(layer_number):
#     template = "  "
#     return template
