"""
Copyright (C) 2021-2025 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Authors: Davide Nadalini, Calin Diaconu

LAYER TEMPLATES
"""


def linear_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_linear_fp32_fw_cl(&linear_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_linear_fp16_fw_cl(&linear_args);\n"
    else:
        print("[net_templates.linear_template_FW]: Invalid data type!")
        exit()
    return template


def linear_template_coeff_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_linear_fp32_bw_param_grads_cl(&linear_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_linear_fp16_bw_param_grads_cl(&linear_args);\n"
    else:
        print("[net_templates.linear_template_BW]: Invalid data type!")
        exit()
    return template


def linear_template_in_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_linear_fp32_bw_input_grads_cl(&linear_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_linear_fp16_bw_input_grads_cl(&linear_args);\n"
    else:
        print("[net_templates.linear_template_BW]: Invalid data type!")
        exit()
    return template


def conv2d_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv2d_fp32_fw_cl(&conv2d_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv2d_fp16_fw_cl(&conv2d_args);\n"
    else:
        print("[net_templates.conv2d_template_FW]: Invalid data type!")
        exit()
    return template


def conv2d_template_coeff_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv2d_fp32_bw_param_grads_cl(&conv2d_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv2d_fp16_bw_param_grads_cl(&conv2d_args);\n"
    else:
        print("[net_templates.conv2d_template_BW]: Invalid data type!")
        exit()
    return template


def conv2d_template_in_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv2d_fp32_bw_input_grads_cl(&conv2d_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv2d_fp16_bw_input_grads_cl(&conv2d_args);\n"
    else:
        print("[net_templates.conv2d_template_BW]: Invalid data type!")
        exit()
    return template


def DW_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_dw_fp32_fw_cl(&DW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_dw_fp16_fw_cl(&DW_args);\n"
    else:
        print("[net_templates.DW_template_FW]: Invalid data type!")
        exit()
    return template


def DW_template_coeff_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_dw_fp32_bw_param_grads_cl(&DW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_dw_fp16_bw_param_grads_cl(&DW_args);\n"
    else:
        print("[net_templates.DW_template_BW]: Invalid data type!")
        exit()
    return template


def DW_template_in_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_dw_fp32_bw_input_grads_cl(&DW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_dw_fp16_bw_input_grads_cl(&DW_args);\n"
    else:
        print("[net_templates.DW_template_BW]: Invalid data type!")
        exit()
    return template


def PW_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_pw_fp32_fw_cl(&PW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_pw_fp16_fw_cl(&PW_args);\n"
    else:
        print("[net_templates.PW_template_FW]: Invalid data type!")
        exit()
    return template


def PW_template_coeff_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_pw_fp32_bw_param_grads_cl(&PW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_pw_fp16_bw_param_grads_cl(&PW_args);\n"
    else:
        print("[net_templates.PW_template_BW]: Invalid data type!")
        exit()
    return template


def PW_template_in_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_conv_pw_fp32_bw_input_grads_cl(&PW_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_conv_pw_fp16_bw_input_grads_cl(&PW_args);\n"
    else:
        print("[net_templates.PW_template_BW]: Invalid data type!")
        exit()
    return template


"""
RESIDUAL CONNECTIONS TEMPLATE
"""


def residualconn_template_FW(layer_number, DATA_TYPE):
    template = "\tresconn_args.skip = &wgt;\n"
    template += "\tresconn_args.output = &out;\n"
    template += "\tresconn_args.lout = &in;\n"
    if DATA_TYPE == "FP32":
        template += "\tpulp_residualconn_fp32_fw(&resconn_args);\n"
    elif DATA_TYPE == "FP16":
        template += "\tpulp_residualconn_fp16_fw(&resconn_args);\n"
    else:
        print("[net_templates.residualconn_template_FW]: Invalid data type!")
        exit()
    return template


def residualconn_template_copy_BW(layer_number, DATA_TYPE):
    template = "\tresconn_args.skip = &wgt;\n"
    template += "\tresconn_args.output = &out;\n"
    template += "\tresconn_args.lout = &in;\n"
    if DATA_TYPE == "FP32":
        template += "\tpulp_residualconn_fp32_bw(&resconn_args);\n"
    elif DATA_TYPE == "FP16":
        template += "\tpulp_residualconn_fp16_bw(&resconn_args);\n"
    else:
        print("[net_templates.residualconn_template_copy_BW]: Invalid data type!")
        exit()
    return template


def residualconn_template_sum_BW(layer_number, DATA_TYPE, target):
    template = "\tresconn_args.skip = &in;\n"
    template += "\tresconn_args.output = &out;\n"
    template += "\tresconn_args.lout = &wgt;\n"
    template += f"\tload_input(&layer{target}_in, 0);\n"
    if DATA_TYPE == "FP32":
        template += "\tpulp_sumnode_fp32_bw(&resconn_args);\n"
    elif DATA_TYPE == "FP16":
        template += "\tpulp_sumnode_fp16_bw(&resconn_args);\n"
    else:
        print("[net_templates.residualconn_template_sum_BW]: Invalid data type!")
        exit()
    return template


"""
ACTIVATIONS TEMPLATES
"""


def ReLU_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_relu_fp32_fw_cl(&act_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_relu_fp16_fw_cl(&act_args);\n"
    else:
        print("[net_templates.ReLU_template_FW]: Invalid data type!")
        exit()
    return template


def ReLU_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = "\tpulp_relu_fp32_bw_cl(&act_args);\n"
    elif DATA_TYPE == "FP16":
        template = "\tpulp_relu_fp16_bw_cl(&act_args);\n"
    else:
        print("[net_templates.ReLU_template_BW]: Invalid data type!")
        exit()
    return template


"""
POOLING TEMPLATES
"""


def AvgPool_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_fw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    elif DATA_TYPE == "FP16":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_fw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    else:
        print("[net_templates.AvgPool_template_FW]: Invalid data type!")
        exit()
    return template


def AvgPool_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_avgpool_fp32_bw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    elif DATA_TYPE == "FP16":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_avgpool_fp16_bw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    else:
        print("[net_templates.AvgPool_template_BW]: Invalid data type!")
        exit()
    return template


def MaxPool_template_FW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_fw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    elif DATA_TYPE == "FP16":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_maxpool_fp16_fw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    else:
        print("[net_templates.MaxPool_template_FW]: Invalid data type!")
        exit()
    return template


def MaxPool_template_BW(layer_number, DATA_TYPE):
    if DATA_TYPE == "FP32":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_maxpool_fp32_bw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    elif DATA_TYPE == "FP16":
        template = (
            "\tpi_cl_team_fork(NUM_CORES, pulp_maxpool_fp16_bw_cl, &l"
            + str(layer_number)
            + "_args);\n"
        )
    else:
        print("[net_templates.MaxPool_template_BW]: Invalid data type!")
        exit()
    return template


"""
NORM TEMPLATES
"""


def InstNorm_template_FW(layer_number, data_type):
    if data_type == "FP32":
        template = "\tpulp_instnorm_fp32_fw_cl(&InstNorm_args);\n"
    elif data_type == "FP16":
        template = "\tpulp_instnorm_fp16_fw_cl(&InstNorm_args);\n"
    else:
        print("[net_templates.InstNorm_template_FW]: Invalid data type!")
        exit()
    return template


def InstNorm_template_BW(layer_number, data_type):
    if data_type == "FP32":
        template = "\tpulp_instnorm_fp32_bw_cl(&InstNorm_args);\n"
    elif data_type == "FP16":
        template = "\tpulp_instnorm_fp16_bw_cl(&InstNorm_args);\n"
    else:
        print("[net_templates.InstNorm_template_BW]: Invalid data type!")
        exit()
    return template


"""
TYPE CHANGE TEMPLATES
"""


def cast_fp32_to_fp16_template(layer_number, STEP, DATA_TYPE):
    if STEP == "FW":
        template = "  // Propagate FP32 layer " + str(layer_number) + " to FP16\n"
        template += "  struct cast_32t16_args cast_l" + str(layer_number) + "_args;\n"
        template += (
            "  cast_l" + str(layer_number) + "_args.source = (float*) cast_buffer;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.destination = l"
            + str(layer_number + 1)
            + "_in;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.size = Tout_C_l"
            + str(layer_number)
            + " * Tout_H_l"
            + str(layer_number)
            + " * Tout_W_l"
            + str(layer_number)
            + ";\n"
        )
        template += (
            "  pi_cl_team_fork(NUM_CORES, cast_fp32_tensor_to_fp16, &cast_l"
            + str(layer_number)
            + "_args);\n"
        )
        template += "  // End of casting\n"
    elif STEP == "BW":
        template = "  // Propagate FP32 layer " + str(layer_number) + " back to FP16\n"
        template += "  struct cast_32t16_args cast_l" + str(layer_number) + "_args;\n"
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.source = l"
            + str(layer_number)
            + "_in_diff;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.destination = (fp16*) cast_buffer;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.size = Tin_C_l"
            + str(layer_number)
            + " * Tin_H_l"
            + str(layer_number)
            + " * Tin_W_l"
            + str(layer_number)
            + ";\n"
        )
        template += (
            "  pi_cl_team_fork(NUM_CORES, cast_fp32_tensor_to_fp16, &cast_l"
            + str(layer_number)
            + "_args);\n"
        )
        template += "  // End of casting\n"
    else:
        print(
            "[net_templates.cast_fp32_to_fp16_template]: Invalid training step for template generation @layer{}!".format(
                layer_number
            )
        )
    return template


def cast_fp16_to_fp32_template(layer_number, STEP, DATA_TYPE):
    if STEP == "FW":
        template = "  // Propagate FP16 layer " + str(layer_number) + " to FP32\n"
        template += "  struct cast_16t32_args cast_l" + str(layer_number) + "_args;\n"
        template += (
            "  cast_l" + str(layer_number) + "_args.source = (fp16*) cast_buffer;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.destination = l"
            + str(layer_number + 1)
            + "_in;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.size = Tout_C_l"
            + str(layer_number)
            + " * Tout_H_l"
            + str(layer_number)
            + " * Tout_W_l"
            + str(layer_number)
            + ";\n"
        )
        template += (
            "  pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &cast_l"
            + str(layer_number)
            + "_args);\n"
        )
        template += "  // End of casting\n"
    elif STEP == "BW":
        template = "  // Propagate FP16 layer " + str(layer_number) + " back to FP32\n"
        template += "  struct cast_16t32_args cast_l" + str(layer_number) + "_args;\n"
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.source = l"
            + str(layer_number)
            + "_in_diff;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.destination = (float*) cast_buffer;\n"
        )
        template += (
            "  cast_l"
            + str(layer_number)
            + "_args.size = Tin_C_l"
            + str(layer_number)
            + " * Tin_H_l"
            + str(layer_number)
            + " * Tin_W_l"
            + str(layer_number)
            + ";\n"
        )
        template += (
            "  pi_cl_team_fork(NUM_CORES, cast_fp16_tensor_to_fp32, &cast_l"
            + str(layer_number)
            + "_args);\n"
        )
        template += "  // End of casting\n"
    else:
        print(
            "[net_templates.cast_fp32_to_fp16_template]: Invalid training step for template generation @layer{}!".format(
                layer_number
            )
        )
    return template


"""
CONFIGURATION STRUCTURE TEMPLATES
"""


def linear_config_template(layer_number, skip_in_grad, DATA_TYPE, update_layer):
    skip_wg_grad = 0
    if update_layer == 0:
        skip_wg_grad = 1
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.coeff = &wgt;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    template += (
        "  l" + str(layer_number) + "_args.skip_wg_grad = " + str(skip_wg_grad) + ";\n"
    )
    template += (
        "  l" + str(layer_number) + "_args.skip_in_grad = " + str(skip_in_grad) + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"
        + str(layer_number)
        + ";\n"
    )
    return template


def conv2d_config_template(
    layer_number,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    skip_in_grad,
    DATA_TYPE,
    use_bias,
    CONV2D_USE_IM2COL,
    update_layer,
):
    skip_wg_grad = 0
    if update_layer == 0:
        skip_wg_grad = 1
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.coeff = &wgt;\n"
    if use_bias:
        template += "  l" + str(layer_number) + "_args.bias = &bias;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    template += (
        "  l" + str(layer_number) + "_args.skip_wg_grad = " + str(skip_wg_grad) + ";\n"
    )
    template += (
        "  l" + str(layer_number) + "_args.skip_in_grad = " + str(skip_in_grad) + ";\n"
    )
    template += "  l" + str(layer_number) + "_args.Lpad = " + str(pad_w) + ";\n"
    template += "  l" + str(layer_number) + "_args.Rpad = " + str(pad_w) + ";\n"
    template += "  l" + str(layer_number) + "_args.Upad = " + str(pad_h) + ";\n"
    template += "  l" + str(layer_number) + "_args.Dpad = " + str(pad_h) + ";\n"
    template += "  l" + str(layer_number) + "_args.stride_h = " + str(stride_h) + ";\n"
    template += "  l" + str(layer_number) + "_args.stride_w = " + str(stride_w) + ";\n"
    if DATA_TYPE == "FP32":
        template += (
            "  l" + str(layer_number) + "_args.i2c_buffer = (float*) im2col_buffer;\n"
        )
        template += (
            "  l" + str(layer_number) + "_args.bt_buffer = (float*) bt_buffer;\n"
        )
    elif DATA_TYPE == "FP16":
        template += (
            "  l" + str(layer_number) + "_args.i2c_buffer = (fp16*) im2col_buffer;\n"
        )
        template += "  l" + str(layer_number) + "_args.bt_buffer = (fp16*) bt_buffer;\n"
    else:
        print("[net_templates.conv2d_config_template]: Invalid data type!")
        exit()
    template += "  l" + str(layer_number) + "_args.HWC = 0;\n"
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"
        + str(layer_number)
        + ";\n"
    )
    # Temporary fix to use padding/stride (naive kernel)
    if (pad_h > 0 or pad_w > 0) or (stride_h > 1 or stride_w > 1):
        template += "  l" + str(layer_number) + "_args.USE_IM2COL = 0;\n"
    else:
        template += (
            "  l"
            + str(layer_number)
            + "_args.USE_IM2COL = "
            + str(CONV2D_USE_IM2COL)
            + ";\n"
        )
    template += "  l" + str(layer_number) + "_args.USE_DMA_IM2COL = 0;\n"
    return template


def DW_config_template(
    layer_number,
    pad_h,
    pad_w,
    stride_h,
    stride_w,
    skip_in_grad,
    DATA_TYPE,
    update_layer,
):
    skip_wg_grad = 0
    if update_layer == 0:
        skip_wg_grad = 1
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.coeff = &wgt;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    template += (
        "  l" + str(layer_number) + "_args.skip_wg_grad = " + str(skip_wg_grad) + ";\n"
    )
    template += (
        "  l" + str(layer_number) + "_args.skip_in_grad = " + str(skip_in_grad) + ";\n"
    )
    template += "  l" + str(layer_number) + "_args.Lpad = " + str(pad_w) + ";\n"
    template += "  l" + str(layer_number) + "_args.Rpad = " + str(pad_w) + ";\n"
    template += "  l" + str(layer_number) + "_args.Upad = " + str(pad_h) + ";\n"
    template += "  l" + str(layer_number) + "_args.Dpad = " + str(pad_h) + ";\n"
    # if DATA_TYPE == 'FP32':
    #    template += "  l"+str(layer_number)+"_args.i2c_buffer = (float*) im2col_buffer;\n"
    # elif DATA_TYPE == 'FP16':
    #    template += "  l"+str(layer_number)+"_args.i2c_buffer = (fp16*) im2col_buffer;\n"
    # else:
    #    print("[net_templates.DW_config_template]: Invalid data type!")
    #    exit()
    template += "  l" + str(layer_number) + "_args.HWC = 0;\n"
    # template += "  l"+str(layer_number)+"_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"+str(layer_number)+";\n"
    # template += "  l"+str(layer_number)+"_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"+str(layer_number)+";\n"
    # template += "  l"+str(layer_number)+"_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"+str(layer_number)+";\n"
    return template


def PW_config_template(layer_number, skip_in_grad, DATA_TYPE, update_layer):
    skip_wg_grad = 0
    if update_layer == 0:
        skip_wg_grad = 1
    # &layer"+str(layer_number)+"_in, &layer"+str(layer_number)+"_wgt, &layer"+str(layer_number)+"_out, "+str(pad)+", MATMUL_TYPE_FW_L"+str(layer_number)+"
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.coeff = &wgt;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    if DATA_TYPE == "FP32":
        template += (
            "  l" + str(layer_number) + "_args.transpose_buffer = (float*) bt_buffer;\n"
        )
    elif DATA_TYPE == "FP16":
        template += (
            "  l" + str(layer_number) + "_args.transpose_buffer = (fp16*) bt_buffer;\n"
        )
    else:
        print("[net_templates.PW_config_template]: Invalid data type!")
        exit()
    template += (
        "  l" + str(layer_number) + "_args.skip_wg_grad = " + str(skip_wg_grad) + ";\n"
    )
    template += (
        "  l" + str(layer_number) + "_args.skip_in_grad = " + str(skip_in_grad) + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L"
        + str(layer_number)
        + ";\n"
    )
    template += (
        "  l"
        + str(layer_number)
        + "_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L"
        + str(layer_number)
        + ";\n"
    )
    template += "  l" + str(layer_number) + "_args.HWC = 0;\n"
    return template


def ReLU_config_template(layer_number, DATA_TYPE):
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    return template


def resconn_config_template(layer_number, skip_node, skip_input):
    template = "  l" + str(layer_number) + "_args.lout = &in;\n"
    template += "  l" + str(layer_number) + "_args.skip = &wgt;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    if skip_input:
        template += f"  l{layer_number}_args.skip_in_grad = 1;\n"
    else:
        template += f"  l{layer_number}_args.skip_in_grad = 0;\n"
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


def sum(layer, is_skipnode, current, output, data_type):
    template = "\tint dims[] = { d" + output + "_blob.dim };"

    if is_skipnode:
        template += f"\tvect_sum_args.op_1 = d{current}_blob.diff;\n"
    else:
        template += f"\tvect_sum_args.op_1 = in.diff;\n"

    template += f"\tvect_sum_args.op_2 = d{output}_blob.diff;\n"
    template += f"\tvect_sum_args.dest = d{output}_blob.diff;\n"

    template += "\tvect_sum_args.op_1_dims = dims;\n"
    template += "\tvect_sum_args.op_2_dims = dims;\n"

    template += "\tvect_sum_args.op_1_dims_len = 1;\n"
    template += "\tvect_sum_args.op_2_dims_len = 1;\n"

    if data_type == "FP32":
        template += (
            "\tpi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &vect_sum_args);\n"
        )
    elif data_type == "FP16":
        template += (
            "\tpi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp16, &vect_sum_args);\n"
        )
    else:
        print("\n[net_templates.py - sum] Invalid Data Type\n")
        exit()

    return template


def InstNorm_config_template(layer_number, skip_in_grad):
    template = "  l" + str(layer_number) + "_args.input = &in;\n"
    template += "  l" + str(layer_number) + "_args.coeff = &wgt;\n"
    template += "  l" + str(layer_number) + "_args.output = &out;\n"
    template += (
        "  l" + str(layer_number) + "_args.skip_in_grad = " + str(skip_in_grad) + ";\n"
    )
    return template
