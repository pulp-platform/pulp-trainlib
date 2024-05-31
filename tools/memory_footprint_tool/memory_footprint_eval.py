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
Evaluate the memory footprint of the current trainlib
"""

# =====>    USER SETTING    <=====
# This section is available for the user to set the data type and 
# sizes of the available layers

# Data type (fp32, fp16, bf16, fp8)
data_type = "fp32"

# Conv2D
conv2d_in_ch = 32
conv2d_in_W = 16
conv2d_in_H = 16
conv2d_ker_W = 1
conv2d_ker_H = 1
conv2d_out_ch = 64
conv2d_use_bias = 1

# Depthwise Convolution
dw_inout_ch = 64
dw_in_H = 25
dw_in_W = 5
dw_ker_H = 3
dw_ker_W = 3

# Pointwise Convolution
pw_in_H = 5
pw_in_W = 5
pw_in_ch = 64
pw_out_ch = 64

# Fully-Connected
lin_in_size = 640
lin_out_size = 32
lin_use_bias = 1

if data_type not in ["fp32"]:
    print("Bias not implemented for selected data type. It will be ignored.")
    conv2d_use_bias = 0
    lin_use_bias = 0

# =====> END OF USER SETTINGS <=====









"""
BACKEND CODE
"""

f = open("memreport.txt", "w")

# Define number of bytes per float
if data_type == "fp32":
    data_size = 4
elif data_type == "fp16" or data_type == "bf16":
    data_size = 2
elif data_type == "fp8":
    data_size = 1

# Initialize file
f.write("------------------------------------------------\n")
f.write("------------ MEMORY OCCUPATION TOOL ------------\n")
f.write("------------------------------------------------\n\n\n")

f.write("Available layers:\n")
f.write("- Conv2D\n")
f.write("- Depthwise Convolution\n")
f.write("- Pointwise Convolution\n")
f.write("- Fully-Connected\n")
f.write("\n\n")

"""
CONV2D 
"""

# Compute Conv2D memory occupation (FORWARD)
in_act  = conv2d_in_ch * conv2d_in_H * conv2d_in_W
ker     = conv2d_ker_H * conv2d_ker_W * conv2d_in_ch * conv2d_out_ch
bias    = conv2d_use_bias * conv2d_out_ch
im2colF = conv2d_in_ch * conv2d_ker_H * conv2d_ker_W * (conv2d_in_W-conv2d_ker_W+1) * (conv2d_in_H-conv2d_ker_H+1) 
out_act = (conv2d_in_W-conv2d_ker_W+1) * (conv2d_in_H-conv2d_ker_H+1) * conv2d_out_ch
tot_FW  = in_act + ker + bias + im2colF + out_act

f.write("-------------------------------------------\n")
f.write("###              CONV2D LAYER           ###\n")
f.write("-------------------------------------------\n")
f.write("| ### SIZES ###\n|\n")
f.write("| IN: \tH={}, W={}, C={}\n".format(conv2d_in_H, conv2d_in_W, conv2d_in_ch))
f.write("| KER: \tH={}, W={}, C_IN={}, C_OUT={}\n".format(conv2d_ker_H, conv2d_ker_W, conv2d_in_ch, conv2d_out_ch))
if conv2d_use_bias == 1:
    f.write("| BIAS: \tSIZE={}\n".format(conv2d_out_ch))
f.write("| OUT: \tH={}, W={}, C={}\n".format((conv2d_in_H-conv2d_ker_H+1), (conv2d_in_W-conv2d_ker_W+1), conv2d_in_ch))
f.write("-------------------------------------------\n")
f.write("| ### FORWARD ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| IM2COL BUFFER: \t{} ({} bytes)\n".format(im2colF, im2colF*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
if conv2d_use_bias == 1:
    f.write("| BIAS: \t\t\t\t{} ({} bytes)\n".format(bias, bias*data_size))
f.write("| OUT: \t\t\t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL FORWARD: \t{} ({} bytes)\n".format(tot_FW, tot_FW*data_size))
f.write("-------------------------------------------\n")
# Compute Conv2D memory occupation (WEIGHT GRADIENT)
in_act  = conv2d_in_ch * conv2d_in_H * conv2d_in_W
ker     = conv2d_ker_H * conv2d_ker_W * conv2d_in_ch * conv2d_out_ch
im2colW = conv2d_in_ch * conv2d_ker_H * conv2d_ker_W * (conv2d_in_W-conv2d_ker_W+1) * (conv2d_in_H-conv2d_ker_H+1)
out_act = (conv2d_in_W-conv2d_ker_W+1) * (conv2d_in_H-conv2d_ker_H+1) * conv2d_out_ch
tot_WGT = in_act + ker + im2colW + out_act
f.write("| ### WEIGHT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| IM2COL BUFFER: \t{} ({} bytes)\n".format(im2colW, im2colW*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
if conv2d_use_bias == 1:
    f.write("| BIAS: \t\t\t\t{} ({} bytes)\n".format(bias, bias*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL WGT GRAD: \t{} ({} bytes)\n".format(tot_WGT, tot_WGT*data_size))
f.write("-------------------------------------------\n")
# Compute Conv2D memory occupation (IN GRADIENT)
in_act  = conv2d_in_ch * conv2d_in_H * conv2d_in_W
ker     = conv2d_ker_H * conv2d_ker_W * conv2d_in_ch * conv2d_out_ch
im2colI = conv2d_out_ch * conv2d_ker_H * conv2d_ker_W * conv2d_in_H * conv2d_in_W
out_act = (conv2d_in_W-conv2d_ker_W+1) * (conv2d_in_H-conv2d_ker_H+1) * conv2d_out_ch
tot_ING = in_act + ker + im2colI + out_act
f.write("| ### INPUT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| IM2COL BUFFER: \t{} ({} bytes)\n".format(im2colI, im2colI*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
if conv2d_use_bias == 1:
    f.write("| BIAS: \t\t\t\t{} ({} bytes)\n".format(bias, bias*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL IN GRAD: \t{} ({} bytes)\n".format(tot_ING, tot_ING*data_size))
f.write("-------------------------------------------\n")
tot_MEM = tot_FW + tot_WGT + tot_ING - im2colF - im2colW - im2colI + max(im2colF, im2colW, im2colI)
f.write("CONV2D TOTAL OCCUPATION: \t{} ({} bytes)\n".format(tot_MEM, (tot_MEM)*data_size))



"""
DEPTHWISE CONVOLUTION
"""

f.write("\n\n\n\n")
# Compute DW memory occupation (FORWARD)
in_act  = dw_in_H * dw_in_W * dw_inout_ch
ker     = dw_ker_H * dw_ker_W * dw_inout_ch
out_act = dw_inout_ch * (dw_in_H-dw_ker_H+1) * (dw_in_W-dw_ker_W+1)
tot_FW  = in_act + ker + im2colF + out_act
f.write("-------------------------------------------\n")
f.write("###             DEPTHWISE CONV          ###\n")
f.write("-------------------------------------------\n")
f.write("| ### SIZES ###\n|\n")
f.write("| IN: \tH={}, W={}, C={}\n".format(dw_in_H, dw_in_W, dw_inout_ch))
f.write("| KER: \tH={}, W={}, C={}\n".format(dw_ker_H, dw_ker_W, dw_inout_ch))
f.write("| OUT: \tH={}, W={}, C={}\n".format((dw_in_H-dw_ker_H+1), (dw_in_W-dw_ker_W+1), dw_inout_ch))
f.write("-------------------------------------------\n")
f.write("| ### FORWARD ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT: \t\t\t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL FORWARD: \t{} ({} bytes)\n".format(tot_FW, tot_FW*data_size))
f.write("-------------------------------------------\n")
# Compute DW memory occupation (WEIGHT GRADIENT)
in_act  = dw_in_H * dw_in_W * dw_inout_ch
ker     = dw_ker_H * dw_ker_W * dw_inout_ch
out_act = dw_inout_ch * (dw_in_H-dw_ker_H+1) * (dw_in_W-dw_ker_W+1)
tot_WGT = in_act + ker + im2colW + out_act
f.write("| ### WEIGHT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL WGT GRAD: \t{} ({} bytes)\n".format(tot_WGT, tot_WGT*data_size))
f.write("-------------------------------------------\n")
# Compute DW memory occupation (IN GRADIENT)
in_act  = dw_in_H * dw_in_W * dw_inout_ch
ker     = dw_ker_H * dw_ker_W * dw_inout_ch
out_act = dw_inout_ch * (dw_in_H-dw_ker_H+1) * (dw_in_W-dw_ker_W+1)
tot_ING = in_act + ker + im2colI + out_act
f.write("| ### INPUT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL IN GRAD: \t{} ({} bytes)\n".format(tot_ING, tot_ING*data_size))
f.write("-------------------------------------------\n")
tot_MEM = tot_FW + tot_WGT
f.write("DEPTHWISE CONV TOTAL OCCUPATION: \t{} ({} bytes)\n".format(tot_MEM, (tot_MEM)*data_size))



"""
POINTWISE CONVOLUTION
"""

f.write("\n\n\n\n")
# Compute PW memory occupation (FORWARD)
in_act  = pw_in_H * pw_in_W * pw_in_ch
ker     = pw_in_ch * pw_out_ch
out_act = pw_in_H * pw_in_W * pw_out_ch
tot_FW  = in_act + ker + out_act
f.write("-------------------------------------------\n")
f.write("###             POINTWISE CONV          ###\n")
f.write("-------------------------------------------\n")
f.write("| ### SIZES ###\n|\n")
f.write("| IN: \tH={}, W={}, C={}\n".format(pw_in_H, pw_in_W, pw_in_ch))
f.write("| KER: \tH={}, W={}, C_IN={}, C_OUT={}\n".format(1, 1, pw_in_ch, pw_out_ch))
f.write("| OUT: \tH={}, W={}, C={}\n".format(pw_in_H, pw_in_W, pw_out_ch))
f.write("-------------------------------------------\n")
f.write("| ### FORWARD ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT: \t\t\t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL FORWARD: \t{} ({} bytes)\n".format(tot_FW, tot_FW*data_size))
f.write("-------------------------------------------\n")
# Compute PW memory occupation (WEIGHT GRADIENT)
in_act  = pw_in_H * pw_in_W * pw_in_ch
ker     = pw_in_ch * pw_out_ch
out_act = pw_in_H * pw_in_W * pw_out_ch
tot_WGT = in_act + ker + out_act
f.write("| ### WEIGHT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL WGT GRAD: \t{} ({} bytes)\n".format(tot_WGT, tot_WGT*data_size))
f.write("-------------------------------------------\n")
# Compute PW memory occupation (IN GRADIENT)
in_act  = pw_in_H * pw_in_W * pw_in_ch
ker     = pw_in_ch * pw_out_ch
out_act = pw_in_H * pw_in_W * pw_out_ch
tot_ING = in_act + ker + out_act
f.write("| ### INPUT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL IN GRAD: \t{} ({} bytes)\n".format(tot_ING, tot_ING*data_size))
f.write("-------------------------------------------\n")
f.write("POINTWISE CONV TOTAL OCCUPATION: \t{} ({} bytes)\n".format(tot_FW+tot_ING+tot_WGT, (tot_FW+tot_ING+tot_WGT)*data_size))



"""
FULLY-CONNECTED
"""

f.write("\n\n\n\n")
# Compute FC memory occupation (FORWARD)
in_act  = lin_in_size
ker     = lin_in_size * lin_out_size
bias    = lin_use_bias * lin_out_size
out_act = lin_out_size
tot_FW  = in_act + ker + bias + out_act
f.write("-------------------------------------------\n")
f.write("###            FULLY-CONNECTED          ###\n")
f.write("-------------------------------------------\n")
f.write("| ### SIZES ###\n|\n")
f.write("| IN: \tH={}, W={}\n".format(lin_in_size, 1))
f.write("| KER: \tH={}, W={}\n".format(lin_out_size, lin_in_size))
if lin_use_bias == 1:
    f.write("| BIAS: \tH={}, W={}\n".format(1, lin_out_size))
f.write("| OUT: \tH={}, W={}\n".format(1, lin_out_size))
f.write("-------------------------------------------\n")
f.write("| ### FORWARD ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
if lin_use_bias == 1:
    f.write("| BIAS: \t\t\t\t{} ({} bytes)\n".format(bias, bias*data_size))
f.write("| OUT: \t\t\t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL FORWARD: \t{} ({} bytes)\n".format(tot_FW, tot_FW*data_size))
f.write("-------------------------------------------\n")
# Compute FC memory occupation (WEIGHT GRADIENT)
in_act  = lin_in_size
ker     = lin_in_size * lin_out_size
bias    = lin_use_bias * lin_out_size
out_act = lin_out_size
tot_WGT = in_act + ker + bias + out_act
f.write("| ### WEIGHT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
if lin_use_bias == 1:
    f.write("| BIAS: \t\t\t\t{} ({} bytes)\n".format(bias, bias*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL WGT GRAD: \t{} ({} bytes)\n".format(tot_WGT, tot_WGT*data_size))
f.write("-------------------------------------------\n")
# Compute FC memory occupation (IN GRADIENT)
in_act  = lin_in_size
ker     = lin_in_size * lin_out_size
out_act = lin_out_size
tot_ING = in_act + ker + out_act
f.write("| ### INPUT GRADIENT ###\n|\n")
f.write("| IN: \t\t\t\t{} ({} bytes)\n".format(in_act, in_act*data_size))
f.write("| KER: \t\t\t\t{} ({} bytes)\n".format(ker, ker*data_size))
f.write("| OUT DIFF: \t\t{} ({} bytes)\n".format(out_act, out_act*data_size))
f.write("| \n| TOTAL IN GRAD: \t{} ({} bytes)\n".format(tot_ING, tot_ING*data_size))
f.write("-------------------------------------------\n")
f.write("FULLY-CONNECTED TOTAL OCCUPATION: \t{} ({} bytes)\n".format(tot_FW+tot_ING+tot_WGT, (tot_FW+tot_ING+tot_WGT)*data_size))


f.close()