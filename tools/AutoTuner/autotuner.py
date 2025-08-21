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
Authors: Davide Nadalini, Leonardo Ravaglia
'''


from math import floor
import os

from tiling_utils import get_tiling
from tiling_utils import find_best_perf
from tiling_utils import sort_results
from tiling_utils import compute_memory_footprint
from tiling_utils import write_error_file
from tiling_utils import write_raw_file

"""
Tiler (Naive or DORY-based) which finds tiling schemes depending on the problem, and then finds the fastest one for the problem
"""

#                       >>> NOTES <<<
# - To tile a LINEAR layer, set INPUT_H, INPUT_W, KER_H, KER_W to 1, 
#   then act on the IN_CH and OUT_CH parameters. 
# - The program does not tile input channels, but only output ones!!


# =====> USER SETTINGS <=====
# Network setting                                         
layer_type  = 'LINEAR'    # Options: 'PW', 'DW', 'LINEAR', 'CONV2D'  
IN_CH       = 128 * 512
INPUT_H     = 1               
INPUT_W     = 1                       
KER_H       = 1                                       
KER_W       = 1                                   
OUT_CH      = 128 * 128
PADDING     = 0                                            
STRIDE      = 1
USE_BIAS    = 1
NUM_CORES   = 8
# Tiler settings
NUM_TILING_SOLUTIONS    = 5
NUM_INPUT_BITS          = 32
NUM_ACTIVATION_BITS     = NUM_INPUT_BITS
NUM_KERNEL_BITS         = 32
NUM_OUTPUT_BITS         = 32
# Select if to ignore FW and WGT GRAD
IGNORE_FW = False
IGNORE_WGT_GRAD = True
# Select if to ignore input grads
IGNORE_IN_GRAD = False
# Select if to use either the naive or the DORY-based tiler
USE_NAIVE_TILER = True
# Select if to compile locally after finding the tiling
FIND_FASTEST_MATMUL = False
# Select if to write the file for server execution (specify trainlib's folder location on server)
WRITE_YML_FILE = True
trainlib_path = '../../'
# PULP settings
NUM_STD_MATMUL      = 24
NUM_DW_MATMUL       = 7
TILING_BUFFER_SIZE  = 128*1024     # Standard = 128k
# =====> END OF USER SETTINGS <=====








# -------------------------
# ----- BACKEND CODE ------
# -------------------------


# Starting path
os.chdir('..')
base_path = os.getcwd()
os.chdir('tools/')
#os.chdir(base_path)
# Output files
sim_result_file = str(base_path + '/tools/AutoTuner/fastest_tiling.txt')
raw_result_file = str(base_path + '/tools/AutoTuner/raw_data_tiling.txt')
err_log_file = str(base_path + '/tools/AutoTuner/error_log.txt')
source_file = 'runs.txt'
temp_file = 'temp.txt'
# For server execution
if WRITE_YML_FILE == True:
    yml_file = str(base_path + '/tools/AutoTuner/basic.yml')
    makefile = str(base_path + '/tools/AutoTuner/Makefile')
    dir_tree_gen_file = str(base_path + '/tools/AutoTuner/treegen.py')


# Output list
C_in    = []
C_out   = []
W_in    = []
W_out   = []
H_in    = []
H_out   = []
Obj     = []

# Set options for tiling algorithm
if layer_type == 'DW':
    name='conv'
    DW_flag = 1
    conv_groups = IN_CH
else:
    if layer_type == 'LINEAR':
        name='MatMul'
    elif layer_type == 'PW':
        name='PW'
    else:
        name='conv'
    DW_flag = 0
    conv_groups = 1

# Extract the tiling schemes depending on input settings
print("Evaluating tiling schemes..\n")
C_in, C_out, H_in, H_out, W_in, W_out, Obj, NUM_FOUND_SOLUTIONS = get_tiling(
                                                                DW=DW_flag,
                                                                filter_size1=KER_H,
                                                                filter_size2=KER_W,
                                                                stride=STRIDE,
                                                                use_bias=USE_BIAS,
                                                                padding_top=PADDING,
                                                                padding_bottom=PADDING,
                                                                padding_left=PADDING,
                                                                padding_right=PADDING,
                                                                groups=conv_groups,
                                                                BN=0,
                                                                in_channels=IN_CH,
                                                                out_channels=OUT_CH,
                                                                x_shape=INPUT_W,
                                                                y_shape=INPUT_H,
                                                                buffer_size=TILING_BUFFER_SIZE,
                                                                BitIn=NUM_INPUT_BITS,
                                                                BitW=NUM_KERNEL_BITS,
                                                                BitActivation=NUM_ACTIVATION_BITS,
                                                                BitOut=NUM_OUTPUT_BITS,
                                                                NUM_RESULTS=NUM_TILING_SOLUTIONS,
                                                                name=name,
                                                                layer_type=layer_type,
                                                                NAIVE=USE_NAIVE_TILER,
                                                                NUM_CORES=NUM_CORES,
                                                                IGNORE_IN_GRAD=IGNORE_IN_GRAD
                                                                )

print("\nReporting {} solutions..".format(NUM_FOUND_SOLUTIONS))
print('C_in: '+str(C_in))
print('C_out: '+str(C_out))
print('H_in: '+str(H_in))
print('H_out: '+str(H_out))
print('W_in: '+str(W_in))
print('W_out: '+str(W_out))
print('OBJECTIVE: '+str(Obj))

# Once the solutions are found, list them on the output file
f = open(sim_result_file, 'w')
f.write("=====> TILING SCHEMES WITH OPTIMIZED MATMULS <=====\n\n")
f.write("--------------- NETWORK STRUCTURE: ----------------\n")
f.write("Layer type: " + layer_type + '\n')
f.write("Input ({} bit): C = {}, H = {}, W = {}\n".format(NUM_INPUT_BITS, IN_CH, INPUT_H, INPUT_W))
f.write("Kernel ({} bit, with {} bit activation): Cin = {}, H = {}, W = {}, Cout = {}\n".format(NUM_KERNEL_BITS, NUM_ACTIVATION_BITS, IN_CH, KER_H, KER_W, OUT_CH))
if USE_BIAS == 1:
    f.write("Bias ({} bit, with {} bit activation): Size = {}\n".format(NUM_KERNEL_BITS, NUM_ACTIVATION_BITS, OUT_CH))
f.write("Output ({} bit): C = {}, H = {}, W = {}\n".format(NUM_OUTPUT_BITS, OUT_CH, (INPUT_H-KER_H+1), (INPUT_W-KER_W+1)))
f.write("---------------------------------------------------\n\n\n")
f.write("---------------- TILING CANDIDATES ----------------\n")
print("\nWriting {} solutions to file..".format(NUM_FOUND_SOLUTIONS))
for idx in range(NUM_FOUND_SOLUTIONS):
    # Write solutions to file
    if NUM_FOUND_SOLUTIONS == 1:
        # Compute the memory occupation of the layer
        memocc_bytes = compute_memory_footprint(layer_type, C_in, H_in, W_in, C_out, H_out, W_out, int(NUM_INPUT_BITS/8), int(NUM_KERNEL_BITS/8), int(NUM_OUTPUT_BITS/8), USE_BIAS=USE_BIAS)
        f.write("{})\tInput: C={}, H={}, W={},\t\tOutput: C={}, H={}, W={}\t\t\tMemory footprint (bytes): FW={}, WGT_G={}, IN_G={}".format(idx, C_in, H_in, W_in, C_out, H_out, W_out, memocc_bytes[0], memocc_bytes[1], memocc_bytes[2]))
        # Compute the total number of tiles
        Cin_pieces = int(floor(IN_CH/C_in)); Cin_remainder = IN_CH % C_in
        if layer_type == 'CONV2D' or layer_type == 'DW':
            if INPUT_H == H_in:
                H_pieces = 1 ; H_remainder = 0 # TO BE FIXED THE REMAINDER
            else :
                H_pieces = 1 + (INPUT_H - H_in) / (H_in - KER_H + 1) ; H_remainder = 0 # TO BE FIXED THE REMAINDER
            if INPUT_W == W_in:
                W_pieces = 1 ; W_remainder = 0 # TO BE FIXED THE REMAINDER
            else :
                W_pieces = 1 + (INPUT_W - W_in) / (W_in - KER_W + 1) ; W_remainder = 0 # TO BE FIXED THE REMAINDER
        else :
            H_pieces = int(floor(INPUT_H/H_in)); H_remainder = INPUT_H % H_in
            W_pieces = int(floor(INPUT_W/W_in)); W_remainder = INPUT_W % W_in
        if (layer_type == 'DW'):
            Cout_pieces = 1; Cout_remainder = 0
        else:
            Cout_pieces = int(floor(OUT_CH/C_out)); Cout_remainder = OUT_CH % C_out
        total_full_tiles = Cin_pieces * H_pieces * W_pieces * Cout_pieces
        if Cin_remainder != 0 or H_remainder != 0 or W_remainder != 0 or Cout_remainder != 0:
            total_border_tiles = 'PRESENT'
        else: 
            total_border_tiles = 'NONE'
        f.write("\t\tNUM_FULL_TILES={}, BORDER_TILES={}\n".format(total_full_tiles, total_border_tiles))
    else:
        # Compute the memory occupation of the layer
        memocc_bytes = compute_memory_footprint(layer_type, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx], int(NUM_INPUT_BITS/8), int(NUM_KERNEL_BITS/8), int(NUM_OUTPUT_BITS/8), USE_BIAS=USE_BIAS)
        f.write("{})\t\tInput: C={}, H={}, W={},\t\tOutput: C={}, H={}, W={}\t\t\tMemory footprint (bytes): FW={}, WGT_G={}, IN_G={}".format(idx, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx], memocc_bytes[0], memocc_bytes[1], memocc_bytes[2]))
        # Compute the total number of tiles
        Cin_pieces = int(floor(IN_CH/C_in[idx])); Cin_remainder = IN_CH % C_in[idx]
        if layer_type == 'CONV2D' or layer_type == 'DW':
            if INPUT_H == H_in[idx]:
                H_pieces = 1 ; H_remainder = 0 # TO BE FIXED THE REMAINDER
            else :
                H_pieces = int(1 + (INPUT_H - H_in[idx]) / (H_in[idx] - KER_H + 1)) ; H_remainder = 0 # TO BE FIXED THE REMAINDER 
            if INPUT_W == W_in[idx]:
                W_pieces = 1 ; W_remainder = 0 # TO BE FIXED THE REMAINDER
            else :
                W_pieces = int(1 + (INPUT_W - W_in[idx]) / (W_in[idx] - KER_W + 1)) ; W_remainder = 0 # TO BE FIXED THE REMAINDER
        else :
            H_pieces = int(floor(INPUT_H/H_in[idx])); H_remainder = INPUT_H % H_in[idx]
            W_pieces = int(floor(INPUT_W/W_in[idx])); W_remainder = INPUT_W % W_in[idx]
        if (layer_type == 'DW'):
            Cout_pieces = 1; Cout_remainder = 0
        else:
            Cout_pieces = int(floor(OUT_CH/C_out[idx])); Cout_remainder = OUT_CH % C_out[idx]
        total_full_tiles = Cin_pieces * H_pieces * W_pieces * Cout_pieces
        if Cin_remainder != 0 or H_remainder != 0 or W_remainder != 0 or Cout_remainder != 0:
            total_border_tiles = 'PRESENT'    
        else:
            total_border_tiles = 'NONE'   
        f.write("\t\tNUM_FULL_TILES={}, BORDER_TILES={}\n".format(total_full_tiles, total_border_tiles))
f.write("---------------------------------------------------\n\n\n")
f.write("------------- SINGLE TILE PERFORMANCES ------------\n")
f.write("-------- (Best matmul with relative cycles) -------\n")
f.write("---------------------------------------------------\n")
f.close()


raw_f = open(raw_result_file, 'w')
raw_f.write("=====> RAW RESULT FILE FOR {} LAYER <=====\n\n".format(layer_type))
raw_f.close()


err_f = open(err_log_file, 'w')
err_f.write("=====> ERROR LOG FOR {} LAYER <=====\n".format(layer_type))
err_f.write("      (Notify broken matmuls)     \n\n")
err_f.close()













# -------------------------------------
# ----- FILE FOR SERVER EXECUTION -----
# -------------------------------------

if WRITE_YML_FILE == True:
    # Select layer type
    if layer_type == 'DW' or layer_type == 'PW':
        yml_f = open(yml_file, 'w')
        py_f = open(dir_tree_gen_file, 'w')
        make_f = open(makefile, 'w')
        yml_f.write("trainlib_test_matmul_tiling:\n")
        py_f.write("import os\n# Create directory tree\n")

        def dw_template(num_cores=1, step='FW'):
            py_f.write("os.system('mkdir ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            py_f.write("os.system('cp -R {}/* ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(trainlib_path, idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("  TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("    path: .\n")
            yml_f.write("    command: make C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("\tcd ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}/tests/test_conv_pw_dw_fp32 && ".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            if step=='FW':
                step_int = "DW_FORWARD"
            elif step=='WGT_G':
                step_int = "DW_BACKWARD_GRAD"
            elif step=="IN_G":
                step_int = "DW_BACKWARD_ERROR"
            make_f.write("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H={} DW_KER_W={} DW_IN_CH={} PW_OUT_CH=1 STEP='{}' BYPASS=1\n".format(num_cores, NUM_DW_MATMUL, H_in[idx], W_in[idx], KER_H, KER_W, C_in[idx], step_int))

        def pw_template(num_cores=1, step='FW'):
            py_f.write("os.system('mkdir ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            py_f.write("os.system('cp -R {}/* ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(trainlib_path, idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("  TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("    path: .\n")
            yml_f.write("    command: make C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("\tcd ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}/tests/_fp32 && ".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            if step=='FW':
                step_int = "PW_FORWARD"
            elif step=='WGT_G':
                step_int = "PW_BACKWARD_GRAD"
            elif step=="IN_G":
                step_int = "PW_BACKWARD_ERROR"
            make_f.write("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H=3 DW_KER_W=3 DW_IN_CH={} PW_OUT_CH={} STEP='{}' BYPASS=1\n".format(num_cores, NUM_STD_MATMUL, H_in[idx]+2, W_in[idx]+2, C_in[idx], C_out[idx], step_int))


        if layer_type == 'DW':
            # FW N Cores
            if IGNORE_FW == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    dw_template(num_cores=NUM_CORES, step='FW')
            # WGT_G N Cores
            if IGNORE_WGT_GRAD == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    dw_template(num_cores=NUM_CORES, step='WGT_G')
            # IN_G N Cores
            if IGNORE_IN_GRAD == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    dw_template(num_cores=NUM_CORES, step='IN_G')

        elif layer_type == 'PW':
            # FW N Cores
            if IGNORE_FW == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    pw_template(num_cores=NUM_CORES, step='FW')
            # WGT_G N Cores
            if IGNORE_WGT_GRAD == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    pw_template(num_cores=NUM_CORES, step='WGT_G')
            # IN_G N Cores
            if IGNORE_IN_GRAD == False:
                for idx in range(NUM_FOUND_SOLUTIONS):
                    pw_template(num_cores=NUM_CORES, step='IN_G')

        yml_f.close()
        py_f.close()
        make_f.close()



    elif layer_type == 'LINEAR':
        yml_f = open(yml_file, 'w')
        py_f = open(dir_tree_gen_file, 'w')
        make_f = open(makefile, 'w')
        yml_f.write("trainlib_test_matmul_tiling:\n")
        py_f.write("import os\n# Create directory tree\n")

        def lin_template(num_cores=1, step='FW'):
            py_f.write("os.system('mkdir ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            py_f.write("os.system('cp -R {}/* ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(trainlib_path, idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("  TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("    path: .\n")
            yml_f.write("    command: make C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("\tcd ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}/tests/test_linear_fp32 && ".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            if step=='FW':
                step_int = "FORWARD"
            elif step=='WGT_G':
                step_int = "BACKWARD_GRAD"
            elif step=="IN_G":
                step_int = "BACKWARD_ERROR"
            make_f.write("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} STEP='{}' USE_BIASES={}\n".format(num_cores, NUM_STD_MATMUL, C_in[idx], C_out[idx], step_int, USE_BIAS))

        # FW N Cores
        if IGNORE_FW == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                lin_template(num_cores=NUM_CORES, step='FW')
        # WGT_G N Cores
        if IGNORE_WGT_GRAD == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                lin_template(num_cores=NUM_CORES, step='WGT_G')
        # IN_G N Cores
        if IGNORE_IN_GRAD == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                lin_template(num_cores=NUM_CORES, step='IN_G')

        yml_f.close()
        py_f.close()


    elif layer_type == 'CONV2D':
        yml_f = open(yml_file, 'w')
        py_f = open(dir_tree_gen_file, 'w')
        make_f = open(makefile, 'w')
        yml_f.write("trainlib_test_matmul_tiling:\n")
        py_f.write("import os\n# Create directory tree\n")

        def conv2d_template(num_cores=1, step='FW'):
            py_f.write("os.system('mkdir ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            py_f.write("os.system('cp -R {}/* ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}')\n".format(trainlib_path, idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("  TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            yml_f.write("    path: .\n")
            yml_f.write("    command: make C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("C_TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}:\n".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            make_f.write("\tcd ./TIL{}_PASS{}_CORES{}_IN{}_{}_{}_OUT_{}_{}_{}/tests/test_conv2d_fp32 && ".format(idx, step, num_cores, C_in[idx], H_in[idx], W_in[idx], C_out[idx], H_out[idx], W_out[idx]))
            if step=='FW':
                step_int = "FORWARD"
            elif step=='WGT_G':
                step_int = "BACKWARD_GRAD"
            elif step=="IN_G":
                step_int = "BACKWARD_ERROR"
            make_f.write("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} IMAGE_H={} IMAGE_W={} KER_H={} KER_W={} STEP='{}' USE_BIASES={}\n".format(num_cores, NUM_STD_MATMUL, C_in[idx], C_out[idx], H_in[idx], W_in[idx], KER_H, KER_W, step_int, USE_BIAS))

        # FW N Cores
        if IGNORE_FW == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                conv2d_template(num_cores=NUM_CORES, step='FW')
        # WGT_G N Cores
        if IGNORE_WGT_GRAD == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                conv2d_template(num_cores=NUM_CORES, step='WGT_G')
        # IN_G N Cores
        if IGNORE_IN_GRAD == False:
            for idx in range(NUM_FOUND_SOLUTIONS):
                conv2d_template(num_cores=NUM_CORES, step='IN_G')


        yml_f.close()
        py_f.close()











# ------------------------------
# ----- LOCAL PC EXECUTION -----
# ------------------------------

test_base_folder = "../tests/" #"/home/lanmei/pulp-trainlib_ori/tests/" # "../tests/"
return_folder = "../../tools" # "/home/lanmei/pulp-trainlib_ori/tools/" # "../../tools"

# Select if to compile layers or not
if FIND_FASTEST_MATMUL == True:
    # Launch simulations on GVSOC to test the tiling schemes
    if layer_type == 'DW' or layer_type == 'PW':
        os.chdir(test_base_folder+"test_conv_pw_dw_fp32")

        # DEPTHWISE ------------------------------------------------------------------------------------------------------------------------------------
        if layer_type == 'DW':
            if IGNORE_FW == False:
                # Lists for output data
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # Results with N CORES
                    # FWD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H={} DW_KER_W={} DW_IN_CH={} PW_OUT_CH=1 STEP='DW_FORWARD' BYPASS=1".format(NUM_CORES, NUM_DW_MATMUL, H_in[idx], W_in[idx], KER_H, KER_W, C_in[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'FW', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('FW')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            # Lists for output data
            if IGNORE_WGT_GRAD == False:
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # WGT GRAD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H={} DW_KER_W={} DW_IN_CH={} PW_OUT_CH=1 STEP='DW_BACKWARD_GRAD' BYPASS=1".format(NUM_CORES, NUM_DW_MATMUL, H_in[idx], W_in[idx], KER_H, KER_W, C_in[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'WGT_G', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('WGT_G')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            # Lists for output data
            if IGNORE_IN_GRAD == False:
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []        
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # IN GRAD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H={} DW_KER_W={} DW_IN_CH={} PW_OUT_CH=1 STEP='DW_BACKWARD_ERROR' BYPASS=1".format(NUM_CORES, NUM_DW_MATMUL, H_in[idx], W_in[idx], KER_H, KER_W, C_in[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'IN_G', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('IN_G')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            os.chdir(return_folder)
        
        
        # POINTWISE -------------------------------------------------------------------------------------------------------------------------------------
        else:
            if IGNORE_FW == False:
                # Lists for output data
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # Results with N CORES
                    # FWD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H=1 DW_KER_W=1 DW_IN_CH={} PW_OUT_CH={} STEP='PW_FORWARD' BYPASS=1".format(NUM_CORES, NUM_STD_MATMUL, H_in[idx], W_in[idx], C_in[idx], C_out[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'FW', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('FW')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            if IGNORE_WGT_GRAD == False:
                # Lists for output data
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # WGT GRAD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H=1 DW_KER_W=1 DW_IN_CH={} PW_OUT_CH={} STEP='PW_BACKWARD_GRAD' BYPASS=1".format(NUM_CORES, NUM_STD_MATMUL, H_in[idx], W_in[idx], C_in[idx], C_out[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'WGT_G', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('WGT_G')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            # Lists for output data
            if IGNORE_IN_GRAD == False:
                tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []        
                for idx in range(NUM_FOUND_SOLUTIONS):
                    # IN GRAD
                    os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IMAGE_H={} IMAGE_W={} DW_KER_H=1 DW_KER_W=1 DW_IN_CH={} PW_OUT_CH={} STEP='PW_BACKWARD_ERROR' BYPASS=1".format(NUM_CORES, NUM_STD_MATMUL, H_in[idx], W_in[idx], C_in[idx], C_out[idx]))
                    write_raw_file(source_file, raw_result_file)
                    tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                    if (errors > 0):
                        write_error_file(err_log_file, 'IN_G', tiling_idx, errors, broken_mm)
                    tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('IN_G')
                sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

            os.chdir(return_folder)


    # LINEAR ------------------------------------------------------------------------------------------------------------------------------------------------------------
    if layer_type == 'LINEAR':
        os.chdir(test_base_folder+"test_linear_fp32")

        if IGNORE_FW == False:
            # Lists for output data
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
            for idx in range(NUM_FOUND_SOLUTIONS):
                # Results with N CORES
                # FWD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} STEP='FORWARD' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'FW', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('FW')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        if IGNORE_WGT_GRAD == False:
            # Lists for output data
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
            for idx in range(NUM_FOUND_SOLUTIONS):
                # WGT GRAD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} STEP='BACKWARD_GRAD' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'WGT_G', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('WGT_G')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        # Lists for output data
        if IGNORE_IN_GRAD == False:
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []        
            for idx in range(NUM_FOUND_SOLUTIONS):
                # IN GRAD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} STEP='BACKWARD_ERROR' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'IN_G', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('IN_G')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        os.chdir(return_folder)


    # CONV2D ----------------------------------------------------------------------------------------------------------------------------------------------------
    if layer_type == 'CONV2D':
        os.chdir(test_base_folder+"test_conv2d_fp32")

        if IGNORE_FW == False:
            # Lists for output data
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
            for idx in range(NUM_FOUND_SOLUTIONS):
                # Results with N CORES
                # FWD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} IMAGE_H={} IMAGE_W={} KER_H={} KER_W={} STEP='FORWARD' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], H_in[idx], W_in[idx], KER_H, KER_W, USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'FW', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('FW')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        if IGNORE_WGT_GRAD == False:
            # Lists for output data
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []
            for idx in range(NUM_FOUND_SOLUTIONS):
                # WGT GRAD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} IMAGE_H={} IMAGE_W={} KER_H={} KER_W={} STEP='BACKWARD_GRAD' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], H_in[idx], W_in[idx], KER_H, KER_W, USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'WGT_G', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('WGT_G')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        if IGNORE_IN_GRAD == False:
            # Lists for output data
            tiling_idx_list = []; matmul_names_list = []; matmul_cycles_list = []; num_cores_list = []; passes_list = []        
            for idx in range(NUM_FOUND_SOLUTIONS):
                # IN GRAD
                os.system("make profile_all_optim NUM_CORES={} NUM_MATMULS={} IN_CH={} OUT_CH={} IMAGE_H={} IMAGE_W={} KER_H={} KER_W={} STEP='BACKWARD_ERROR' USE_BIASES={}".format(NUM_CORES, NUM_STD_MATMUL, C_in[idx], C_out[idx], H_in[idx], W_in[idx], KER_H, KER_W, USE_BIAS))
                write_raw_file(source_file, raw_result_file)
                tiling_idx, mm, cyc, cores, errors, broken_mm = find_best_perf(source_file, idx, NUM_CORES)
                if (errors > 0):
                    write_error_file(err_log_file, 'IN_G', tiling_idx, errors, broken_mm)
                tiling_idx_list.append(tiling_idx); matmul_names_list.append(mm); matmul_cycles_list.append(cyc); num_cores_list.append(cores); passes_list.append('IN_G')
            sort_results(sim_result_file, tiling_idx_list, matmul_names_list, matmul_cycles_list, num_cores_list, passes_list)

        os.chdir(return_folder)


# Write file ender
f = open(sim_result_file, 'a')
f.write("---------------------------------------------------\n\n")
f.close()


