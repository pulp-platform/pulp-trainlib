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


'''
Profile multiple network sizes in a single call
'''

# =====>    USER CODE    <=====
# Arrays of the network sizes to be simulated. These arrays have to be all the same length, 
# since the code compiles over the sequence of tests:
# TEST 0: (im_size[0], ker_size[0],..)
# TEST 1: (im_size[1], ker_size[1],..)
# TEST 2: (im_size[2], ker_size[2],..)
im_height = [3, 3, 7]
im_width = [3, 3, 7]
ker_height = [1, 3, 3]
ker_width = [1, 3, 3]
ch_in = [8, 8, 16]
ch_out = [1, 8, 16]
# =====> END OF USER CODE <=====

array_size = len(im_height)
if (len(im_width) != array_size) or (len(ker_height) != array_size) or(len(ker_width) != array_size) or (len(ch_in) != array_size) or (len(ch_out) != array_size):
    print("Arrays for multiple size test are not equally sized!!")

import os
import subprocess
import argparse
import profile_utils as prof

# Take arguments for profiling
parser = argparse.ArgumentParser("Multiple Sizes Layer Profiling")
parser.add_argument( '--num_sizes', type=int, default=array_size)
parser.add_argument( '--perf_file_name', type=str, default='runs.txt' )
parser.add_argument( '--step', type=str, default="PW_FORWARD")
parser.add_argument( '--cores', type=int, default=1)
parser.add_argument( '--data_type', type=str, default='fp32')
parser.add_argument( '--use_biases', type=int, default='int')

parser.add_argument( '--matmul_type', type=int, default=0)  # Selects a matmul algorithm

args = parser.parse_args()

num_sizes = args.num_sizes
step_type = args.step
filename = args.perf_file_name
cores = args.cores
data_type = args.data_type
use_biases = args.use_biases
matmul_alg = args.matmul_type

print("\n=====> ENTERING TEST SEQUENCE.. <=====\n")

# Prepare log file for the measured performances
f = open(filename, "w")
f.write("[ PERFORMANCES OVER DIFFERENT NETWORK SIZES ]\n")
f.write("---------------------------------------------\n")
f.write("STEP TYPE: {}\n".format(step_type))
f.write("NUM_CORES: {}\n".format(cores))
f.write("DATA_TYPE: {}\n".format(data_type))
f.write("Number of different layer sizes: {}\n".format(num_sizes))
f.write("---------------------------------------------\n")
f.write("\n=====> NETWORK RUNS <=====")
f.close()

# Execute multiple make commands and report performances
for compile_idx in range(num_sizes) :
    print("Executing build {}".format(compile_idx))
    # Execute build
    os.system("rm -r BUILD/")
    if (step_type == "FORWARD" or "BACKWARD_GRAD" or "BW_ERROR"):
        os.system("make clean get_golden all run STEP={} NUM_CORES={} MATMUL_TYPE={} IMAGE_W={} IMAGE_H={} KER_W={} KER_H={} IN_CH={} OUT_CH={} USE_BIASES={} > log.txt".format(step_type, cores, matmul_alg, im_height[compile_idx], im_width[compile_idx], ker_height[compile_idx], ker_width[compile_idx], ch_in[compile_idx], ch_out[compile_idx], use_biases))
    else: 
        print("Invalid step!!")
        exit()
    # Find profiling and write it to file
    f = open(filename, "a")
    f.write("\nRUN {}: MATMUL_ALG= {}, IMAGE_H={}, IMAGE_W={}, KER_H={}, KER_W={}, IN_CH={}, OUT_CH={}, USE_BIASES={}".format(compile_idx, matmul_alg, im_height[compile_idx], im_width[compile_idx], ker_height[compile_idx], ker_width[compile_idx], ch_in[compile_idx], ch_out[compile_idx], use_biases))
    f.close()
    prof.extract_size_performance(step_type, filename)

print("\n=====> TERMINATING TEST SEQUENCE.. <=====\n")
os.system("rm -r BUILD/")

