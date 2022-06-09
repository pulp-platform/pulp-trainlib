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
# TEST 0: (in_size[0], out_size[0],..)
# TEST 1: (in_size[1], out_size[1],..)
# TEST 2: (in_size[2], out_size[2],..)
in_size = [1024, 2048, 4096]
out_size = [8, 8, 8]
# =====> END OF USER CODE <=====

array_size = len(in_size)
if (len(out_size) != array_size):
    print("Arrays for multiple size test are not equally sized!!")

import os
import argparse
import profile_utils as prof

# Take arguments for profiling
parser = argparse.ArgumentParser("Multiple Sizes Layer Profiling")
parser.add_argument( '--num_sizes', type=int, default=array_size)
parser.add_argument( '--perf_file_name', type=str, default='runs.txt' )
parser.add_argument( '--step', type=str, default="PW_FORWARD")
parser.add_argument( '--cores', type=int, default=1)
parser.add_argument( '--data_type', type=str, default='fp32')

parser.add_argument( '--matmul_type', type=int, default=0)  # Selects a matmul algorithm

args = parser.parse_args()

num_sizes = args.num_sizes
step_type = args.step
filename = args.perf_file_name
cores = args.cores
data_type = args.data_type
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
        os.system("make clean get_golden all run STEP={} NUM_CORES={} MATMUL_TYPE={} IN_CH={} OUT_CH={} > log.txt".format(step_type, cores, matmul_alg, in_size[compile_idx], out_size[compile_idx]))
    else: 
        print("Invalid step!!")
        exit()
    # Find profiling and write it to file
    f = open(filename, "a")
    f.write("\nRUN {}: MATMUL_ALG= {}, IN_CH={}, OUT_CH={}".format(compile_idx, matmul_alg, in_size[compile_idx], out_size[compile_idx]))
    f.close()
    prof.extract_size_performance(step_type, filename)

print("\n=====> TERMINATING TEST SEQUENCE.. <=====\n")
os.system("rm -r BUILD/")

