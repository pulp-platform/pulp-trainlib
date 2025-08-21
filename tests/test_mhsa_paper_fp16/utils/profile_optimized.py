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
Profile and sort all the available optimizations over the layers
by compiling multiple times with all matmuls.
'''

import os
import argparse
import profile_utils as prof

# Take arguments for profiling
parser = argparse.ArgumentParser("Optimized Layer Profiling")
parser.add_argument( '--num_matmuls', type=int, default=2)
parser.add_argument( '--perf_file_name', type=str, default='runs.txt' )
parser.add_argument( '--step', type=str, default="FORWARD")
parser.add_argument( '--cores', type=int, default=1)
parser.add_argument( '--data_type', type=str, default='fp32')

parser.add_argument( '--in_width', type=int, default=8 )
parser.add_argument( '--in_height', type=int, default=8 )
parser.add_argument( '--ch_in', type=int, default=1 )
parser.add_argument( '--ch_out', type=int, default=1 )
parser.add_argument( '--n_heads', type=int, default=8 )
parser.add_argument( '--att_dim', type=int, default=64 )


args = parser.parse_args()

num_matmuls = args.num_matmuls
step_type = args.step
filename = args.perf_file_name
cores = args.cores
data_type = args.data_type

in_width = args.in_width
in_height = args.in_height
ch_in = args.ch_in
ch_out = args.ch_out
n_heads = args.n_heads
att_dim = args.att_dim

print("\n=====> ENTERING TEST SEQUENCE.. <=====\n")

# Prepare log file for the measured performances
f = open(filename, "w")
f.write("[ PERFORMANCE COMPARISON WITH MULTIPLE MATMULS ]\n")
f.write("------------------------------------------------\n")
f.write("STEP TYPE: LINEAR {}\n".format(step_type))
f.write("NUM_CORES: {}\n".format(cores))
f.write("DATA_TYPE: {}\n".format(data_type))
f.write("NUM_MATMUL algorithms: {}\n".format(num_matmuls))
f.write("SIZES ARE:\n  Sequence Length (in_height): {}\n  Vector Length (in_width): {}\n Channels In: {} Out: {}\n Number of Attention Heads: {}\n Hidden Attention Dimension: {}\n".format(in_height, in_width, ch_in, ch_out, n_heads, att_dim))
f.write("------------------------------------------------\n")
f.write("\n=====> UNSORTED RESULTS <=====")
f.close()

# Execute multiple make commands and report performances
for compile_idx in range(num_matmuls) :
    print("Executing build {}".format(compile_idx))
    # Execute build
    os.system("rm -r BUILD/")
    os.system("make clean get_golden all run STEP={} NUM_CORES={} MATMUL_TYPE={} IN_H={} IN_W={} IN_CH={} OUT_CH={} N_HEADS={} ATT_DIM={}> log.txt".format(step_type, cores, compile_idx, in_height, in_width, ch_in, ch_out, n_heads, att_dim))
    # Find profiling and write it to file
    prof.extract_performance(compile_idx, filename)

print("\n=====> TERMINATING TEST SEQUENCE.. <=====\n")
os.system("rm -r BUILD/")


# Sort the executions from best to worst
matmul_group = "STANDARD"
prof.sort_best_performances(filename, matmul_group)