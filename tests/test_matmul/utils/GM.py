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


"""
    This script generates matrices for matrix multiply tests
"""

import torch
import torch.nn as nn
import argparse
import dump_utils as dump


# COMPUTE A(in_size x mid_size) * B(mid_size x out_size) 
#       = C(in_size x out_size)

parser = argparse.ArgumentParser("mm Operation Test")
# Matmul arguments
parser.add_argument( '--in_size', type=int, default=3 )
parser.add_argument( '--out_size', type=int, default=3 )
parser.add_argument( '--mid_size', type=int, default=3 )
# DW Conv arguments
parser.add_argument( '--in_image_size', type=int, default=5 )
parser.add_argument( '--ker_size', type=int, default=3 )
# General arguments
parser.add_argument( '--file_name', type=str, default='matmul_data.h')
parser.add_argument( '--type', type=str, default='float')       # float, fp16 to select the desired format
parser.add_argument( '--init_value_div', type=float, default=1)
parser.add_argument( '--transpose', type=str, default=0)    # Matrix B is transposed if = 1
args = parser.parse_args()

# Network parametersin_size
in_size = args.in_size
out_size = args.out_size
mid_size = args.mid_size
in_image_size = args.in_image_size
ker_size = args.ker_size
data_type = args.type
matmul_alg = 'STANDARD'
divider = args.init_value_div
transp = args.transpose



"""
Generate matrices for standard matrix multiply
"""

if matmul_alg == 'STANDARD':

    if transp == '1':
        print ("B matrix is transposed!")

    if (data_type == 'float'):
        # Matrices to be multiplied
        A = torch.FloatTensor(in_size, mid_size)
        if transp == '1':
            B = torch.FloatTensor(out_size, mid_size)
        else:    
            B = torch.FloatTensor(mid_size, out_size)
        C = torch.FloatTensor(in_size, out_size)

        A = torch.div(torch.ones(in_size, mid_size), divider)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] += (i+j+0.1)/divider

        if transp == '1':
            B = torch.zeros(out_size, mid_size)
        else:
            B = torch.zeros(mid_size, out_size)

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = i*j+0.1

        if transp == '1':
            C = torch.mm(input=A, mat2=B.transpose(0, 1), out=C)
            B.transpose(0, 1)
        else:
            C = torch.mm(input=A, mat2=B, out=C)

    # FP16 data
    elif (data_type == 'fp16'):
        # Matrices to be multiplied
        A = torch.HalfTensor(in_size, mid_size)
        if transp == '1':
            B = torch.HalfTensor(out_size, mid_size)
        else:
            B = torch.HalfTensor(mid_size, out_size)
        C = torch.HalfTensor(in_size, out_size)

        A = torch.div(torch.ones(in_size, mid_size), divider).half()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] += (i+j+0.1)/divider

        if transp == '1':
            B = torch.zeros(out_size, mid_size).half()
        else:
            B = torch.zeros(mid_size, out_size).half()

        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = i*j+0.1

        if transp == '1':
            C = torch.mm(input=A, mat2=B.transpose(0, 1), out=C)
            B.transpose(0, 1)
        else:
            C = torch.mm(input=A, mat2=B, out=C)

    else :  # Error message
        print('Invalid data type selection!!')
        exit()



    # Print data and create data header file
    f = open('net_args.h', "w") 

    # Setup the compilation parameter for the data type
    if data_type == 'float':
        f.write('// Float32 matmuls\n#define FLOAT32\n\n')
        f.write('// Matmul algorithm\n#define ' + matmul_alg + '\n')
    elif data_type == 'fp16':
        f.write('// Float16 matmuls\n#define FLOAT16\n\n')
        f.write('// Matmul algorithm\n#define ' + matmul_alg + '\n')
    else: 
        print("Invalid data type selection!!")

    # Define if matrix is transposed or not
    if transp == '1':
        f.write('// Matrix B is transposed\n#define TRANSPOSE_B 1')
    else:
        f.write('// Matrix B is not transposed\n#define TRANSPOSE_B 0')
    f.write('\n')

    # Write sizes in header file
    f.write('#define IN_CH ' + str(in_size) + '\n')
    f.write('#define MID_CH ' + str(mid_size) + '\n')
    f.write('#define OUT_CH ' + str(out_size) + '\n')
    f.write('\n')

    f.close()

    # Write data to file
    f = open(args.file_name, "w")

    print("\nInput Data: ")
    print("\nA is: ", A, A.shape, A.dtype)
    f.write('PI_L1 ' + data_type + ' A[IN_CH*MID_CH] = {'+dump.tensor_to_string(A)+'};\n')

    print("\nB is: ", B, B.shape, B.dtype)
    f.write('PI_L1 ' + data_type + ' B[MID_CH*OUT_CH] = {'+dump.tensor_to_string(B)+'};\n')

    print("\nC is: ", C, C.shape, C.dtype)
    f.write('PI_L2 ' + data_type + ' C[IN_CH*OUT_CH] = {'+dump.tensor_to_string(C)+'};\n')

    print("\n\n")

    f.close()
