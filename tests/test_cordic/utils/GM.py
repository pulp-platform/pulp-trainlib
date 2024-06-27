import torch
import dump_utils as dump
import argparse
import math

parser = argparse.ArgumentParser("Cordic test")
parser.add_argument( '--n_test', type=int, default=300)
args = parser.parse_args()

n_test = args.n_test

# angles = torch.empty(n_test).uniform_(-math.pi/2, math.pi/2)
angles = torch.empty(n_test).uniform_(0, 10)

cos = torch.cos(angles)
sin = torch.sin(angles)

# Write data to file
f = open("cordic_data.h", "w")
# f.write("#define N_TEST "+str(n_test)+"\n")
f.write("PI_L1 float gm_angles["+str(n_test)+"] = {"+dump.tensor_to_string(angles)+"};\n")
f.write("PI_L2 float gm_cos["+str(n_test)+"] = {"+dump.tensor_to_string(cos)+"};\n")
f.write("PI_L2 float gm_sin["+str(n_test)+"] = {"+dump.tensor_to_string(sin)+"};\n")

f.close()


def print_constant(N):
    print("atan_pow_2: \n")
    for i in range(0, N):
        print(f"{math.atan(2**(-i))}, ")

    sf = 1
    for i in range(0, N):
        sf *= math.cos(math.atan(2**(-i)))

    print(f"\nscaling factor: {sf}")