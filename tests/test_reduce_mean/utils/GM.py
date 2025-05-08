import argparse

import numpy as np


def get_args():
    # Create arguments
    parser = argparse.ArgumentParser()

    # Get dimensions of first element
    parser.add_argument("--input_dims", type=int, nargs="+", default=[2, 3, 4])

    # Get dimensions of second element
    parser.add_argument("--reduce_axis", type=int, default=1)

    # Set data type
    parser.add_argument("--dtype", type=int, default=32)

    args = parser.parse_args()

    return args


def main():
    # Get args
    args = get_args()

    # Fix seed
    np.random.seed(42)

    # Set data marker
    if args.dtype == 32:
        data_marker = "float"
    elif args.dtype == 16:
        data_marker = "fp16"
    else:
        raise ValueError("Invalid data type")

    # Create test matrices
    in_matrix = np.random.rand(*args.input_dims)

    # Half if fp16
    if args.dtype == 16:
        in_matrix = in_matrix.astype(np.float16)

    # Compute mean
    out_matrix = in_matrix.mean(axis=args.reduce_axis)

    # Compute total dimensions
    total_dim = 1
    for dim in in_matrix.shape:
        total_dim *= dim

    total_dim_out = 1
    for dim in out_matrix.shape:
        total_dim_out *= dim

    # Write info to files
    with open("test_data.h", "w") as f:
        f.write("#define N_DIMS " + str(len(in_matrix.shape)) + "\n")
        f.write("#define REDUCE_AXIS " + str(args.reduce_axis) + "\n")

        f.write("#define TOTAL_SIZE " + str(total_dim) + "\n")
        f.write("#define TOTAL_SIZE_OUT " + str(total_dim_out) + "\n\n")

        f.write("PI_L2 int DIMS[] = {" + ", ".join(map(str, in_matrix.shape)) + "};\n")

        f.write("PI_L2 " + data_marker + " OUT_MATRIX[TOTAL_SIZE_OUT] = {0};\n\n")

        f.write(
            "PI_L2 "
            + data_marker
            + " IN_MATRIX[] = {"
            + "f, ".join(map(str, in_matrix.flatten()))
            + "};\n\n"
        )

        f.write(
            "PI_L2 "
            + data_marker
            + " TEST_OUT[] = {"
            + "f, ".join(map(str, out_matrix.flatten()))
            + "};\n"
        )

    return None


if __name__ == "__main__":
    main()
