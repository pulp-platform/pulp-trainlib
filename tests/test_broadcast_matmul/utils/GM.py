import argparse

import numpy as np


def get_args():
    # Create arguments
    parser = argparse.ArgumentParser()

    # Get dimensions of first element
    parser.add_argument("--dims_1", type=int, nargs="+", default=[3, 4])

    # Get dimensions of second element
    parser.add_argument("--dims_2", type=int, nargs="+", default=[3, 4])

    # Set data type
    parser.add_argument("--dtype", type=int, default=32)

    args = parser.parse_args()

    # Check arguments
    try:
        np.broadcast_shapes(args.dims_1[:-2], args.dims_2[:-2])
    except ValueError:
        raise ValueError("Dimensions not compatible for broadcasting")

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
    in_matrix_1 = np.random.rand(*args.dims_1)
    in_matrix_2 = np.random.rand(*args.dims_2)
    out_matrix = in_matrix_1 @ in_matrix_2

    # Half if fp16
    if args.dtype == 16:
        in_matrix_1 = in_matrix_1.astype(np.float16)
        in_matrix_2 = in_matrix_2.astype(np.float16)

    # Compute total dimensions
    total_dim_1 = 1
    for dim in in_matrix_1.shape:
        total_dim_1 *= dim

    total_dim_2 = 1
    for dim in in_matrix_2.shape:
        total_dim_2 *= dim

    total_dim_out = 1
    for dim in out_matrix.shape:
        total_dim_out *= dim

    # Write info to files
    with open("test_data.h", "w") as f:
        f.write("#define N_DIMS_1 " + str(len(in_matrix_1.shape)) + "\n")
        f.write("#define N_DIMS_2 " + str(len(in_matrix_2.shape)) + "\n")
        f.write("#define N_DIMS_OUT " + str(len(out_matrix.shape)) + "\n\n")

        f.write("#define TOTAL_SIZE_1 " + str(total_dim_1) + "\n")
        f.write("#define TOTAL_SIZE_2 " + str(total_dim_2) + "\n")
        f.write("#define TOTAL_SIZE_OUT " + str(total_dim_out) + "\n\n")

        f.write("PI_L2 int DIMS_1[] = {" + ", ".join(map(str, in_matrix_1.shape)) + "};\n")
        f.write("PI_L2 int DIMS_2[] = {" + ", ".join(map(str, in_matrix_2.shape)) + "};\n")
        f.write("PI_L2 int DIMS_OUT[] = {" + ", ".join(map(str, out_matrix.shape)) + "};\n\n")

        f.write("PI_L2 " + data_marker + " OUT_MATRIX[TOTAL_SIZE_OUT];\n\n")

        f.write(
            "PI_L2 "
            + data_marker
            + " IN_MATRIX_1[] = {"
            + "f, ".join(map(str, in_matrix_1.flatten()))
            + "};\n\n"
        )

        f.write(
            "PI_L2 "
            + data_marker
            + " IN_MATRIX_2[] = {"
            + "f, ".join(map(str, in_matrix_2.flatten()))
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
