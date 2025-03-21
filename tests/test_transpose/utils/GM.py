import argparse

import numpy as np


def get_args():
    # Create arguments
    parser = argparse.ArgumentParser()

    # Get dimensions
    parser.add_argument("--dims", type=int, nargs="+", default=[3, 4])

    # Get transposed axes
    parser.add_argument("--transposed_axes", type=int, nargs="+", default=[1, 0])

    # Set data type
    parser.add_argument("--dtype", type=int, default=32)

    args = parser.parse_args()

    # Check arguments
    assert len(args.dims) == len(args.transposed_axes)

    return args


def main():
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

    # Get number of dimensions
    n_dims = len(args.dims)

    # Create test matrices
    in_matrix = np.random.rand(*args.dims)
    out_matrix = np.transpose(in_matrix, args.transposed_axes)

    # Half if fp16
    if args.dtype == 16:
        in_matrix = in_matrix.astype(np.float16)
        out_matrix = out_matrix.astype(np.float16)

    # Compute total dimension
    total_dim = 1
    for dim in args.dims:
        total_dim *= dim

    # Write info to files
    with open("test_data.h", "w") as f:
        f.write("#define N_DIMS " + str(n_dims) + "\n")
        f.write("#define TOTAL_SIZE " + str(total_dim) + "\n\n")

        f.write("PI_L2 int DIMS[" + str(n_dims) + "] = {" + ", ".join(map(str, args.dims)) + "};\n")
        f.write(
            "PI_L2 int TRANSPOSED_AXES[" + str(n_dims) + "] = {"
            + ", ".join(map(str, args.transposed_axes))
            + "};\n\n"
        )

        f.write("PI_L2 " + data_marker + " OUT_M[" + str(total_dim) + "];\n\n")

        f.write(
            "PI_L2 "
            + data_marker
            + " IN_M[" + str(total_dim) + "] = {"
            + "f, ".join(map(str, in_matrix.flatten()))
            + "};\n\n"
        )

        f.write(
            "PI_L2 "
            + data_marker
            + " TEST_TRANSPOSE_OUT[" + str(total_dim) + "] = {"
            + "f, ".join(map(str, out_matrix.flatten()))
            + "};\n"
        )

    return None


if __name__ == "__main__":
    main()
