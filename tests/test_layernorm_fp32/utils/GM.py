import argparse
import torch
import torch.nn as nn
import dump_utils as dump


def create_arg_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_shape_width",
        help="Integer - the width of the input shape.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--input_shape_height",
        help="Integer - the height of the input shape.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--data_type",
        help="Data type to be used.",
        type=str,
        required=False,
        default="fp32",
    )

    return parser


def write_initial_defines(x, step_size):
    f = open("layer_norm_init_defines.h", "w")

    f.write("#define SHAPE " + str(x.numel()) + "\n")
    f.write("#define STEP_SIZE " + str(step_size) + "\n")

    f.close()


def write_input(x, data_identifier):
    f = open("layer_norm_input.h", "w")

    f.write("PI_L2 " + data_identifier + " INPUT[" + str(x.numel()) + "] = {" + dump.tensor_to_string(x) + "};\n")

    f.close()


def write_wb(layer, data_identifier):
    f = open("layer_norm_wb.h", "w")

    f.write("PI_L2 " + data_identifier + " WEIGHT[" + str(layer.weight.numel()) + "] = {" + dump.tensor_to_string(layer.weight) + "};\n")
    f.write("PI_L2 " + data_identifier + " BIAS[" + str(layer.bias.numel()) + "] = {" + dump.tensor_to_string(layer.bias) + "};\n")

    f.close()


def write_output(output, data_identifier):
    f = open("layer_norm_output.h", "w")

    f.write("PI_L2 " + data_identifier + " OUTPUT[" + str(output.numel()) + "] = {" + dump.tensor_to_string(output) + "};\n")

    f.close()


def main():
    # Set the seed for reproducibility
    torch.manual_seed(0)

    # Visualize data with more precision
    torch.set_printoptions(precision=10, sci_mode=False)

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    input_shape = (args.input_shape_width, args.input_shape_height)
    data_type = args.data_type

    if data_type == "fp32":
        data_identifier = "float"

    # Generate input
    x = torch.rand(input_shape)

    # Define layer
    layer = nn.LayerNorm(normalized_shape=[input_shape[1]])

    # Randomize the weight and bias of the layer
    layer.weight = nn.Parameter(torch.rand(layer.weight.shape))
    layer.bias = nn.Parameter(torch.rand(layer.bias.shape))

    # Compute output of layer
    output = layer(x)

    # Write to files
    write_initial_defines(x=x, step_size=input_shape[1])
    write_input(x=x, data_identifier=data_identifier)
    write_wb(layer=layer, data_identifier=data_identifier)
    write_output(output, data_identifier=data_identifier)

    return None


if __name__ == "__main__":
    main()
