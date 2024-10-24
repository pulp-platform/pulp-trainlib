import torch

from utils.dump_utils import tensor_to_string

IMPLEMENTED_DATA_TYPES = ["fp32"]


def header_writer(data_type):
    # Check passed arguments
    assert data_type in IMPLEMENTED_DATA_TYPES, "Invalid data type"

    # Open file
    f = open("net_args.h", "w")

    # Introductory elements
    if data_type == "fp32":
        f.write("// FLOAT 32 ViT model\n")
        f.write("#define FLOAT32\n\n")

    # Write necessary dimensions

    # Close file
    f.close()

    return None


def input_writer(data, data_type):
    # Check passed arguments
    assert isinstance(data, torch.Tensor), "Invalid data"
    assert data_type in IMPLEMENTED_DATA_TYPES, "Invalid data type"

    # Open file
    f = open("input-sequence.h", "w")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    f.write("#define INPUT_SIZE " + str(data.numel()) + "\n\n")

    # Write actual data
    f.write(
        "PI_L2 "
        + data_marker
        + " INPUT[INPUT_SIZE] = {"
        + tensor_to_string(data)
        + "};\n\n"
    )

    # Close file
    f.close()

    return None


def model_writer(model, data_type):
    # Check passed arguments
    assert data_type in IMPLEMENTED_DATA_TYPES, "Invalid data type"

    # Open file
    f = open("model-defines.h", "w")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"

    # Write actual data
    total_params = len(list(model.parameters()))
    for i, (name, el) in enumerate(model.named_parameters()):
        print("[" + str(i + 1) + "/" + str(total_params) + "] Writing " + name + "...")
        f.write(
            "PI_L2 "
            + data_marker
            + " "
            + name.replace(".", "_").upper()
            + "["
            + str(el.numel())
            + "] = {"
            + tensor_to_string(el)
            + "};\n\n"
        )

    # Close file
    f.close()

    return None


def output_writer(data, data_type):
    # Check passed arguments
    assert isinstance(data, torch.Tensor), "Invalid data"
    assert data_type in IMPLEMENTED_DATA_TYPES, "Invalid data type"

    # Open file
    f = open("output-sequence.h", "w")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    f.write("#define OUTPUT_SIZE " + str(data.numel()) + "\n")

    # Write actual data
    f.write(
        "PI_L2 "
        + data_marker
        + " OUTPUT[OUTPUT_SIZE] = {"
        + tensor_to_string(data)
        + "};\n\n"
    )

    # Close file
    f.close()

    return None
