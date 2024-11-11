import torch

from utils.dump_utils import tensor_to_string
from utils.torch_to_trainlib import VIT_COMPONENTS_WRITERS

IMPLEMENTED_DATA_TYPES = ["fp32"]


def header_writer(data_type):
    # Open file
    f = open("net_args.h", "w")

    f.write("#ifndef NET_ARGS_H\n")
    f.write("#define NET_ARGS_H\n\n")

    # Introductory elements
    if data_type == "fp32":
        f.write("// FLOAT 32 ViT model\n")
        f.write("#define FLOAT32\n")

    # Write necessary dimensions

    # Close file
    f.write("\n#endif\n")
    f.close()

    return None


def input_writer(data, data_type):
    # Check passed arguments
    assert isinstance(data, torch.Tensor), "Invalid data"

    # Open file
    f = open("input_sequence.h", "w")

    f.write("#ifndef INPUT_SEQUENCE_H\n")
    f.write("#define INPUT_SEQUENCE_H\n\n")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    else:
        data_marker = None
    f.write("#define INPUT_SIZE " + str(data.numel()) + "\n\n")

    # Write actual data
    f.write(
        "PI_L2 "
        + data_marker
        + " INPUT[INPUT_SIZE] = {"
        + tensor_to_string(data)
        + "};\n"
    )

    # Close file
    f.write("\n#endif\n")
    f.close()

    return None


def model_writer(model, data_type):
    # Check passed arguments
    assert isinstance(model, torch.nn.Module), "Invalid model"

    # Open file
    f = open("model_defines.h", "w")

    f.write("#ifndef MODEL_DEFINES_H\n")
    f.write("#define MODEL_DEFINES_H\n\n")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    else:
        data_marker = None

    # Write actual data
    for i, (name, el) in enumerate(model.named_parameters()):
        print("[" + str(i) + "] Working on: " + name)

        # Get the variable name
        var_name = name.replace(".", "_").upper()

        # Flag to transpose input q, k, v weights, in the mhsa layer
        to_transpose = False
        to_be_transposed = ["ATTN_PROJ_Q_WEIGHT", "ATTN_PROJ_K_WEIGHT", "ATTN_PROJ_V_WEIGHT"]
        for tbt in to_be_transposed:
            if tbt in var_name:
                to_transpose = True
                break

        # Write to file
        f.write(
            "PI_L2 "
            + data_marker
            + " "
            + var_name
            + "["
            + str(el.numel())
            + "] = {"
            + tensor_to_string(el if not to_transpose else el.t())
            + "};\n\n"
        )

    # Close file
    f.write("\n#endif\n")
    f.close()

    return None


def output_writer(data, data_type):
    # Check passed arguments
    assert isinstance(data, torch.Tensor), "Invalid data"

    # Open file
    f = open("output_sequence.h", "w")

    f.write("#ifndef OUTPUT_SEQUENCE_H\n")
    f.write("#define OUTPUT_SEQUENCE_H\n\n")

    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    else:
        data_marker = None
    f.write("#define OUTPUT_SIZE " + str(data.numel()) + "\n\n")

    # Write actual data
    f.write(
        "PI_L2 "
        + data_marker
        + " OUTPUT[OUTPUT_SIZE] = {"
        + tensor_to_string(data)
        + "};\n"
    )

    # Close file
    f.write("\n#endif\n")
    f.close()

    return None


def model_components_writer(ordered_nodes, all_nodes, data_type):
    # Introductory elements
    if data_type == "fp32":
        data_marker = "float"
    else:
        data_marker = None

    structures_and_blobs = ""
    blob_initializations = ""
    blob_connect = ""
    forward_function = ""

    # Write actual data
    for node in ordered_nodes:
        if node in VIT_COMPONENTS_WRITERS.keys():
            text_content = VIT_COMPONENTS_WRITERS[node](
                component_name=node,
                component=all_nodes[node],
                data_marker=data_marker,
            )

            if text_content is not None:
                structures_and_blobs += text_content[0]
                blob_initializations += text_content[1]
                blob_connect += text_content[2]
                forward_function += text_content[3]

    # Write to header
    f = open("model_components.h", "w")
    f.write("#ifndef MODEL_COMPONENTS_H\n")
    f.write("#define MODEL_COMPONENTS_H\n\n")
    f.write("\n")

    f.write("// =============== Includes ===============\n")
    f.write("#include \"input_sequence.h\"\n")
    f.write("#include \"output_sequence.h\"\n")
    f.write("\n")
    f.write("#include \"model_defines.h\"\n")
    f.write("\n")

    f.write("// =============== Constants definition ===============\n")
    f.write("PI_L1 float zero_init = 0.0f;\n")
    f.write("PI_L1 float min_float = -340282346638528859811704183484516925440.0f;\n")
    f.write("\n")

    f.write("// =============== Structures and blobs definition ===============\n")
    f.write(structures_and_blobs)

    # f.write("void init_and_connect_blobs();\n")
    #
    # f.close()
    #
    # # Write to file
    # f = open("model_components.c", "w")
    #
    # f.write("#include \"model_components.h\"\n")
    # f.write("\n")

    f.write("void init_and_connect_blobs() {\n")
    f.write("\t// =============== Initializations ===============\n")
    f.write(blob_initializations)

    f.write(
        "\t// =============== Populating and connecting blobs to structures ===============\n"
    )
    f.write(blob_connect)

    f.write("}\n")

    # Write forward function
    f.write(
        "\n\n// =============== The forward function ===============\n"
    )
    f.write("void forward() {\n")
    f.write(forward_function)
    f.write("}\n")

    # FIXME: Remove if splitting back into header-source
    f.write("\n#endif\n")
    f.close()

    return None
