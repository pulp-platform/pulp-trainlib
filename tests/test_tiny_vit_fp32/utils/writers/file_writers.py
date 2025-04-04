import os

import numpy as np
import torch


def add_f(x):
    return str(float(x)) + "f"


def tensor_to_string(tensor):
    if isinstance(tensor, torch.Tensor) or isinstance(tensor, np.ndarray):
        return ", ".join(map(add_f, tensor.flatten().tolist()))
    elif isinstance(tensor, int):
        return str(int)
    else:
        raise ValueError("Unsupported tensor type: " + str(type(tensor)))


def model_components_writer(
    file_root_dir,
    structures_and_blobs,
    blob_initializations,
    blob_connect,
    forward_function,
    output_array_name,
):
    # Write to header
    f = open(os.path.join(file_root_dir, "model_components.h"), "w")

    f.write("#ifndef MODEL_COMPONENTS_H\n")
    f.write("#define MODEL_COMPONENTS_H\n\n")

    f.write("\n")

    f.write("// =============== Includes ===============\n")
    f.write('#include "input_sequence.h"\n')
    f.write('#include "output_sequence.h"\n')
    f.write('#include "tensor_checkers.h"\n')
    f.write('#include "model_defines.h"\n')
    f.write("\n")

    f.write("// =============== Constants definition ===============\n")
    f.write("PI_L1 float zero_init = 0.0f;\n")
    f.write("PI_L1 float min_float = -340282346638528859811704183484516925440.0f;\n")
    f.write("\n\n")

    f.write("// =============== Structures and blobs definition ===============\n")
    f.write(structures_and_blobs)

    f.write("void init_and_connect_blobs() {\n")
    f.write("\t// =============== Initializations ===============\n")
    f.write(blob_initializations)

    f.write(
        "\t// =============== Populating and connecting blobs to structures ===============\n"
    )
    f.write(blob_connect)

    f.write("}\n")

    # Write forward function
    f.write("\n\n// =============== The forward function ===============\n")
    f.write("void forward() {\n")
    f.write(forward_function)
    f.write("}\n")

    # Write output check function
    f.write("\n\n// =============== The output check function ===============\n")
    f.write("void check_output() {\n")
    f.write("\tmean_error_checker(" + output_array_name + ", OUTPUT, OUTPUT_SIZE);\n")
    f.write("\telementwise_checker(" + output_array_name + ", OUTPUT, OUTPUT_SIZE);\n")
    f.write("}\n")

    f.write("\n#endif\n")
    f.close()


def input_writer(file_root_dir, input_name, input_array):
    f = open(os.path.join(file_root_dir, "input_sequence.h"), "w")

    f.write("#ifndef INPUT_SEQUENCE_H\n")
    f.write("#define INPUT_SEQUENCE_H\n\n")

    f.write("#define INPUT_SIZE " + str(input_array.numel()) + "\n\n")

    f.write(
        "PI_L2 float "
        + input_name
        + "[INPUT_SIZE] = { "
        + tensor_to_string(input_array)
        + " };\n"
    )

    f.write("\n#endif\n")
    f.close()


def output_writer(file_root_dir, output_array):
    # Open file
    f = open(os.path.join(file_root_dir, "output_sequence.h"), "w")

    f.write("#ifndef OUTPUT_SEQUENCE_H\n")
    f.write("#define OUTPUT_SEQUENCE_H\n\n")

    # Introductory elements
    f.write("#define OUTPUT_SIZE " + str(output_array.numel()) + "\n\n")

    # Write actual data
    f.write(
        "PI_L2 float OUTPUT[OUTPUT_SIZE] = {" + tensor_to_string(output_array) + "};\n"
    )

    # Close file
    f.write("\n#endif\n")
    f.close()

    return None


def parameters_writer(file_root_dir, parameter_arrays):
    f = open(os.path.join(file_root_dir, "model_defines.h"), "w")

    f.write("#ifndef MODEL_DEFINES_H\n")
    f.write("#define MODEL_DEFINES_H\n\n")

    for key in parameter_arrays.keys():
        el = parameter_arrays[key]

        if isinstance(el, torch.Tensor):
            el_size = el.numel()
        elif isinstance(el, np.ndarray):
            el_size = el.size
        elif isinstance(el, int):
            el_size = 1
        else:
            raise ValueError("Unsupported parameter type: " + str(type(el)))

        f.write(
            "PI_L2 float "
            + key
            + "["
            + str(int(el_size))
            + "] = {"
            + tensor_to_string(el)
            + "};\n\n"
        )

    f.write("\n#endif\n")
    f.close()
