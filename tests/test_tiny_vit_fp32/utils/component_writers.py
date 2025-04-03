import numpy as np


def get_initialization_text(dim, data_name, filler):
    to_return = (
        "\tfor (int i = 0; i < " + str(dim) + "; i++) " + data_name + "[i] = " + filler
    )
    to_return += "[i];\n" if filler not in ["zero_init", "min_float"] else ";\n"

    return to_return


def get_connect_text(blob_name, elements):
    text = ""

    for key in elements.keys():
        text += "\t" + blob_name + "." + key + " = " + str(elements[key]) + ";\n"

    text += "\n"

    return text


def adapt_onnx_name(name):
    return "_" + str(name).replace("/", "_").replace(".", "_").replace(":", "_")


def conv_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    # TODO: Implement bias
    # TODO: Implement dilations
    # TODO: Implement group

    # Initial values
    strides = None
    pads = None

    # Clean up component name
    component_name = adapt_onnx_name(node.name)

    # Extract input
    if len(node.input) == 3:
        x, w, b = node.input
    else:
        x, w = node.input
        b = None

    # Extract conv input shape
    # TODO: Turn this into function and use it in every writer
    if isinstance(all_elements[x], dict):
        x_data = all_elements[x]["data"]

        in_b, in_c, in_h, in_w = all_elements[x]["shape"]
        input_shape = (in_c, in_h, in_w)
    else:
        input_shape = tuple(
            [val.dim_value for val in all_elements[x].type.tensor_type.shape.dim][-3:]
        )
        in_b = all_elements[x].type.tensor_type.shape.dim[0].dim_value
        x_data = x

    # Extract conv weights shape
    # out_c, in_c, k_w, k_h
    if isinstance(all_elements[w], np.ndarray):
        weights_shape = all_elements[w].shape
    elif isinstance(all_elements[w], dict):
        weights_shape = all_elements[w]["shape"]
    else:
        weights_shape = tuple(all_elements[w].dims)

    # Extract padding and strides
    for attr in node.attribute:
        if attr.name == "pads":
            pads = tuple(attr.ints)
        elif attr.name == "strides":
            strides = tuple(attr.ints)

    # Compute output shape
    output_shape = (
        int(weights_shape[0]),
        int((input_shape[1] + pads[0] + pads[1] - weights_shape[2]) // strides[0] + 1),
        int((input_shape[2] + pads[2] + pads[3] - weights_shape[3]) // strides[1] + 1),
    )

    # Extract data names
    input_data_name = adapt_onnx_name(x_data)
    weight_data_name = adapt_onnx_name(w)
    output_data_name = adapt_onnx_name(node.output[0])

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": (in_b,) + output_shape,
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_conv2d_args"

    input_c, input_w, input_h = input_shape
    input_dim = input_c * input_w * input_h

    output_c, output_w, output_h = output_shape
    output_dim = output_c * output_w * output_h

    weight_c = input_c
    weight_w, weight_h = weights_shape[-2:]
    weight_dim = weight_c * weight_w * weight_h * output_c

    input_name = component_name + "_input_blob"
    weight_name = component_name + "_weight_blob"
    output_name = component_name + "_output_blob"

    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct Conv2D_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define element blobs
    structures_and_blobs += "PI_L2 struct blob " + input_name + ";\n"
    structures_and_blobs += "PI_L2 struct blob " + weight_name + ";\n"
    structures_and_blobs += "PI_L2 struct blob " + output_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_dim) + "];\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        input_name,
        {
            "data": input_data_name,
            "dim": input_dim,
            "C": input_c,
            "W": input_w,
            "H": input_h,
        },
    )

    blob_connect += get_connect_text(
        weight_name,
        {
            "data": weight_data_name,
            "dim": weight_dim,
            "C": weight_c,
            "W": weight_w,
            "H": weight_h,
        },
    )

    blob_connect += get_connect_text(
        output_name,
        {
            "data": output_data_name,
            "dim": output_dim,
            "C": output_c,
            "W": output_w,
            "H": output_h,
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = &" + input_name + ";\n"
    blob_connect += "\t" + args_name + ".coeff = &" + weight_name + ";\n"
    blob_connect += "\t" + args_name + ".output = &" + output_name + ";\n"

    blob_connect += "\t" + args_name + ".stride_h = " + str(strides[0]) + ";\n"
    blob_connect += "\t" + args_name + ".stride_w = " + str(strides[1]) + ";\n"

    blob_connect += "\t" + args_name + ".Lpad = " + str(pads[0]) + ";\n"
    blob_connect += "\t" + args_name + ".Rpad = " + str(pads[1]) + ";\n"
    blob_connect += "\t" + args_name + ".Upad = " + str(pads[2]) + ";\n"
    blob_connect += "\t" + args_name + ".Dpad = " + str(pads[3]) + ";\n"

    blob_connect += "\t" + args_name + ".USE_BIASES = 0;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += "\tpulp_conv2d_fp32_fw_cl(&" + args_name + ");\n\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function
