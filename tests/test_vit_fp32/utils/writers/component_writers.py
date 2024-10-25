def get_initialization_text(dim, data_name, filler):
    to_return = (
            "\tfor (int i = 0; i < " + str(dim) + "; i++) " + data_name + "[i] = " + filler
    )
    to_return += "[i];\n" if filler != "zero_init" else ";\n"

    return to_return


def get_connect_text(name, data_name, dim, c, w, h):
    return (
            ("\t" + name + ".data = " + data_name + ";\n")
            + ("\t" + name + ".dim = " + str(dim) + ";\n")
            + ("\t" + name + ".C = " + str(c) + ";\n")
            + ("\t" + name + ".W = " + str(w) + ";\n")
            + ("\t" + name + ".H = " + str(h) + ";\n\n")
    )


def parameter_writer(component, data_marker):
    return None


def concat_writer(component, data_marker):
    return None


def conv2d_writer(
        component_name,
        component,
        data_marker,
):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_conv2d_args"

    input_c, input_w, input_h = component["input_shape"]
    input_dim = input_c * input_w * input_h

    output_c, output_w, output_h = component["output_shape"]
    output_dim = output_c * output_w * output_h

    weight_c = input_c
    weight_w, weight_h = component["weights_shape"][-2:]
    weight_dim = weight_c * weight_w * weight_h * output_c

    bias_c = bias_w = bias_h = bias_dim = 0

    input_name = component_name + "_input"
    weight_name = component_name + "_weight"
    output_name = component_name + "_output"
    bias_name = component_name + "_bias"

    input_data_name = input_name + "_data"
    weight_data_name = weight_name + "_data"
    output_data_name = output_name + "_data"
    bias_data_name = bias_name + "_data"

    if "input_from" in component.keys():
        input_filler = component["input_from"].upper()
    else:
        input_filler = "zero_init"

    weight_filler = component_name.upper() + "_WEIGHT"
    bias_filler = component_name.upper() + "_BIAS"

    output_filler = "zero_init"

    if "bias_shape" in component.keys():
        bias_shape = list(component["bias_shape"])
        if len(bias_shape) == 1:
            bias_shape += [1, 1]

        bias_c, bias_w, bias_h = bias_shape
        bias_dim = bias_c * bias_w * bias_h

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct Conv2D_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define element blobs
    structures_and_blobs += (
            "PI_L2 struct blob "
            + input_name
            + ", "
            + weight_name
            + ", "
            + output_name
            + ";\n"
    )

    if "bias_shape" in component.keys():
        structures_and_blobs += "PI_L2 struct blob " + bias_name + ";\n"
    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + input_data_name + "[" + str(input_dim) + "];\n"
    )
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + weight_data_name + "[" + str(weight_dim) + "];\n"
    )
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_dim) + "];\n"
    )
    if "bias_shape" in component.keys():
        structures_and_blobs += (
                "PI_L2 " + data_marker + " " + bias_data_name + "[" + str(bias_dim) + "];\n"
        )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        input_dim, input_data_name, input_filler
    )
    blob_initializations += get_initialization_text(
        weight_dim, weight_data_name, weight_filler
    )
    blob_initializations += get_initialization_text(
        output_dim, output_data_name, output_filler
    )
    if "bias_shape" in component.keys():
        blob_initializations += get_initialization_text(
            bias_dim, bias_data_name, bias_filler
        )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    # Populate blobs
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        input_name, input_data_name, input_dim, input_c, input_w, input_h
    )
    blob_connect += get_connect_text(
        weight_name, weight_data_name, weight_dim, weight_c, weight_w, weight_h
    )
    blob_connect += get_connect_text(
        output_name, output_data_name, output_dim, output_c, output_w, output_h
    )
    if "bias_shape" in component.keys():
        blob_connect += get_connect_text(
            bias_name, bias_data_name, bias_dim, bias_c, bias_w, bias_h
        )

    # Connect blobs
    blob_connect += "\t" + args_name + ".input = &" + input_name + ";\n"
    blob_connect += "\t" + args_name + ".coeff = &" + weight_name + ";\n"
    blob_connect += "\t" + args_name + ".output = &" + output_name + ";\n"

    blob_connect += "\t" + args_name + ".stride_h = " + str(component["stride_h"]) + ";\n"
    blob_connect += "\t" + args_name + ".stride_w = " + str(component["stride_w"]) + ";\n"

    if "bias_shape" in component.keys():
        blob_connect += "\t" + args_name + ".bias = &" + bias_name + ";\n"
        blob_connect += "\t" + args_name + ".USE_BIASES = 1;\n"
    else:
        blob_connect += "\t" + args_name + ".USE_BIASES = 0;\n"

    return structures_and_blobs, blob_initializations, blob_connect


def add_writer(component, data_marker):
    return None


def layer_norm_writer(component, data_marker):
    return None


def mhsa_writer(component, data_marker):
    return None


def linear_writer(component, data_marker):
    return None


def tanh_writer(component, data_marker):
    return None


def flatten_writer(component, data_marker):
    return None


def transpose_writer(component, data_marker):
    return None
