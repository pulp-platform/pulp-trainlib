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


def extract_sizes(shape):
    if len(shape) == 1:
        w = shape[0]
        h = 1
        c = 1
    elif len(shape) == 2:
        c = 1
        w, h = shape
    else:
        c, w, h = shape

    return c * w * h, w, h, c


def copy_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    if "copy" in component_name:
        args_name = component_name + "_args"
    else:
        args_name = component_name + "_copy_args"

    if component["available_input"]:
        input_data_name = component["input_from"].upper()
    else:
        input_data_name = component["input_from"] + "_output_data"

    if len(component["output_shape"]) == 2:
        output_w, output_h = component["output_shape"]
        output_c = 1
    else:
        output_c, output_w, output_h = component["output_shape"]
    output_dim = output_c * output_w * output_h

    output_data_name = component["output_target"]

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct copy_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".from = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".to = " + output_data_name + " + " + str(component["output_offset"]) + ";\n"
    blob_connect += "\t" + args_name + ".size = " + str(output_dim) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, copy, &" + args_name + ");\n\n"

    return structures_and_blobs, "", blob_connect, forward_function


def concat_writer(component_name, component, data_marker):
    # Initialize
    structures_and_blobs, blob_initializations, blob_connect, forward_function = "", "", "", ""

    # Prepare output target
    output_data_name = component_name + "_output_data"
    output_dim = 0
    output_filler = "zero_init"
    for i, el in enumerate(component["input_from"]):
        element_size = 1
        for size in component["input_shape"][i]:
            element_size *= size
        output_dim += element_size

    structures_and_blobs += "// " + component_name.upper() + "\n"
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_dim) + "];\n"
    )
    structures_and_blobs += "\n"

    blob_initializations = "\t// " + component_name.upper() + "\n"
    blob_initializations += get_initialization_text(
        output_dim, output_data_name, output_filler
    )
    blob_initializations += "\n"

    output_offset = 0
    for i, el in enumerate(component["input_from"]):
        copy_name = component_name + "_copy_" + str(i)
        copy_component = {
            "input_from": el,
            "output_target": output_data_name,
            "available_input": component["available_input"][i],
            "output_shape": component["input_shape"][i],
            "output_offset": output_offset,
        }

        r1, r2, r3, r4 = copy_writer(copy_name, copy_component, data_marker)

        # Compute new offset
        current_size = 1
        for size in component["input_shape"][i]:
            current_size *= size
        output_offset += current_size

        # Store text
        structures_and_blobs += r1
        blob_initializations += r2
        blob_connect += r3
        forward_function += r4

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


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
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        input_name,
        {
            "data": input_data_name,
            "dim": input_dim,
            "C": input_c,
            "W": input_w,
            "H": input_h,
        }
    )
    blob_connect += get_connect_text(
        weight_name,
        {
            "data": weight_data_name,
            "dim": weight_dim,
            "C": weight_c,
            "W": weight_w,
            "H": weight_h,
        }
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
    if "bias_shape" in component.keys():
        blob_connect += get_connect_text(
            bias_name,
            {
                "data": bias_data_name,
                "dim": bias_dim,
                "C": bias_c,
                "W": bias_w,
                "H": bias_h,
            },
        )

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
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

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpulp_conv2d_fp32_fw_cl(&" + args_name + ");\n\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def vector_sum_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_vect_sum_args"

    if component["available_input"][0]:
        input_0_data_name = component["input_from"][0].upper()
    elif component["input_from"][0].endswith("output_data"):
        input_0_data_name = component["input_from"][0]
    else:
        input_0_data_name = component["input_from"][0] + "_output_data"

    if component["available_input"][1]:
        input_1_data_name = component["input_from"][1].upper()
    elif component["input_from"][1].endswith("output_data"):
        input_1_data_name = component["input_from"][1]
    else:
        input_1_data_name = component["input_from"][1] + "_output_data"

    if len(component["shape"]) == 2:
        w, h = component["shape"]
        c = 1
    else:
        c, w, h = component["shape"]
    dim = c * w * h

    output_data_name = component_name + "_output_data"
    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct array_broadcast_sum_fp32_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
            "PI_L2 "
            + data_marker
            + " "
            + output_data_name
            + "["
            + str(dim)
            + "];\n"
    )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".op_1 = " + input_0_data_name + ";\n"
    blob_connect += "\t" + args_name + ".op_2 = " + input_1_data_name + ";\n"
    blob_connect += "\t" + args_name + ".dest = " + output_data_name + ";\n\n"

    blob_connect += (
            "\t"
            + args_name
            + ".op_1_dims = {"
            + ", ".join(map(str, [dim,]))
            + "};\n"
    )

    blob_connect += (
            "\t"
            + args_name
            + ".op_2_dims = {"
            + ", ".join(map(str, [dim,]))
            + "};\n\n"
    )

    blob_connect += "\t" + args_name + ".op_1_dims_len = 1;\n"
    blob_connect += "\t" + args_name + ".op_2_dims_len = 1;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += (
            "\tpi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &"
            + args_name
            + ");\n\n"
    )

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def layer_norm_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    total_dimension = 1
    for el in list(component["shape"]):
        total_dimension *= el

    # TODO: This only works for 2-dimensional matrices, where the normalized shape is the second dimension!
    step_size = list(component["shape"])[-1]

    component_blob_name = component_name + "_args"

    output_data_name = component_name + "_output_data"
    eps_data_name = component_name + "_eps"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct LayerNorm_args_fp32 " + component_blob_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += "PI_L1 " + data_marker + " " + eps_data_name + "[1] = {" + str(component["eps"]) + "};\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + output_data_name + "[" + str(total_dimension) + "];\n"

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initialization = "\t// " + component_name.upper() + "\n"

    blob_initialization += get_initialization_text(total_dimension, component_name + "_output_data", "zero_init")

    blob_initialization += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        component_blob_name,
        {
            "x": component["input"],
            "weight": component_name.upper() + "_WEIGHT",
            "bias": component_name.upper() + "_BIAS",
            "output": output_data_name,
            "eps": eps_data_name,
            "size": total_dimension,
            "step_size": step_size,
        }
    )

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, pulp_layerNorm_fp32_fw_cl, &" + component_blob_name + ");\n\n"

    return structures_and_blobs, blob_initialization, blob_connect, forward_function


def mhsa_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    n_heads = component["n_heads"]

    struct_name = component_name + "_args"

    in_blob_name = component_name + "_in"
    wgt_in_q_blob_name = component_name + "_wgt_in_q"
    wgt_in_k_blob_name = component_name + "_wgt_in_k"
    wgt_in_v_blob_name = component_name + "_wgt_in_v"
    bias_in_q_blob_name = component_name + "_bias_in_q"
    bias_in_k_blob_name = component_name + "_bias_in_k"
    bias_in_v_blob_name = component_name + "_bias_in_v"
    wgt_out_blob_name = component_name + "_wgt_out"
    q_blob_name = component_name + "_q"
    k_blob_name = component_name + "_k"
    v_blob_name = component_name + "_v"
    att_map_blob_name = component_name + "_att_map"
    softmax_buffer_blob_name = component_name + "_softmax_buffer"
    out_blob_name = component_name + "_out"

    q_data_name = component_name + "_q_data"
    k_data_name = component_name + "_k_data"
    v_data_name = component_name + "_v_data"

    att_map_data_name = component_name + "_att_map_data"
    softmax_buffer_data_name = component_name + "_softmax_buffer_data"
    out_data_name = component_name + "_output_data"
    temp_data_name = component_name + "_temp_data"

    sums_data_name = component_name + "_sums_data"
    maxes_data_name = component_name + "_maxes_data"

    q_data_size, q_data_w, q_data_h, q_data_c = extract_sizes(component["q_shape"])
    k_data_size, k_data_w, k_data_h, k_data_c = extract_sizes(component["k_shape"])
    v_data_size, v_data_w, v_data_h, v_data_c = extract_sizes(component["v_shape"])

    att_map_data_size, att_map_data_w, att_map_data_h, att_map_data_c = extract_sizes(component["att_map_shape"])
    softmax_buffer_data_size, softmax_buffer_data_w, softmax_buffer_data_h, softmax_buffer_data_c = extract_sizes(component["softmax_buffer_shape"])
    out_data_size, out_data_w, out_data_h, out_data_c = extract_sizes(component["output_shape"])

    temp_data_size = extract_sizes(component["temp_shape"])[0]

    sums_data_size, sums_data_w, sums_data_h, sums_data_c = extract_sizes(component["sm_shape"])
    maxes_data_size, maxes_data_w, maxes_data_h, maxes_data_c = extract_sizes(component["sm_shape"])

    in_data_size, in_data_w, in_data_h, in_data_c = extract_sizes(component["input_shape"])

    wgt_in_q_data_size, wgt_in_q_data_w, wgt_in_q_data_h, wgt_in_q_data_c = extract_sizes(component["wgt_in_q_shape"])
    wgt_in_k_data_size, wgt_in_k_data_w, wgt_in_k_data_h, wgt_in_k_data_c = extract_sizes(component["wgt_in_k_shape"])
    wgt_in_v_data_size, wgt_in_v_data_w, wgt_in_v_data_h, wgt_in_v_data_c = extract_sizes(component["wgt_in_v_shape"])

    bias_in_q_data_size, bias_in_q_data_w, bias_in_q_data_h, bias_in_q_data_c = extract_sizes(component["bias_in_q_shape"])
    bias_in_k_data_size, bias_in_k_data_w, bias_in_k_data_h, bias_in_k_data_c = extract_sizes(component["bias_in_k_shape"])
    bias_in_v_data_size, bias_in_v_data_w, bias_in_v_data_h, bias_in_v_data_c = extract_sizes(component["bias_in_v_shape"])

    wgt_out_data_size, wgt_out_data_w, wgt_out_data_h, wgt_out_data_c = extract_sizes(component["wgt_proj_out_shape"])

    zero_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct Mhsa_args " + struct_name + ";\n"

    structures_and_blobs += "\n"

    # Define element blobs
    structures_and_blobs += ("PI_L2 struct blob\n" +
                             "\t" + in_blob_name + ",\n" +
                             "\t" + wgt_in_q_blob_name + ",\n" +
                             "\t" + wgt_in_k_blob_name + ",\n" +
                             "\t" + wgt_in_v_blob_name + ",\n" +
                             "\t" + bias_in_q_blob_name + ",\n" +
                             "\t" + bias_in_k_blob_name + ",\n" +
                             "\t" + bias_in_v_blob_name + ",\n" +
                             "\t" + wgt_out_blob_name + ",\n" +
                             "\t" + q_blob_name + ",\n" +
                             "\t" + k_blob_name + ",\n" +
                             "\t" + v_blob_name + ",\n" +
                             "\t" + att_map_blob_name + ",\n" +
                             "\t" + softmax_buffer_blob_name + ",\n" +
                             "\t" + out_blob_name + ";\n"
                             )

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += "PI_L2 " + data_marker + " " + q_data_name + "[" + str(q_data_size) + "];\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + k_data_name + "[" + str(k_data_size) + "];\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + v_data_name + "[" + str(v_data_size) + "];\n"

    structures_and_blobs += "PI_L2 " + data_marker + " " + att_map_data_name + "[" + str(att_map_data_size) + "];\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + softmax_buffer_data_name + "[" + str(softmax_buffer_data_size) + "];\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + out_data_name + "[" + str(out_data_size) + "];\n"

    structures_and_blobs += "PI_L2 " + data_marker + " " + temp_data_name + "[" + str(temp_data_size) + "];\n"

    structures_and_blobs += "PI_L2 " + data_marker + " " + sums_data_name + "[" + str(sums_data_size) + "];\n"
    structures_and_blobs += "PI_L2 " + data_marker + " " + maxes_data_name + "[" + str(maxes_data_size) + "];\n"

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(q_data_size, q_data_name, zero_filler)
    blob_initializations += get_initialization_text(k_data_size, k_data_name, zero_filler)
    blob_initializations += get_initialization_text(v_data_size, v_data_name, zero_filler)
    blob_initializations += get_initialization_text(att_map_data_size, att_map_data_name, zero_filler)
    blob_initializations += get_initialization_text(softmax_buffer_data_size, softmax_buffer_data_name, zero_filler)
    blob_initializations += get_initialization_text(out_data_size, out_data_name, zero_filler)
    blob_initializations += get_initialization_text(temp_data_size, temp_data_name, zero_filler)
    blob_initializations += get_initialization_text(sums_data_size, sums_data_name, zero_filler)
    blob_initializations += get_initialization_text(maxes_data_size, maxes_data_name, "min_float")

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        in_blob_name,
        {
            "data": component["input"],
            "dim": in_data_size,
            "W": in_data_w,
            "H": in_data_h,
            "C": in_data_c,
        },
    )

    blob_connect += get_connect_text(
        wgt_in_q_blob_name,
        {
            "data": component_name.upper() + "_PROJ_Q_WEIGHT",
            "dim": wgt_in_q_data_size,
            "W": wgt_in_q_data_w,
            "H": wgt_in_q_data_h,
            "C": wgt_in_q_data_c,
        },
    )

    blob_connect += get_connect_text(
        wgt_in_k_blob_name,
        {
            "data": component_name.upper() + "_PROJ_K_WEIGHT",
            "dim": wgt_in_k_data_size,
            "W": wgt_in_k_data_w,
            "H": wgt_in_k_data_h,
            "C": wgt_in_k_data_c,
        },
    )

    blob_connect += get_connect_text(
        wgt_in_v_blob_name,
        {
            "data": component_name.upper() + "_PROJ_V_WEIGHT",
            "dim": wgt_in_v_data_size,
            "W": wgt_in_v_data_w,
            "H": wgt_in_v_data_h,
            "C": wgt_in_v_data_c,
        },
    )

    blob_connect += get_connect_text(
        bias_in_q_blob_name,
        {
            "data": component_name.upper() + "_PROJ_Q_BIAS",
            "dim": bias_in_q_data_size,
            "W": bias_in_q_data_w,
            "H": bias_in_q_data_h,
            "C": bias_in_q_data_c,
        },
    )

    blob_connect += get_connect_text(
        bias_in_k_blob_name,
        {
            "data": component_name.upper() + "_PROJ_K_BIAS",
            "dim": bias_in_k_data_size,
            "W": bias_in_k_data_w,
            "H": bias_in_k_data_h,
            "C": bias_in_k_data_c,
        },
    )

    blob_connect += get_connect_text(
        bias_in_v_blob_name,
        {
            "data": component_name.upper() + "_PROJ_V_BIAS",
            "dim": bias_in_v_data_size,
            "W": bias_in_v_data_w,
            "H": bias_in_v_data_h,
            "C": bias_in_v_data_c,
        },
    )

    blob_connect += get_connect_text(
        wgt_out_blob_name,
        {
            "data": component_name.upper() + "_PROJ_OUT_WEIGHT",
            "dim": wgt_out_data_size,
            "W": wgt_out_data_w,
            "H": wgt_out_data_h,
            "C": wgt_out_data_c,
        },
    )

    blob_connect += get_connect_text(
        q_blob_name,
        {
            "data": q_data_name,
            "dim": q_data_size,
            "W": q_data_w,
            "H": q_data_h,
            "C": q_data_c,
        },
    )

    blob_connect += get_connect_text(
        k_blob_name,
        {
            "data": k_data_name,
            "dim": k_data_size,
            "W": k_data_w,
            "H": k_data_h,
            "C": k_data_c,
        },
    )

    blob_connect += get_connect_text(
        v_blob_name,
        {
            "data": v_data_name,
            "dim": v_data_size,
            "W": v_data_w,
            "H": v_data_h,
            "C": v_data_c,
        },
    )

    blob_connect += get_connect_text(
        out_blob_name,
        {
            "data": out_data_name,
            "dim": out_data_size,
            "W": out_data_w,
            "H": out_data_h,
            "C": out_data_c,
        },
    )

    blob_connect += get_connect_text(
        att_map_blob_name,
        {
            "data": att_map_data_name,
            "dim": att_map_data_size,
            "W": att_map_data_w,
            "H": att_map_data_h,
            "C": att_map_data_c,
        },
    )

    blob_connect += get_connect_text(
        softmax_buffer_blob_name,
        {
            "data": softmax_buffer_data_name,
            "dim": softmax_buffer_data_size,
            "W": softmax_buffer_data_w,
            "H": softmax_buffer_data_h,
            "C": softmax_buffer_data_c,
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + struct_name + ".input = &" + in_blob_name + ";\n"

    blob_connect += "\t" + struct_name + ".n_heads = " + str(n_heads) + ";\n\n"

    blob_connect += "\t" + struct_name + ".q = &" + q_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".k = &" + k_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".v = &" + v_blob_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".output = &" + out_blob_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".coeff_in_q = &" + wgt_in_q_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".coeff_in_k = &" + wgt_in_k_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".coeff_in_v = &" + wgt_in_v_blob_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".bias_in_q = &" + bias_in_q_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".bias_in_k = &" + bias_in_k_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".bias_in_v = &" + bias_in_v_blob_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".coeff_out = &" + wgt_out_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".attention_map = &" + att_map_blob_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".temp_buffer = " + temp_data_name + ";\n"
    blob_connect += "\t" + struct_name + ".softmax_buffer = &" + softmax_buffer_blob_name + ";\n"
    blob_connect += "\t" + struct_name + ".sums = " + sums_data_name + ";\n"
    blob_connect += "\t" + struct_name + ".maxes = " + maxes_data_name + ";\n\n"

    blob_connect += "\t" + struct_name + ".opt_matmul_type_fw = MATMUL_TYPE;\n"
    blob_connect += "\t" + struct_name + ".opt_matmul_type_wg = MATMUL_TYPE;\n"
    blob_connect += "\t" + struct_name + ".opt_matmul_type_ig = MATMUL_TYPE;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpulp_mhsa_fp32_fw_cl(&" + struct_name + ");\n\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def linear_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    input_data_a_name = component["input_a"]
    input_data_b_name = component_name.upper() + "_WEIGHT"

    _, n_size, k_size, __ = extract_sizes(component["input_a_shape"])
    output_dim, _, m_size, __ = extract_sizes(component["output_shape"])
    bias_size, _, __, ___ = extract_sizes(component["bias_shape"])

    man_args_name = component_name + "_man_args"
    args_name = component_name + "_args"

    bias_data_name = component_name.upper() + "_BIAS"

    output_data_name = component_name + "_output_data"
    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct matMul_args " + args_name + ";\n"
    structures_and_blobs += "PI_L2 struct mm_manager_args " + man_args_name + ";\n"

    structures_and_blobs += "\n"

    # Define blobs

    # Define data variables
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_dim) + "];\n"
    )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += get_connect_text(
        args_name,
        {
            "A": input_data_a_name,
            "B": input_data_b_name,
            "C": output_data_name,
            "bias": bias_data_name,
            "N": n_size,
            "K": k_size,
            "M": m_size,
            "trans_B": 1,
            "USE_BIASES": 1,
            "bias_dim": bias_size,
        }
    )

    blob_connect += "\t" + man_args_name + ".mm_args = &" + args_name + ";\n"
    blob_connect += "\t" + man_args_name + ".layer_type = LAYER_LINEAR;\n"
    blob_connect += "\t" + man_args_name + ".step_type = STEP_FW;\n"
    blob_connect += "\t" + man_args_name + ".matmul_type = MATMUL_TYPE;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"

    forward_function += "\tpi_cl_team_fork(NUM_CORES, pulp_linear_fp32_fw_cl_kernel, &" + man_args_name + ");\n"

    forward_function += "\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def tanh_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    input_data_name = component["input"]

    output_data_name = component_name + "_output_data"
    output_filler = "zero_init"

    dim = extract_sizes(component["shape"])[0]

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct tanh_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(dim) + "];\n"
    )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".output = " + output_data_name + ";\n"
    blob_connect += "\t" + args_name + ".dim = " + str(dim) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, tanh_prll, &" + args_name + ");\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def transpose_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_transpose_args"

    if "input_from" in component.keys():
        input_data_name = component["input_from"] + "_output_data"
    else:
        raise NotImplementedError("Transpose component must have an input_from key")

    output_w, output_h = component["output_shape"]
    output_c = 1
    output_dim = output_c * output_w * output_h

    output_data_name = component_name + "_output_data"

    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct transp_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_dim) + "];\n"
    )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + "_dims = {" + str(output_w) + ", " + str(output_h) + "};\n"
    blob_connect += "\t" + args_name + "_tr_axes = {1, 0};\n"

    blob_connect += "\t" + args_name + ".in_matrix = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".out_matrix = " + output_data_name + ";\n"
    blob_connect += "\t" + args_name + ".dim = " + args_name + "_dims;\n"
    blob_connect += "\t" + args_name + ".transposed_axes = " + args_name + "_tr_axes;\n"
    blob_connect += "\t" + args_name + ".n_dim = 2;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, transpose, &" + args_name + ");\n\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def gelu_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    input_blob_name = component_name + "_input_blob"
    output_blob_name = component_name + "_output_blob"

    output_data_name = component_name + "_output_data"

    dim, w, h, c = extract_sizes(component["shape"])

    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct act_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define element blobs
    structures_and_blobs += (
            "PI_L2 struct blob "
            + input_blob_name
            + ", "
            + output_blob_name
            + ";\n"
    )
    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
            "PI_L2 " + data_marker + " " + output_data_name + "[" + str(dim) + "];\n"
    )

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        dim, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        input_blob_name,
        {
            "data": component["input"],
            "dim": dim,
            "C": c,
            "W": w,
            "H": h,
        }
    )

    blob_connect += get_connect_text(
        output_blob_name,
        {
            "data": output_data_name,
            "dim": dim,
            "C": c,
            "W": w,
            "H": h,
        }
    )

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = &" + input_blob_name + ";\n"
    blob_connect += "\t" + args_name + ".output = &" + output_blob_name + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, pulp_gelu_tanh_approx_fp32_fw_cl, &" + args_name + ");\n\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function


def matmul_writer(component_name, component, data_marker):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_mm_args"
    man_args_name = "man_" + args_name

    input_data_a_name = component["input_a"]
    input_data_b_name = component["input_b"]

    output_data_name = component_name + "_output_data"

    _, n_size, k_size, __ = extract_sizes(component["input_a_shape"])
    output_size, _, m_size, __ = extract_sizes(component["output_shape"])

    filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct matMul_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define blobs

    # Define data variables
    structures_and_blobs += "PI_L2 " + data_marker + " " + output_data_name + "[" + str(output_size) + "];\n"

    structures_and_blobs += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_size, output_data_name, filler
    )

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += get_connect_text(
        args_name,
        {
            "A": input_data_a_name,
            "B": input_data_b_name,
            "C": output_data_name,
            "N": n_size,
            "K": k_size,
            "M": m_size,
            # TODO: Make these dynamic
            "trans_B": 0,
        }
    )

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"

    forward_function += "\t#ifdef DEBUG\n"
    forward_function += "\tprintf(\"Working on " + component_name + "...\\n\");\n"
    forward_function += "\t#endif\n\n"

    forward_function += "\t#ifndef OPTIMIZE\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, mm, &" + args_name + ");\n"
    forward_function += "\t#else\n"
    forward_function += "\tstruct mm_manager_args " + man_args_name + ";\n"
    forward_function += "\t" + man_args_name + ".mm_args = &" + args_name + ";\n"
    forward_function += "\t" + man_args_name + ".layer_type = LAYER_LINEAR;\n"
    forward_function += "\t" + man_args_name + ".step_type = STEP_FW;\n"
    forward_function += "\t" + man_args_name + ".matmul_type = MATMUL_TYPE;\n"
    forward_function += "\tpi_cl_team_fork(NUM_CORES, mm_manager, &" + man_args_name + ");\n"
    forward_function += "\t#endif\n"

    forward_function += "\n"

    return structures_and_blobs, blob_initializations, blob_connect, forward_function
