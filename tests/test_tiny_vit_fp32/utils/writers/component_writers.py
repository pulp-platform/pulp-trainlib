import numpy as np

from writers.writers_utils import (
    get_initialization_text,
    get_connect_text,
    adapt_onnx_name,
    extract_input_information,
)


def conv_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    # TODO: Implement bias
    # TODO: Implement dilations
    # TODO: Extend group implementation

    # Initial values
    strides = None
    pads = None
    groups = 1

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
        elif attr.name == "group":
            groups = attr.i

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

    used_data = [input_data_name, weight_data_name, output_data_name]

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

    weight_c = input_c // groups
    weight_w, weight_h = weights_shape[-2:]
    weight_dim = weight_c * weight_w * weight_h * output_c

    input_name = component_name + "_input_blob"
    weight_name = component_name + "_weight_blob"
    output_name = component_name + "_output_blob"

    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    if groups == 1:
        structures_and_blobs += "PI_L2 struct Conv2D_args " + args_name + ";\n"
    elif groups == output_c:
        structures_and_blobs += "PI_L2 struct DepthWise_Conv_args " + args_name + ";\n"
    else:
        raise NotImplementedError("Group convolutions not implemented!")

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

    blob_connect += "\t" + args_name + ".Lpad = " + str(pads[0]) + ";\n"
    blob_connect += "\t" + args_name + ".Rpad = " + str(pads[1]) + ";\n"
    blob_connect += "\t" + args_name + ".Upad = " + str(pads[2]) + ";\n"
    blob_connect += "\t" + args_name + ".Dpad = " + str(pads[3]) + ";\n"

    blob_connect += "\t" + args_name + ".stride_h = " + str(strides[0]) + ";\n"
    blob_connect += "\t" + args_name + ".stride_w = " + str(strides[1]) + ";\n"

    if groups == 1:
        blob_connect += "\t" + args_name + ".USE_BIASES = 0;\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"

    if groups == 1:
        forward_function += "\tpulp_conv2d_fp32_fw_cl(&" + args_name + ");\n\n"
    elif groups == output_c:
        forward_function += "\tpulp_conv_dw_fp32_fw_cl(&" + args_name + ");\n\n"

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def gelu_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_data_name = output_data_name = adapt_onnx_name(
        all_elements[node.input[0]]["data"]
    )

    input_shape = all_elements[node.input[0]]["shape"]
    dim = 1
    for el in input_shape:
        dim *= el

    used_data = [
        input_data_name,
    ]

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": all_elements[node.input[0]]["shape"],
        "data": all_elements[node.input[0]]["data"],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    input_blob_name = component_name + "_input_blob"
    output_blob_name = component_name + "_output_blob"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct act_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define element blobs
    structures_and_blobs += "PI_L2 struct blob " + input_blob_name + ";\n"
    structures_and_blobs += "PI_L2 struct blob " + output_blob_name + ";\n"

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        input_blob_name,
        {
            "data": input_data_name,
            "dim": dim,
            "C": 1,
            "W": 1,
            "H": dim,
        },
    )

    blob_connect += get_connect_text(
        output_blob_name,
        {
            "data": output_data_name,
            "dim": dim,
            "C": 1,
            "W": 1,
            "H": dim,
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = &" + input_blob_name + ";\n"
    blob_connect += "\t" + args_name + ".output = &" + output_blob_name + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"

    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, pulp_gelu_tanh_approx_fp32_fw_cl, &"
        + args_name
        + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def add_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_1_data, input_1_shape = extract_input_information(all_elements[node.input[0]])
    input_2_data, input_2_shape = extract_input_information(all_elements[node.input[1]])

    input_1_data_name = adapt_onnx_name(input_1_data)
    input_2_data_name = adapt_onnx_name(input_2_data)
    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_1_data_name, input_2_data_name, output_data_name]

    op1_dims_name = component_name + "_op1_dims"
    op2_dims_name = component_name + "_op2_dims"

    output_dims = tuple(np.broadcast_shapes(input_1_shape, input_2_shape))
    output_total_size = np.prod(output_dims)

    output_filler = "zero_init"

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": output_dims,
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_broadcast_add_args"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += (
        "PI_L2 struct array_broadcast_sum_fp32_args " + args_name + ";\n"
    )

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L2 "
        + data_marker
        + " "
        + output_data_name
        + "["
        + str(output_total_size)
        + "];\n\n"
    )

    structures_and_blobs += (
        "PI_L2 int "
        + op1_dims_name
        + "[] = {"
        + ", ".join(map(str, input_1_shape))
        + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 int "
        + op2_dims_name
        + "[] = {"
        + ", ".join(map(str, input_2_shape))
        + "};\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_total_size, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".op_1 = " + input_1_data_name + ";\n"
    blob_connect += "\t" + args_name + ".op_2 = " + input_2_data_name + ";\n"
    blob_connect += "\t" + args_name + ".dest = " + output_data_name + ";\n\n"

    blob_connect += "\t" + args_name + ".op_1_dims = " + op1_dims_name + ";\n"
    blob_connect += "\t" + args_name + ".op_2_dims = " + op2_dims_name + ";\n\n"

    blob_connect += (
        "\t" + args_name + ".op_1_dims_len = " + str(len(input_1_shape)) + ";\n"
    )
    blob_connect += (
        "\t" + args_name + ".op_2_dims_len = " + str(len(input_2_shape)) + ";\n"
    )

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, array_broadcast_sum_fp32, &"
        + args_name
        + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def transpose_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_data_name = adapt_onnx_name(all_elements[node.input[0]]["data"])
    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_data_name, output_data_name]

    transposed_axes = node.attribute[0].ints
    input_shape = all_elements[node.input[0]]["shape"]

    assert len(input_shape) == len(
        transposed_axes
    ), "Input and transposed number of axes not matching for transposition writer!"

    output_shape = list()
    output_size = 1
    for i in transposed_axes:
        output_shape.append(input_shape[i])
        output_size *= input_shape[i]

    assert np.prod(input_shape) == np.prod(
        output_shape
    ), "Input and output shapes not matching for transposition writer!"

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": tuple(output_shape),
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_transpose_args"
    dim_name = component_name + "_dim"
    transposed_axes_name = component_name + "_transposed_axes"

    output_filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct transp_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L2 "
        + data_marker
        + " "
        + output_data_name
        + "["
        + str(output_size)
        + "];\n\n"
    )

    structures_and_blobs += (
        "PI_L2 int " + dim_name + "[] = {" + ", ".join(map(str, input_shape)) + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 int "
        + transposed_axes_name
        + "[] = {"
        + ", ".join(map(str, transposed_axes))
        + "};\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_size, output_data_name, output_filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".in_matrix = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".out_matrix = " + output_data_name + ";\n"
    blob_connect += "\t" + args_name + ".dim = " + dim_name + ";\n"
    blob_connect += (
        "\t" + args_name + ".transposed_axes = " + transposed_axes_name + ";\n"
    )
    blob_connect += "\t" + args_name + ".n_dim = " + str(len(output_shape)) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, transpose, &" + args_name + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def layer_norm_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract and define component information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    weight_data_name = adapt_onnx_name(node.input[1])
    bias_data_name = adapt_onnx_name(node.input[2])

    input_data_name = adapt_onnx_name(all_elements[node.input[0]]["data"])
    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_data_name, weight_data_name, bias_data_name, output_data_name]

    axis = node.attribute[0].i
    epsilon = node.attribute[1].f

    component_shape = all_elements[node.input[0]]["shape"]
    total_size = 1
    for el in component_shape:
        total_size *= el

    step_size = 1
    for el in component_shape[axis:]:
        step_size *= el

    component_blob_name = component_name + "_args"
    eps_data_name = component_name + "_eps"

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": component_shape,
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += (
        "PI_L2 struct LayerNorm_args_fp32 " + component_blob_name + ";\n"
    )

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L1 " + data_marker + " " + eps_data_name + "[1] = {" + str(epsilon) + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 " + data_marker + " " + output_data_name + "[" + str(total_size) + "];\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initialization = "\t// " + component_name.upper() + "\n"

    blob_initialization += get_initialization_text(
        total_size, output_data_name, "zero_init"
    )

    blob_initialization += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    blob_connect += get_connect_text(
        component_blob_name,
        {
            "x": input_data_name,
            "weight": weight_data_name,
            "bias": bias_data_name,
            "output": output_data_name,
            "eps": eps_data_name,
            "size": total_size,
            "step_size": step_size,
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, pulp_layerNorm_fp32_fw_cl, &"
        + component_blob_name
        + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initialization,
        blob_connect,
        forward_function,
        used_data,
    )


def matmul_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_a_name = node.input[0]
    input_b_name = node.input[1]

    input_data_a, input_a_shape = extract_input_information(all_elements[input_a_name])
    input_data_b, input_b_shape = extract_input_information(all_elements[input_b_name])

    input_data_a_name = adapt_onnx_name(input_data_a)
    input_data_b_name = adapt_onnx_name(input_data_b)

    input_a_dims_name = component_name + "_input_a_dims"
    input_b_dims_name = component_name + "_input_b_dims"

    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_data_a_name, input_data_b_name, output_data_name]

    output_shape = np.concatenate(
        (
            np.broadcast_shapes(input_a_shape[:-2], input_b_shape[:-2]),
            (input_a_shape[-2],),
            (input_b_shape[-1],),
        )
    )

    output_total_size = 1
    for el in output_shape:
        output_total_size *= el

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": tuple(output_shape),
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += (
        "PI_L2 struct broadcastMatMul_args_fp32 " + args_name + ";\n"
    )

    structures_and_blobs += "\n"

    # Define blobs

    # Define data variables
    structures_and_blobs += (
        "PI_L2 int "
        + input_a_dims_name
        + "[] = {"
        + ", ".join(map(str, input_a_shape))
        + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 int "
        + input_b_dims_name
        + "[] = {"
        + ", ".join(map(str, input_b_shape))
        + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 "
        + data_marker
        + " "
        + output_data_name
        + "["
        + str(output_total_size)
        + "];\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_total_size, output_data_name, filler
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
            "A_dims": input_a_dims_name,
            "B_dims": input_b_dims_name,
            "A_dims_len": len(input_a_shape),
            "B_dims_len": len(input_b_shape),
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"

    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"

    forward_function += "\tmm_broadcast_fp32(&" + args_name + ");\n"

    forward_function += "\n"

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def split_writer(node, all_elements, data_marker="float"):
    # Extract components of interest
    input_shape = all_elements[node.input[0]]["shape"]
    input_data = all_elements[node.input[0]]["data"]

    split_sizes = all_elements[node.input[1]]["val"]
    split_axis = node.attribute[0].i

    assert input_shape[split_axis] == sum(split_sizes), (
        "Split sizes do not match input shape for " + node.name
    )

    # Compute output offsets and sizes
    root_size = 1
    for i, el in enumerate(input_shape):
        if i != split_axis:
            root_size *= el

    current_offset = 0
    out_sizes = list()
    out_offsets = list()
    for el in split_sizes:
        out_sizes.append(root_size * el)

        out_offsets.append(current_offset)
        current_offset += root_size * el

    structures_and_blobs = "// " + adapt_onnx_name(node.name).upper() + "\n"

    for i, el in enumerate(node.output):
        structures_and_blobs += (
            "PI_L2 "
            + data_marker
            + " *"
            + adapt_onnx_name(el)
            + " = "
            + adapt_onnx_name(input_data)
            + " + "
            + str(out_offsets[i])
            + ";\n"
        )

        # Store output dimensions
        all_elements[node.output[i]] = {
            "shape": input_shape[:split_axis]
            + (split_sizes[i],)
            + input_shape[split_axis + 1 :],
            "data": el,
        }

    structures_and_blobs += "\n\n"

    return structures_and_blobs, "", "", "", list()


def mul_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_data_name = adapt_onnx_name(all_elements[node.input[0]]["data"])

    used_data = [
        input_data_name,
    ]

    # Compute input shape and total dimension
    input_shape = all_elements[node.input[0]]["shape"]
    dim = 1
    for el in input_shape:
        dim *= el

    # Get multiplication factor
    if isinstance(all_elements[node.input[1]], dict):
        factor = all_elements[node.input[1]]["val"]
    else:
        factor = all_elements[node.input[1]]

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": input_shape,
        "data": all_elements[node.input[0]]["data"],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct scalar_mul_args " + args_name + ";\n"

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".scalar = " + str(factor) + ";\n"
    blob_connect += "\t" + args_name + ".dim = " + str(dim) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"

    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"

    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, pulp_scalar_mul_fp32_cl, &" + args_name + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def softmax_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_data_name = adapt_onnx_name(all_elements[node.input[0]]["data"])

    maxes_name = component_name + "_maxes"
    sums_name = component_name + "_sums"

    used_data = [input_data_name, maxes_name, sums_name]

    input_shape = all_elements[node.input[0]]["shape"]

    w = input_shape[-1]
    h = 1
    for el in input_shape[:-1]:
        h *= el

    filler = "zero_init"

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": input_shape,
        "data": all_elements[node.input[0]]["data"],
    }

    all_elements[maxes_name] = {
        "shape": (h,),
        "data": maxes_name,
    }

    all_elements[sums_name] = {
        "shape": (h,),
        "data": sums_name,
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct softmax_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L2 " + data_marker + " " + maxes_name + "[" + str(h) + "];\n"
    )

    structures_and_blobs += (
        "PI_L2 " + data_marker + " " + sums_name + "[" + str(h) + "];\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(h, sums_name, filler)
    blob_initializations += get_initialization_text(h, maxes_name, filler)

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input_data = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".output_data = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".maxes = " + maxes_name + ";\n"
    blob_connect += "\t" + args_name + ".sums = " + sums_name + ";\n"
    blob_connect += "\t" + args_name + ".H = " + str(h) + ";\n"
    blob_connect += "\t" + args_name + ".W = " + str(w) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += "\tpulp_softmax_fp32_fw_cl(&" + args_name + ");\n\n"

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def reduce_mean_writer(node, all_elements, data_marker="float"):
    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    # FIXME: Change primitive to support multiple-dim reduction
    assert (
        len(all_elements[node.input[1]]["val"]) == 1
    ), "Only single dimension reduction supported at the moment."

    component_name = adapt_onnx_name(node.name)
    dims_name = component_name + "_dims"

    input_data_name = adapt_onnx_name(all_elements[node.input[0]]["data"])
    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_data_name, output_data_name]

    input_shape = all_elements[node.input[0]]["shape"]
    dims_to_reduce = all_elements[node.input[1]]["val"]
    output_shape = list()
    output_size = 1

    if node.attribute[0].i == 0:
        # Don't keep dimensions that are reduced
        for i, el in enumerate(input_shape):
            if i not in dims_to_reduce:
                output_shape.append(el)
                output_size *= el

        output_shape = tuple(output_shape)
    else:
        # Keep dimensions that are reduced
        for i, el in enumerate(input_shape):
            if i in dims_to_reduce:
                output_shape.append(1)
            else:
                output_shape.append(el)

        output_shape = tuple(output_shape)

    filler = "zero_init"

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": output_shape,
        "data": all_elements[node.input[0]]["data"],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct reduce_mean_args_fp32 " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define data variables
    structures_and_blobs += (
        "PI_L2 int " + dims_name + "[] = {" + ", ".join(map(str, input_shape)) + "};\n"
    )

    structures_and_blobs += (
        "PI_L2 "
        + data_marker
        + " "
        + output_data_name
        + "["
        + str(output_size)
        + "];\n"
    )

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        np.prod(output_shape), output_data_name, filler
    )

    blob_initializations += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".input = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".output = " + output_data_name + ";\n"

    blob_connect += "\t" + args_name + ".dims = " + dims_name + ";\n"
    blob_connect += "\t" + args_name + ".dims_len = " + str(len(input_shape)) + ";\n"
    blob_connect += (
        "\t" + args_name + ".reduce_axis = " + str(dims_to_reduce[0]) + ";\n"
    )

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"
    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"
    forward_function += (
        "\tpi_cl_team_fork(NUM_CORES, reduce_mean_fp32, &" + args_name + ");\n\n"
    )

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )


def gemm_writer(node, all_elements, data_marker="float"):
    # FIXME: Currently only supports simple version found in TinyViT (C considered 0, alpha and beta considered 1,
    #  no support for A transposition)

    # ~~~~~~~~~~~~~~~~~~~~ Extract node information ~~~~~~~~~~~~~~~~~~~~
    component_name = adapt_onnx_name(node.name)

    input_name = node.input[0]
    weight_name = node.input[1]
    bias_name = node.input[2]

    input_data, input_shape = extract_input_information(all_elements[input_name])
    weight_data, weight_shape = extract_input_information(all_elements[weight_name])
    bias_data, bias_shape = extract_input_information(all_elements[bias_name])

    input_data_name = adapt_onnx_name(input_data)
    weight_data_name = adapt_onnx_name(weight_data)
    bias_data_name = adapt_onnx_name(bias_data)
    output_data_name = adapt_onnx_name(node.output[0])

    used_data = [input_data_name, weight_data_name, bias_data_name, output_data_name]

    # Check if transposed B and compute output shape
    trans_b = False
    for el in node.attribute:
        if el.name == "transB":
            trans_b = el.i == 1

    if trans_b:
        output_shape = tuple([input_shape[0], weight_shape[0]])
    else:
        output_shape = tuple([input_shape[0], weight_shape[1]])

    output_total_size = np.prod(output_shape)

    # Store output dimension
    all_elements[node.output[0]] = {
        "shape": output_shape,
        "data": node.output[0],
    }

    # ~~~~~~~~~~~~~~~~~~~~ Define component information ~~~~~~~~~~~~~~~~~~~~
    args_name = component_name + "_args"
    man_args_name = component_name + "_man_args"

    filler = "zero_init"

    # ~~~~~~~~~~~~~~~~~~~~ Define components ~~~~~~~~~~~~~~~~~~~~
    # Define structures
    structures_and_blobs = "// " + component_name.upper() + "\n"

    structures_and_blobs += "PI_L2 struct mm_manager_args " + man_args_name + ";\n"
    structures_and_blobs += "PI_L2 struct matMul_args " + args_name + ";\n"

    structures_and_blobs += "\n"

    # Define blobs

    # Define data variables
    structures_and_blobs += (
        "PI_L2 "
        + data_marker
        + " "
        + output_data_name
        + "["
        + str(output_total_size)
        + "];\n"
    )

    structures_and_blobs += "\n\n"

    # ~~~~~~~~~~~~~~~~~~~~ Perform initializations ~~~~~~~~~~~~~~~~~~~~
    blob_initializations = "\t// " + component_name.upper() + "\n"

    blob_initializations += get_initialization_text(
        output_total_size, output_data_name, filler
    )

    # ~~~~~~~~~~~~~~~~~~~~ Populate blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect = "\t// " + component_name.upper() + "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Connect blobs ~~~~~~~~~~~~~~~~~~~~
    blob_connect += "\t" + args_name + ".A = " + input_data_name + ";\n"
    blob_connect += "\t" + args_name + ".B = " + weight_data_name + ";\n"
    blob_connect += "\t" + args_name + ".bias = " + bias_data_name + ";\n"
    blob_connect += "\t" + args_name + ".C = " + output_data_name + ";\n"

    blob_connect += "\t" + args_name + ".N = " + str(input_shape[0]) + ";\n"
    blob_connect += "\t" + args_name + ".K = " + str(input_shape[1]) + ";\n"
    blob_connect += "\t" + args_name + ".M = " + str(weight_shape[1]) + ";\n"

    blob_connect += "\t" + args_name + ".trans_B = " + str(int(trans_b)) + ";\n"
    blob_connect += "\t" + args_name + ".USE_BIASES = " + str(1) + ";\n"

    blob_connect += "\n"

    # ~~~~~~~~~~~~~~~~~~~~ Forward function ~~~~~~~~~~~~~~~~~~~~
    forward_function = "\t// " + component_name.upper() + "\n"

    forward_function += "\t#ifdef DEBUG\n"
    forward_function += '\tprintf("Working on ' + component_name + '...\\n");\n'
    forward_function += "\t#endif\n\n"

    forward_function += "\t#ifndef OPTIMIZE\n"
    forward_function += "\tmm(" + args_name + ");\n"
    forward_function += "\t#else\n"
    forward_function += "\tmm_manager(&" + man_args_name + ");\n"
    forward_function += "\t#endif\n"

    forward_function += "\n"

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        used_data,
    )
