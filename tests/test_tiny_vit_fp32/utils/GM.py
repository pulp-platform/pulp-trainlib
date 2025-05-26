import argparse
import random

import numpy as np
import onnx
import torch
from utils.writers.component_writers import (
    adapt_onnx_name,
    conv_writer,
    gelu_writer,
    add_writer,
    transpose_writer,
    layer_norm_writer,
    matmul_writer,
    split_writer,
    mul_writer,
    softmax_writer,
    reduce_mean_writer,
    gemm_writer,
)
from utils.writers.file_writers import (
    model_components_writer,
    input_writer,
    output_writer,
    parameters_writer,
)

from model.TinyViT import TinyViT
from model_configs import MODEL_CONFIGS


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="DEMO_TINY_VIT_CONFIG",
        help="Name of the config",
    )

    return parser.parse_args()


def add_used_data(used_data, all_used_data, all_elements, current_node):
    for el in used_data:
        for key in all_elements.keys():
            if el in adapt_onnx_name(key):
                if key in all_used_data.keys():
                    # Update last node that uses this data
                    all_used_data[key]["end_node"] = current_node
                else:
                    # Introduce the new data with the birth node
                    if (
                        isinstance(all_elements[key], dict)
                        and "shape" in all_elements[key].keys()
                    ):
                        element_shape = all_elements[key]["shape"]
                    elif isinstance(all_elements[key], np.ndarray):
                        element_shape = tuple(all_elements[key].shape)
                    else:
                        raise ValueError("Problem at element: ", all_elements[key])

                    all_used_data[key] = {
                        "shape": element_shape,
                        "start_node": current_node,
                        "end_node": current_node,
                    }

                break


def onnx_parser(onnx_model):
    # Prepare nodes of interest
    op_types_of_interest = {
        "Conv": conv_writer,
        "Gelu": gelu_writer,
        "Add": add_writer,
        "Transpose": transpose_writer,
        "LayerNormalization": layer_norm_writer,
        "MatMul": matmul_writer,
        "Split": split_writer,
        "Mul": mul_writer,
        "Softmax": softmax_writer,
        "ReduceMean": reduce_mean_writer,
        "Gemm": gemm_writer,
    }

    # Prepare output strings and element storage dict
    structures_and_blobs = ""
    blob_initializations = ""
    blob_connect = ""
    forward_function = ""

    all_elements = {}
    ignore_parameter_arrays = list()
    all_used_data = dict()

    # Iterate through inputs of the ONNX model
    for _input in onnx_model.graph.input:
        all_elements[_input.name] = {
            "shape": tuple(el.dim_value for el in _input.type.tensor_type.shape.dim),
            "data": _input.name,
        }

    # Iterate through given values in the ONNX model
    for el in onnx_model.graph.initializer:
        element_array = onnx.numpy_helper.to_array(el)

        all_elements[el.name] = {
            "val": element_array,
            "shape": tuple(element_array.shape),
            "data": el.name,
        }

    # Iterate through onnx model nodes and perform necessary operations
    for node in onnx_model.graph.node:
        if node.op_type in op_types_of_interest.keys():
            s1, s2, s3, s4, used_data = op_types_of_interest[node.op_type](
                node=node,
                all_elements=all_elements,
                data_marker="float",
            )

            structures_and_blobs += s1
            blob_initializations += s2
            blob_connect += s3
            forward_function += s4

            add_used_data(
                used_data, all_used_data, all_elements, adapt_onnx_name(node.name)
            )

            if node.op_type == "Transpose":
                new_shape = list(all_elements[node.output[0]]["shape"])

                for reshape_output in onnx_model.graph.value_info:
                    if reshape_output.name == node.output[0]:
                        for iii, d in enumerate(reshape_output.type.tensor_type.shape.dim):
                            d.dim_value = new_shape[iii]

        elif node.op_type == "Identity":
            the_data = all_elements[node.input[0]]["val"]

            all_elements[node.output[0]] = {
                "shape": tuple(the_data.shape),
                "data": node.output[0],
                "val": the_data,
            }
        elif node.op_type == "Shape":
            # FIXME: Implement for general case
            #  (now only for TinyViT scenario, when part of flatten and operating on shape array)
            all_elements[node.output[0]] = {"val": all_elements[node.input[0]]["shape"]}
        elif node.op_type == "Constant":
            # FIXME: Implement for general case
            #  (now only for TinyViT scenario, when part of flatten and operating on shape array)
            val = onnx.numpy_helper.to_array(node.attribute[0].t)

            all_elements[node.output[0]] = {
                "val": val,
                "shape": val.shape,
                "data": node.output[0],
            }
        elif node.op_type == "Slice":
            # FIXME: Implement for general case
            #  (now only for TinyViT scenario, when part of flatten and operating on shape array)
            inp = np.array(all_elements[node.input[0]]["val"])

            starts = all_elements[node.input[1]]["val"]
            ends = all_elements[node.input[2]]["val"]
            axes = all_elements[node.input[3]]["val"]

            slice_list = list()

            for i in range(axes.max() + 1):
                if i not in axes:
                    slice_list.append(slice(inp.shape[i]))
                else:
                    slice_list.append(slice(starts[i], ends[i]))

            all_elements[node.output[0]] = {"val": inp[tuple(slice_list)]}

            ignore_parameter_arrays.append(node.input[1])
            ignore_parameter_arrays.append(node.input[2])
            ignore_parameter_arrays.append(node.input[3])
        elif node.op_type == "Concat":
            # FIXME: Implement for general case
            #  (now only for TinyViT scenario, when part of flatten and operating on shape array)
            val = np.concatenate(
                [all_elements[inp]["val"] for inp in node.input],
                axis=node.attribute[0].i,
            )

            all_elements[node.output[0]] = {
                "val": val,
                "shape": tuple(val.shape),
                "data": node.output[0],
            }

            ignore_parameter_arrays.append(node.input[1])
        elif node.op_type == "Reshape":
            # FIXME: Implement for general case
            #  (now only for TinyViT scenario, when part of flatten and operating on shape array)
            original_shape = all_elements[node.input[0]]["shape"]

            if (
                isinstance(all_elements[node.input[1]], dict)
                and "val" in all_elements[node.input[1]].keys()
            ):
                target_shape = all_elements[node.input[1]]["val"]
            else:
                target_shape = all_elements[node.input[1]]

            if -1 in target_shape:
                min_1_index = list(target_shape).index(-1)

                shape_prefix = target_shape[:min_1_index]
                shape_postfix = target_shape[min_1_index + 1 :]

                remaining_shape = np.prod(original_shape) // (
                    np.prod(shape_prefix) * np.prod(shape_postfix)
                )

                new_shape = np.concatenate(
                    [shape_prefix, [remaining_shape], shape_postfix]
                )

                ignore_parameter_arrays.append(node.input[1])
            else:
                new_shape = target_shape

            all_elements[node.output[0]] = {
                "shape": tuple(new_shape),
                "data": all_elements[node.input[0]]["data"],
            }

            for reshape_output in onnx_model.graph.value_info:
                if reshape_output.name == node.output[0]:
                    for iii, d in enumerate(reshape_output.type.tensor_type.shape.dim):
                        d.dim_value = new_shape[iii]
        else:
            raise NotImplementedError(
                f"Operation {node.op_type} is not implemented in the parser."
            )

    # Find how much data needs to be kept for each node
    active_data = list()
    node_required_data = dict()
    max_data_needed = 0
    max_data_node = ""

    for node in onnx_model.graph.node:
        if node.op_type in op_types_of_interest.keys():
            node_required_data[adapt_onnx_name(node.name)] = {
                "required_data": list(),
                "total_size": 0,
            }

            for el in active_data:
                node_required_data[adapt_onnx_name(node.name)]["required_data"].append(
                    all_used_data[el]
                )
                node_required_data[adapt_onnx_name(node.name)]["total_size"] += np.prod(
                    all_used_data[el]["shape"]
                )

            for el in all_used_data.keys():
                if adapt_onnx_name(node.name) == all_used_data[el]["start_node"]:
                    active_data.append(el)

                    node_required_data[adapt_onnx_name(node.name)][
                        "required_data"
                    ].append(all_used_data[el])
                    node_required_data[adapt_onnx_name(node.name)][
                        "total_size"
                    ] += np.prod(all_used_data[el]["shape"])

                if adapt_onnx_name(node.name) == all_used_data[el]["end_node"]:
                    active_data.remove(el)

            if (
                node_required_data[adapt_onnx_name(node.name)]["total_size"]
                > max_data_needed
            ):
                max_data_needed = node_required_data[adapt_onnx_name(node.name)][
                    "total_size"
                ]
                max_data_node = node.name

    assert len(active_data) == 0, "Error in data management, some end node was skipped."

    print("Maximum data needed by an operation:", max_data_needed, "by", max_data_node)

    # Extract output name
    output_array_name = adapt_onnx_name(
        all_elements[onnx_model.graph.output[0].name]["data"]
    )

    # Extract parameter arrays
    parameter_arrays = dict()
    for el in all_elements.keys():
        if (
            isinstance(all_elements[el], dict)
            and "val" in all_elements[el].keys()
            and "data" in all_elements[el].keys()
            and el not in ignore_parameter_arrays
        ):
            parameter_arrays[adapt_onnx_name(all_elements[el]["data"])] = all_elements[
                el
            ]["val"]

    # Extract input name
    input_name = adapt_onnx_name(onnx_model.graph.input[0].name)

    return (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        output_array_name,
        parameter_arrays,
        input_name,
    )


def main():
    # Fix seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Set root directory
    root_dir = "."

    # Parse arguments
    args = arg_parse()

    # Create model
    cfg = MODEL_CONFIGS[args.config]
    model = TinyViT(
        img_size=cfg["IN_IMG_SIZE"],
        in_chans=cfg["IN_CHANS"],
        num_classes=cfg["NUM_CLASSES"],
        embed_dims=cfg["EMBED_DIMS"],
        depths=cfg["DEPTHS"],
        num_heads=cfg["NUM_HEADS"],
        window_sizes=cfg["WINDOW_SIZES"],
        mlp_ratio=cfg["MLP_RATIO"],
        drop_rate=cfg["DROP_RATE"],
        drop_path_rate=cfg["DROP_PATH_RATE"],
        use_checkpoint=cfg["USE_CHECKPOINT"],
        mbconv_expand_ratio=cfg["MBCONV_EXPAND_RATIO"],
        local_conv_size=cfg["LOCAL_CONV_SIZE"],
    )
    model.eval()

    # Model to ONNX
    sample_input = torch.randn(
        1, cfg["IN_CHANS"], cfg["IN_IMG_SIZE"], cfg["IN_IMG_SIZE"]
    )
    torch.onnx.export(
        model,
        sample_input,
        "TinyViT.onnx",
        verbose=False,
        opset_version=20,
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,
    )

    # Infer node output dimensions
    onnx_model = onnx.load("TinyViT.onnx")
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.save_model(onnx_model, "TinyViT.onnx")
    onnx_model = onnx.load("TinyViT.onnx")

    # Parse onnx
    (
        structures_and_blobs,
        blob_initializations,
        blob_connect,
        forward_function,
        output_array_name,
        parameter_arrays,
        input_name,
    ) = onnx_parser(onnx_model)

    inputs_to_save = [
        sample_input.numpy(),
    ] + list(parameter_arrays.values())

    np.savez("inputs.npz", *inputs_to_save)
    np.savez("outputs.npz", input=model(sample_input).detach())

    onnx.save_model(onnx_model, "TinyViT.onnx")

    input("Stop if only exporting ONNX model.")

    # Write input sequence
    input_writer(
        file_root_dir=root_dir, input_name=input_name, input_array=sample_input
    )

    # Write output sequence
    output_writer(file_root_dir=root_dir, output_array=model(sample_input).detach())

    # Write model components
    model_components_writer(
        file_root_dir=root_dir,
        structures_and_blobs=structures_and_blobs,
        blob_initializations=blob_initializations,
        blob_connect=blob_connect,
        forward_function=forward_function,
        output_array_name=output_array_name,
    )

    # Write parameter arrays
    parameters_writer(file_root_dir=root_dir, parameter_arrays=parameter_arrays)


if __name__ == "__main__":
    main()
