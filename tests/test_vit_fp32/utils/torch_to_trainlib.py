from utils.writers.component_writers import (
    parameter_writer,
    concat_writer,
    conv2d_writer,
    add_writer,
    layer_norm_writer,
    mhsa_writer,
    linear_writer,
    tanh_writer,
    flatten_writer,
    transpose_writer,
)

VIT_COMPONENTS_WRITERS = {
    "patch_embedding": conv2d_writer,
    "flatten_and_transpose": transpose_writer,
    # "class_token": [
    #     {
    #         "trainlib_name": None,  # "Parameter",
    #         "write_function": parameter_writer,
    #     },
    #     {
    #         "trainlib_name": None,  # "Concat",
    #         "write_function": concat_writer,
    #     },
    # ],
    # "positional_embedding": [
    #     {
    #         "trainlib_name": None,  # "Parameter",
    #         "write_function": parameter_writer,
    #     },
    #     {
    #         "trainlib_name": None,  # "Add",
    #         "write_function": add_writer,
    #     },
    # ],
    # "transformer_blocks": [
    #     {
    #         "trainlib_name": None,  # "LayerNorm",
    #         "write_function": layer_norm_writer,
    #     },
    #     {
    #         "trainlib_name": "mhsa",
    #         "write_function": mhsa_writer,
    #     },
    #     {
    #         "trainlib_name": "Linear",
    #         "write_function": linear_writer,
    #     },
    #     # TODO: Dropout
    #     {
    #         "trainlib_name": None,  # "LayerNorm",
    #         "write_function": layer_norm_writer,
    #     },
    #     {
    #         "trainlib_name": "Linear",
    #         "write_function": linear_writer,
    #     },
    #     {
    #         "trainlib_name": "Linear",
    #         "write_function": linear_writer,
    #     },
    # ],
    # "norm": [
    #     {
    #         "trainlib_name": None,  # "LayerNorm",
    #         "write_function": layer_norm_writer,
    #     },
    # ],
    # "fc": [
    #     {
    #         "trainlib_name": "Linear",
    #         "write_function": linear_writer,
    #     },
    # ],
    # "others": [
    #     {
    #         "trainlib_name": None,  # "TanH",
    #         "write_function": tanh_writer,
    #     },
    #     {
    #         "trainlib_name": None,  # "Flatten",
    #         "write_function": flatten_writer,
    #     },
    #     {
    #         "trainlib_name": "Transpose",
    #         "write_function": transpose_writer,
    #     },
    # ],
}
