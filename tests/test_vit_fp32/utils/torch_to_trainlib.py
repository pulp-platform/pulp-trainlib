from utils.writers.component_writers import (
    parameter_writer,
    concat_writer,
    conv2d_writer,
    vector_sum_writer,
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
    "concat": concat_writer,
    "positional_embedding": vector_sum_writer,
}
