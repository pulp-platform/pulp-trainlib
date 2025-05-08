from utils.writers.component_writers import (
    concat_writer,
    conv2d_writer,
    gelu_writer,
    layer_norm_writer,
    linear_writer,
    mhsa_writer,
    tanh_writer,
    transpose_writer,
    vector_sum_writer,
)


VIT_COMPONENTS_WRITERS = {
    "patch_embedding": conv2d_writer,
    "flatten_and_transpose": transpose_writer,
    "concat": concat_writer,
    "positional_embedding": vector_sum_writer,
    "tanh": tanh_writer,
    "norm": layer_norm_writer,
    "fc": linear_writer,
}

for i in range(12):
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_norm1"] = layer_norm_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_pre_attn_transpose"] = (
        transpose_writer
    )
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_attn"] = mhsa_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_post_attn_transpose"] = (
        transpose_writer
    )
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_proj"] = linear_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_add_1"] = vector_sum_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_norm2"] = layer_norm_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_pwff_fc1"] = linear_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_pwff_gelu"] = gelu_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_pwff_fc2"] = linear_writer
    VIT_COMPONENTS_WRITERS[f"transformer_blocks_{i}_add_2"] = vector_sum_writer
