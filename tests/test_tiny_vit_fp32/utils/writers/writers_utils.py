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


def extract_input_information(node):
    if isinstance(node, dict):
        return node["data"], tuple(node["shape"])
    else:
        try:
            return node.name, tuple(node.dims)
        except:
            raise NotImplementedError("Node structure not recognized")
