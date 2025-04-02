import os


def model_components_writer(file_root_dir):
    # Write to header
    f = open(os.path.join(file_root_dir, "model_components.h"), "w")

    f.write("#ifndef MODEL_COMPONENTS_H\n")
    f.write("#define MODEL_COMPONENTS_H\n\n")

    f.write("\n")

    f.write("// =============== Includes ===============\n")
    # f.write('#include "input_sequence.h"\n')
    # f.write('#include "output_sequence.h"\n')
    f.write("\n")

    # f.write('#include "model_defines.h"\n')
    f.write("\n\n")

    f.write("// =============== Constants definition ===============\n")
    f.write("PI_L1 float zero_init = 0.0f;\n")
    f.write("PI_L1 float min_float = -340282346638528859811704183484516925440.0f;\n")
    f.write("\n\n")

    f.write("\n#endif\n")
    f.close()
