import re
import struct
import numpy as np

# Function to parse bfloat16 vectors from multiple header files
def parse_bfloat16_vectors(filenames):
    vectors = {}
    # Regular expression to match bfloat16 vector definitions
    vector_regex = re.compile(r'PI_L2 fp16 (\w+)\[(\w+)\] = \{([^\}]*)\}')
    
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                match = vector_regex.search(line)
                if match:
                    # Extract vector name, size, and values
                    name = match.group(1)
                    values = match.group(3).split(',')
                    # Convert string values to float32 (standard Python float)
                    float_values = [float(value.strip().replace('f', '')) for value in values if value.strip()]
                    # Store as numpy float32 array for easy bfloat16 conversion later
                    vectors[name] = np.array(float_values, dtype=np.float32)
    
    return vectors

# Convert a float32 number to bfloat16 format
def float32_to_bfloat16(value):
    # Interpret float32 as 32-bit integer, mask out lower 16 bits (truncate mantissa)
    int_repr = np.frombuffer(struct.pack('f', value), dtype=np.uint32)[0]
    bfloat16_repr = int_repr >> 16  # Keep top 16 bits (8 for exponent, 7 for mantissa, 1 for sign)
    return bfloat16_repr

# Function to write vectors to a binary file in bfloat16 format
def write_vectors_to_binary_bfloat16(vectors, ordered_names, output_filename):
    with open(output_filename, 'wb') as bin_file:
        for name in ordered_names:
            if name in vectors:
                # Convert each float32 value to bfloat16 and write it as 2 bytes
                for value in vectors[name]:
                    bfloat16_value = float32_to_bfloat16(value)
                    bin_file.write(struct.pack('H', bfloat16_value))  # 'H' is for unsigned short (2 bytes)
            else:
                print(f"Warning: {name} not found in the parsed vectors.")

if __name__ == "__main__":
    header_filenames = ['attention-defines.h', 'bottleneck-defines.h', 'embeddings.h', 'ffn-defines.h', 'input-sequence.h', 'intermediate-defines.h', 'output-defines.h', 'output-sequence.h']  # List of header files
    output_filename = 'weights.bin'  # Output binary file
    ordered_names = ['BOTTLENECK_ATTENTION_NORM_WEIGHTS', 'BOTTLENECK_ATTENTION_NORM_BIASES', 'BOTTLENECK_INPUT_NORM_WEIGHTS', 'BOTTLENECK_INPUT_NORM_BIASES', 'INPUT', 'ATTENTION_OUTPUT_WEIGHTS', 'ATTENTION_OUTPUT_BIASES', 'INPUT_WEIGHTS_Q', 'INPUT_BIASES_Q', 'INPUT_WEIGHTS_K', 'INPUT_BIASES_K', 'INPUT_WEIGHTS_V', 'INPUT_BIASES_V', 'ATTENTION_OUTPUT_NORM_WEIGHTS', 'ATTENTION_OUTPUT_NORM_BIASES', 'FFN0_OUTPUT_NORM_WEIGHTS', 'FFN0_OUTPUT_NORM_BIASES', 'FFN1_OUTPUT_NORM_WEIGHTS', 'FFN1_OUTPUT_NORM_BIASES', 'FFN2_OUTPUT_NORM_WEIGHTS', 'FFN2_OUTPUT_NORM_BIASES', 'OUTPUT_NORM_WEIGHTS', 'OUTPUT_NORM_BIASES', 'OUTPUT_BOTTLENECK_NORM_WEIGHTS', 'OUTPUT_BOTTLENECK_NORM_BIASES', 'BOTTLENECK_ATTENTION_WEIGHTS', 'BOTTLENECK_ATTENTION_BIASES', 'BOTTLENECK_INPUT_WEIGHTS', 'BOTTLENECK_INPUT_BIASES', 'FFN0_INTERMEDIATE_WEIGHTS', 'FFN0_INTERMEDIATE_BIASES', 'FFN0_OUTPUT_WEIGHTS', 'FFN0_OUTPUT_BIASES', 'FFN1_INTERMEDIATE_WEIGHTS', 'FFN1_INTERMEDIATE_BIASES', 'FFN1_OUTPUT_WEIGHTS', 'FFN1_OUTPUT_BIASES', 'FFN2_INTERMEDIATE_WEIGHTS', 'FFN2_INTERMEDIATE_BIASES', 'FFN2_OUTPUT_WEIGHTS', 'FFN2_OUTPUT_BIASES', 'INTERMEDIATE_WEIGHTS', 'INTERMEDIATE_BIASES', 'OUTPUT_WEIGHTS', 'OUTPUT_BIASES', 'OUTPUT_BOTTLENECK_WEIGHTS', 'OUTPUT_BOTTLENECK_BIASES', 'OUTPUT']


    # Parse the vectors from the list of header files
    vectors = parse_bfloat16_vectors(header_filenames)
    for name, v in vectors.items():
        print(name)
    import pdb; pdb.set_trace()
    
    # Write vectors to binary file in bfloat16 format in the specified order
    write_vectors_to_binary_bfloat16(vectors, ordered_names, output_filename)

    print(f"Binary file '{output_filename}' created successfully.")