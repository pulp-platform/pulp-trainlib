import argparse
import os

import numpy as np
import torch
from PIL import Image

from utils.vit_lr.ViTLR_model import ViTLR
from utils.vit_lr.vit_lr_utils import bordering_resize
from utils.vit_lr.vit_lr_utils import vit_lr_image_preprocessing
from utils.writers.file_writers import (
    header_writer,
    input_writer,
    model_writer,
    output_writer,
    model_components_writer,
    IMPLEMENTED_DATA_TYPES,
)


def create_arg_parser():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights_path",
        help="Path to the saved model.",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--input_path",
        help="Path to a single input file.",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--data_type",
        help="Data type to be used.",
        type=str,
        required=False,
        default="fp32",
    )

    parser.add_argument(
        "--in_channels",
        help="Number of input channels.",
        type=int,
        required=False,
        default=3,
    )

    parser.add_argument(
        "--num_classes",
        help="Number of classes.",
        type=int,
        required=False,
        default=50,
    )

    parser.add_argument(
        "--input_image_width",
        help="Width of the input image.",
        type=int,
        required=False,
        default=350,
    )

    parser.add_argument(
        "--input_image_height",
        help="Height of the input image.",
        type=int,
        required=False,
        default=350,
    )

    parser.add_argument(
        "--original_image_width",
        help="Width of the original image.",
        type=int,
        required=False,
        default=384,
    )

    parser.add_argument(
        "--original_image_height",
        help="Height of the original image.",
        type=int,
        required=False,
        default=384,
    )

    return parser


def get_cuda_device():
    cuda_device = None
    if torch.cuda.is_available():
        if cuda_device is None:
            cuda_device = torch.cuda.device_count() - 1
        device = torch.device("cuda:" + str(cuda_device))
        print("DEVICE SET TO GPU " + str(cuda_device) + "!\n")
    else:
        print("DEVICE SET TO CPU!\n")
        device = torch.device("cpu")

    return device


def load_model(weights_path, num_classes, device, input_size):
    """
    A None weights_path will result in defining a minimal model with one block, for testing purposes.
    """
    print("Loading weights...")
    if weights_path is not None:
        weights = torch.load(weights_path, weights_only=False, map_location=device)

        if "model_state_dict" in weights.keys():
            weights = weights["model_state_dict"]

        # Find number of blocks
        max_block = 0
        for el in weights.keys():
            if "blocks" in el:
                current_block = int(el.split(".")[2])

                if current_block > max_block:
                    max_block = current_block

        n_blocks = max_block + 1

        # Find patch size
        patch_size = tuple(list(weights["patch_embedding.weight"].shape)[-2:])

        # Find hidden dimension
        hidden_dimension = list(weights["patch_embedding.weight"].shape)[0]

        # Find ff_dim
        ff_dim = list(weights["transformer.blocks.0.pwff.fc1.weight"].shape)[0]

        # Find num_heads
        num_heads = list(weights["transformer.blocks.0.attn.proj_q.weight"].shape)[0]
    else:
        # Generate demo values
        n_blocks = 3
        patch_size = (2, 2)
        hidden_dimension = 12
        num_heads = 12
        ff_dim = 4

    print("Creating model object...")
    model = ViTLR(
        device=device,
        num_blocks=n_blocks,
        input_size=input_size,
        num_classes=num_classes,
        # TODO: Adapt to > 0 for training
        dropout_rate=0.0,
        patch_size=patch_size,
        hidden_dimension=hidden_dimension,
        ff_dim=ff_dim,
        num_heads=num_heads,
    )

    if weights_path is not None:
        model.load_state_dict(weights)

    model.to(device)
    model.eval()

    return model


def get_input(path, original_image_size, input_image_size, in_channels, device):
    # Prepare variable to store the image
    x = np.zeros(
        (
            1,
            original_image_size[0],
            original_image_size[1],
            in_channels,
        ),
        dtype=np.uint8,
    )

    if path is not None:
        # Load as grayscale
        if in_channels == 1:
            x[0, :, :, 0] = np.array(Image.open(path).convert("L"))
        # Load normally
        else:
            x[0] = np.array(Image.open(path))
    else:
        # Generate random image if no path provided
        x = np.random.randint(0, 255, x.shape)

    x = x.astype(np.uint8)
    x = bordering_resize(
        x,
        input_image_size=input_image_size,
        original_image_size=original_image_size,
    )
    x = (True, x)
    x = vit_lr_image_preprocessing(x=x, device=device)

    return x


def main():
    # Set seed
    print("Setting seed...")
    np.random.seed(seed=42)
    torch.manual_seed(42)

    # Set higher printing precision
    torch.set_printoptions(precision=10, sci_mode=False)

    # Parse and preprocess arguments
    print("Parsing arguments...")
    parser = create_arg_parser()
    args = parser.parse_args()

    data_type = args.data_type
    assert data_type in IMPLEMENTED_DATA_TYPES, "Invalid data type"

    weights_path = args.weights_path
    input_path = args.input_path
    in_channels = args.in_channels
    num_classes = args.num_classes

    input_image_size = (args.input_image_width, args.input_image_height)
    original_image_size = (args.original_image_width, args.original_image_height)

    if (weights_path is not None) and (weights_path != "None"):
        if "utils" in os.getcwd() and "utils" in weights_path:
            weights_path = weights_path.replace("utils/", "")
        elif "utils" not in os.getcwd() and "utils" not in weights_path:
            weights_path = os.path.join("utils", weights_path)
    else:
        # Demo data
        weights_path = None
        num_classes = 50
        input_image_size = (8, 8)

    if (input_path is not None) and (input_path != "None"):
        if "utils" in os.getcwd() and "utils" in input_path:
            input_path = input_path.replace("utils/", "")
        elif "utils" not in os.getcwd() and "utils" not in input_path:
            input_path = os.path.join("utils", input_path)
    else:
        input_path = None
        original_image_size = (6, 6)

    # Get cuda device
    print("Setting device...")
    device = get_cuda_device()

    # Load model and weights
    model = load_model(
        weights_path=weights_path, num_classes=num_classes, device=device, input_size=input_image_size
    )

    # Load sample input data - currently from class 32
    x = get_input(
        path=input_path,
        original_image_size=original_image_size,
        input_image_size=input_image_size,
        in_channels=in_channels,
        device=device,
    )

    y_pred = model(x=x, get_activation=False)

    ordered_nodes, all_nodes = model.get_model_graph_information(
        x=x, get_activation=False
    )

    # Write header file
    print("\n------------------------------ Writing header file...")
    header_writer(data_type=data_type)

    # Write input data file
    print("\n------------------------------ Writing input data file...")
    input_writer(data=x[1][0], data_type=data_type)

    # Write model file
    print("\n------------------------------ Writing model file...")
    model_writer(model=model, data_type=data_type)

    # Write output data file
    print("\n------------------------------ Writing output file...")
    output_writer(data=y_pred, data_type=data_type)

    # Write header with structures and blobs for model components
    print("\n------------------------------ Writing model components file...")
    model_components_writer(
        ordered_nodes=ordered_nodes, all_nodes=all_nodes, data_type=data_type
    )

    return None


if __name__ == "__main__":
    main()
