import numpy as np
import torch
from PIL import Image

from utils.file_writers import (
    header_writer,
    input_writer,
    model_writer,
    output_writer
)
from utils.vit_lr.ViTLR_model import ViTLR
from utils.vit_lr.vit_lr_utils import bordering_resize
from utils.vit_lr.vit_lr_utils import vit_lr_image_preprocessing


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


def load_model(weights_path, num_classes, device):
    print("Loading weights...")
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

    print("Creating model object...")
    model = ViTLR(
        device=device,
        num_blocks=max_block + 1,
        input_size=(384, 384),
        num_classes=num_classes,
        dropout_rate=0.0,
    )
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

    # Load as grayscale
    if in_channels == 1:
        x[0, :, :, 0] = np.array(Image.open(path).convert("L"))
    # Load normally
    else:
        x[0] = np.array(Image.open(path))

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
    # Parse arguments
    print("Parsing arguments...")
    weights_path = "utils/sample_data/lite_weights_sample.pth"
    num_classes = 50
    input_path = "utils/sample_data/input_sample.png"
    original_image_size = (350, 350)
    input_image_size = (384, 384)
    in_channels = 3

    # Set seed
    print("Setting seed...")
    torch.manual_seed(42)

    # Set higher printing precision
    torch.set_printoptions(precision=10, sci_mode=False)

    # Get cuda device
    print("Setting device...")
    device = get_cuda_device()

    # Load model and weights
    model = load_model(
        weights_path=weights_path, num_classes=num_classes, device=device
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

    # Write header file
    print("\n------------------------------ Writing header file...")
    header_writer(data_type="fp32")

    # Write input data file
    print("\n------------------------------ Writing input data file...")
    input_writer(data=x[1][0], data_type="fp32")

    # Write model file
    print("\n------------------------------ Writing model file...")
    model_writer(model=model, data_type="fp32")

    # Write output data file
    print("\n------------------------------ Writing output file...")
    output_writer(data=y_pred, data_type="fp32")

    return None


if __name__ == "__main__":
    main()
