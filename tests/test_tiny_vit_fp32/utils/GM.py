import argparse

import onnx
import torch

from file_writers import model_components_writer
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


def onnx_parser(onnx_model):
    return None


def main():
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

    # Model to ONNX
    sample_input = torch.randn(1, cfg["IN_CHANS"], cfg["IN_IMG_SIZE"], cfg["IN_IMG_SIZE"])
    torch.onnx.export(
        model,
        sample_input,
        "TinyViT.onnx",
        verbose=False,
        opset_version=20,
        training=torch.onnx.TrainingMode.EVAL,
    )
    onnx_model = onnx.load("TinyViT.onnx")

    # Parse onnx
    onnx_parser(onnx_model)

    # Write model components
    model_components_writer(file_root_dir=root_dir)


if __name__ == "__main__":
    main()
