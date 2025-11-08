# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import glob
import sys
from typing import Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from qai_hub_models.models.ddrnet23_slim.app import DDRNetApp
from qai_hub_models.models.ddrnet23_slim.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    DDRNet,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image
from qai_hub_models.utils.image_processing import pil_resize_pad
from torchvision import transforms
from torch.nn.functional import interpolate, pad
from tqdm import tqdm

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.png"
)

def undo_resize_pad(
    image: torch.Tensor,
    orig_size_wh: Tuple[int, int],
    scale: float,
    padding: Tuple[int, int],
):
    """
    Undos the efffect of resize_pad. Instead of scale, the original size
    (in order width, height) is provided to prevent an off-by-one size.
    """
    width, height = orig_size_wh

    rescaled_image = interpolate(image.unsqueeze(0), scale_factor=1 / scale, mode="bilinear")

    scaled_padding = [int(round(padding[0] / scale)), int(round(padding[1] / scale))]

    cropped_image = rescaled_image[
        ...,
        scaled_padding[1] : scaled_padding[1] + height,
        scaled_padding[0] : scaled_padding[0] + width,
    ]

    return cropped_image


def pil_undo_resize_pad(
    image: Image, orig_size_wh: Tuple[int, int], scale: float, padding: Tuple[int, int]
) -> Image:
    transform = transforms.Compose([transforms.PILToTensor()])  # bgr image
    torch_image: torch.Tensor = transform(image)
    torch_out_image = undo_resize_pad(torch_image, orig_size_wh, scale, padding)
    return torch_out_image

def get_model():
    parser = get_model_cli_parser(DDRNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([])
    model = demo_model_from_cli_args(DDRNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    ds_dir = "datasets/Cambridge_GreatCourt"
    images_folder = f"{ds_dir}/train/rgb"

    # Load image
    app = DDRNetApp(model)
    classes = [2]
    return app

# Run DDRNet end-to-end on a sample image.
# The demo will display a image with the predicted segmentation map overlaid.
def main(is_test: bool = False):
    # Demo parameters
    parser = get_model_cli_parser(DDRNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(DDRNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    ds_dir = "datasets/Cambridge_StMarysChurch"
    images_folder = f"{ds_dir}/train/rgb"
    files = glob.glob(f"{images_folder}/*.png")

    # Load image
    app = DDRNetApp(model)
    classes = [2]

    with h5py.File(f"{ds_dir}/selected_points.h5", "w") as h5f:
        for count, img in enumerate(tqdm(files, desc="Processing images")):
            (_, _, height, width) = DDRNet.get_input_spec()["image"][0]
            orig_image = load_image(img)
            image, scale, padding = pil_resize_pad(orig_image, (height, width))

            # segmentation
            mask = app.segment_image(image, raw_output=True)[0]
            mask = np.argmax(mask, 0).astype(np.uint8)
            mask = pil_undo_resize_pad(Image.fromarray(mask), orig_image.size, scale, padding)
            mask = np.array(mask)[0, 0]

            orig_image = np.array(orig_image)

            # process each class
            for cls in classes:
                binary_mask = (mask == cls).astype(np.uint8) * 255
                rows, cols = np.nonzero(binary_mask)
                selected = np.stack((rows, cols), axis=1)

                # Save selected coordinates to h5 file under group per image
                grp = h5f.create_group(img)
                grp.create_dataset("selected", data=selected, compression="gzip")

                # Visualize with removed pixels
                orig_image[selected[:, 0], selected[:, 1]] = 0
                Image.fromarray(orig_image).save(f"debug/{count}.png")

if __name__ == "__main__":
    main()
