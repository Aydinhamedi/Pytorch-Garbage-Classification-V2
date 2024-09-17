# Libs >>>
import os
import cv2
import numpy as np
from tqdm import tqdm
from ..print_color import print_colored as cprint


# Main Func >>>
def load_dataset(
    directory,
    img_size=(224, 224),
    color_mode="grayscale",
    dtype=np.uint8,
    verbose=True,
    **kwargs,
):
    """
    Loads image data and labels from a directory.

    Args:
        directory (str): Path to data directory.
        img_size (tuple): Size to resize images to (width, height). Defaults to (224, 224).
        color_mode (str): Color mode for loading images ('grayscale', 'rgb', 'bgr'). Defaults to 'grayscale'.
        dtype (type): Data type for the loaded images. Defaults to np.uint8.
        verbose (bool): If True, print detailed information. Defaults to True.

    Returns:
        np.ndarray: Array of loaded image data.
        np.ndarray: Array of labels for each image.
    """
    # Prep func print signature
    print_sig = kwargs.get(
        "print_sig",
        cprint(
            f"\\<Func.<Fore.LIGHTMAGENTA_EX>{load_dataset.__name__}<Style.RESET_ALL>\\> ",
            end="",
            return_string=True,
        ),
    )
    # Main logic
    cprint(f"{print_sig}Loading data from directory: {directory}")

    x_data, y_data = [], []
    label_names = sorted(os.listdir(directory))
    total_images = sum(len(files) for _, _, files in os.walk(directory))

    with tqdm(
        total=total_images, desc=kwargs.get("Progbar_desc", "Loading images")
    ) as pbar:
        for label_idx, label_name in enumerate(label_names):
            label_dir = os.path.join(directory, label_name)
            if not os.path.isdir(label_dir):
                continue

            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if color_mode == "grayscale":
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                elif color_mode == "rgb":
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif color_mode == "bgr":
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                else:
                    raise ValueError(
                        "Invalid color_mode. Choose from 'grayscale', 'rgb', 'bgr'."
                    )

                if img is None:
                    continue

                img = cv2.resize(img, img_size)
                x_data.append(img.astype(dtype))
                y_data.append(label_idx)
                pbar.update(1)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data
