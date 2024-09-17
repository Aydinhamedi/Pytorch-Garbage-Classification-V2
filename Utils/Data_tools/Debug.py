# Libs >>>
import os
import shutil
import tarfile
import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, List
from ..print_color import print_colored as cprint


# Main Func >>>
def save_samples(
    images: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    label_names: Optional[List[str]] = None,
    jpeg_quality: int = 95,
    verbose: bool = False,
) -> None:
    """
    Saves sample images as JPEG along with their labels, and compresses them into a tar archive.

    This function supports multi-class one-hot encoded labels.

    Args:
        images (np.ndarray): Array of images with shape (n_samples, height, width, channels).
        labels (np.ndarray): Corresponding labels with shape (n_samples, n_classes) for multi-class.
        save_dir (str): Directory to save the samples.
        label_names (Optional[List[str]]): List of label names corresponding to one-hot positions.
        jpeg_quality (int): JPEG compression quality (0-100). Defaults to 95.
        verbose (bool): If True, print detailed information. Defaults to False.

    Returns:
        None

    Raises:
        ValueError: If images and labels have mismatched lengths.
    """
    # Prep func print signature
    print_sig = cprint(
        f"\\<Func.<Fore.LIGHTMAGENTA_EX>{save_samples.__name__}<Style.RESET_ALL>\\>",
        end="",
        return_string=True,
    )

    # Main logic
    if len(images) != len(labels):
        raise ValueError("Number of images and labels must match.")

    os.makedirs(save_dir, exist_ok=True)
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_dir = os.path.join(save_dir, date_time)
    os.makedirs(sample_dir, exist_ok=True)

    for i, (image, label) in enumerate(
        tqdm(zip(images, labels), desc="Saving images", disable=not verbose)
    ):
        # Ensure image is in the correct range for saving
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Handle multi-class one-hot data
        label_index = np.argmax(label)
        label_str = label_names[label_index] if label_names else str(label_index)

        # Save image
        pil_image = Image.fromarray(image.squeeze())
        image_path = os.path.join(sample_dir, f"sample_{i}_label_{label_str}.jpg")
        pil_image.save(image_path, format="JPEG", quality=jpeg_quality)

        if verbose:
            cprint(f"{print_sig} Saved image {i} with label: {label_str}")

    # Create tar archive
    tar_path = f"{sample_dir}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(sample_dir, arcname=os.path.basename(sample_dir))

    # Remove the original folder
    shutil.rmtree(sample_dir)

    if verbose:
        cprint(f"{print_sig} Samples compressed and saved to {tar_path}")
        cprint(f"{print_sig} Original folder removed: {sample_dir}")
