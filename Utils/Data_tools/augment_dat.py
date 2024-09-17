# Libs >>>
import torch
from tqdm import tqdm
from typing import Tuple
from functools import partial
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed


# Funcs >>>
def worker(
    input_data: Tuple[int, torch.Tensor],
    transformer: transforms.Compose,
    mode: str = "cpu",
) -> Tuple[int, torch.Tensor]:
    """
    Worker function to apply transformations to an image.

    Args:
        input_data (Tuple[int, torch.Tensor]): A tuple containing an index and an image tensor.
        transformer (transforms.Compose): The Compose object containing transformations to apply.
        mode (str, optional): The mode to use for processing ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing the index and the transformed image tensor.
    """
    index, image = input_data
    try:
        with torch.no_grad():
            if mode == "cuda":
                image = image.pin_memory().to("cuda", non_blocking=True)
            transformed_image = transformer(image)
            if mode == "cuda":
                transformed_image = transformed_image.to("cpu", non_blocking=True)
        return index, transformed_image
    except Exception as e:
        print(f"Error processing image at index {index}: {e}")
        return index, image  # Return original image if transformation fails


def augment_tensor(
    tensor: torch.Tensor,
    transformer: transforms.Compose,
    verbose: bool = True,
    proc_count: int = torch.get_num_threads(),
    batch_size: int = 64,
    mode: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Augments a PyTorch tensor using the provided Compose transformer in parallel.


    Args:
        tensor (torch.Tensor): The input tensor to augment.
        transformer (transforms.Compose): The Compose object containing transformations to apply.
        verbose (bool, optional): If True, shows the progress bar. Defaults to True.
        proc_count (int, optional): Number of threads to use. Defaults to (available CPU threads).
        batch_size (int, optional): Number of images to process in each batch. Defaults to 128.
        mode (str, optional): The mode to use for processing ('cpu' or 'cuda'). Defaults to 'cpu'.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.Tensor: The augmented tensor with the same order as input.

    Raises:
        RuntimeError: If an error occurs during the augmentation process.
    """
    worker_with_transformer = partial(worker, transformer=transformer, mode=mode)
    augmented_images = []

    try:
        with ThreadPoolExecutor(max_workers=proc_count) as executor:
            for i in tqdm(
                range(0, len(tensor), batch_size),
                disable=not verbose,
                desc=kwargs.get("Progbar_desc", "Processing images"),
            ):
                batch = tensor[i : i + batch_size]
                indexed_data = list(enumerate(batch, start=i))

                futures = [
                    executor.submit(worker_with_transformer, item)
                    for item in indexed_data
                ]

                batch_results = [future.result() for future in as_completed(futures)]
                batch_results.sort(key=lambda x: x[0])

                augmented_images.extend([img for _, img in batch_results])

        return torch.stack(augmented_images)

    except Exception as e:
        raise RuntimeError(f"Error during augmentation process: {e}")
