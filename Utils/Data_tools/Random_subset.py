# Libs >>>
import numpy as np
from typing import List, Tuple


# Main Func >>>
def random_subset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    num: int,
    seed: int = 42,
    label_list: List[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a random subset of data from x_data and y_data.

    This function ensures that the selected indices for x_data and y_data are the same,
    preventing any mismatch between the features and labels.

    Args:
        x_data (np.ndarray): The input data array with shape (n_samples, ...).
        y_data (np.ndarray): The corresponding labels array with shape (n_samples, num_classes).
        num (int): The number of samples to select.
        seed (int, optional): The seed for the random indices generator to ensure reproducibility. Default is 42.
        label_list (List[int], optional): A list of label indices to filter the data. If None, all labels are considered.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the randomly selected subset of x_data and y_data.

    Raises:
        ValueError: If num is greater than the number of samples in x_data or y_data.
        ValueError: If label_list is provided but fewer than num samples match the labels.
    """

    if seed is not None:
        np.random.seed(seed)

    if label_list is not None:
        # Convert one-hot encoded labels to indices
        label_indices = np.argmax(y_data, axis=1)
        # Find indices where labels match the label_list
        valid_indices = np.where(np.isin(label_indices, label_list))[0]
        if len(valid_indices) < num:
            raise ValueError(
                f"Not enough samples with labels in {label_list} to select {num} samples. Total: {len(valid_indices)}"
            )
        indices = np.random.choice(valid_indices, num, replace=False)
    else:
        indices = np.random.choice(x_data.shape[0], num, replace=False)

    return x_data[indices], y_data[indices]
