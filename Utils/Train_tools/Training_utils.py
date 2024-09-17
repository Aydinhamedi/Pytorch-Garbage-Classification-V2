# Libs >>>
import os
import glob
import torch
import datetime
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, Union, Tuple
from ..print_color import print_colored as cprint
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
)


# Funcs >>>
def EarlyStopping(
    Cache_dict,
    input_monitor,
    Epoch,
    model,
    cache_dir,
    patience=32,
    min_delta=0.0001,
    verbose=True,
):
    # Prep func print signature
    print_sig = cprint(
        f"\\<Func.<Fore.LIGHTMAGENTA_EX>{EarlyStopping.__name__}<Style.RESET_ALL>\\>",
        end="",
        return_string=True,
    )

    # Prep Auxelary Funcs
    def improve_function(new, old):
        return (
            new >= old + min_delta
            if not input_monitor[1] == "min"
            else new <= old - min_delta
        )

    # Prep
    if "EarlyStopping" not in Cache_dict:
        cprint(
            f"{print_sig} <Fore.YELLOW>Initializing Cache_dict..."
        ) if verbose else None
        Cache_dict["EarlyStopping"] = {
            "Best_results": {
                "Epoch": 0,
                "monitor_val": float("inf")
                if input_monitor[1] == "min"
                else -float("inf"),
                "model_cache_dir": None,
            }
        }
    ES_Cache_dict = Cache_dict["EarlyStopping"]
    # main proc
    if improve_function(
        input_monitor[0], ES_Cache_dict["Best_results"]["monitor_val"]
    ):  # if improvement
        cprint(
            f"{print_sig} <Fore.YELLOW>Model Improvement detected: <Fore.GREEN>{input_monitor[2]}[{ES_Cache_dict["Best_results"]["monitor_val"]} to {input_monitor[0]}]"
        ) if verbose else None
        # Saving the best reults
        ES_Cache_dict["Best_results"]["monitor_val"] = input_monitor[0]
        ES_Cache_dict["Best_results"]["Epoch"] = Epoch
        # Caching the model
        [os.remove(file) for file in glob.glob(f"{cache_dir}/*.pth")]
        ES_Cache_dict["Best_results"]["model_cache_dir"] = (
            f"{cache_dir}\\E{Epoch}_{datetime.datetime.now().strftime("y%Y_m%m_d%d-h%H_m%M_s%S")}.pth"
        )
        cprint(
            f"{print_sig} <Fore.YELLOW>Caching the model... <Fore.GREEN>dir[{ES_Cache_dict['Best_results']['model_cache_dir']}]"
        ) if verbose else None
        torch.save(model.state_dict(), ES_Cache_dict["Best_results"]["model_cache_dir"])
        # End
        return False
    else:
        # if no improvement
        cprint(
            f"{print_sig} <Fore.YELLOW>Model Improvement not detected: <Fore.RED>{input_monitor[2]}[{input_monitor[0]} !!! {ES_Cache_dict['Best_results']['monitor_val']}]"
        ) if verbose else None
        # check the patience
        if Epoch - ES_Cache_dict["Best_results"]["Epoch"] >= patience:
            cprint(
                f"{print_sig} <Fore.YELLOW>Early Stopping: <Fore.RED>Patience exceeded"
            ) if verbose else None
            # Loading the best model from cache
            cprint(
                f"{print_sig} <Fore.YELLOW>Loading the best model from cache... <Fore.GREEN>dir[{ES_Cache_dict['Best_results']['model_cache_dir']}]"
            ) if verbose else None
            model.load_state_dict(
                torch.load(
                    ES_Cache_dict["Best_results"]["model_cache_dir"], weights_only=True
                )
            )
            # End
            return True
        else:
            # If patience not exceeded
            # End
            return False


def EarlyStopping_LoadBest(model, Cache_dict, verbose=True):
    # Prep func print signature
    print_sig = cprint(
        f"\\<Func.<Fore.LIGHTMAGENTA_EX>{EarlyStopping_LoadBest.__name__}<Style.RESET_ALL>\\>",
        end="",
        return_string=True,
    )
    # Check if EarlyStopping is initialized
    if "EarlyStopping" not in Cache_dict:
        cprint(
            f"{print_sig} <Fore.YELLOW>EarlyStopping is not initialized"
        ) if verbose else None
        # End
        return None
    # Loading the best model from cache
    cprint(
        f"{print_sig} <Fore.YELLOW>Loading the best model from cache... <Fore.GREEN>dir[{Cache_dict['EarlyStopping']['Best_results']['model_cache_dir']}]"
    ) if verbose else None
    model.load_state_dict(
        torch.load(Cache_dict["EarlyStopping"]["Best_results"]["model_cache_dir"])
    )
    # End
    return None


def calc_metrics(y, y_pred, loss_fn, averaging="macro"):
    """
    Calculate various metrics for multi-class classification.

    Args:
        y (torch.Tensor): Ground truth labels, shape (batch_size, num_classes)
        y_pred (torch.Tensor): Model predictions, shape (batch_size, num_classes)
        loss_fn (callable): The loss function used during training

    Returns:
        dict: A dictionary containing various evaluation metrics
    """
    # Define a small epsilon value
    epsilon = 1e-10

    # Function to safely calculate a metric
    def safe_metric_calculation(metric_fn, *args, **kwargs):
        try:
            return metric_fn(*args, **kwargs)
        except Exception:
            return epsilon

    # Convert tensors to numpy arrays
    y = y.numpy()
    y_pred = y_pred.numpy()

    # Convert predictions to class labels
    y_pred_labels = y_pred.argmax(axis=1)
    y_labels = y.argmax(axis=1)

    # Calculating the metrics
    metrics_dict = {
        "Loss": safe_metric_calculation(
            loss_fn, torch.tensor(y_pred), torch.tensor(y)
        ).item(),
        f"F1 Score ({averaging})": safe_metric_calculation(
            f1_score, y_labels, y_pred_labels, average=averaging
        ),
        f"Precision ({averaging})": safe_metric_calculation(
            precision_score, y_labels, y_pred_labels, average=averaging, zero_division=0
        ),
        f"Recall ({averaging})": safe_metric_calculation(
            recall_score, y_labels, y_pred_labels, average=averaging
        ),
        "AUROC": safe_metric_calculation(roc_auc_score, y, y_pred, multi_class="ovr"),
        "Accuracy": safe_metric_calculation(accuracy_score, y_labels, y_pred_labels),
        "Cohen's Kappa": safe_metric_calculation(
            cohen_kappa_score, y_labels, y_pred_labels
        ),
        "Matthews Correlation Coefficient": safe_metric_calculation(
            matthews_corrcoef, y_labels, y_pred_labels
        ),
    }

    return metrics_dict


def eval(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    loss_fn: Optional[Callable] = None,
    verbose: bool = True,
    return_preds: bool = False,
    **kwargs,
) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]:
    """
    Evaluates the model on the provided dataloader for multi-class classification.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing evaluation data.
        model (torch.nn.Module): The PyTorch model to evaluate.
        loss_fn (Optional[Callable]): The loss function for evaluation (e.g., CrossEntropyLoss). If None, loss is not calculated.
        device (torch.device): The device to run the evaluation on.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        return_preds (bool, optional): Whether to return model predictions and original labels. Defaults to False.
        **kwargs: Additional keyword arguments.
            - TQDM_desc (str): Custom description for the progress bar.

    Returns:
        Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor, torch.Tensor]]: A dictionary containing various evaluation metrics, and optionally the model predictions and original labels.

    Example:
        >>> eval_metrics = eval(test_dataloader, model, nn.CrossEntropyLoss(), device)
        >>> print(f"Test Accuracy: {eval_metrics['Accuracy']:.2f}%")
    """
    model.eval()
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for x, y in tqdm(
            dataloader,
            disable=not verbose,
            unit="batch",
            desc=kwargs.get("TQDM_desc", "Evaluation"),
        ):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            all_y.append(y.detach().cpu())
            all_y_pred.append(y_pred.detach().cpu())

    all_y = torch.cat(all_y)
    all_y_pred = torch.cat(all_y_pred)

    metrics = calc_metrics(all_y, all_y_pred, loss_fn.cpu() if loss_fn else None)

    if return_preds:
        return metrics, all_y_pred, all_y
    else:
        return metrics


class TensorDataset_rtDTC(Dataset[Tuple[torch.Tensor, ...]]):
    """Runtime Data Type Conversion (rtDTC) dataset for efficient data type conversion.

    Args:
        image_tensors (Tensor): A tensor containing the image data.
        label_tensors (Tensor): A tensor containing the labels corresponding to the images.
        dtype (torch.dtype, optional): The desired data type for tensor (Img tensor). Defaults to torch.float32.
    """

    def __init__(
        self,
        image_tensors: torch.Tensor,
        label_tensors: torch.Tensor,
        dtype=torch.float32,
    ) -> None:
        assert image_tensors.size(0) == label_tensors.size(
            0
        ), "Size mismatch between image and label tensors"
        self.image_tensors = image_tensors
        self.label_tensors = label_tensors
        self.dtype = dtype

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.image_tensors[index].type(dtype=self.dtype, non_blocking=True),
            self.label_tensors[index],
        )

    def __len__(self) -> int:
        return self.label_tensors.size(0)


class TensorDataset_rtIDT(Dataset[Tuple[torch.Tensor, ...]]):
    """Runtime Image Data Transformation (rtIDT) dataset for efficient image data processing.

    This dataset applies transformations only to image tensors, changes their dtype,
    and saves memory by augmenting images on-the-fly using torchvision.transforms.v2.

    Args:
        image_tensors (Tensor): A tensor containing the image data.
        label_tensors (Tensor): A tensor containing the labels corresponding to the images.
        transformer: A torchvision.transforms.v2.Compose object.
        dtype (torch.dtype, optional): The desired data type for tensor (Img tensor). Defaults to torch.float32.
    """

    def __init__(
        self,
        image_tensors: torch.Tensor,
        label_tensors: torch.Tensor,
        transformer,
        dtype=torch.float32,
    ) -> None:
        assert image_tensors.size(0) == label_tensors.size(
            0
        ), "Size mismatch between image and label tensors"
        self.image_tensors = image_tensors
        self.label_tensors = label_tensors
        self.transformer = transformer
        self.dtype = dtype

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.transformer(self.image_tensors[index]).type(
                self.dtype, non_blocking=True
            ),
            self.label_tensors[index],
        )

    def __len__(self) -> int:
        return self.label_tensors.size(0)
