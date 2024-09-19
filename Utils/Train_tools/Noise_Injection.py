# Libs >>>
import torch

# Fucns >>>
def Add_gaussian_noise(input, input_dim, mean=0, std=0.1):
    """
    Applies Gaussian noise to the input tensor.
    
    Args:
        input (torch.Tensor): The input tensor to apply Gaussian noise to.
        input_dim (int): The number of dimensions in the input tensor.
        mean (float, optional): The mean of the Gaussian noise distribution. Defaults to 0.
        std (float, optional): The standard deviation of the Gaussian noise distribution. Defaults to 0.1.
    
    Returns:
        torch.Tensor: The input tensor with Gaussian noise applied.
    """
    return torch.randn(input_dim, pin_memory=True) * std + mean + input