import inspect
import sys

import torch

def get_device():
    """
    Determine the optimal device for computation based on availability.

    This function checks the availability of hardware acceleration devices 
    in the following order of preference:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Metal Performance Shaders on Apple devices)
    3. CPU (fallback if no GPU is available)

    Returns:
        str: A string representing the device to be used for computation.
             Possible values are:
             - 'cuda': Indicates that an NVIDIA GPU is available.
             - 'mps': Indicates that Metal Performance Shaders are available on
                      macOS.
             - 'cpu': Indicates that neither CUDA nor MPS is available,
                      defaulting to CPU.
    """    
    return(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def empty_device(device):
    """
    Clear the memory cache of a specified device.

    This function frees up memory on the specified device by clearing its cache.
    It supports CUDA (NVIDIA GPUs) and MPS (Metal Performance Shaders on Apple 
    devices).

    Parameters:
        device (str): The device whose cache should be cleared.
                      Supported values are:
                      - 'cuda': Clears the CUDA cache for NVIDIA GPUs.
                      - 'mps': Clears the MPS cache for macOS devices.

    Returns:
        None
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
