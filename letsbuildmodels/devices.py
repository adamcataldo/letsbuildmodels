import inspect
import sys

import torch

def try_free(obj, device):
    """Try to free object memory from a device.

    This method only works if the object has only a single reference, defined in
    the namespace of the calling method. It is designed for interactive use 
    cases, like jupyter notebooks, where a user may wish to experiment with
    multiple models and wish to free memory on a GPU from a previous model
    before creating the next model.

    Parameters:
        obj - A (string) variable name to be deleted.
        device - A string, like 'cuda' or 'mps', representing the device to free
                 memory from.

    Retruns:
        True if the object was freed from the device, False otherwise.
    """
    p_frame = inspect.currentframe().f_back
    if obj in p_frame.f_locals and sys.getrefcount(p_frame.f_locals[obj]) == 2:
        del p_frame.f_locals[obj]
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
        return True
    return False