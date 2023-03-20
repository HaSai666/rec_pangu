# -*- ecoding: utf-8 -*-
# @ModuleName: gpu_utils
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/9/19 7:27 PM
import torch
def get_gpu_usage(device=None):
    r""" Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    return '{:.2f} G/{:.2f} G'.format(reserved, total)


def set_device(device_id: int = -1) -> torch.device:
    """
    Sets the device to be used for tensor computations.

    Args:
        device_id: int, optional
            An integer indicating the index of the GPU device to use for tensor computations.
            `device_id` < 0 indicates that the computation should be performed on the CPU. Default -1.

    Returns:
        torch.device
            A device representing the device to be used for tensor computations.

    Raises:
        TypeError
            If `device_id` is not an integer.
    """
    # Check if `device_id` is of integer type
    if not isinstance(device_id, int):
        raise TypeError("Device ID should be an integer.")

    # Set device to CPU if `device_id` is less than zero
    if device_id < 0:
        return torch.device('cpu')
    # Otherwise, return the specified GPU device
    else:
        return torch.device(f'cuda:{device_id}')
