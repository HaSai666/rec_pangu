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

def set_device(device_id):
    device_id = int(device_id)
    if device_id<0:
        return torch.device('cpu')
    else:
        return  torch.device(f'cuda:{device_id}')