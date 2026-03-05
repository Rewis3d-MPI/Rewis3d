"""
Misc

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module

import subprocess
import datetime
import psutil
import glob


class TrainingStopException(Exception):
    """Custom exception to stop training gracefully."""

    pass


def clear_shared_cache(prefix="pointcept-"):
    for shm_file in glob.glob(f"/dev/shm/{prefix}*"):
        try:
            os.remove(shm_file)
        except Exception as e:
            print(f"Could not delete {shm_file}: {e}")


def get_remaining_slurm_time():
    """
    Returns remaining time (in seconds) for the Slurm job.
    Returns `None` if not running under Slurm.
    """
    try:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return None

        # Get job end time using `scontrol`
        cmd = f"scontrol show job {job_id} -o"
        output = subprocess.check_output(cmd, shell=True).decode()

        # Parse the `EndTime` field
        end_time_str = [
            x.split("=")[1] for x in output.split() if x.startswith("EndTime=")
        ][0]
        end_time = datetime.datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S")

        # Calculate remaining time
        now = datetime.datetime.now()
        remaining = (end_time - now).total_seconds()
        return max(0, remaining)  # Avoid negative values

    except Exception as e:
        print(f"Error checking Slurm time: {e}")
        return None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1).clone()
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_2d_gpu(output, target, k, ignore_index=-1):
    # Ensure output and target are 1D tensors
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)

    # Create a mask to ignore specified indices
    valid_mask = target != ignore_index

    # Apply the mask to filter out invalid indices
    output = output[valid_mask]
    target = target[valid_mask]

    # Compute intersection by selecting indices where predictions match targets
    intersection = output[output == target]

    # Use torch.bincount to count occurrences of each class
    area_intersection = torch.bincount(intersection, minlength=k)
    area_output = torch.bincount(output, minlength=k)
    area_target = torch.bincount(target, minlength=k)

    # Compute the union for each class
    area_union = area_output + area_target - area_intersection

    # Convert counts to float tensors
    return area_intersection.float(), area_union.float(), area_target.float()


def error_map(prediction, target, ignore_index=255):
    """
    Create an error map based on the comparison of prediction and target arrays.

    Parameters:
    prediction (np.ndarray): The predicted labels, shape (N,)
    target (np.ndarray): The ground truth labels, shape (N,)
    ignore_index (int): The index in the target array that should be ignored

    Returns:
    np.ndarray: The error map, shape (N,)
    """

    error_map = np.zeros_like(prediction, dtype=np.uint8)

    error_map[prediction == target] = 1

    error_map[target == ignore_index] = ignore_index

    error_map[prediction == ignore_index] = ignore_index

    return error_map


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass
