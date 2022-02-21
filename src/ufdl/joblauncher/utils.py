from typing import IO, Union

import numpy as np


def np_min(a):
    """
    Returns the minimum from the array, NaN if length is 0.

    :param a: the array to use for the calculation
    :type a: np.array
    :return: the minimum or NaN if failed to calculate
    :rtype: float
    """
    if len(a) == 0:
        return float("NaN")
    try:
        return float(np.min(a))
    except:
        return float("NaN")


def np_max(a):
    """
    Returns the maximum from the array, NaN if length is 0.

    :param a: the array to use for the calculation
    :type a: np.array
    :return: the maximum or NaN if failed to calculate
    :rtype: float
    """
    if len(a) == 0:
        return float("NaN")
    try:
        return float(np.max(a))
    except:
        return float("NaN")


def np_mean(a):
    """
    Returns the mean from the array, NaN if length is 0.

    :param a: the array to use for the calculation
    :type a: np.array
    :return: the mean or NaN if failed to calculate
    :rtype: float
    """
    if len(a) == 0:
        return float("NaN")
    try:
        return float(np.mean(a))
    except:
        return float("NaN")


def np_median(a):
    """
    Returns the median from the array, NaN if length is 0.

    :param a: the array to use for the calculation
    :type a: np.array
    :return: the median or NaN if failed to calculate
    :rtype: float
    """
    if len(a) == 0:
        return float("NaN")
    try:
        return float(np.median(a))
    except:
        return float("NaN")


def np_std(a):
    """
    Returns the stdev from the array, NaN if length is 0.

    :param a: the array to use for the calculation
    :type a: np.array
    :return: the stdev or NaN if failed to calculate
    :rtype: float
    """
    if len(a) == 0:
        return float("NaN")
    try:
        return float(np.std(a))
    except:
        return float("NaN")


def write_to_file(file: IO[bytes], data: Union[bytes, IO[bytes]]):
    """
    Writes the given data to the given file.

    :param file:
                The file to write to.
    :param data:
                The data to write.
    """
    if isinstance(data, bytes):
        file.write(data)
    else:
        for chunk in data:
            file.write(chunk)
