"""
Only for testing purposes. Delete in the future.
"""

from pathlib import Path
import numpy as np
from typing import Tuple

def load_eeg_record(eeg_record_path: Path) -> np.ndarray:
    """
    Loads an EEG record in numpy format.

    Args:
        eeg_record_path (Path): Path to the EEG record file in numpy format.

    Returns:
        np.ndarray: The loaded EEG data with the singleton dimension removed.

    Notes:
        The EEG record should have the shape (1, DESIRED_NUM_OF_CHANS, TIME_POINTS), e.g., (1, 18, 250).
    """
    data = np.load(eeg_record_path)
    
    if len(data.shape) == 3:
        data = np.squeeze(data)  # Remove singleton dimension
    
    return data


def generate_random_heatmap(data_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generate a random heatmap with the specified shape.

    Args:
        data_shape (tuple): Shape of the data for the heatmap.

    Returns:
        numpy.ndarray: A random heatmap with values between 0 and 1.
    """
    return np.random.rand(*data_shape)