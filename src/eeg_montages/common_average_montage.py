import numpy as np

from src.constants import EEGChannels1020System
from enum import Enum
from typing import Type, Any
from src.signal_preprocessing.validating.eeg_validator import EEGValidator


class CommonAverageMontage:
    """
    Default montage other montages are derived from this.

    This class uses EEGChannels1020System from constants.py as the default set of EEG channels
    for computations.
    """

    def __init__(
        self, eeg_signal: np.ndarray, channel_names: Type[Enum] = EEGChannels1020System
    ):
        self.eeg_signal = EEGValidator.validate_eeg_signal(eeg_signal)
        self.channel_names = channel_names
        self.channel_data = self._get_channel_data()

    def compute_montage(self) -> np.ndarray:
        """
        Computes the common average montage from the EEG data.

        The common average montage is computed by subtracting the average of all channels from each channel.

        Returns:
            np.ndarray: A 2D NumPy array representing the EEG data in the common average montage format,
                        where each row corresponds to a channel and each column corresponds to time points.
        """
        common_average = np.mean(self.eeg_signal, axis=0)
        common_average_montage = self.eeg_signal - common_average

        return common_average_montage

    def _get_channel_data(self) -> dict:
        """
        Creates a dictionary mapping each channel name to its respective EEG data.

        Returns:
            Dict[str, np.ndarray]: A dictionary where the keys are channel names and the values are 1D NumPy arrays
                                   representing the EEG data for each channel.
        """
        return {
            channel.name: self.eeg_signal[channel.value - 1, :]
            for channel in self.channel_names
        }

    def __getattribute__(self, name: str) -> Any:
        """
        Overrides attribute access to enable direct access to channel data via channel names.

        If the attribute name matches a channel name, the corresponding EEG data is returned.
        Otherwise, standard attribute access is performed.

        Args:
            name (str): The name of the attribute or channel to access.

        Returns:
            Any: The attribute value or EEG data corresponding to the given channel name.

        Raises:
            AttributeError: If the attribute or channel name does not exist.
        """
        try:
            # Try to retrieve attribute normally
            return object.__getattribute__(self, name)
        except AttributeError:
            # If attribute is not found, check in channel_data
            channel_data = object.__getattribute__(self, "channel_data")
            if name in channel_data:
                return channel_data[name]
            # If still not found, raise AttributeError
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
