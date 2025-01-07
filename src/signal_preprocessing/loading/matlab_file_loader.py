import logging
import mat73
import pendulum
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import List

from src.signal_preprocessing.loading.eeg_file_loader import EEGFileLoader
from src.constants import MATLAB_FILE_MANDATATORY_VARS

logger = logging.getLogger(__name__)


class MatlabFileLoader(EEGFileLoader):
    """
    A class designed to load and validate MATLAB files (specifically version 7.3) that contain EEG data.

    This class is responsible for:
    - Loading the MATLAB file and extracting its contents into a structured format.
    - Performing mandatory checks to ensure that all required variables are present and contain valid data.
    - Providing easy access to key EEG data and metadata such as sampling frequency, channel names, and recording duration.

    The loaded data is returned as a dictionary containing the raw EEG signal and its associated metadata.

    Attributes:
        file_path (str): The path to the MATLAB file to be loaded.
        data (dict): The loaded data from the MATLAB file, structured as a dictionary.
    """

    def __init__(self, file_path: str):
        """
        Initializes the MatlabFileLoader with the provided file path.

        Args:
            file_path (str): The path to the MATLAB file to be loaded.
        """
        super().__init__(file_path)
        self.data = self._load_matlab_file_v73()
        self._check_mandatory_vars()

    def load(self) -> dict:
        """
        Loads and validates the MATLAB file, returning the raw data.

        Returns:
            dict: A dictionary containing the raw EEG data and associated metadata.
        """
        logger.info(f"Loading of MATLAB file {self.file_name} was successful.")

        return {
            "file_name": self.file_name,
            "eeg_signal": self.eeg_signal,
            "fs": self.fs,
            "start_date": self.data["header"]["startdate"],
            "formatted_start_time": self.formatted_start_time,
            "channel_names": self.channel_names,
            "duration": self.duration,
        }

    @staticmethod
    def load_single_mat_epoch(file_path: str) -> np.ndarray:
        """
        Loads EEG epoch data from a MATLAB file.

        This static method reads a MATLAB file from the given file path and extracts the EEG signal
        data stored under the key 'data'. If the 'data' key is not found in the MATLAB file, an error
        is logged, and a KeyError is raised.

        Parameters:
            file_path (str): The path to the MATLAB (.mat) file to load.

        Returns:
            np.ndarray: A NumPy array containing the EEG data extracted from the 'data' field
                        in the MATLAB file.

        Raises:
            KeyError: If the 'data' key is not present in the MATLAB file.
        """
        matlab_data = sio.loadmat(file_path)

        # Direct loading from Brainstorm
        if "F" in matlab_data:
            data = matlab_data["F"]
        else:
            logger.error(
                "You need to specify where the EEG signal is saved in the .mat file."
            )
            raise KeyError("The key 'data' was not found in the MATLAB file.")

        return data

    def _load_matlab_file_v73(self) -> dict:
        """
        Load a MATLAB file (version 7.3) using the mat73 library.

        Returns:
            dict: The contents of the MATLAB file as a dictionary.
        """
        logger.info(f"Loading MATLAB file from {self.file_path}")

        return mat73.loadmat(self.file_path, use_attrdict=True)

    def _check_mandatory_vars(self) -> None:
        """
        Validates the presence of mandatory variables in the loaded MATLAB file.

        Raises:
            KeyError: If any of the mandatory keys specified in `MATLAB_FILE_MANDATATORY_VARS` is missing or contains an empty value.
        """
        for path, description in MATLAB_FILE_MANDATATORY_VARS:
            current_data = self.data
            for i, key in enumerate(path):
                if key not in current_data:
                    raise KeyError(
                        f"Matlab file does not contain '{'/'.join(path[:i+1])}'. This variable should contain {description}."
                    )
                value = current_data[key]
                if (isinstance(value, np.ndarray) and value.size == 0) or (
                    isinstance(value, list) and len(value) == 0
                ):
                    raise KeyError(
                        f"Matlab file contains an empty '{'/'.join(path[:i+1])}'. This variable should contain {description}."
                    )
                current_data = current_data[key]

    @property
    def file_name(self) -> str:
        """
        Extracts the file name from the file path.

        Returns:
            str: The file name without extension.
        """
        return Path(self.file_path).stem

    @property
    def eeg_signal(self) -> np.ndarray:
        """
        Extracts and transposes the EEG signal from the loaded MATLAB file data.

        Returns:
            np.ndarray: The EEG signal matrix with channels as the first dimension and time as the second dimension.
        """
        eeg_signal = self.data["d"]
        if eeg_signal.shape[0] != len(self.channel_names):
            eeg_signal = np.transpose(eeg_signal)

        return eeg_signal

    @property
    def fs(self) -> float:
        """
        Returns the sampling frequency from the MATLAB file data.

        Returns:
            float: The sampling frequency.
        """
        return self.data["fs"]

    @property
    def formatted_start_time(self) -> str:
        """
        Formats the MATLAB serial date number to a readable date string.

        Returns:
            str: The formatted start date and time.
        """
        serial_date_number = self.data["header"]["startdate"]

        if isinstance(serial_date_number, np.ndarray):
            serial_date_number = float(serial_date_number)

        reference_date = pendulum.datetime(1, 1, 1)
        converted_date = reference_date.add(days=serial_date_number)
        year_adjusted_date = converted_date.subtract(years=1, days=2)

        return year_adjusted_date.format("DD-MMM-YYYY HH:mm:ss")

    @property
    def channel_names(self) -> List[str]:
        """
        Extracts channel names from the loaded MATLAB file data.
        Get the original order and names of channels.

        Returns:
            List[str]: A list of channel names.
        """
        raw_channel_names = self.data["header"]["label"]

        return [item for sublist in raw_channel_names for item in sublist]

    @property
    def duration(self) -> float:
        """
        Computes the duration of the EEG record.

        Returns:
            float: The duration of the record in seconds.
        """
        num_samples = self.eeg_signal.shape[1]
        duration = num_samples / self.fs

        return duration
