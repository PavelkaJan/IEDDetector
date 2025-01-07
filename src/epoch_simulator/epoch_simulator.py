import numpy as np
import pandas as pd
from scipy.io import savemat
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import logging
import random
from src.constants import (
    EEGChannels1020System,
    SAMPLING_FREQUENCY,
    SAMPLING_FREQUENCY_INTERPOLATION,
)
from src.patient.patient import Patient
from src.signal_preprocessing.loading.eeg_data_io import EEGDataIO
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class EpochSimulator:
    """
    Class for creating SIMMEEG signal matrix.
    """

    def __init__(
        self,
        patient_file_path: str,
        output_folder: Path = Path("spike_matrix_simmeeg_output"),
    ):
        """
        Initialize the simulation with a patient and output folder.

        Args:
            patient_file_path (str): Path to the patient pickle file.
            output_folder (Path): Directory to save results. Default is `spike_matrix_output`.
        """
        self.patient = self._load_patient(patient_file_path)
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_patient(patient_file_path: str) -> Patient:
        logger.info(f"Loading patient data from {patient_file_path}")
        return Patient.load_patient_instance(patient_file_path)

    def _generate_random_epoch_path(self, montage: str = "CA") -> Path:
        logger.debug(f"Selecting random epoch for montage {montage}")
        print(self.patient.epochs_PY_folder_path)
        montage_path = self.patient.epochs_PY_folder_path["IED_present"] / montage
        print("montage_path", montage_path)
        if not montage_path.exists():
            logger.error(f"Montage folder does not exist: {montage_path}")
            raise FileNotFoundError(
                f"{montage} montage folder does not exist: {montage_path}"
            )

        epoch_files = list(montage_path.glob("epoch_*.npy"))
        if not epoch_files:
            logger.error(f"No epoch files in montage folder: {montage_path}")
            raise FileNotFoundError(f"No epoch files in montage folder: {montage_path}")

        return random.choice(epoch_files)

    @staticmethod
    def _find_channel_with_spike(
        epoch_data: np.ndarray, enum_class=EEGChannels1020System
    ) -> Tuple[int, str, np.ndarray]:
        logger.debug("Finding channel with maximum spike.")
        max_channel_index = np.argmax(np.max(epoch_data, axis=1)) + 1
        channel_name = [
            name
            for name, member in enum_class.__members__.items()
            if member.value == max_channel_index
        ][0]
        channel_data = epoch_data[
            max_channel_index - 1
        ]  # Subtract 1 for zero-based indexing
        return max_channel_index, channel_name, channel_data

    @staticmethod
    def _extend_channel_with_spikes_with_zeros(
        channel_data: np.ndarray, original_fs: int = SAMPLING_FREQUENCY
    ) -> np.ndarray:
        logger.debug("Extending signal with zero padding (1 second before/after).")
        zero_padding = np.zeros(original_fs)
        return np.concatenate((zero_padding, channel_data, zero_padding))

    @staticmethod
    def _interpolate_channel_with_spike(
        channel_data: np.ndarray,
        original_fs: int = SAMPLING_FREQUENCY,
        target_fs: int = SAMPLING_FREQUENCY_INTERPOLATION,
    ) -> np.ndarray:
        logger.debug(
            f"Interpolating channel data from {original_fs} Hz to {target_fs} Hz."
        )
        original_length = len(channel_data)
        duration = original_length / original_fs
        target_length = int(duration * target_fs)
        x_original = np.linspace(0, duration, original_length)
        x_target = np.linspace(0, duration, target_length)
        interpolator = interp1d(x_original, channel_data, kind="linear")
        return interpolator(x_target)

    def create_spike_matrix(
        self, num_epochs: int = 15
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Create a spike matrix from randomly selected EEG epochs.

        This function processes a specified number of EEG epochs. For each epoch, the channel with the largest spike is identified,
        the signal is extended with zero-padding, interpolated to a target sampling frequency, and added as a column to the spike matrix.
        Metadata about the processed epochs is also collected.

        Args:
            num_epochs (int, optional): The number of epochs to process. Defaults to 15.

        Returns:
            Tuple[np.ndarray, List[dict]]:
                - np.ndarray: A 2D spike matrix where each column represents the processed data from a channel with the largest spike.
                            Shape: (timepoints, num_epochs).
                - List[dict]: Metadata for each processed epoch, including:
                    - "Patient ID": The ID of the patient.
                    - "Epoch": The filename of the epoch.
                    - "Channel Index": The index of the channel with the spike.
                    - "Channel Name": The name of the channel with the spike.

        Raises:
            FileNotFoundError: If an epoch file or montage folder cannot be found.
        """
        logger.info(f"Creating spike matrix with {num_epochs} epochs.")
        spike_matrix = []
        metadata = []

        for i in range(num_epochs):
            epoch_path = self._generate_random_epoch_path()
            logger.debug(f"Processing epoch {i + 1}/{num_epochs}: {epoch_path}")
            epoch_data = EEGDataIO.load_eeg_epoch(epoch_path)

            max_channel_index, channel_name, channel_data = (
                self._find_channel_with_spike(epoch_data)
            )
            extended_signal = self._extend_channel_with_spikes_with_zeros(channel_data)
            interpolated_signal = self._interpolate_channel_with_spike(
                extended_signal, SAMPLING_FREQUENCY, SAMPLING_FREQUENCY_INTERPOLATION
            )

            spike_matrix.append(interpolated_signal)
            metadata.append(
                {
                    "Patient ID": self.patient.id,
                    "Epoch": epoch_path.name,
                    "Channel Index": max_channel_index,
                    "Channel Name": channel_name,
                }
            )

        spike_matrix = np.array(spike_matrix).T
        return spike_matrix, metadata

    def save_spike_metadata_to_csv(
        self, metadata: List[dict], output_path: str = None
    ) -> None:
        """
        Save metadata for the spike matrix to a CSV file.

        This function saves the metadata collected during spike matrix creation to a CSV file. If no output path is
        provided, the method generates a filename using the patient ID and a timestamp.

        Args:
            metadata (List[dict]): A list of dictionaries containing metadata about the processed epochs, including:
                - "Patient ID": The ID of the patient.
                - "Epoch": The filename of the epoch.
                - "Channel Index": The index of the channel with the spike.
                - "Channel Name": The name of the channel with the spike.
            output_path (str, optional): The path to save the CSV file. If None, a filename is generated automatically
                                        using the patient ID and a timestamp.

        Returns:
            None

        Saves:
            A CSV file containing the metadata to the specified location or the default output path.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.output_folder
                / f"template_{self.patient.id}_spike_metadata_{timestamp}.csv"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(metadata).to_csv(output_path, index=False)
        logger.info(f"Saved spike metadata to {output_path}")

    def transfer_spike_matrix_into_simmmeeg_format(
        self,
        spike_matrix: np.ndarray,
        spike_multiplicator: int = 10,
        output_path: str = None,
    ) -> None:
        """
        Save the spike matrix in SIMMEEG format as a MATLAB file.

        This function replicates the spike matrix to simulate multiple spikes and saves it in a format compatible
        with SIMMEEG. The output file includes the spike data, latency, and sampling rate.

        Args:
            spike_matrix (np.ndarray): The spike matrix to save. Shape: (timepoints, num_epochs).
            spike_multiplicator (int, optional): The factor by which the spike matrix is replicated. Defaults to 10.
            output_path (str, optional): The output path for the saved file. If None, a default name is generated
                                        in the output folder based on the patient ID.

        Returns:
            None

        Saves:
            A MATLAB (.mat) file containing the spike matrix, sampling rate, and latency in the output folder.
        """
        NUM_OF_SOURCES = 3  # Should be 3, num of sources in SIMMEG. If you want use just one source, set 0 other sources in SIMMEEG not here

        logger.debug("Saving spike matrix in SIMMEEG format.")

        if output_path is None:
            output_path = (
                self.output_folder
                / f"template_{self.patient.id}_spike_matrix_simmeeg.mat"
            )
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        srate = SAMPLING_FREQUENCY_INTERPOLATION
        lat = np.linspace(
            0, spike_matrix.shape[0] / srate, spike_matrix.shape[0], endpoint=False
        )
        replicated_data = np.tile(
            spike_matrix, (NUM_OF_SOURCES, 1, spike_multiplicator)
        )

        savemat(
            output_path, {"lat": lat, "srate": srate, "source_data": replicated_data}
        )
        logger.info(f"Spike matrix saved in SIMMEEG format to {output_path}")
