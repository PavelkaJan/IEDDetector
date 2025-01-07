import numpy as np
import scipy.signal
import logging
from typing import List, Union
from scipy.signal import butter, filtfilt
from src.signal_preprocessing.validating.eeg_validator import EEGValidator
from src.patient.patient import Patient
from src.constants import (
    SAMPLING_FREQUENCY,
    TIME,
    TRANSLATION_EEG_CHANS_MAP,
    CUTOFF_FREQUENCY,
    EEGChannels1020System,
)

logger = logging.getLogger(__name__)


class EEGProcessor:
    """
    A class responsible for processing EEG data, including resampling,
    reordering channels, and splitting the signal into epochs.
    """

    @staticmethod
    def resample_eeg_signal(
        patient: Union[Patient, int],
        eeg_signal: np.ndarray,
        target_fs: int = SAMPLING_FREQUENCY,
    ) -> np.ndarray:
        """
        Resamples the EEG signal from the patient's original sampling frequency to the target sampling frequency
        and updates the patient's sampling frequency attribute (patient.fs) if a Patient object is provided.

        Args:
            eeg_signal (np.ndarray): The original EEG signal as a 2D numpy array where rows correspond to channels.
            patient (Union[Patient, int]): The patient object containing original sampling frequency, or an int for original_fs.
            target_fs (int, optional): The target sampling frequency. Defaults to SAMPLING_FREQUENCY.

        Returns:
            np.ndarray: The resampled EEG signal.

        Raises:
            ValueError: If the EEG signal is not a 2D array or if the original sampling frequency is invalid.
        """
        if isinstance(patient, Patient):
            original_fs = patient.original_fs
        elif isinstance(patient, int):
            original_fs = patient
        else:
            raise ValueError(
                "The 'patient' argument must be either a Patient object or an integer representing the original_fs."
            )

        # Validate the EEG signal and original sampling frequency
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        EEGValidator.validate_positive_value(original_fs, "Original sampling frequency")

        logger.debug(f"Resampling EEG signal from {original_fs} Hz to {target_fs} Hz.")

        # Calculate the number of samples for resampling
        num_samples = int(eeg_signal.shape[1] * target_fs / original_fs)
        resampled_signal = scipy.signal.resample(eeg_signal, num_samples, axis=1)

        # If a Patient object was provided, update the patient's sampling frequency
        if isinstance(patient, Patient):
            patient.fs = target_fs
            logger.debug(
                f"Updated patient {patient.id}'s sampling frequency to {patient.fs} Hz."
            )

        logger.debug(f"Shape of resampled signal: {resampled_signal.shape}")
        return resampled_signal

    @staticmethod
    def reorder_and_select_eeg_channels(
        eeg_signal: np.ndarray, current_order: List[str]
    ) -> np.ndarray:
        """
        Reorders and selects EEG channels in the provided EEG signal matrix according to a standard set of channels.

        Args:
            eeg_signal (np.ndarray): The EEG signal matrix where the first dimension corresponds to channels.
            current_order (List[str]): The current order of channel names in the EEG signal.

        Returns:
            np.ndarray: The EEG signal reordered and reduced to the standard channel set.

        Raises:
            ValueError: If one or more channels in the standard set are not found in the current order.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        # Translate the current order of channels
        translated_current_order = EEGProcessor._translate_channel_names(current_order)
        # Define the standard order of channels
        standard_channel_names = [channel.name for channel in EEGChannels1020System]

        # Find the indices of the channels in the standard order
        standard_channel_indices = [
            translated_current_order.index(name)
            if name in translated_current_order
            else -1
            for name in standard_channel_names
        ]

        # Check if any standard channels are missing from the translated current order
        if any(index == -1 for index in standard_channel_indices):
            missing_channels = [
                name
                for name, index in zip(standard_channel_names, standard_channel_indices)
                if index == -1
            ]
            logger.error(f"Missing channels: {', '.join(missing_channels)}")
            raise ValueError(
                f"One or more channels in the standard set were not found in the current order: {', '.join(missing_channels)}"
            )

        logger.debug(
            "Channels were reordered and shrinked according to standard 10-20 system specified in constanst.py."
        )

        # Reorder and reduce the EEG signal to the standard channel set
        return eeg_signal[standard_channel_indices, :]

    @staticmethod
    def _translate_channel_names(channels: List[str]) -> List[str]:
        """
        Translates EEG channel names using a provided translation map.

        Args:
            channels (List[str]): A list of EEG channel names to be translated.

        Returns:
            List[str]: A list of translated EEG channel names.
        """
        return [TRANSLATION_EEG_CHANS_MAP.get(channel, channel) for channel in channels]

    @staticmethod
    def split_eeg_to_epochs(
        eeg_signal: np.ndarray, fs: int = SAMPLING_FREQUENCY, epoch_duration: int = TIME
    ) -> np.ndarray:
        """
        Splits the EEG signal into epochs of a specified duration.

        Args:
            eeg_signal (np.ndarray): The EEG signal as a 2D array (channels x samples).
            fs (int, optional): Sampling frequency of the EEG signal. Defaults to SAMPLING_FREQUENCY.
            epoch_duration (int, optional): Duration of each epoch in seconds. Defaults to TIME.

        Returns:
            np.ndarray: A 3D array of shape (n_epochs, channels, epoch_samples) containing the EEG epochs.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        epoch_samples = int(epoch_duration * fs)
        n_epochs = eeg_signal.shape[1] // epoch_samples
        reshaped_signal = eeg_signal[:, : n_epochs * epoch_samples]
        epochs = reshaped_signal.reshape(n_epochs, eeg_signal.shape[0], epoch_samples)

        logger.info(
            f"EEG signal was splited into {n_epochs} epochs. The output shape is {epochs.shape}."
        )
        return epochs

    @staticmethod
    def center_eeg_signal(eeg_signal: np.ndarray) -> np.ndarray:
        """
        Centers the EEG signal by subtracting the mean of each channel.

        Args:
            eeg_signal (np.ndarray): The EEG signal as a 2D array (channels x samples).

        Returns:
            np.ndarray: The centered EEG signal.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        mean_values = np.mean(eeg_signal, axis=1, keepdims=True)
        centered_signal = eeg_signal - mean_values

        logger.debug(
            f"EEG signal was centered. Shape of centred signal: {centered_signal.shape}"
        )

        return centered_signal

    @staticmethod
    def remove_low_frequency_drift(
        eeg_signal: np.ndarray,
        fs: int = SAMPLING_FREQUENCY,
        cutoff_freq: float = CUTOFF_FREQUENCY,
    ) -> np.ndarray:
        """
        Removes drift from EEG data using a high-pass filter.

        Args:
            eeg_signal (np.ndarray): EEG data with channels as rows and samples as columns.
            fs (int. optional): Sampling frequency of the EEG data in Hz.
            cutoff_freq (float, optional): Cutoff frequency for the high-pass filter in Hz.
                                        Default value is specified in constants.py.

        Returns:
            np.ndarray: Drift-corrected EEG data.
        """

        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        EEGValidator.validate_positive_value(fs, "Sampling frequency")
        EEGValidator.validate_positive_value(cutoff_freq, "Cutoff frequency")

        # Design the high-pass filter
        Wn = cutoff_freq / (fs / 2)
        b, a = butter(2, Wn, btype="high")

        filtred_signal = filtfilt(b, a, eeg_signal, axis=1)

        logger.debug("Drift was removed from EEG signal.")
        logger.debug(f"Shape of filtred signal: {filtred_signal.shape}")

        return filtred_signal

    @staticmethod
    def apply_hann_window(eeg_signal: np.ndarray) -> np.ndarray:
        """
        Applies a Hann window to each channel of an EEG signal.

        Args:
            eeg_signal (np.ndarray): A 2D numpy array representing the EEG signal,
                                    with channels as rows and time points as columns.

        Returns:
            np.ndarray: Output EEG data after application of the Hann window.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        # Apply the Hann window across all channels using broadcasting
        window = np.hanning(eeg_signal.shape[1])

        windowed_signal = eeg_signal * window

        logger.debug("Hanning window was apllied to EEG signal.")
        logger.debug(f"Shape of windowed signal: {windowed_signal.shape}")

        return windowed_signal

    @staticmethod
    def add_dimension_to_eeg_signal(eeg_signal: np.ndarray) -> np.ndarray:
        """
        Adds a new dimension to the EEG signal, reshaping it to (1, n_chans, n_samples).

        Args:
            eeg_signal (np.ndarray): The EEG signal as a 2D numpy array (n_chans, n_samples).

        Returns:
            np.ndarray: The EEG signal with an additional dimension (1, n_chans, n_samples).

        Raises:
            ValueError: If the EEG signal is not a 2D array.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        # Add a new dimension to the EEG signal
        reshaped_signal = eeg_signal[np.newaxis, :, :]

        logger.debug(
            f"Added a new dimension to EEG signal. New shape is {reshaped_signal.shape}. Dimensions should be (1, n_chans, n_samples)."
        )

        return reshaped_signal

    @staticmethod
    def standardize_epoch_length(
        eeg_signal: np.ndarray,
        desired_duration: int = TIME,
        desired_fs: int = SAMPLING_FREQUENCY,
    ) -> np.ndarray:
        """
        Standardizes the length of an EEG epoch by trimming it to the desired duration.

        If the number of samples in the EEG signal does not match the expected number of samples for the desired duration
        (based on the sampling frequency), the signal is trimmed to match the desired length. This may result in a loss of information.

        Args:
            eeg_signal (np.ndarray): The input EEG signal as a 2D numpy array, where the first dimension represents channels and the second dimension represents samples.
            desired_duration (int, optional): The desired epoch duration in seconds. Defaults to the global TIME variable.
            desired_fs (int, optional): The desired sampling frequency in Hz. Defaults to the global SAMPLING_FREQUENCY variable.

        Returns:
            np.ndarray: The trimmed EEG signal with the standardized length (channels x samples).
        """
        EEGValidator.validate_eeg_signal_n_samples(eeg_signal)

        [n_chans, n_samples] = eeg_signal.shape
        if n_samples != desired_duration * desired_fs:
            eeg_signal = eeg_signal[:, 0 : desired_duration * desired_fs]

            logging.warning(
                f"EEG signal was forcely shrinked to epoch with shape {eeg_signal.shape}. Possible lost of information. Adjust resampling."
            )
        return eeg_signal

    @staticmethod
    def apply_low_pass_filter(
        eeg_signal: np.ndarray,
        fs: int = SAMPLING_FREQUENCY,
        cutoff: int = 50,
        order: int = 5,
    ) -> np.ndarray:
        """
        Applies a low-pass Butterworth filter to an EEG signal.

        Args:
            eeg_signal (np.ndarray): The input EEG signal to filter.
            fs (int): The sampling frequency of the EEG signal in Hz. Default set to SAMPLING_FREQUENCY.
            cutoff (int): The cutoff frequency in Hz (default is 50 Hz).
            order (int): The order of the filter (default is 5).

        Returns:
            np.ndarray: The filtered EEG signal.
        """
        nyquist = 0.5 * fs
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype="low", analog=False)
        filtered_signal = filtfilt(b, a, eeg_signal)

        logger.debug(
            f"Shape of signal after low pass filtering: {filtered_signal.shape}"
        )
        return filtered_signal

    @staticmethod
    def apply_z_score(eeg_signal: np.ndarray) -> np.ndarray:
        """
        Applies Z-score normalization to an entire EEG epoch.

        Args:
            eeg_signal (np.ndarray): EEG epoch array with shape (n_channels, n_samples),
                                    where n_channels is the number of channels, and
                                    n_samples is the number of time points.

        Returns:
            np.ndarray: Z-score EEG epoch with the same shape as input.
        """
        # Calculate mean and standard deviation across all values in the epoch
        mean = eeg_signal.mean()
        std_dev = eeg_signal.std()

        # Apply Z-score normalization to the whole epoch
        z_scored_epoch = (eeg_signal - mean) / std_dev

        logger.debug(f"Shape of Z-scored signal: {z_scored_epoch.shape}")
        return z_scored_epoch

    @staticmethod
    def create_2_epoch_with_shift(
        original_4s_eeg_signal: np.ndarray, fs: int, start_time: float, end_time: float
    ) -> np.ndarray:
        """
        Creates a 2-second epoch from a 4-second EEG signal matrix created in SIMMEEG.
        Note: This function is specifically designed for simulated epileptic.

        Args:
        - original_4s_eeg_signal (np.ndarray): The 19x1024 matrix representing the 4-second EEG signal
        (19 trials or channels, 1024 samples).
        - fs (int): Sampling frequency in Hz (e.g., 256 Hz).
        - start_time (float): Start time for the new 2-second epoch (e.g., -1.0).
        - end_time (float): End time for the new 2-second epoch (e.g., 1.0).

        Returns:
        - np.ndarray: The (n_chans, n_samples) matrix representing the new 2-second epoch, where n_samples = 2 * fs.
        """
        if end_time - start_time != 2:
            raise ValueError("The time range must be exactly 2 seconds.")

        num_samples = original_4s_eeg_signal.shape[1]
        total_time = num_samples / fs
        time_array = np.linspace(-total_time / 2, total_time / 2, num_samples)

        start_idx = np.searchsorted(time_array, start_time, side="left")
        end_idx = np.searchsorted(time_array, end_time, side="right")

        new_2s_eeg_signal = original_4s_eeg_signal[:, start_idx:end_idx]

        return new_2s_eeg_signal
