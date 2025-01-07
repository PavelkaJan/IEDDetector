import numpy as np
import logging
from typing import Optional
from src.patient.patient import Patient
from src.constants import (
    NUM_OF_CHANS_10_20,
    TIME,
    SAMPLING_FREQUENCY,
    NN_INPUT_DIMENSIONS,
    EEGChannels1020System,
)

logger = logging.getLogger(__name__)


class EEGValidator:
    """
    Provides methods for validating the EEG signal, including its dimensions and channel count.
    """

    @staticmethod
    def validate_eeg_signal(
        eeg_signal: np.ndarray, num_of_chans: int = NUM_OF_CHANS_10_20
    ) -> np.ndarray:
        """
        Validates both the dimensions and channel count of the EEG signal.

        Args:
            eeg_signal (np.ndarray): The EEG signal data to be validated.
            num_of_chans (int): The expected number of channels (default is NUM_OF_CHANS_10_20).

        Returns:
            np.ndarray: The validated EEG signal.

        Raises:
            ValueError: If the EEG signal is not a 2D array or if the number of channels is incorrect.
        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        EEGValidator.validate_eeg_signal_n_chans(eeg_signal, num_of_chans)
        return eeg_signal

    @staticmethod
    def validate_eeg_signal_dims(eeg_signal: np.ndarray) -> np.ndarray:
        """
        Ensures the EEG signal is a non-empty 2D array, squeezing if necessary.

        Args:
            eeg_signal (np.ndarray): The EEG data to be validated.

        Returns:
            np.ndarray: The validated 2D EEG signal.

        Raises:
            ValueError: If the EEG data is not 2D or is empty.
        """
        if eeg_signal.ndim == 3:
            eeg_signal = np.squeeze(eeg_signal)
            logger.info(f"EEG signal squeezed to 2D array. Shape is {eeg_signal.shape}")

        if eeg_signal.ndim != 2 or eeg_signal.size == 0:
            logger.error(
                f"Invalid EEG signal dimensions: {eeg_signal.shape}. Expected a non-empty 2D array."
            )
            raise ValueError(
                "EEG signal must be a non-empty 2D array with shape (n_channels, n_samples)."
            )

        return eeg_signal

    @staticmethod
    def validate_eeg_signal_n_chans(
        eeg_signal: np.ndarray, num_of_chans: int = NUM_OF_CHANS_10_20
    ) -> None:
        """
        Validates that the EEG signal contains the expected number of channels.

        Args:
            eeg_signal (np.ndarray): The EEG signal data to validate.
            num_of_chans (int): The expected number of channels.

        Raises:
            ValueError: If the number of channels does not match the expected value.
        """
        actual_channels = eeg_signal.shape[0]

        if actual_channels not in {num_of_chans, num_of_chans - 1}:
            logger.error(
                f"Invalid number of EEG channels: {actual_channels}. Expected {num_of_chans} or {num_of_chans - 1}."
            )
            raise ValueError(
                f"EEG signal should have {num_of_chans} or {num_of_chans - 1} channels (depending on montage), but has {actual_channels}."
            )

    @staticmethod
    def validate_eeg_signal_n_samples(
        eeg_signal: np.ndarray,
        desired_duration: int = TIME,
        desired_fs: int = SAMPLING_FREQUENCY,
    ) -> None:
        """
        Validates the number of samples in an EEG signal to ensure it matches the expected number of samples
        based on the desired duration and sampling frequency.

        This function first validates the dimensions of the EEG signal. It then checks whether the number of samples
        in the EEG signal matches the expected number, which is the product of the desired duration and sampling frequency.
        If the number of samples does not match, a warning is logged.

        Args:
            eeg_signal (np.ndarray): The EEG signal as a 2D numpy array, where the first dimension represents channels and the second dimension represents samples.
            desired_duration (int, optional): The desired duration of the EEG epoch in seconds. Defaults to the global `TIME` variable.
            desired_fs (int, optional): The desired sampling frequency in Hz. Defaults to the global `SAMPLING_FREQUENCY` variable.

        """
        eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)
        [n_chans, n_samples] = eeg_signal.shape
        desired_n_samples = desired_duration * desired_fs

        if n_samples != desired_n_samples:
            logger.warning(
                f"Invalid number of samples. Actual number is {n_samples} should be {desired_n_samples}."
            )

    @staticmethod
    def validate_input_epoch_to_nn(
        epoch: np.ndarray,
        desired_n_chans: int = NUM_OF_CHANS_10_20,
        desired_duration: int = TIME,
        patient: Optional[Patient] = None,
    ):
        logger.debug("Starting input epoch validation.")

        # Validate epoch dimensions
        if epoch.ndim != NN_INPUT_DIMENSIONS:
            logger.error(
                f"Invalid epoch dimensions: expected {NN_INPUT_DIMENSIONS} dimensions, but got {epoch.ndim}."
            )
            raise ValueError(
                f"Input epoch must have {NN_INPUT_DIMENSIONS} dimensions but has {epoch.ndim}."
            )

        logger.info(f"Epoch dimensions validated: {epoch.shape}")

        # Extract epoch shape
        n_epoch, n_chans, n_samples = epoch.shape
        expected_shape = (1, desired_n_chans, desired_duration * SAMPLING_FREQUENCY)

        # Validate shape of the input epoch
        if (n_epoch, n_chans, n_samples) != expected_shape:
            logger.error(
                f"Invalid epoch shape: expected {expected_shape}, but got {epoch.shape}."
            )
            raise ValueError(
                f"Input epoch should have shape {expected_shape} but has {epoch.shape}."
            )

        logger.debug(f"Epoch shape validated: {epoch.shape}")

        # If a patient object is provided, validate patient properties
        if patient:
            expected_channel_names = [channel.name for channel in EEGChannels1020System]

            # Validate patient channel names
            if not patient.channel_names_CA:
                patient.channel_names_CA = expected_channel_names
                logger.debug(
                    "Patient's channel names for Common Average montage set to default 10-20 system channel names."
                )
            elif patient.channel_names_CA != expected_channel_names:
                logger.error(
                    "Patient's channel names do not match the expected 10-20 system channel names."
                )
                raise ValueError(
                    "Patient's channel names do not match the expected 10-20 system channels."
                )

        # Validate patient sampling frequency
        if patient.fs is None:
            patient.fs = SAMPLING_FREQUENCY
            logger.info(f"Patient's sampling frequency set to {SAMPLING_FREQUENCY}.")
        elif patient.fs != SAMPLING_FREQUENCY:
            logger.error(
                f"Invalid patient sampling frequency: expected {SAMPLING_FREQUENCY}, but got {patient.fs}."
            )
            raise ValueError(
                f"Patient's sampling frequency ({patient.fs}) does not match the expected {SAMPLING_FREQUENCY}."
            )

        # Update patient attributes if validation passes
        patient.epoch_duration = desired_duration
        patient.num_of_chans = desired_n_chans

        logger.info(
            f"Patient attributes updated: epoch_duration={desired_duration} and num_of_chans={desired_n_chans}."
        )

        logger.info("Input epoch to neural network validation completed successfully.")

    @staticmethod
    def validate_positive_value(value: float, name: str) -> None:
        """
        Validates that a given value is positive.

        Args:
            value (float): The value to validate.
            name (str): The name of the variable being validated.

        Raises:
            ValueError: If the value is not positive.
        """
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
