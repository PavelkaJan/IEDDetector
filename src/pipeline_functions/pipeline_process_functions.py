from src.patient.patient import Patient
from pathlib import Path
import logging
from typing import List, Dict, Union, Optional
import numpy as np
from src.signal_preprocessing.loading.matlab_file_loader import MatlabFileLoader
from src.signal_preprocessing.loading.eeg_data_io import EEGDataIO
from src.signal_preprocessing.processing.eeg_processor import EEGProcessor
from src.plotting.eeg_plotter import EEGPlotter
from src.signal_preprocessing.validating.eeg_validator import EEGValidator
from src.eeg_montages.common_average_montage import CommonAverageMontage
from src.eeg_montages.double_banana_montage import DoubleBananaMontage
from src.eeg_montages.source_derivation_montage import SourceDerivationMontage
from src.constants import PatientType
from src.constants import (
    EEGChannels1020System,
    DoubleBananaMontageChannels,
    SourceDerivationMontageChannels,
    EPOCH_SHIFTS,
)

logger = logging.getLogger(__name__)


# Dictionary to map montage names to their corresponding classes
montage_classes = {
    "CA": CommonAverageMontage,
    "DB": DoubleBananaMontage,
    "SD": SourceDerivationMontage,
}


def process_epoch(
    patient: Patient, epoch_file_path: Path, ied_state: Optional[str] = None
) -> None:
    """
    Processes and saves an EEG epoch for a patient by applying different montages.
    The IED state is mandatory for epileptic_real patients.

    Args:
        patient (Patient): The patient object.
        epoch_file_path (Path): Path to the EEG epoch file.
        ied_state (Optional[str]): 'IED_present' or 'IED_absent' for epileptic_real patients.
    """
    logger.info(f"Processing {epoch_file_path} for patient {patient.id}")

    raw_epoch, original_epoch_name = MatlabFileLoader.load_single_mat_epoch(str(epoch_file_path))

    processed_signal = preprocess_signal(patient, raw_epoch)

    ca_montage_signal = compute_common_average_montage(processed_signal, patient)
    ca_montage_signal = EEGProcessor.add_dimension_to_eeg_signal(ca_montage_signal)
    EEGValidator.validate_input_epoch_to_nn(epoch=ca_montage_signal, patient=patient)
    save_epoch(ca_montage_signal, patient, "CA", ied_state, original_epoch_name)

    db_montage_signal = compute_double_banana_montage(processed_signal, patient)
    db_montage_signal = EEGProcessor.add_dimension_to_eeg_signal(db_montage_signal)
    EEGValidator.validate_input_epoch_to_nn(epoch=db_montage_signal, patient=patient)
    save_epoch(db_montage_signal, patient, "DB", ied_state, original_epoch_name)

    sd_montage_signal = compute_source_derivation_montage(processed_signal, patient)
    sd_montage_signal = EEGProcessor.add_dimension_to_eeg_signal(sd_montage_signal)
    EEGValidator.validate_input_epoch_to_nn(epoch=sd_montage_signal, patient=patient)
    save_epoch(sd_montage_signal, patient, "SD", ied_state, original_epoch_name)

    logger.info(
        f"Processed and saved all montages for {epoch_file_path} for patient {patient.id}"
    )


def save_epoch(
    signal: np.ndarray,
    patient: Patient,
    montage_name: str,
    ied_state: Optional[str] = None,
    original_file_name: Optional[str] = None
) -> None:
    """
    Saves the processed EEG epoch with a specific montage to the appropriate folder.

    Args:
        signal (np.ndarray): The processed EEG signal.
        patient (Patient): The patient object.
        montage_name (str): The montage name ('CA', 'DB', 'SD').
        ied_state (Optional[str]): 'IED_present' or 'IED_absent' for epileptic_real patients.
        original_file_name (Optional[str]): The name to use for saving the file. If not
            provided, the function will use a dynamic naming strategy based on the
            EEGDataIO's 'save_epoch' logic. Default is `None`.
    """
    save_folder = determine_save_folder(patient, montage_name, ied_state)
    save_folder.mkdir(parents=True, exist_ok=True)

    EEGDataIO.save_epoch(signal, save_folder, file_extenstion="npy", use_original_file_name=True, original_file_name=original_file_name)
    logger.info(
        f"Saved {montage_name} montage epoch for patient {patient.id} in folder: {save_folder}"
    )


def preprocess_signal(patient: Patient, raw_epoch: Union[List, Dict]) -> List:
    """
    Applies a sequence of preprocessing steps to the EEG signal.

    Parameters:
    - patient (Patient): The patient object.
    - raw_epoch (List or Dict): The raw EEG signal.

    Returns:
    - List: The processed EEG signal.
    """
    resampled_signal = EEGProcessor.resample_eeg_signal(patient, raw_epoch)
    reordered_signal = EEGProcessor.reorder_and_select_eeg_channels(
        resampled_signal, patient.original_channel_names_CA
    )
    shorted_signal = EEGProcessor.standardize_epoch_length(reordered_signal)
    filtered_signal = EEGProcessor.remove_low_frequency_drift(shorted_signal)
    cutoffed_signal = EEGProcessor.apply_low_pass_filter(filtered_signal, cutoff=50)
    z_scored_signal = EEGProcessor.apply_z_score(cutoffed_signal)
    centered_signal = EEGProcessor.center_eeg_signal(z_scored_signal)
    output_signal = EEGProcessor.apply_hann_window(centered_signal)

    return output_signal


def compute_common_average_montage(signal: np.ndarray, patient: Patient) -> np.ndarray:
    """
    Computes the Common Average Montage (CA) on a preprocessed EEG signal and updates the patient's channel names.

    Args:
        signal (np.ndarray): The preprocessed EEG signal.
        patient (Patient): The patient object to update channel names.

    Returns:
        np.ndarray: The EEG signal after applying the CA montage.
    """
    ca_montage = CommonAverageMontage(signal)
    patient.channel_names_CA = [channel.name for channel in EEGChannels1020System]
    return ca_montage.compute_montage()


def compute_double_banana_montage(signal: np.ndarray, patient: Patient) -> np.ndarray:
    """
    Computes the Double Banana Montage (DB) on a preprocessed EEG signal and updates the patient's channel names.

    Args:
        signal (np.ndarray): The preprocessed EEG signal.
        patient (Patient): The patient object to update channel names.

    Returns:
        np.ndarray: The EEG signal after applying the DB montage.
    """
    db_montage = DoubleBananaMontage(signal)
    patient.channel_names_DB = [
        channel.label for channel in DoubleBananaMontageChannels
    ]
    return db_montage.compute_montage()


def compute_source_derivation_montage(
    signal: np.ndarray, patient: Patient
) -> np.ndarray:
    """
    Computes the Source Derivation Montage (SD) on a preprocessed EEG signal and updates the patient's channel names.

    Args:
        signal (np.ndarray): The preprocessed EEG signal.
        patient (Patient): The patient object to update channel names.

    Returns:
        np.ndarray: The EEG signal after applying the SD montage.
    """
    sd_montage = SourceDerivationMontage(signal)
    patient.channel_names_SD = [
        channel.name for channel in SourceDerivationMontageChannels
    ]
    return sd_montage.compute_montage()


def determine_save_folder(
    patient: Patient, montage_name: str, ied_state: Optional[str] = None
) -> Path:
    """
    Determines the folder where the processed epoch will be saved.

    Args:
        patient (Patient): The patient object.
        montage_name (str): The name of the montage.
        ied_state (Optional[str]): 'IED_present' or 'IED_absent' for epileptic_real patients.

    Returns:
        Path: The folder path where the epoch should be saved.
    """
    if patient.patient_type == PatientType.EPILEPTIC_REAL and ied_state:
        return Path(patient.epochs_PY_folder_path[ied_state]) / montage_name
    return Path(patient.epochs_PY_folder_path) / montage_name


def process_patients(patients: List[Patient]):
    """
    Processes EEG data for a list of patients. Deletes previously processed epochs, processes new epochs,
    saves labels, and updates the patient instance.

    Parameters:
    - patients (List[Patient]): A list of patients to process.
    """
    for patient in patients:
        logger.info(f"Processing patient {patient.id}")
        patient.delete_processed_epochs()
        patient.delete_labels_folder()

        if patient.patient_type == PatientType.EPILEPTIC_REAL:
            process_real_epileptic_patient(patient)
        else:
            process_other_patient_types(patient)

        patient.create_and_save_labels()
        patient.save_patient_instance()


def process_real_epileptic_patient(patient: Patient):
    """
    Processes EEG data for a real epileptic patient, handling both IED_present and IED_absent states.

    Parameters:
    - patient (Patient): The patient object.
    """
    for ied_state, folder_path in patient.epochs_BS_folder_path.items():
        epoch_files = list_files_in_folder(folder_path, ied_state, patient.id)
        for epoch_file in epoch_files:
            process_epoch(patient, epoch_file, ied_state)


def process_other_patient_types(patient: Patient):
    """
    Processes EEG data for non-real-epileptic patients.

    Parameters:
    - patient (Patient): The patient object.
    """
    epochs_folder_path = Path(patient.epochs_BS_folder_path)
    epoch_files = list_files_in_folder(epochs_folder_path, None, patient.id)
    for epoch_file in epoch_files:
        process_epoch(patient, epoch_file)


def list_files_in_folder(
    folder_path: Union[str, Path], ied_state: Optional[str], patient_id: str
) -> List[Path]:
    """
    Lists all .mat files in a given folder.

    Parameters:
    - folder_path (Union[str, Path]): Path to the folder containing EEG epoch files.
    - ied_state (Optional[str]): The IED state for logger ('IED_present' or 'IED_absent' if applicable).
    - patient_id (str): The patient ID for logger.

    Returns:
    - List[Path]: A list of paths to the EEG epoch files.
    """
    folder = Path(folder_path)
    if not folder.exists():
        if ied_state:
            logger.warning(
                f"Epochs folder '{ied_state}' does not exist for patient {patient_id}: {folder}"
            )
        else:
            logger.error(
                f"Epochs folder does not exist for patient {patient_id}: {folder}"
            )
        return []

    epoch_files = list(folder.glob("*.mat"))
    if not epoch_files:
        if ied_state:
            logger.warning(
                f"No epoch files found in '{ied_state}' for patient {patient_id}: {folder}"
            )
        else:
            logger.warning(f"No epoch files found for patient {patient_id}: {folder}")
    else:
        if ied_state:
            logger.info(
                f"Found {len(epoch_files)} epoch files in '{ied_state}' for patient {patient_id}"
            )
        else:
            logger.info(
                f"Found {len(epoch_files)} epoch files for patient {patient_id}"
            )

    return epoch_files


def load_and_process_4s_epochs(input_folder_path, output_folder_path, fs):
    epochs_4s = list_mat_files(input_folder_path)
    process_4s_epochs(epochs_4s, fs, output_folder_path)


def list_mat_files(folder_path: str):
    """
    Lists all .mat files in the specified folder.

    Args:
        folder_path (str): The path to the folder to search for .mat files.

    Returns:
        list: A list of .mat file paths in the folder.
    """
    folder = Path(folder_path)
    mat_files = sorted(folder.glob("*.mat"))
    return mat_files


def process_4s_epochs(
    epochs_4s: List[str], fs: float, output_dir: str = "processed_4s_epochs_with_shift"
):
    """
    Processes 4-second epochs into overlapping 2-second epochs using predefined time shifts.

    Args:
        epochs_4s (List[str]): List of file paths to 4-second epoch MATLAB files.
        fs (float): Sampling frequency of the EEG data.
        output_dir (str, optional): Directory where the processed 2-second epochs will be saved. Default is "output_dir2".

    Returns:
        None
    """
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    for idx, epoch_4s_path in enumerate(epochs_4s):
        raw_epoch_4s = MatlabFileLoader.load_single_mat_epoch(epoch_4s_path)

        for start_shift, end_shift in EPOCH_SHIFTS:
            epoch_2s = EEGProcessor.create_2_epoch_with_shift(
                raw_epoch_4s, fs, start_shift, end_shift
            )
            EEGDataIO.save_epoch(epoch_2s, output_folder, "mat")

        logger.info(
            f"Processed and saved all 2s epochs for file {idx + 1}/{len(epochs_4s)}: {epoch_4s_path}"
        )
