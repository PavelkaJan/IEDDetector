import logging
import shutil
from pathlib import Path
from typing import Dict, List, Union
from src.constants import PatientType

logger = logging.getLogger(__name__)


def validate_patient_type(
    patient_type: Union[str, PatientType], PatientType: type
) -> PatientType:
    """
    Validate the patient type and ensure it is either a valid string or a PatientType Enum.

    Args:
        patient_type (str or PatientType): A string or instance of PatientType enum.
        PatientType (type): Enum class to validate against.

    Returns:
        PatientType: Validated patient type as an instance of PatientType enum.

    Raises:
        ValueError: If `patient_type` is not valid.
    """
    if isinstance(patient_type, str):
        try:
            return PatientType[patient_type.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid patient type. Must be one of {[t.value for t in PatientType]}. You inserted {patient_type}."
            )
    elif isinstance(patient_type, PatientType):
        return patient_type
    else:
        raise ValueError(
            f"Invalid patient type. Must be a string or PatientType Enum. You inserted {patient_type}."
        )


def generate_labels(
    patient_type: PatientType, epochs_from_PY_folder_path: Union[Path, dict]
) -> Dict[str, List[int]]:
    """
    Generates labels for each montage (CA, DB, SD) and for each IED state (IED_present, IED_absent)
    for 'epileptic_real' patients.

    Args:
        patient_type (PatientType): The type of patient to determine label values.
        epochs_from_PY_folder_path (Union[Path, dict]): The path or dictionary where processed epochs are stored.

    Returns:
        Dict[str, List[int]]: A dictionary where the key is the montage (and IED state for 'epileptic_real')
        and the value is the list of labels for that montage.
    """
    labels_dict: Dict[str, List[int]] = {}
    logger.info(f"Generating labels for patient type: {patient_type}")

    if patient_type == PatientType.EPILEPTIC_REAL:
        # For 'epileptic_real' patients: IED_present and IED_absent, each with CA, DB, SD montages
        for ied_state in ["IED_present", "IED_absent"]:
            for montage in ["CA", "DB", "SD"]:
                folder_path = epochs_from_PY_folder_path[ied_state] / montage
                labels_key = f"{ied_state}_{montage}"
                labels_dict[labels_key] = _generate_labels_for_folder(
                    folder_path, 1 if ied_state == "IED_present" else 0
                )
                logger.debug(
                    f"Generated labels for {labels_key} with {len(labels_dict[labels_key])} epochs."
                )
    else:
        # For other patients: CA, DB, SD montages with one label for the whole set (0 for healthy, 1 for epileptic_simulated)
        label_value = (
            0
            if patient_type in [PatientType.HEALTHY_REAL, PatientType.HEALTHY_SIMULATED]
            else 1
        )
        for montage in ["CA", "DB", "SD"]:
            folder_path = epochs_from_PY_folder_path / montage
            labels_key = montage
            labels_dict[labels_key] = _generate_labels_for_folder(
                folder_path, label_value
            )
            logger.info(
                f"Generated labels for {labels_key} with {len(labels_dict[labels_key])} epochs."
            )

    return labels_dict


def _generate_labels_for_folder(folder_path: Path, label_value: int) -> List[int]:
    """
    Generates labels for all files in a given folder for a specific montage or IED state.

    Args:
        folder_path (Path): Path to the folder containing the files.
        label_value (int): The label value to assign to all files.

    Returns:
        List[int]: A list of label values corresponding to the number of files in the folder.
    """
    if not folder_path.exists():
        logger.warning(
            f"Folder {folder_path} does not exist. Skipping label generation."
        )
        return []

    num_files = len(list(folder_path.glob("*.npy")))
    logger.debug(f"Found {num_files} files in {folder_path} for label {label_value}.")
    return [label_value] * num_files


def count_epochs_in_folders(
    folders: Union[Dict[str, Path], Path], file_extension: str = "*.npy"
) -> Union[Dict[str, int], int]:
    """
    Helper function that counts the number of files in specified folders.
    - For 'epileptic_real' patients, it expects a dictionary with IED states as keys and paths as values.
    - For other patient types, it expects a single Path object.

    Returns:
    - A dictionary of counts if the input is a dictionary (for 'epileptic_real').
    - A single integer count if the input is a single Path (for other patients).
    """
    if isinstance(folders, dict):
        counts = {}
        for state, folder_path in folders.items():
            if not folder_path.exists():
                raise FileNotFoundError(f"Input folder does not exist: {folder_path}")
            counts[state] = len(list(folder_path.glob(file_extension)))
        return counts
    else:
        if not folders.exists():
            raise FileNotFoundError(f"Input folder does not exist: {folders}")
        return len(list(folders.glob(file_extension)))


def delete_files_in_folder(folder_path: Path) -> None:
    """
    Helper function to delete all files in the specified folder, keeping the folder structure intact.
    Raises an error if there is an issue deleting a file.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"The path {folder_path} does not exist.")

    for item in folder_path.glob("**/*"):
        if item.is_file():
            item.unlink()
            logger.debug(f"Deleted file: {item}")
