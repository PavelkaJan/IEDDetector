import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Union
from src.patient.patient import Patient


class EEGDataset(Dataset):
    def __init__(self, patients: List[Patient]):
        """
        Initializes the dataset with a list of Patient instances.
        Collects .npy files from the CA, SD, and DB subfolders for each patient, along with corresponding labels.

        Args:
            patients (List[Patient]): A list of Patient instances from which EEG data will be loaded.
        """
        self.file_names: List[Path] = []
        self.labels: List[Union[int, float]] = []
        self.patient_ids: List[str] = []

        # Loop through each patient and load data and labels by montage type
        for patient in patients:
            self._load_patient_data(patient)

    def _load_patient_data(self, patient: Patient) -> None:
        """
        Loads data and labels for a single patient across all montage types.

        Args:
            patient (Patient): The patient instance from which EEG data will be loaded.
        """
        montage_folders = ["CA", "SD", "DB"]
        label_paths = {
            montage: Path(path)
            for montage, path in zip(montage_folders, patient.labels_path)
        }

        # Determine if the patient's path structure is a dictionary (epileptic_real) or a single path
        if isinstance(patient.epochs_PY_folder_path, dict):
            for ied_state, base_folder in patient.epochs_PY_folder_path.items():
                self._load_montage_data(
                    base_folder, montage_folders, label_paths, patient
                )
        else:
            base_folder = patient.epochs_PY_folder_path
            self._load_montage_data(base_folder, montage_folders, label_paths, patient)

    def _load_montage_data(
        self,
        base_folder: Path,
        montage_folders: List[str],
        label_paths: dict,
        patient: Patient,
    ) -> None:
        """
        Loads EEG data and labels for specified montages within a given folder path.

        Args:
            base_folder (Path): The base path for EEG epochs of a specific montage type.
            montage_folders (List[str]): List of montage types (e.g., ["CA", "SD", "DB"]).
            label_paths (dict): Dictionary of label file paths for each montage.
            patient (Patient): The patient instance being processed.
        """
        for montage in montage_folders:
            montage_folder_path = base_folder / montage
            label_path = label_paths.get(montage)

            # Load labels for each montage if the file exists
            current_labels = self._load_labels(label_path, montage, patient)

            # Collect EEG epoch files from the montage folder
            if montage_folder_path.exists():
                files = [
                    f
                    for f in sorted(montage_folder_path.iterdir())
                    if f.suffix == ".npy"
                ]
                self.file_names.extend(files)
                self.labels.extend(current_labels[: len(files)])
                self.patient_ids.extend([patient.id] * len(files))
            else:
                raise ValueError(
                    f"Montage folder {montage_folder_path} does not exist for patient {patient.id}."
                )

    def _load_labels(
        self, label_path: Union[Path, str], montage: str, patient: Patient
    ) -> np.ndarray:
        """
        Loads labels for a specified montage, ensuring the label file exists.

        Args:
            label_path (Union[Path, str]): Path to the label file.
            montage (str): The montage type (e.g., "CA", "SD", "DB").
            patient (Patient): The patient instance for which labels are being loaded.

        Returns:
            np.ndarray: The loaded labels.

        Raises:
            ValueError: If the label file is not found.
        """
        # Ensure label_path is a Path object
        label_path = Path(label_path) if isinstance(label_path, str) else label_path

        if label_path.exists():
            return np.load(label_path)
        else:
            raise ValueError(
                f"Label file for {montage} not found for patient {patient.id}."
            )

    def __len__(self) -> int:
        """
        Returns the total number of samples (EEG epochs) in the dataset.

        Returns:
            int: The number of EEG epochs.
        """
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor]:
        """
        Fetches the data and label for a specific index.

        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: A tuple containing:
                - data (torch.Tensor): The EEG epoch data.
                - label (torch.Tensor): The corresponding label (IED or non-IED).
        """
        data_path = self.file_names[idx]
        data = np.load(data_path)
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            patient_id,
        )
