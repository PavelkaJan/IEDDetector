from pathlib import Path
import numpy as np
import logging
from typing import Optional
import pickle
from src.constants import PatientType
import src.patient.patient_functions as pf
from typing import List, Dict, Union, Optional
import scipy.io

logger = logging.getLogger(__name__)


class Patient:
    def __init__(
        self,
        id: str=None,
        patient_type: PatientType=None,
        base_folder_path: str=None,
        original_channel_names_CA: List[str]=None,
        original_fs: int =None,
    ):
        # Mandatory attributes
        self.id = id
        self.patient_type = pf.validate_patient_type(patient_type, PatientType)
        self.base_folder_path = Path(base_folder_path) / id
        self.original_channel_names_CA = original_channel_names_CA
        self.original_fs = original_fs

        # Attribute only for "epileptic_smilated"
        self.epochs_BS_4s_folder_path = self.base_folder_path / "epochs_BS_4s"

        # Original epochs from BS
        if self.patient_type == PatientType.EPILEPTIC_REAL:
            self.epochs_BS_folder_path = {
                "IED_present": self.base_folder_path / "epochs_BS" / "IED_present",
                "IED_absent": self.base_folder_path / "epochs_BS" / "IED_absent",
            }
        else:
            self.epochs_BS_folder_path = self.base_folder_path / "epochs_BS"

        # Processed epochs in PY
        if self.patient_type == PatientType.EPILEPTIC_REAL:
            self.epochs_PY_folder_path = {
                "IED_present": self.base_folder_path / "epochs_PY" / "IED_present",
                "IED_absent": self.base_folder_path / "epochs_PY" / "IED_absent",
            }
        else:
            self.epochs_PY_folder_path = self.base_folder_path / "epochs_PY"
        self.create_epochs_py_folders()

        # Attributes get after processing:
        self.fs = None

        # Channel names for processed epochs
        self.channel_names_CA: List[str] = []
        self.channel_names_DB: List[str] = []
        self.channel_names_SD: List[str] = []

        # Neural network input
        self.labels_path = []

    def __repr__(self):
        return f"{self.id}"

    def create_epochs_py_folders(self):
        """
        Creates the 'epochs_PY' folder structure based on the patient type.
        For 'epileptic_real', creates additional subfolders for 'IED_absent' and 'IED_present'.
        """
        if not self.base_folder_path:
            raise ValueError("Base folder path must be specified.")

        if self.patient_type == PatientType.EPILEPTIC_REAL:
            for ied_state in ["IED_absent", "IED_present"]:
                # Access the correct path from the dictionary using the ied_state key
                ied_folder_path = Path(self.epochs_PY_folder_path[ied_state])
                ied_folder_path.mkdir(parents=True, exist_ok=True)
                for montage in ["CA", "DB", "SD"]:
                    montage_folder = ied_folder_path / montage
                    montage_folder.mkdir(parents=True, exist_ok=True)
        else:
            # When patient type is not 'epileptic_real', handle a single epochs_PY folder
            for montage in ["CA", "DB", "SD"]:
                montage_folder = Path(self.epochs_PY_folder_path) / montage
                montage_folder.mkdir(parents=True, exist_ok=True)

    def delete_processed_epochs(self) -> None:
        """
        Deletes all files in the folder specified by epochs_PY_folder_path, but keeps the folder structure intact.
        Raises an error if there is an issue deleting a file.
        """
        if isinstance(self.epochs_PY_folder_path, dict):
            # For 'epileptic_real' patients, handle both IED_present and IED_absent
            for folder_type, folder_path in self.epochs_PY_folder_path.items():
                pf.delete_files_in_folder(Path(folder_path))
        else:
            # For other patient types, handle single epochs_PY folder
            pf.delete_files_in_folder(Path(self.epochs_PY_folder_path))

        logger.info(
            f"All files deleted from {self.epochs_PY_folder_path}, but folder structure kept intact."
        )

    def create_and_save_labels(self) -> None:
        """
        Creates and saves labels for the processed EEG epochs into the 'labels' folder for each montage (CA, DB, SD).
        For 'epileptic_real' patients, it creates labels separately for IED_present and IED_absent in respective subfolders.
        The labels_path attribute will store a list of all paths to the saved labels.
        """
        labels_folder = self.base_folder_path / "labels"
        labels_folder.mkdir(parents=True, exist_ok=True)

        self.labels_path = []

        labels_dict = pf.generate_labels(self.patient_type, self.epochs_PY_folder_path)

        # Save labels for each montage and collect the paths
        for label_key, labels in labels_dict.items():
            parts = label_key.rsplit("_", 1)

            if len(parts) == 2:
                ied_state, montage = parts
                # Handle epileptic_real patients with subfolders for IED_present and IED_absent
                save_folder = labels_folder / ied_state
                save_folder.mkdir(parents=True, exist_ok=True)
            else:
                # For non-epileptic patients, just use the main labels folder
                montage = parts[0]
                save_folder = labels_folder

            save_path = save_folder / f"labels_{montage}.npy"
            np.save(save_path, np.array(labels))
            logger.info(f"Saved labels to {save_path} for montage {montage}.")

            self.labels_path.append(str(save_path))

    def delete_labels_folder(self) -> None:
        """
        Recursively deletes the 'labels' folder and all its contents for the patient.
        """
        labels_folder = self.base_folder_path / "labels"

        if not labels_folder.exists():
            logger.warning(
                f"Labels folder does not exist for patient {self.id}: {labels_folder}"
            )
            return

        for item in labels_folder.glob("**/*"):
            if item.is_file():
                item.unlink()
                logger.debug(f"Deleted file: {item}")
            elif item.is_dir():
                # Recursively delete all files inside the subdirectory
                for sub_item in item.glob("**/*"):
                    if sub_item.is_file():
                        sub_item.unlink()
                        logger.debug(f"Deleted file: {sub_item}")
                # Once the subdirectory is empty, delete it
                item.rmdir()
                logger.debug(f"Deleted directory: {item}")

        # Finally, remove the labels folder itself
        labels_folder.rmdir()
        logger.info(f"Deleted labels folder for patient {self.id}: {labels_folder}")

    def save_patient_instance(self, output_path: Optional[str] = None):
        """
        Saves the patient instance to a file. The save location depends on the patient's folder structure.
        For 'epileptic_real' patients, it uses the 'IED_present' path to determine the output folder.
        """
        if self.id is None:
            logger.error("Patient ID is not set. Cannot save instance.")
            return

        if output_path is None:
            output_folder_path = self.base_folder_path
        else:
            output_folder_path = Path(output_path)

        output_folder_path.mkdir(parents=True, exist_ok=True)

        file_path = output_folder_path / f"{self.id}_instance.pkl"
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

        logger.info(f"Patient instance saved to {file_path}")

    @staticmethod
    def load_patient_instance(file_path: str) -> "Patient":
        """
        Loads a patient instance from a pickle file.

        Args:
            file_path (str): The path to the pickle file containing the patient instance.

        Returns:
            Patient: The loaded patient instance.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            EOFError, pickle.UnpicklingError: If there is an issue loading the pickle file.

        Notes:
            This method does not use a try-except block to handle exceptions.
            Any errors raised during loading will propagate for the caller to handle.
        """
        path = Path(file_path)
        if not path.is_file():
            logger.error(f"Patient instance file not found: {file_path}")
            raise FileNotFoundError(f"File does not exist: {file_path}")

        with path.open("rb") as file:
            instance = pickle.load(file)
            logger.info(f"Patient instance: {instance.id} was successfully loaded.")
        return instance

    @property
    def processed_epoch_count_by_montage(
        self,
    ) -> Union[Dict[str, Dict[str, int]], Dict[str, int]]:
        """
        Dynamically calculates and returns the number of montage epochs (CA, DB, SD) in the folder.
        - For 'epileptic_real' patients, it returns a dictionary with counts for 'IED_present' and 'IED_absent'.
        - For other patient types, it returns a dictionary with counts for the montages directly.
        """
        if isinstance(self.epochs_PY_folder_path, dict):
            counts = {
                "IED_present": pf.count_epochs_in_folders(
                    {
                        montage: self.epochs_PY_folder_path["IED_present"] / montage
                        for montage in ["CA", "DB", "SD"]
                    }
                ),
                "IED_absent": pf.count_epochs_in_folders(
                    {
                        montage: self.epochs_PY_folder_path["IED_absent"] / montage
                        for montage in ["CA", "DB", "SD"]
                    }
                ),
            }
        else:
            counts = pf.count_epochs_in_folders(
                {
                    montage: self.epochs_PY_folder_path / montage
                    for montage in ["CA", "DB", "SD"]
                }
            )
        return counts

    @property
    def raw_epoch_count(self) -> Union[int, Dict[str, int]]:
        """
        Property that returns the number of raw input epochs.
        - For 'epileptic_real' patients, it returns a dictionary with counts for 'IED_present' and 'IED_absent'.
        - For other patient types, it returns a single integer count.
        """
        return pf.count_epochs_in_folders(
            self.epochs_BS_folder_path, file_extension="*.mat"
        )

    @property
    def processed_epoch_count(self) -> int:
        """
        Property that returns the total number of processed epochs across all montages (CA, DB, SD).
        - For 'epileptic_real' patients, it sums the processed epochs in both 'IED_present' and 'IED_absent' folders.
        - For other patient types, it sums the processed epochs across the montages directly.
        """
        epoch_counts = self.processed_epoch_count_by_montage

        if isinstance(epoch_counts, dict) and "IED_present" in epoch_counts:
            total_epochs = sum(
                sum(montage_counts.values()) for montage_counts in epoch_counts.values()
            )
        else:
            total_epochs = sum(epoch_counts.values())

        return total_epochs

    @staticmethod
    def get_channel_names_from_mat(mat_file_path: Path) -> list:
        """
        Loads the first column of the 'Channel' variable from a MATLAB .mat file and returns it as a list.

        Parameters:
            mat_file_path (str): Path to the MATLAB .mat file.

        Returns:
            list: The first column of the 'Channel' variable.
        """
        logger.debug(f"Loading MATLAB file from: {mat_file_path}")

        mat_data = scipy.io.loadmat(mat_file_path)
        channel_data = mat_data.get("Channel")

        if channel_data is None:
            logger.error("The 'Channel' variable was not found in the MATLAB file.")
            raise ValueError("The 'Channel' variable was not found in the MATLAB file.")

        if channel_data.dtype.names and "Name" in channel_data.dtype.names:
            name_column = [str(name[0]) for name in channel_data["Name"][0]]
            logger.debug(f"Extracted channel names: {name_column}")
            return name_column

        logger.error("The 'Channel' variable does not contain a 'Name' field.")
        raise ValueError("The 'Channel' variable does not contain a 'Name' field.")
