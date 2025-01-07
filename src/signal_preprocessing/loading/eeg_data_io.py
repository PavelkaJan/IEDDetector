import h5py
import numpy as np
import logging
from pathlib import Path
from scipy.io import savemat
from src.signal_preprocessing.validating.eeg_validator import EEGValidator

logger = logging.getLogger(__name__)


class EEGDataIO:
    """
    A class responsible for handling file operations, including saving EEG data
    to HDF5 format, loading EEG data from HDF5 files.
    """

    @staticmethod
    def save_to_hdf5(data: dict, output_folder: str) -> None:
        """
        Saves EEG data and associated metadata to an HDF5 file.

        Args:
            data (dict): A dictionary containing the EEG data and metadata to be saved.
            output_folder (str): The folder where the HDF5 file will be saved.
        """
        file_name = data.get("file_name", "default_name")
        logger.info(f"Saving EEG data to HDF5 file at {output_folder}/{file_name}.h5")

        output_folder_path = Path(output_folder)
        output_folder_path.mkdir(parents=True, exist_ok=True)
        full_file_path = output_folder_path / f"{file_name}.h5"

        with h5py.File(full_file_path, "w") as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    # Convert Unicode strings in numpy arrays to ASCII strings
                    if value.dtype.kind == "U":
                        value = value.astype("S")
                    f.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    # Convert lists or tuples to numpy arrays
                    value = np.array(value)
                    if value.dtype.kind == "U":
                        value = value.astype("S")
                    f.create_dataset(key, data=value)
                elif isinstance(value, str):
                    # Save strings as variable-length UTF-8 strings
                    f.create_dataset(key, data=np.string_(value))
                elif isinstance(value, float):
                    f.create_dataset(key, data=value, dtype=float)
                else:
                    f.create_dataset(key, data=value)

        logger.info(f"EEG data successfully saved to {full_file_path}")

    @staticmethod
    def load_eeg_from_hdf5(hdf5_file_path: str) -> dict:
        """
        Loads EEG data from an HDF5 file as a dictionary.

        Args:
            hdf5_file_path (str): The path to the HDF5 file to be loaded.

        Returns:
            dict: A dictionary containing the EEG data and associated metadata.
        """
        logger.info(f"Loading EEG data from HDF5 file at {hdf5_file_path}")

        data = {}
        with h5py.File(hdf5_file_path, "r") as f:
            for key in f.keys():
                dataset = f[key]

                # Handle scalar datasets differently
                if dataset.shape == ():
                    value = dataset[()]
                else:
                    value = dataset[:]

                # Convert byte strings to regular strings
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                elif isinstance(value, np.ndarray) and value.dtype.kind == "S":
                    value = value.astype(str)

                # Store the dataset in the dictionary
                data[key] = value

        logger.info("EEG data successfully loaded from HDF5 file")

        return data

    @staticmethod
    def save_epochs(epochs: np.ndarray, output_folder_path: str) -> None:
        """
        Saves each epoch in the provided array to individual .npy files in the specified folder.
        The naming continues based on the existing files in the directory to avoid overwriting.

        Args:
            epochs (np.ndarray): A 3D array of EEG epochs to save. Should have shape (n_epochs, n_chans, n_samples).
            output_folder_path (str): The directory path where the epoch files will be saved.
        """
        output_folder_path = Path(output_folder_path)

        # Ensure output directory exists
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Determine the starting index based on existing epoch files
        existing_epochs = sorted(output_folder_path.glob("epoch_*.npy"))
        if existing_epochs:
            # Extract the highest epoch number from existing files
            last_epoch_num = max(
                int(file.stem.split("_")[1]) for file in existing_epochs
            )
            start_index = last_epoch_num + 1
        else:
            start_index = 1

        for i, epoch in enumerate(epochs, start=start_index):
            file_name = output_folder_path / f"epoch_{i}.npy"
            np.save(file_name, epoch)
            logger.info(
                f"Epoch number {i} was saved as {file_name.name} in {output_folder_path}."
            )

    @staticmethod
    def load_eeg_epoch(eeg_file_path: str) -> np.ndarray:
        """
        Loads EEG signal data from a .npy file.

        Args:
            eeg_file_path (str): The file path to a .npy file containing the EEG data.

        Returns:
            np.ndarray: The EEG data loaded as a NumPy array.

        Raises:
            ValueError: If the loaded data is not a valid NumPy array.
        """
        logger.debug(f"Loading EEG data from file: {eeg_file_path}")
        eeg_data = np.load(eeg_file_path)
        eeg_data = EEGValidator.validate_eeg_signal_dims(eeg_data)

        if not isinstance(eeg_data, np.ndarray):
            logger.error(f"Loaded data is not a valid NumPy array: {eeg_file_path}")
            raise ValueError(f"Loaded data is not a valid NumPy array: {eeg_file_path}")

        logger.info("EEG data successfully loaded.")
        return eeg_data

    @staticmethod
    def save_epoch(
        epoch: np.ndarray, output_folder_path: str, file_extenstion="npy"
    ) -> None:
        """
        Saves the provided epoch to a file in the specified folder. The file can be saved
        as either a `.npy` (NumPy binary format) or a `.mat` (MATLAB format) file.

        The naming of the files is dynamic, ensuring that new files do not overwrite
        existing ones in the directory. The files are named sequentially in the format
        `epoch_<index>.<extension>`.

        Args:
            epoch (np.ndarray): An EEG epoch to save. The data should be provided as a NumPy array.
            output_folder_path (str): The directory path where the epoch file will be saved.
            file_extenstion (str): The file format to save the epoch in. Options are:
                - "npy": Saves the epoch in NumPy's binary `.npy` format.
                - "mat": Saves the epoch in MATLAB's `.mat` format with the variable name `F`.
                Default is "npy".
        """
        output_folder_path = Path(output_folder_path)
        output_folder_path.mkdir(parents=True, exist_ok=True)

        existing_epochs = sorted(output_folder_path.glob(f"epoch_*.{file_extenstion}"))
        if existing_epochs:
            # Extract the highest epoch number from existing files
            last_epoch_num = max(
                int(file.stem.split("_")[1]) for file in existing_epochs
            )
            start_index = last_epoch_num + 1
        else:
            start_index = 1

        # Save the provided epoch
        if file_extenstion == "npy":
            file_name = output_folder_path / f"epoch_{start_index}.npy"
            np.save(file_name, epoch)
        elif file_extenstion == "mat":
            file_name = output_folder_path / f"epoch_{start_index}.mat"
            savemat(file_name, {"F": epoch})

        logger.info(
            f"Epoch number {start_index} was saved as '{file_name.name}' in '{output_folder_path}'. The shape is {epoch.shape}."
        )
