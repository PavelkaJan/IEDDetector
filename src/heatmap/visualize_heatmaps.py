import numpy as np
from pathlib import Path
from src.mne_heatmap_plot.mne_utils import create_mne_info, create_mne_raw_array
from src.mne_heatmap_plot.mne_heatmap_plotter import plot_raw_with_rainbow_heatmap
from src.patient.patient import Patient
import os
import mne


def visualize_heatmap(epoch_idx, heatmap_file, epochs_file, patient_file):
    """
    Visualize occlusion heatmap for a specific epoch.

    Args:
        epoch_idx (int): Index of the epoch to visualize.
        heatmap_file (str): Path to the saved heatmap file.
        epochs_file (str): Path to the saved EEG epochs file.
        patient_file (str): Path to the patient instance file.
    """
    # Check if files exist
    if not os.path.exists(heatmap_file):
        raise FileNotFoundError(f"The heatmap file {heatmap_file} does not exist.")
    if not os.path.exists(epochs_file):
        raise FileNotFoundError(f"The epochs file {epochs_file} does not exist.")
    if not os.path.exists(patient_file):
        raise FileNotFoundError(f"The patient file {patient_file} does not exist.")

    # Load heatmaps and raw EEG data
    heatmaps = np.load(heatmap_file)  # Shape: (n_epochs, n_channels, n_samples)
    raw_data = np.load(epochs_file)  # Shape: (n_epochs, n_channels, n_samples)

    # Extract the specific epoch
    heatmap = heatmaps[epoch_idx]  # Shape: (n_channels, n_samples)
    eeg_data = raw_data[epoch_idx]  # Shape: (n_channels, n_samples)

    # Ensure the data has the correct shape (n_channels, n_samples)
    if len(eeg_data.shape) > 2:
        eeg_data = eeg_data.squeeze()  # Remove extra dimensions if present
    if len(heatmap.shape) > 2:
        heatmap = heatmap.squeeze()  # Remove extra dimensions if present

    # Load patient metadata
    patient = Patient.load_patient_instance(Path(patient_file))

    # Create MNE objects
    info = create_mne_info(patient)
    raw = create_mne_raw_array(eeg_data, info)

    # Plot the raw EEG data with heatmap
    plot_raw_with_rainbow_heatmap(raw, heatmap)

# Example usage
if __name__ == "__main__":
    # File paths
    heatmap_file = "occlusion_heatmaps.npy"
    epochs_file = "epochs_tensor.npy"
    patient_file = "D:\DIPLOMKA_DATASET\P322\P322_instance.pkl"
    heatmap = np.load(heatmap_file)
    try:
        epoch_idx = 1
        visualize_heatmap(epoch_idx, heatmap_file, epochs_file, patient_file)
    except FileNotFoundError as e:
        print(e)
    except ValueError:
        print("Invalid input. Please enter a valid epoch index.")
