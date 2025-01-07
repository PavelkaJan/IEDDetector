"""
DEMO file that show whole processing of 4s epoch from SIMMEEG.
"""

from src.signal_preprocessing.loading.matlab_file_loader import MatlabFileLoader
from src.signal_preprocessing.processing.eeg_processor import EEGProcessor
from src.plotting.eeg_plotter import EEGPlotter
from src.signal_preprocessing.loading.eeg_data_io import EEGDataIO
from src.logging_config.logging_setup import setup_logging

setup_logging()

# Load 4s epoch from SIMMEEG
simmeeg_4s_epoch_path = "DEMOs\modules\epoch_simmeg_4s_epileptic_simulated.mat"
original_signal_4s = MatlabFileLoader.load_single_mat_epoch(simmeeg_4s_epoch_path)
fs = 256

current_channel_order = [
    "Fp1",
    "Fp2",
    "F4",
    "F3",
    "C3",
    "C4",
    "P4",
    "P3",
    "O2",
    "O1",
    "F8",
    "F7",
    "T4",
    "T3",
    "T6",
    "T5",
    "Pz",
    "Fz",
    "Cz",
]

# Create epochs with shift
epoch_spike_mid = EEGProcessor.create_2_epoch_with_shift(original_signal_4s, fs, -1, 1)
epoch_spike_left = EEGProcessor.create_2_epoch_with_shift(
    original_signal_4s, fs, -1.5, 0.5
)
epoch_spike_right = EEGProcessor.create_2_epoch_with_shift(
    original_signal_4s, fs, -0.5, 1.5
)

# Process epoch
epochs = [epoch_spike_mid, epoch_spike_left, epoch_spike_right]

# To see debug messages set up logging.ini file
for epoch in epochs:
    resampled_signal = EEGProcessor.resample_eeg_signal(fs, epoch)
    shorted_signal = EEGProcessor.standardize_epoch_length(resampled_signal)
    reordered_signal = EEGProcessor.reorder_and_select_eeg_channels(
        shorted_signal, current_channel_order
    )
    cutoffed_signal = EEGProcessor.apply_low_pass_filter(reordered_signal, cutoff=50)
    z_scored_signal = EEGProcessor.apply_z_score(cutoffed_signal)
    centered_signal = EEGProcessor.center_eeg_signal(z_scored_signal)
    filtered_signal = EEGProcessor.remove_low_frequency_drift(centered_signal)
    output_signal = EEGProcessor.apply_hann_window(filtered_signal)
    EEGDataIO.save_epoch(output_signal, "output_dir", "mat")


# Check that is it possible to load the saved matrix
data = MatlabFileLoader.load_single_mat_epoch("output_dir\epoch_3.mat")

EEGPlotter.plot_eeg(data)
