"""
DEMO file for loading 4s epoch from SIMMEEG.
"""

from src.signal_preprocessing.loading.matlab_file_loader import MatlabFileLoader
from src.signal_preprocessing.processing.eeg_processor import EEGProcessor
from src.plotting.eeg_plotter import EEGPlotter


# Load 4s epoch from SIMMEEG
simmeeg_4s_epoch_path = "DEMOs\modules\epoch_simmeg_4s_epileptic_simulated.mat"
original_signal_4s = MatlabFileLoader.load_single_mat_epoch(simmeeg_4s_epoch_path)
fs = 256
print("Original epoch shape: ", original_signal_4s.shape)
print("Original epoch duration [s]: ", original_signal_4s.shape[1] / fs)

# Plot original epoch
# EEGPlotter.plot_eeg(original_signal_4s)

# Create epochs with shift
epoch_1 = EEGProcessor.create_2_epoch_with_shift(original_signal_4s, fs, -1, 1)
epoch_2 = EEGProcessor.create_2_epoch_with_shift(original_signal_4s, fs, -1.5, 0.5)
epoch_3 = EEGProcessor.create_2_epoch_with_shift(original_signal_4s, fs, -0.5, 1.5)
print("Shifted epoch shape: ", epoch_1.shape)
print("Shifted epoch duration [s]: ", epoch_1.shape[1] / fs)

# Plot shifted epoch
EEGPlotter.plot_eeg(epoch_1)
