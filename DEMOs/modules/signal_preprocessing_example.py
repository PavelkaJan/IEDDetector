import numpy as np
from src.logging_config.logging_setup import setup_logging
from src.signal_preprocessing.processing.eeg_processor import EEGProcessor
from src.plotting.eeg_plotter import EEGPlotter
import matplotlib.pyplot as plt
from src.signal_preprocessing.loading.matlab_file_loader import MatlabFileLoader
from src.eeg_montages.common_average_montage import CommonAverageMontage
from src.eeg_montages.double_banana_montage import DoubleBananaMontage
from src.eeg_montages.source_derivation_montage import SourceDerivationMontage

setup_logging()

# Path to simulated epileptic epoch from Brainstorm
path = "DEMOs/DEMO_data/data_simmeeg_241130_1806_sens-final_trial002.mat"

# ==========================================================
#                  SIGNAL PREPROCESSING
# ==========================================================

"""
Original signal
"""
eeg_signal = MatlabFileLoader.load_single_mat_epoch(path)
# EEGPlotter.plot_eeg(eeg_signal)
"""
Resampled signal to 128 Hz
"""
resampled_signal = EEGProcessor.resample_eeg_signal(256, eeg_signal, 128)
# EEGPlotter.plot_eeg(resampled_signal)

"""
Filtred signal - cutoff frequency = 20 Hz
"""
cutoffed_signal = EEGProcessor.apply_low_pass_filter(resampled_signal, 128, cutoff=20)
# EEGPlotter.plot_eeg(cutoffed_signal)

"""
Z-transformed signal for every channel
"""
z_transformed_signal = EEGProcessor.apply_z_score(cutoffed_signal)
# EEGPlotter.plot_eeg(z_transformed_signal)
# EEGPlotter.plot_single_channel(z_transformed_signal, 11)

"""
Centered signal
"""
centered_signal = EEGProcessor.center_eeg_signal(z_transformed_signal)
# EEGPlotter.plot_eeg(centered_signal)

"""
Signal without drift
"""
filtred_signal = EEGProcessor.remove_low_frequency_drift(centered_signal, 256)
# EEGPlotter.plot_eeg(filtred_signal)

"""
Apply Hann window
"""
windowed_signal = EEGProcessor.apply_hann_window(filtred_signal)
# EEGPlotter.plot_eeg(windowed_signal)

"""
Plot all signals in one graph
"""
time_vector = np.arange(resampled_signal.shape[1]) / 128

# plt.figure(figsize=(10, 4))

one_channel = 10

# plt.plot(time_vector, resampled_signal[one_channel, :], label="Original Signal")
# plt.plot(time_vector, cutoffed_signal[one_channel, :], label="Cutoffed Signal")
# plt.plot(time_vector, z_transformed_signal[one_channel, :], label="Z-transformed Signal")
# plt.plot(time_vector, centered_signal[one_channel, :], label="Centered Signal")
# plt.plot(time_vector, filtred_signal[one_channel, :], label="Filtered Signal")
# plt.plot(time_vector, windowed_signal[one_channel, :], label="Windowed (Final) Signal")


# # Adding title and labels
# plt.title("EEG Processing")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.grid(True)

# # Adding a legend to distinguish the plots
# plt.legend()

# # Show the plot
# plt.show()


# ==========================================================
#                  COMPUTE OTHER MONTAGES
# ==========================================================

ca_montage = CommonAverageMontage(windowed_signal)
signal_in_ca_montage = ca_montage.compute_montage()

db_montage = DoubleBananaMontage(windowed_signal)
signal_in_db_montage = db_montage.compute_montage()

sd_montage = SourceDerivationMontage(windowed_signal)
signal_in_sd_montage = sd_montage.compute_montage()

# EEGPlotter.plot_eeg(signal_in_ca_montage, eeg_montage="CA")
# EEGPlotter.plot_eeg(signal_in_db_montage, eeg_montage="DB")
EEGPlotter.plot_eeg(signal_in_sd_montage, eeg_montage="SD")
