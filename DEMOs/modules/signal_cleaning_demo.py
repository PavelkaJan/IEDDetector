import logging
import numpy as np
from src.logging_config.logging_setup import setup_logging
from src.signal_preprocessing.processing.eeg_processor import EEGProcessor
from src.plotting.eeg_plotter import EEGPlotter
import matplotlib.pyplot as plt


setup_logging()

path = "DEMOs/DEMO_data/raw_epoch_19_256_128Hz_CA.npy"


"""
Original signal
"""
eeg_signal = np.load(path)
# EEGPlotter.plot_single_channel(eeg_signal, 15)

"""
Centered signal
"""
centered_signal = EEGProcessor.center_eeg_signal(eeg_signal)
# EEGPlotter.plot_single_channel(centered_signal, 15)

"""
Signal without drift
"""
filtred_signal = EEGProcessor.remove_low_frequency_drift(centered_signal, 128)
# EEGPlotter.plot_single_channel(filtred_signal, 15)

"""
Apply Hann window
"""
windowed_signal = EEGProcessor.apply_hann_window(filtred_signal)


"""
Plot all signals in one graph
"""
# Plotting all three signals on the same graph
time_vector = np.arange(eeg_signal.shape[1]) / 128


plt.figure(figsize=(10, 4))


plt.plot(time_vector, eeg_signal[14, :], label="Original Signal")

plt.plot(time_vector, centered_signal[14, :], label="Centered Signal")

plt.plot(time_vector, filtred_signal[14, :], label="Filtered Signal")

plt.plot(time_vector, windowed_signal[14, :], label="Windowed Signal")

# Adding title and labels
plt.title("EEG Processing")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


# Adding a legend to distinguish the plots
plt.legend()

# Show the plot
plt.show()
