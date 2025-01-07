from src.plotting.eeg_plotter import EEGPlotter

epoch = "DEMOs/DEMO_data/processed_epoch_19_256_128Hz_CA.npy"


# Interactive plot with MNE library
EEGPlotter.plot_eeg(epoch)

# Single channel plot
# EEGPlotter.plot_single_channel(epoch)

