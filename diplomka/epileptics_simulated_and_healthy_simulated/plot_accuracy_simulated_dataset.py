import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from src.neural_network.nn_plots import NNPlots
import pickle

# Define SNR values
snr_values = [-9, -6, -3, 0, 3]

# Load stats for each SNR
sensitivity_stats_dict = {}
specificity_stats_dict = {}
fnr_stats_dict = {}
fpr_stats_dict = {}

for snr in snr_values:
    with open(
        f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_sensitivity_stats_SNR_{snr}.pkl",
        "rb",
    ) as file:
        sensitivity_stats_dict[snr] = pickle.load(file)

    with open(
        f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_specificity_stats_SNR_{snr}.pkl",
        "rb",
    ) as file:
        specificity_stats_dict[snr] = pickle.load(file)


NNPlots.plot_accuracy_across_snr_v2(
    sensitivity_stats_dict,
    specificity_stats_dict,
    snr_values,
    save_path="diplomka20243012/simulated_dataset/plots_to_overleaf/accuracySimulatedDataset135mm.pdf",
)
