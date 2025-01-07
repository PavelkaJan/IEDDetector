import pickle
from src.neural_network.nn_plots import NNPlots


snr_values = [-9, -6, -3, 0, 3]

snr_patient_ids_dict = {}
snr_accuracies_dict = {}

for snr in snr_values:
    with open(
        f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_epileptic_patient_ids_SNR_{snr}.pkl",
        "rb",
    ) as file:
        snr_patient_ids_dict[snr] = pickle.load(file)

    with open(
        f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_epileptic_accuracies_cv_plot_SNR_{snr}.pkl",
        "rb",
    ) as file:
        snr_accuracies_dict[snr] = pickle.load(file)

NNPlots.plot_accuracy_by_location_across_snr(
    snr_patient_ids_dict,
    snr_accuracies_dict,
    snr_values,
    save_path="diplomka20243012/simulated_dataset/plots_to_overleaf/sensitivityByLocation135mm.pdf",
)
