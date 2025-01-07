from src.neural_network.nn_plots import NNPlots
import pickle

with open("diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/all_sensitivity_stats.pkl", 'rb') as file:
    all_sensitivity_stats = pickle.load(file)

with open("diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/all_specificity_stats.pkl", 'rb') as file:
    all_specificity_stats = pickle.load(file)

NNPlots.plot_accuracy_boxplot(
    sensitivity_stats=all_sensitivity_stats,
    specificity_stats=all_specificity_stats,
    save_path="diplomka20243012/epileptics_real_and_healthy_real_4s_epochy/plots_to_overleaf/accuracyRealDataset135mm.pdf",
)

