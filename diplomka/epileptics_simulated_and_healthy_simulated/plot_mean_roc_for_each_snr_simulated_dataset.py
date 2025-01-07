from typing import Dict, Any
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from src.neural_network.nn_plots import NNPlots

def compute_mean_roc_curve(all_labels, all_probs, n_folds):
    """
    Compute mean ROC curve for a specific dataset (e.g., for an SNR level).
    This function is based on the `plot_mean_roc_curve` but returns the
    computed ROC data instead of plotting it.
    """
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    aucs = []

    for fold_idx in range(n_folds):
        fpr, tpr, _ = roc_curve(all_labels[fold_idx], all_probs[fold_idx])
        roc_auc = auc(fpr, tpr)
        aucs.append(float(roc_auc))  # Ensure Python float

        # Interpolate TPR to mean FPR
        tprs.append(np.interp(mean_fpr, fpr, tpr).tolist())
        tprs[-1][0] = 0.0

    # Compute mean and std TPR
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(auc(mean_fpr, mean_tpr))
    std_auc = float(np.std(aucs))

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }

snr_values = [-9, -6, -3, 0, 3]
roc_results = {}

for snr in snr_values:
    # Load fold-specific data for this SNR
    with open(f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_test_labels_folds_SNR_{snr}.pkl", "rb") as file:
        all_test_labels_folds = pickle.load(file)

    with open(f"diplomka/plots2/epileptics_simulated_SNR_{snr}_and_healthy_simulated/all_test_probs_folds_SNR_{snr}.pkl", "rb") as file:
        all_test_probs_folds = pickle.load(file)

    # Compute the mean ROC curve for this SNR
    roc_data = compute_mean_roc_curve(
        all_labels=all_test_labels_folds,
        all_probs=all_test_probs_folds,
        n_folds=len(all_test_labels_folds)
    )

    # Store the ROC data
    roc_results[snr] = roc_data


NNPlots.plot_mean_roc_curves_for_snrs(
    roc_results,
    title=r"\textnormal{ROC křivky pro různé úrovně SNR",
    save_path="diplomka20243012/simulated_dataset/plots_to_overleaf/meanROCForDiffSNRSimuData135mm.pdf",
)