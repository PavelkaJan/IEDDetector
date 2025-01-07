import torch
import logging
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import auc, roc_curve
from src.logging_config.logging_setup import setup_logging
from src.neural_network.nn_control import NNControl
from src.neural_network.nn_diplomka import NNDiplomkaFunctions
from src.neural_network.nn_plots import NNPlots
from src.neural_network.nn_evaluation_metrics import NNEvaluationMetrics
from torch import nn
import numpy as np
from src.constants import PatientType
from pprint import pprint
import pickle


# Init
setup_logging()
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class_counts = torch.tensor([14112, 58212], dtype=torch.float)
total_samples = class_counts.sum()

class_weights = total_samples / class_counts
print(f"Class Weights: {class_weights}")

class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


MODELS_OUTPUT_NAME = "epileptics_real_and_healthy_real_4s_epochy"
MODELS_OUTPUT_PATH = Path("models_20241230") / MODELS_OUTPUT_NAME
PLOTS_OUTPUT_FOLDER = (
    Path("diplomka20243012_v2") / "epileptics_real_and_healthy_real_4s_epochy"
)
os.makedirs(MODELS_OUTPUT_PATH, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)

# Hyperparameters
batch_size = 64
learning_rate = 1e-5
base_path = "D:\\DIPLOMKA_DATASET"

# Load patients and categorize
patient_paths = NNControl.get_patient_file_paths(base_path)
patients = NNControl.load_patients(patient_paths)

patients_dict = NNDiplomkaFunctions.categorize_patients_by_diplomka_category(patients)
cv_splits = NNDiplomkaFunctions.generate_full_coverage_cv_splits(
    patients_dict, n_splits=10
)

for fold_idx, (train_set, test_set) in enumerate(cv_splits):
    print(f"Fold {fold_idx + 1}:")
    print("  Train Set:")
    pprint(train_set)
    print("  Test Set:")
    pprint(test_set)
    print("\n")


# Patient stats
all_sensitivity_stats = []
all_fnr_stats = []
all_specificity_stats = []
all_fpr_stats = []
all_epileptic_accuracies = []
all_healthy_accuracies = []

# ROC
all_auc_scores = []
all_test_labels_folds = []
all_test_probs_folds = []

# Big plot for all SNR
current_snr_metrics = []


for fold_idx, (train_patients, test_patients) in enumerate(cv_splits):
    logger.info(f"Processing fold {fold_idx + 1}/{len(cv_splits)}")

    train_loader, test_loader = NNControl.prepare_data_loaders(
        train_patients, test_patients, batch_size=batch_size
    )

    model, _, optimizer = NNControl.initialize_model(
        device=device, learning_rate=learning_rate
    )

    patient_types = {patient.id: patient.patient_type for patient in test_patients}

    epileptics_real = NNControl.get_patient_ids_by_type(
        patient_types, PatientType.EPILEPTIC_REAL
    )
    healthy_controls_real = NNControl.get_patient_ids_by_type(
        patient_types, PatientType.HEALTHY_REAL
    )

    train_losses, train_accuracies, test_losses, test_accuracies = (
        NNControl.train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=20,  # Maximum number of epochs
            patience=3,  # Stop if no improvement after x epochs
            save_best_model=True,
            best_model_path=MODELS_OUTPUT_PATH / f"fold_{fold_idx + 1}_best_model.pth",
        )
    )

    test_loss, all_test_labels, all_test_probs, patient_metrics = (
        NNControl.evaluate_with_patient_info(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
    )

    # ROC
    all_test_labels_folds.append(all_test_labels)
    all_test_probs_folds.append(all_test_probs)

    fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
    auc_score = auc(fpr, tpr)
    all_auc_scores.append(auc_score)
    print(f"Fold {fold_idx + 1} AUC: {auc_score:.4f}")

    # Patient stats
    fold_sensitivity_stats = (
        NNEvaluationMetrics.compute_sensitivity_stats_for_epileptic(
            patient_metrics, epileptics_real
        )
    )

    fold_fnr_stats = (
        NNEvaluationMetrics.compute_false_negative_rate_stats_for_epileptic(
            patient_metrics, epileptics_real
        )
    )

    fold_specificity_stats = NNEvaluationMetrics.compute_specificity_stats_for_healthy(
        patient_metrics, healthy_controls_real
    )

    fold_fpr_stats = NNEvaluationMetrics.compute_false_positive_rate_stats_for_healthy(
        patient_metrics, healthy_controls_real
    )

    epileptic_accuracy = NNEvaluationMetrics.compute_accuracy_for_patient_group(
        patient_metrics, epileptics_real
    )

    healthy_accuracy = NNEvaluationMetrics.compute_accuracy_for_patient_group(
        patient_metrics, healthy_controls_real
    )

    all_sensitivity_stats.append(fold_sensitivity_stats["mean"])
    all_fnr_stats.append(fold_fnr_stats["mean"])
    all_specificity_stats.append(fold_specificity_stats["mean"])
    all_fpr_stats.append(fold_fpr_stats["mean"])

    logger.info(
        f"Fold {fold_idx + 1} Sensitivity: "
        f"{fold_sensitivity_stats['mean']:.4f} ± {fold_sensitivity_stats['std']:.4f}"
    )

    logger.info(
        f"Fold {fold_idx + 1} False Negative Rate: "
        f"{fold_fnr_stats['mean']:.4f} ± {fold_fnr_stats['std']:.4f}"
    )

    logger.info(
        f"Fold {fold_idx + 1} Specificity: "
        f"{fold_specificity_stats['mean']:.4f} ± {fold_specificity_stats['std']:.4f}"
    )

    logger.info(
        f"Fold {fold_idx + 1} False Positive rate: "
        f"{fold_fpr_stats['mean']:.4f} ± {fold_fpr_stats['std']:.4f}"
    )

    logger.info(f"Fold {fold_idx + 1} Epileptic Accuracy: {epileptic_accuracy:.4f}")
    logger.info(f"Fold {fold_idx + 1} Healthy Accuracy: {healthy_accuracy:.4f}")


# ROC
overall_auc_mean = np.mean(all_auc_scores)
overall_auc_std = np.std(all_auc_scores)

# Patient stats
overall_sensitivity_mean = np.mean(all_sensitivity_stats)
overall_sensitivity_std = np.std(all_sensitivity_stats)

overall_fnr_mean = np.mean(all_fnr_stats)
overall_fnr_std = np.std(all_fnr_stats)

overall_specificity_mean = np.mean(all_specificity_stats)
overall_specificity_std = np.std(all_specificity_stats)

overall_misclassification_mean = np.mean(all_fpr_stats)
overall_misclassification_std = np.std(all_fpr_stats)

overall_epileptic_accuracy_mean = np.mean(all_epileptic_accuracies)
overall_epileptic_accuracy_std = np.std(all_epileptic_accuracies)

overall_healthy_accuracy_mean = np.mean(all_healthy_accuracies)
overall_healthy_accuracy_std = np.std(all_healthy_accuracies)

logger.info("\nOverall AUC Across All Folds:")
logger.info(f"AUC: {overall_auc_mean:.4f} ± {overall_auc_std:.4f}")

# Print overall statistics
logger.info("Overall Patinets Info Across All Folds:")
logger.info(
    f"Sensitivity: {overall_sensitivity_mean:.4f} ± {overall_sensitivity_std:.4f}"
)
logger.info(f"False Negative Rate: {overall_fnr_mean:.4f} ± {overall_fnr_std:.4f}")
logger.info(
    f"Specificity: {overall_specificity_mean:.4f} ± {overall_specificity_std:.4f}"
)
logger.info(
    f"Misclassification: {overall_misclassification_mean:.4f} ± {overall_misclassification_std:.4f}"
)
logger.info(
    f"Epileptic Accuracy: {overall_epileptic_accuracy_mean:.4f} ± {overall_epileptic_accuracy_std:.4f}"
)
logger.info(
    f"Healthy Accuracy: {overall_healthy_accuracy_mean:.4f} ± {overall_healthy_accuracy_std:.4f}"
)


with open(PLOTS_OUTPUT_FOLDER / "all_test_labels_folds.pkl", "wb") as f:
    pickle.dump(all_test_labels_folds, f)

with open(PLOTS_OUTPUT_FOLDER / "all_test_probs_folds.pkl", "wb") as f:
    pickle.dump(all_test_probs_folds, f)

with open(PLOTS_OUTPUT_FOLDER / "n_folds.pkl", "wb") as f:
    pickle.dump(len(cv_splits), f)

NNPlots.plot_mean_roc_curve(
    all_labels=all_test_labels_folds,
    all_probs=all_test_probs_folds,
    n_folds=len(cv_splits),
    save_path=PLOTS_OUTPUT_FOLDER / "roc_curve.pdf",
    save_roc_data_path=PLOTS_OUTPUT_FOLDER / "roc_curve_data",
)

with open(PLOTS_OUTPUT_FOLDER / "all_epileptic_accuracies.pkl", "wb") as f:
    pickle.dump(all_epileptic_accuracies, f)

with open(PLOTS_OUTPUT_FOLDER / "all_healthy_accuracies.pkl", "wb") as f:
    pickle.dump(all_healthy_accuracies, f)

NNPlots.plot_accuracy_boxplot(
    epileptic_accuracies=all_epileptic_accuracies,
    healthy_accuracies=all_healthy_accuracies,
    save_path=PLOTS_OUTPUT_FOLDER / "accuracy_boxplot.pdf",
    save_data_path=PLOTS_OUTPUT_FOLDER / "accuracy_data",
)

with open(PLOTS_OUTPUT_FOLDER / "all_sensitivity_stats.pkl", "wb") as f:
    pickle.dump(all_sensitivity_stats, f)

with open(PLOTS_OUTPUT_FOLDER / "all_specificity_stats.pkl", "wb") as f:
    pickle.dump(all_specificity_stats, f)

with open(PLOTS_OUTPUT_FOLDER / "all_fnr_stats.pkl", "wb") as f:
    pickle.dump(all_fnr_stats, f)

with open(PLOTS_OUTPUT_FOLDER / "all_fpr_stats.pkl", "wb") as f:
    pickle.dump(all_fpr_stats, f)

NNPlots.plot_combined_metrics_boxplot(
    sensitivity_stats=all_sensitivity_stats,
    specificity_stats=all_specificity_stats,
    fnr_stats=all_fnr_stats,
    fpr_stats=all_fpr_stats,
    save_path=PLOTS_OUTPUT_FOLDER / "combined_metrics_boxplots.pdf",
    save_data_path=PLOTS_OUTPUT_FOLDER / "combined_metrics_data",
)
