import logging
from typing import Tuple, List, Dict
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)


class NNEvaluationMetrics:
    @staticmethod
    def compute_metrics(
        y_true: List[int], y_pred_probs: List[float], threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics including Accuracy, Precision, Recall, F1 Score, and AUC.
        """
        if not y_true or not y_pred_probs:
            raise ValueError("Inputs y_true and y_pred_probs cannot be None or empty.")

        y_pred = [1 if prob >= threshold else 0 for prob in y_pred_probs]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        logger.info(
            f"Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}"
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": roc_auc,
        }

    @staticmethod
    def compute_roc_curve(
        y_true: List[int], y_pred_probs: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Compute the ROC curve data points.

        Args:
            y_true (List[int]): True class labels (0 or 1).
            y_pred_probs (List[float]): Predicted probabilities for the positive class.

        Returns:
            Tuple[List[float], List[float], List[float]]: False positive rates, true positive rates, and thresholds.
        """
        if not y_true or not y_pred_probs:
            raise ValueError("Inputs y_true and y_pred_probs cannot be None or empty.")

        logger.info("Computing ROC curve.")
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        return fpr.tolist(), tpr.tolist(), thresholds.tolist()

    @staticmethod
    def compute_average_metrics(
        roc_data_per_fold, epileptic_sensitivities_per_fold=None
    ):
        """
        Compute average ROC curve, AUC, sensitivity (for epileptic patients), specificity, and their
        standard deviations across folds from ROC data.

        Parameters:
            roc_data_per_fold (list of dicts): A list where each element is a dictionary with keys
                'fpr', 'tpr', and 'auc' for each fold.
            epileptic_sensitivities_per_fold (list of float, optional): Sensitivities for epileptic
                patients in each fold, if pre-computed.

        Returns:
            dict: A dictionary containing average metrics:
                - mean_fpr: Fixed FPR points for the mean ROC curve
                - mean_tpr: Mean TPR values at the fixed FPR points
                - std_tpr: Standard deviation of TPR values at the fixed FPR points
                - mean_auc: Mean AUC across folds
                - std_auc: Standard deviation of AUC across folds
                - mean_sensitivity: Mean sensitivity across folds (for epileptic patients)
                - std_sensitivity: Standard deviation of sensitivity across folds
                - mean_specificity: Mean specificity across folds
                - std_specificity: Standard deviation of specificity across folds
        """
        # Fixed FPR points for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        specificities = []
        computed_sensitivities = []

        # Interpolate TPR for each fold and compute metrics
        for roc_data in roc_data_per_fold:
            # Check if the fold has no variability (e.g., all predictions are correct)
            if len(roc_data["fpr"]) == 0 or len(roc_data["tpr"]) == 0:
                # Handle case with perfect predictions
                interp_tpr = np.ones_like(mean_fpr)  # TPR = 1 across all thresholds
                tprs.append(interp_tpr)
                aucs.append(1.0)  # Perfect AUC
                specificities.append(1.0)
            else:
                # Interpolate TPR values
                interp_tpr = np.interp(mean_fpr, roc_data["fpr"], roc_data["tpr"])
                interp_tpr[0] = 0.0  # Ensure TPR starts at 0
                tprs.append(interp_tpr)
                aucs.append(roc_data["auc"])

                # Compute fold-specific specificity
                fold_specificity = 1 - roc_data["fpr"]
                specificities.append(np.mean(fold_specificity))

        # Use the pre-computed epileptic sensitivities if provided
        if epileptic_sensitivities_per_fold is not None:
            final_sensitivities = epileptic_sensitivities_per_fold
        else:
            # Default to using the computed sensitivities from TPR (fallback)
            final_sensitivities = computed_sensitivities

        # Compute averages and standard deviations
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_specificity = np.mean(specificities)
        std_specificity = np.std(specificities)
        mean_sensitivity = np.mean(final_sensitivities) if final_sensitivities else 0.0
        std_sensitivity = np.std(final_sensitivities) if final_sensitivities else 0.0

        return {
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_sensitivity": mean_sensitivity,
            "std_sensitivity": std_sensitivity,
            "mean_specificity": mean_specificity,
            "std_specificity": std_specificity,
        }

    # ----------------------------------------------------------
    @staticmethod
    def compute_sensitivity_stats_for_epileptic(
        patient_metrics: Dict[str, Dict[str, float]], epileptic_patient_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute the average and standard deviation of sensitivity
        for epileptic patients in a fold.

        Args:
            patient_metrics (dict): Dictionary of metrics for all patients in the fold.
                Format: {patient_id: {metric_name: metric_value, ...}}
            epileptic_patient_ids (list): List of patient IDs corresponding to epileptic patients.

        Returns:
            dict: A dictionary containing average and std deviation of sensitivity:
                {"mean": float, "std": float}
        """
        # Extract sensitivities for epileptic patients
        sensitivities = [
            metrics["sensitivity"]
            for patient_id, metrics in patient_metrics.items()
            if patient_id in epileptic_patient_ids and "sensitivity" in metrics
        ]

        # Compute and return the stats
        if sensitivities:
            mean_value = np.mean(sensitivities)
            std_value = np.std(sensitivities)
        else:
            mean_value, std_value = 0.0, 0.0

        return {"mean": mean_value, "std": std_value}

    def compute_false_negative_rate_stats_for_epileptic(
        patient_metrics: Dict[str, Dict[str, float]], epileptic_patient_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute the average and standard deviation of the false negative rate (FNR)
        for epileptic patients in a fold.

        Args:
            patient_metrics (dict): Dictionary of metrics for all patients in the fold.
                Format: {patient_id: {metric_name: metric_value, ...}}
            epileptic_patient_ids (list): List of patient IDs corresponding to epileptic patients.

        Returns:
            dict: A dictionary containing average and std deviation of FNR:
                {"mean": float, "std": float}
        """
        # Extract false negative rates for epileptic patients
        fnr_values = [
            metrics["false_negative_rate"]
            for patient_id, metrics in patient_metrics.items()
            if patient_id in epileptic_patient_ids and "false_negative_rate" in metrics
        ]

        # Compute and return the stats
        if fnr_values:
            mean_value = np.mean(fnr_values)
            std_value = np.std(fnr_values)
        else:
            mean_value, std_value = 0.0, 0.0

        return {"mean": mean_value, "std": std_value}

    @staticmethod
    def compute_specificity_stats_for_healthy(
        patient_metrics: Dict[str, Dict[str, float]], healthy_patient_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute the average and standard deviation of the specificity
        for healthy patients in a fold.

        Args:
            patient_metrics (dict): Dictionary of metrics for all patients in the fold.
                Format: {patient_id: {metric_name: metric_value, ...}}
            healthy_patient_ids (list): List of patient IDs corresponding to healthy patients.

        Returns:
            dict: A dictionary containing average and std deviation of specificity:
                {"mean": float, "std": float}
        """
        # Extract specificity values for healthy patients
        specificity_values = [
            metrics["specificity"]
            for patient_id, metrics in patient_metrics.items()
            if patient_id in healthy_patient_ids and "specificity" in metrics
        ]

        # Compute and return the stats
        if specificity_values:
            mean_value = np.mean(specificity_values)
            std_value = np.std(specificity_values)
        else:
            mean_value, std_value = 0.0, 0.0

        return {"mean": mean_value, "std": std_value}

    @staticmethod
    def compute_false_positive_rate_stats_for_healthy(
        patient_metrics: Dict[str, Dict[str, float]], healthy_patient_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute the average and standard deviation of the misclassification rate
        for healthy patients in a fold.

        Args:
            patient_metrics (dict): Dictionary of metrics for all patients in the fold.
                Format: {patient_id: {metric_name: metric_value, ...}}
            healthy_patient_ids (list): List of patient IDs corresponding to healthy patients.

        Returns:
            dict: A dictionary containing average and std deviation of the misclassification rate:
                {"mean": float, "std": float}
        """
        # Extract misclassification rate values for healthy patients
        misclassification_values = [
            metrics["false_positive_rate"]
            for patient_id, metrics in patient_metrics.items()
            if patient_id in healthy_patient_ids and "false_positive_rate" in metrics
        ]

        # Compute and return the stats
        if misclassification_values:
            mean_value = np.mean(misclassification_values)
            std_value = np.std(misclassification_values)
        else:
            mean_value, std_value = 0.0, 0.0

        return {"mean": mean_value, "std": std_value}

    @staticmethod
    def compute_accuracy_for_patient_group(
        patient_metrics: Dict[str, Dict[str, float]], patient_ids: List[str]
    ) -> float:
        """
        Compute the average accuracy for a specific group of patients (e.g., epileptic or healthy).

        Args:
            patient_metrics (dict): Dictionary of metrics for all patients in the fold.
                Format: {patient_id: {metric_name: metric_value, ...}}
            patient_ids (list): List of patient IDs corresponding to the specific group.

        Returns:
            float: The average accuracy for the specified group. Returns 0 if no patients are found.
        """
        accuracies = [
            patient_metrics[patient_id]["accuracy"]
            for patient_id in patient_ids
            if patient_id in patient_metrics
        ]

        if not accuracies:
            return 0.0

        return sum(accuracies) / len(accuracies)
