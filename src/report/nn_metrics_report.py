from typing import Dict, List, Any
from src.neural_network.nn_evaluation_metrics import NNEvaluationMetrics


class NNMetricsReport:
    @staticmethod
    def prepare_report_data(
        all_test_labels: List[int],
        all_test_probs: List[float],
        train_losses: List[float],
        test_losses: List[float],
        train_accuracies: List[float],
        test_accuracies: List[float],
    ) -> Dict[str, Any]:
        """
        Prepare data for the neural network evaluation report.

        Returns:
            Dict[str, Any]: Metrics and visualizations to include in the report.
        """
        metrics = NNEvaluationMetrics.compute_metrics(all_test_labels, all_test_probs)

        return {
            "roc_auc": metrics["auc"],
            "final_metrics": metrics,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "train_accuracies": train_accuracies,
            "test_accuracies": test_accuracies,
        }
