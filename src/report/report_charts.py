import plotly.graph_objects as go
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, auc

class ReportCharts:
    @staticmethod
    def generate_patient_type_bar_chart(summary_counts: Dict[str, int]) -> str:
        """
        Generate a bar chart of patient types and return it as an HTML string.

        Args:
            summary_counts (Dict[str, int]): A dictionary where keys are patient types
                                            and values are counts of patients.

        Returns:
            str: HTML string representing the generated bar chart.
        """
        bar_chart = go.Figure(
            data=[
                go.Bar(
                    x=list(summary_counts.keys()),
                    y=list(summary_counts.values()),
                    marker_color="skyblue",
                )
            ]
        )
        bar_chart.update_layout(
            title="Number of Patients by Type",
            xaxis_title="Patient Type",
            yaxis_title="Number of Patients",
            template="plotly_white",
        )
        return bar_chart.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def generate_raw_epochs_by_group_chart(summary_data: Dict[str, Any]) -> str:
        """
        Generate a bar chart showing the raw epochs grouped by patient type
        and return it as an HTML string.
        """
        patients_by_type = summary_data.get("patients_by_type", {})
        raw_epochs_by_type = {
            patient_type: sum(
                patient.raw_epoch_count if isinstance(patient.raw_epoch_count, int)
                else sum(patient.raw_epoch_count.values())
                for patient in patients
            )
            for patient_type, patients in patients_by_type.items()
        }

        bar_chart = go.Figure(
            data=[
                go.Bar(
                    x=list(raw_epochs_by_type.keys()),
                    y=list(raw_epochs_by_type.values()),
                    marker_color="orange",
                )
            ]
        )
        bar_chart.update_layout(
            title="Raw Epochs by Patient Type",
            xaxis_title="Patient Type",
            yaxis_title="Total Raw Epochs",
            template="plotly_white",
        )
        return bar_chart.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def generate_raw_epochs_pie_chart(raw_ied_epochs: int, raw_non_ied_epochs: int) -> str:
        """Generate a pie chart for raw IED and non-IED epoch distribution and return it as an HTML string."""
        pie_chart = go.Figure(
            data=[
                go.Pie(
                    labels=["Raw IED Epochs", "Raw Non-IED Epochs"],
                    values=[raw_ied_epochs, raw_non_ied_epochs],
                    hole=0.3,
                    marker=dict(colors=["#ff9999", "#66b3ff"]),
                )
            ]
        )
        pie_chart.update_layout(
            title="Distribution of Raw IED and Non-IED Epochs", template="plotly_white"
        )
        return pie_chart.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def generate_processed_epochs_pie_chart(total_ied_epochs: int, total_non_ied_epochs: int) -> str:
        """Generate a pie chart for processed IED and non-IED epoch distribution and return it as an HTML string."""
        pie_chart = go.Figure(
            data=[
                go.Pie(
                    labels=["Processed IED Epochs", "Processed Non-IED Epochs"],
                    values=[total_ied_epochs, total_non_ied_epochs],
                    hole=0.3,
                    marker=dict(colors=["#ffa07a", "#87ceeb"]),
                )
            ]
        )
        pie_chart.update_layout(
            title="Distribution of Processed IED and Non-IED Epochs", template="plotly_white"
        )
        return pie_chart.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def generate_roc_curve_chart(y_true: List[int], y_probs: List[float]) -> str:
        """
        Generate an interactive ROC curve chart using Plotly and return it as an HTML string.

        Args:
            y_true (List[int]): True labels.
            y_probs (List[float]): Predicted probabilities for the positive class.

        Returns:
            str: HTML string representing the ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        roc_figure = go.Figure()
        roc_figure.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.4f})",
                line=dict(color="blue", width=2),
            )
        )
        roc_figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(color="red", dash="dash"),
            )
        )

        roc_figure.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
        )

        return roc_figure.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def generate_loss_curve(train_losses: List[float], test_losses: List[float]) -> str:
        """
        Generate an interactive loss curve chart using Plotly and return it as an HTML string.

        Args:
            train_losses (List[float]): Training losses per epoch.
            test_losses (List[float]): Testing losses per epoch.

        Returns:
            str: HTML string representing the loss curve.
        """
        epochs = list(range(1, len(train_losses) + 1))

        loss_curve = go.Figure()
        loss_curve.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode="lines+markers",
                name="Training Loss",
                line=dict(color="blue"),
            )
        )
        loss_curve.add_trace(
            go.Scatter(
                x=epochs,
                y=test_losses,
                mode="lines+markers",
                name="Testing Loss",
                line=dict(color="orange"),
            )
        )

        loss_curve.update_layout(
            title="Training and Testing Loss Curve",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            template="plotly_white",
        )

        return loss_curve.to_html(full_html=False, include_plotlyjs="cdn")

    @staticmethod
    def generate_accuracy_curve(train_accuracies: List[float], test_accuracies: List[float]) -> str:
        """
        Generate an interactive accuracy curve chart using Plotly and return it as an HTML string.

        Args:
            train_accuracies (List[float]): Training accuracies per epoch.
            test_accuracies (List[float]): Testing accuracies per epoch.

        Returns:
            str: HTML string representing the accuracy curve.
        """
        epochs = list(range(1, len(train_accuracies) + 1))

        accuracy_curve = go.Figure()
        accuracy_curve.add_trace(
            go.Scatter(
                x=epochs,
                y=train_accuracies,
                mode="lines+markers",
                name="Training Accuracy",
                line=dict(color="green"),
            )
        )
        accuracy_curve.add_trace(
            go.Scatter(
                x=epochs,
                y=test_accuracies,
                mode="lines+markers",
                name="Testing Accuracy",
                line=dict(color="purple"),
            )
        )

        accuracy_curve.update_layout(
            title="Training and Testing Accuracy Curve",
            xaxis_title="Epochs",
            yaxis_title="Accuracy",
            template="plotly_white",
        )

        return accuracy_curve.to_html(full_html=False, include_plotlyjs="cdn")
        