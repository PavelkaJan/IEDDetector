import logging
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from src.patient.patient import Patient
from src.report.report_charts import ReportCharts
from src.report.patient_statistics import PatientStatistics
from src.report.nn_metrics_report import NNMetricsReport


logger = logging.getLogger(__name__)


class Report:
    def __init__(self, patients: List[Patient], output_path: str):
        """
        Initialize the Report with patient data and output path.

        Args:
            patients (List[Patient]): List of patient instances.
            output_path (str): Path where the generated report will be saved.
        """
        self.patients = patients
        self.output_path = output_path
        logger.debug("Report initialized with %d patients", len(patients))

    def summarize_patient_data(self) -> Dict[str, Any]:
        """
        Organize patients by type and generate summary statistics.
        """
        logger.debug("Summarizing patient data")
        statistics = PatientStatistics(self.patients)
        summary_data = statistics.summarize()

        logger.debug("Patient data summarized with %d total patients", summary_data["total_number_of_patients"])
        return summary_data

    def _load_html_template(self) -> Template:
        """
        Load and return the main HTML report template using the FileSystemLoader.
        """
        template_dir = "src/report/templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template_name = "report_template.html"
        return env.get_template(template_name)

    def _render_html_report(
        self, summary_data: Dict[str, Any], nn_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render the HTML report using Jinja2 with provided patient data and evaluation results.
        """
        logger.debug("Rendering HTML report")
        template = self._load_html_template()
        nn_results = nn_results or {}
        charts = self._create_charts(summary_data)

        return template.render(
            patients_by_type=summary_data.get("patients_by_type", {}),
            summary_counts=summary_data.get("summary_counts", {}),
            total_epochs=summary_data.get("total_epochs", 0),
            total_ied_epochs=summary_data.get("total_ied_epochs", 0),
            total_non_ied_epochs=summary_data.get("total_non_ied_epochs", 0),
            raw_ied_epochs=summary_data.get("raw_ied_epochs", 0),
            raw_non_ied_epochs=summary_data.get("raw_non_ied_epochs", 0),
            total_number_of_patients=summary_data.get("total_number_of_patients", 0),
            total_processed_epochs=summary_data.get("total_processed_epochs", 0),
            roc_auc=nn_results.get("roc_auc", 0),
            final_metrics=nn_results.get("final_metrics", {}),
            roc_curve_html=nn_results.get("roc_curve_html", ""),  
            loss_curve_html=nn_results.get("loss_curve_html", ""), 
            accuracy_curve_html=nn_results.get("accuracy_curve_html", ""),
            **charts,
        )

    def generate_and_save(
        self,
        nn_report_inputs: Optional[Dict[str, Any]] = None
    ):
        """
        Generate, render, and save the HTML report with metrics.

        Args:
            nn_report_inputs (Optional[Dict[str, Any]]): Inputs for computing metrics.
        """
        logger.info("Starting report generation process")

        summary_data = self.summarize_patient_data()

        if nn_report_inputs is None:
            logger.error("No neural network inputs provided for the report.")
            raise ValueError("Cannot generate report without neural network inputs.")

        nn_results = NNMetricsReport.prepare_report_data(
            nn_report_inputs["all_test_labels"],
            nn_report_inputs["all_test_probs"],
            nn_report_inputs["train_losses"],
            nn_report_inputs["test_losses"],
            nn_report_inputs["train_accuracies"],
            nn_report_inputs["test_accuracies"]
        )

        logger.debug("Generating ROC curve chart.")
        nn_results["roc_curve_html"] = ReportCharts.generate_roc_curve_chart(
            nn_report_inputs["all_test_labels"],
            nn_report_inputs["all_test_probs"]
        )

        logger.debug("Generating loss curve chart.")
        nn_results["loss_curve_html"] = ReportCharts.generate_loss_curve(
            nn_report_inputs["train_losses"],
            nn_report_inputs["test_losses"]
        )

        logger.debug("Generating accuracy curve chart.")
        nn_results["accuracy_curve_html"] = ReportCharts.generate_accuracy_curve(
            nn_report_inputs["train_accuracies"],
            nn_report_inputs["test_accuracies"]
        )

        report_html = self._render_html_report(summary_data, nn_results)

        self._save_report_to_file(report_html)

        logger.info("HTML report generation completed successfully.")

    def _save_report_to_file(self, content: str) -> None:
        """
        Save the plain-text content to the output path.

        Args:
            content (str): Plain-text content to save.
        """
        logger.debug("Saving report to %s", self.output_path)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Report successfully saved to %s", self.output_path)

    def _create_charts(self, summary_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate the charts for the report and return them as HTML strings.
        """
        logger.debug("Creating charts for the report")

        return {
            "bar_chart_html": ReportCharts.generate_patient_type_bar_chart(summary_data.get("summary_counts", {})),
            "raw_epochs_chart_html": ReportCharts.generate_raw_epochs_by_group_chart(summary_data),
            "raw_epochs_pie_chart_html": ReportCharts.generate_raw_epochs_pie_chart(
                summary_data.get("raw_ied_epochs", 0),
                summary_data.get("raw_non_ied_epochs", 0),
            ),
            "processed_epochs_pie_chart_html": ReportCharts.generate_processed_epochs_pie_chart(
                summary_data.get("total_ied_epochs", 0),
                summary_data.get("total_non_ied_epochs", 0),
            ),
        }