import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from sklearn.metrics import roc_curve, auc
import json
import re


class NNPlots:
    # Set LaTeX fonts globally for consistent style
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    @staticmethod
    def plot_combined_metrics_boxplot(
        sensitivity_stats: list,
        specificity_stats: list,
        fnr_stats: list,
        fpr_stats: list,
        snr: float = None,  # Optional parameter
        title: str = r"\textnormal{Porovnání metrik přesnosti a chybovosti}",
        save_path: str = None,
        save_data_path: str = None,
    ):
        """
        Plot sensitivity, specificity, FNR, and misclassification metrics and save boxplot stats.

        Args:
            sensitivity_stats (list): Sensitivity values for epileptic patients.
            specificity_stats (list): Specificity values for healthy patients.
            fnr_stats (list): FNR values for epileptic patients.
            fpr_stats (list): FPR values for healthy patients.
            snr (float): SNR value for this dataset.
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF.
            save_data_path (str): Path to save the boxplot statistics.

        Returns:
            None
        """

        # Convert dimensions to inches (13 cm x 9 cm)
        width_in_inches = 13.5 / 2.54
        height_in_inches = 10 / 2.54

        # Prepare the data
        data = pd.DataFrame(
            {
                "Metric": (
                    [r"\textnormal{Sensitivita}"] * len(sensitivity_stats)
                    + [r"\textnormal{Specificita}"] * len(specificity_stats)
                    + [r"\textnormal{FNR}"] * len(fnr_stats)
                    + [r"\textnormal{FPR}"] * len(fpr_stats)
                ),
                "Value": sensitivity_stats + specificity_stats + fnr_stats + fpr_stats,
                "Patient Group": (
                    [r"\textnormal{Epileptici}"] * len(sensitivity_stats)
                    + [r"\textnormal{Kontrolní skupina}"] * len(specificity_stats)
                    + [r"\textnormal{Epileptici}"] * len(fnr_stats)
                    + [r"\textnormal{Kontrolní skupina}"] * len(fpr_stats)
                ),
            }
        )

        # Define custom colors
        deep_palette = sns.color_palette("deep")
        custom_palette = {
            r"\textnormal{Epileptici}": deep_palette[4],
            r"\textnormal{Kontrolní skupina}": deep_palette[0],
        }

        # Set Seaborn theme
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        # Plot
        plt.figure(figsize=(width_in_inches, height_in_inches))
        ax = sns.boxplot(
            x="Metric",
            y="Value",
            hue="Patient Group",
            data=data,
            palette=custom_palette,
            showfliers=True,
        )

        # Customizations
        plt.title(title, fontsize=10)
        plt.ylabel(r"\textnormal{Hodnota metriky [-]}", fontsize=8)
        plt.xlabel(r"\textnormal{Typ metriky}", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        sns.despine()

        # Adjust legend
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            title=None,
            frameon=False,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot if save_path is provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

        # Show the plot
        plt.show()

        # Save metrics data to a file if save_data_path is provided
        if save_data_path:
            # Create a structured dictionary for saving
            metrics_data = {
                "SNR": snr,
                "Metrics": {
                    "Sensitivity": sensitivity_stats,
                    "Specificity": specificity_stats,
                    "FNR": fnr_stats,
                    "FPR": fpr_stats,
                },
            }

            # Save as a JSON file
            with open(save_data_path, "w") as f:
                json.dump(metrics_data, f, indent=4)

    @staticmethod
    def plot_accuracy_boxplot(
        sensitivity_stats: list,
        specificity_stats: list,
        snr: float = None,  # Optional parameter
        title: str = r"\textnormal{Přesnost klasifikace}",
        save_path: str = None,
        save_data_path: str = None,
    ):
        """
        Plot sensitivity, specificity, FNR, and misclassification metrics and save boxplot stats.

        Args:
            sensitivity_stats (list): Sensitivity values for epileptic patients.
            specificity_stats (list): Specificity values for healthy patients.
            fnr_stats (list): FNR values for epileptic patients.
            fpr_stats (list): FPR values for healthy patients.
            snr (float): SNR value for this dataset.
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF.
            save_data_path (str): Path to save the boxplot statistics.

        Returns:
            None
        """

        # Convert dimensions to inches (13 cm x 9 cm)
        width_in_inches = 13.5 / 2.54
        height_in_inches = 10 / 2.54

        # Prepare the data
        data = pd.DataFrame(
            {
                "Metric": (
                    [r"\textnormal{Sensitivita}"] * len(sensitivity_stats)
                    + [r"\textnormal{Specificita}"] * len(specificity_stats)
                ),
                "Value": sensitivity_stats + specificity_stats,
                "Patient Group": (
                    [r"\textnormal{Epileptici}"] * len(sensitivity_stats)
                    + [r"\textnormal{Kontrolní skupina}"] * len(specificity_stats)
                ),
            }
        )

        # Define custom colors
        deep_palette = sns.color_palette("deep")
        custom_palette = {
            r"\textnormal{Epileptici}": deep_palette[4],
            r"\textnormal{Kontrolní skupina}": deep_palette[0],
        }

        # Set Seaborn theme
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        # Plot
        plt.figure(figsize=(width_in_inches, height_in_inches))
        ax = sns.boxplot(
            x="Metric",
            y="Value",
            hue="Patient Group",
            data=data,
            palette=custom_palette,
            showfliers=True,
        )

        # Customizations
        plt.title(title, fontsize=10)
        plt.ylabel(r"\textnormal{Hodnota metriky [-]}", fontsize=10)
        plt.xlabel(r"\textnormal{Typ metriky}", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        sns.despine()

        # Adjust legend
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            title=None,
            frameon=False,
            prop={"size": 8},
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot if save_path is provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

        # Show the plot
        plt.show()

        # Save metrics data to a file if save_data_path is provided
        if save_data_path:
            # Create a structured dictionary for saving
            metrics_data = {
                "SNR": snr,
                "Metrics": {
                    "Sensitivity": sensitivity_stats,
                    "Specificity": specificity_stats,
                },
            }

            # Save as a JSON file
            with open(save_data_path, "w") as f:
                json.dump(metrics_data, f, indent=4)

    @staticmethod
    def plot_mean_roc_curve(
        all_labels,
        all_probs,
        n_folds,
        title: str = r"\textnormal{ROC křivka pro 10 skupin CV}",
        save_path=None,
        save_roc_data_path=None,
    ):
        """
        Plot ROC curves for all folds, compute a mean ROC curve, and save the mean ROC data.

        Args:
            all_labels (list of lists): True labels for each fold.
            all_probs (list of lists): Predicted probabilities for each fold.
            n_folds (int): Number of folds.
            title (str): Title of the plot. Defaults to None.
            save_path (str): Path to save the plot as a PDF. Defaults to None.
            save_roc_data_path (str): Path to save the mean ROC data along with all input data as a JSON file.

        Returns:
            None
        """
        sns.set_theme(style="whitegrid")  # Apply whitegrid style
        sns.set_context("paper")

        # Convert dimensions to inches (13 cm width x 13 cm height)
        graph_width_in_inches = 13.5 / 2.54  # 5.12 inches
        graph_height_in_inches = 10 / 2.54  # 5.12 inches

        # Initialize variables for computing the mean ROC curve
        mean_fpr = np.linspace(0, 1, 1000)
        tprs = []
        aucs = []

        # Generate colors from the Seaborn deep palette
        deep_palette = sns.color_palette("deep", n_colors=n_folds)

        # Create the plot
        plt.figure(figsize=(graph_width_in_inches, graph_height_in_inches + 2))

        for fold_idx in range(n_folds):
            fpr, tpr, _ = roc_curve(all_labels[fold_idx], all_probs[fold_idx])
            roc_auc = auc(fpr, tpr)
            aucs.append(float(roc_auc))  # Ensure AUC is a Python float

            # Interpolate TPR to mean FPR
            tprs.append(
                np.interp(mean_fpr, fpr, tpr).tolist()
            )  # Convert to list for JSON compatibility
            tprs[-1][0] = 0.0

            plt.plot(
                fpr,
                tpr,
                lw=1.0,
                alpha=0.7,
                color=deep_palette[fold_idx],
                # label=rf"\textnormal{{Skupina {fold_idx + 1} (AUC = {roc_auc:.4f})}}",
            )

        # Compute mean and std TPR
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(auc(mean_fpr, mean_tpr))  # Ensure Python float
        std_auc = float(np.std(aucs))  # Ensure Python float

        # Plot the mean ROC curve with a two-line label
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=deep_palette[0],
            lw=2.5,
            # label=(
            #     r"\textnormal{Průměrná ROC křivka}\hfill\\"
            #     r"\hfill\textnormal{(AUC = %.4f ± %.4f)}" % (mean_auc, std_auc)
            # ),
            label=r"\textnormal{Průměrná ROC křivka (AUC = %.4f ± %.4f)}"
            % (mean_auc, std_auc),
        )

        # Plot the random chance line
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            lw=1.0,
            label=r"\textnormal{Náhodný klasifikátor}",
        )

        # Customize plot
        # if not title:
        #     title = r"\textnormal{ROC křivky pro všechny skupiny}"
        plt.title(title, fontsize=10)
        plt.xlabel(r"\textnormal{1 − Specificita}", fontsize=10)
        plt.ylabel(r"\textnormal{Sensitivita}", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        sns.despine()

        # Move legend below the plot and center it
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            frameon=False,
            prop={"size": 8, "family": "serif"},
        )

        # Gridlines
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout to account for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.85])

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

        # Save ROC data and input data if save_roc_data_path is provided
        if save_roc_data_path:
            # Ensure all data is JSON-serializable
            roc_data = {
                "all_labels": [
                    [int(label) for label in fold] for fold in all_labels
                ],  # Convert to Python int
                "all_probs": [
                    [float(prob) for prob in fold] for fold in all_probs
                ],  # Convert to Python float
                "n_folds": int(n_folds),  # Convert to Python int
                "mean_fpr": mean_fpr.tolist(),  # Convert to list
                "mean_tpr": mean_tpr.tolist(),  # Convert to list
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "fold_aucs": aucs,
            }
            with open(save_roc_data_path, "w") as f:
                json.dump(roc_data, f, indent=4)
            print(f"ROC data saved to {save_roc_data_path}")

        # Show the plot
        plt.show()

    @staticmethod
    def plot_accuracy_boxplots_by_snr(
        accuracy_data, snr_values, save_path=None, title=None
    ):
        """
        Create a grouped boxplot with SNR on the x-axis and accuracies on the y-axis,
        showing groups (Epileptic vs Healthy) separately.

        Args:
            accuracy_data (dict): Dictionary where keys are SNR values and values are dictionaries:
                                {"Epileptic": [...], "Healthy": [...]}.
            snr_values (list): List of SNR values in the desired order for the x-axis.
            save_path (str): Path to save the plot as a PDF. Defaults to None.
            title (str): Title of the plot. Defaults to None.
        """
        # Convert dimensions to inches (13 cm x 13 cm for the graph)
        graph_width_in_inches = 13 / 2.54  # 5.12 inches
        graph_height_in_inches = 13 / 2.54  # 5.12 inches

        # Prepare the data for plotting
        plot_data = []
        for snr in snr_values:
            for acc in accuracy_data.get(snr, {}).get("Epileptic", []):
                plot_data.append(
                    {"SNR": f"SNR = {snr}", "Accuracy": acc, "Group": "Epileptic"}
                )
            for acc in accuracy_data.get(snr, {}).get("Healthy", []):
                plot_data.append(
                    {"SNR": f"SNR = {snr}", "Accuracy": acc, "Group": "Healthy"}
                )

        # Convert to DataFrame
        df = pd.DataFrame(plot_data)

        # Create the plot
        plt.figure(figsize=(graph_width_in_inches, graph_height_in_inches))
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        # Boxplot
        ax = sns.boxplot(
            x="SNR", y="Accuracy", hue="Group", data=df, palette="deep", showfliers=True
        )

        # Customize plot
        if not title:
            title = r"\textnormal{Přesnost napříč hodnotami SNR}"
        plt.title(title, fontsize=10)
        plt.xlabel(r"\textnormal{SNR}", fontsize=8)
        plt.ylabel(r"\textnormal{Přesnost [-]}", fontsize=8)
        plt.xticks(fontsize=8, rotation=45)  # Rotate x-axis labels for readability
        plt.yticks(fontsize=8)

        sns.despine()

        # Gridlines
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)

        # Show the plot
        plt.show()

    @staticmethod
    def plot_metrics_across_snr(
        sensitivity_stats_dict,
        specificity_stats_dict,
        fnr_stats_dict,
        fpr_stats_dict,
        snr_values,
        title: str = r"\textnormal{Porovnání metrik přesnosti a chybovosti pro různá SNR}",
        save_path: str = None,
        save_data_path: str = None,
    ):
        """
        Plot sensitivity, specificity, FNR, and FPR metrics across different SNR values.

        Args:
            sensitivity_stats_dict (dict): Dictionary mapping SNR values to sensitivity stats.
            specificity_stats_dict (dict): Dictionary mapping SNR values to specificity stats.
            fnr_stats_dict (dict): Dictionary mapping SNR values to FNR stats.
            fpr_stats_dict (dict): Dictionary mapping SNR values to FPR stats.
            snr_values (list): List of SNR values.
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF.
            save_data_path (str): Path to save metrics statistics as a JSON file.

        Returns:
            None
        """
        # Convert dimensions to inches (13 cm x 9 cm)
        width_in_inches = 13.5 / 2.54
        height_in_inches = 10 / 2.54

        # Prepare the data
        combined_data = []
        for snr in snr_values:
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{Sensitivita}",
                        "Value": val,
                    }
                    for val in sensitivity_stats_dict[snr]
                ]
            )
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{Specificita}",
                        "Value": val,
                    }
                    for val in specificity_stats_dict[snr]
                ]
            )
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{FNR}",
                        "Value": val,
                    }
                    for val in fnr_stats_dict[snr]
                ]
            )
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{FPR}",
                        "Value": val,
                    }
                    for val in fpr_stats_dict[snr]
                ]
            )

        # Convert to DataFrame
        data = pd.DataFrame(combined_data)

        # Define specific colors
        deep_palette = sns.color_palette("deep")
        metric_colors = {
            r"\textnormal{Sensitivita}": deep_palette[4],  # Deep Purple
            r"\textnormal{Specificita}": deep_palette[0],  # Deep Blue
            r"\textnormal{FNR}": deep_palette[6],  # Light Purple
            r"\textnormal{FPR}": deep_palette[-1],  # Light Blue
        }

        # Set Seaborn theme
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        # Plot
        plt.figure(figsize=(width_in_inches, height_in_inches))
        ax = sns.boxplot(
            x="SNR",
            y="Value",
            hue="Metric",
            data=data,
            palette=metric_colors,
            showfliers=True,
        )

        # Customizations
        plt.title(title, fontsize=10)
        plt.ylabel(r"\textnormal{Hodnota metriky [-]}", fontsize=10)
        plt.xlabel(r"\textnormal{SNR}", fontsize=10)

        # Apply LaTeX-style x-axis tick labels
        ax.set_xticklabels(
            [
                r"$\textnormal{" + str(label.get_text()) + r"}$"
                for label in ax.get_xticklabels()
            ],
            fontsize=8,
        )
        plt.yticks(fontsize=8)

        sns.despine()

        # Adjust legend
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=4,
            title=None,
            frameon=False,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot if save_path is provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")

        # Show the plot
        plt.show()

        # Save metrics data if save_data_path is provided
        if save_data_path:
            metrics_data = {
                "SNR Values": snr_values,
                "Metrics": {
                    "Sensitivity": sensitivity_stats_dict,
                    "Specificity": specificity_stats_dict,
                    "FNR": fnr_stats_dict,
                    "FPR": fpr_stats_dict,
                },
            }
            with open(save_data_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
            print(f"Data saved to {save_data_path}")

    @staticmethod
    def plot_accuracy_by_location_across_snr(
        snr_patient_ids_dict,
        snr_accuracies_dict,
        snr_values,
        title: str = r"\textnormal{Sensitivita v~závislosti na lokalizaci zdroje výbojů v~mozku}",
        save_path: str = None,
    ):
        """
        Create a grouped boxplot showing accuracies by location for each SNR value.

        Args:
            snr_patient_ids_dict (dict): Dictionary mapping SNR values to patient IDs folds.
            snr_accuracies_dict (dict): Dictionary mapping SNR values to accuracies folds.
            snr_values (list): List of SNR values.
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF (optional).
        """
        import re
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Mapping of abbreviations to desired names with LaTeX-style \textnormal
        location_mapping = {
            "FL": r"\textnormal{Čelní lalok}",
            "PL": r"\textnormal{Temenní lalok}",
            "OL": r"\textnormal{Týlní lalok}",
            "TL": r"\textnormal{Spánkový lalok}",
            "In": r"\textnormal{Insula}",
        }

        def extract_location_from_id(patient_id):
            # Match only allowed locations: FL, PL, OL, TL, In
            match = re.search(r"Loc(FL|PL|OL|TL|In)", patient_id)
            if match:
                return match.group(1)
            return None

        # Prepare the data
        combined_data = []
        for snr in snr_values:
            patient_ids_folds = snr_patient_ids_dict[snr]
            accuracies_folds = snr_accuracies_dict[snr]
            for fold_idx, (patient_ids, accuracies) in enumerate(
                zip(patient_ids_folds, accuracies_folds)
            ):
                for pid, acc in zip(patient_ids, accuracies):
                    location = extract_location_from_id(pid)
                    if location:
                        combined_data.append(
                            {"SNR": f"SNR {snr}", "Location": location, "Accuracy": acc}
                        )

        # Convert to DataFrame
        df = pd.DataFrame(combined_data)

        # Check if data is present
        if df.empty:
            print("No valid data found for the given inputs.")
            return

        # Replace the 'Location' column values using the mapping
        df["Location"] = df["Location"].map(location_mapping)

        # Define specific colors from the Seaborn deep palette
        deep_palette = sns.color_palette("deep")
        custom_colors = [
            deep_palette[0],
            deep_palette[1],
            deep_palette[2],
            deep_palette[3],
            deep_palette[4],
        ]

        # Plot
        plt.figure(
            figsize=(13.5 / 2.54, 13 / 2.54)
        )  # Convert dimensions to inches (13.5 cm x 10 cm)
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        ax = sns.boxplot(
            x="SNR",
            y="Accuracy",
            hue="Location",
            data=df,
            palette=custom_colors,
            showfliers=True,
        )

        # Update SNR labels with LaTeX style after plot creation
        snr_labels = [r"$\textnormal{" + str(snr) + r"}$" for snr in snr_values]
        ax.set_xticklabels(snr_labels, fontsize=8)

        # Customizations
        plt.title(title, fontsize=10)
        plt.xlabel(r"\textnormal{SNR}", fontsize=10)
        plt.ylabel(r"\textnormal{Sensitivita [-]}", fontsize=10)
        plt.yticks(fontsize=8)
        sns.despine()
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust legend (split into two lines)
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=3,
            frameon=False,
            title=r"\textnormal{Oblast mozku}",
            prop={"size": 8},
        )

        # Save the plot
        if save_path:
            try:
                plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_accuracy_across_snr_v2(
        sensitivity_stats_dict,
        specificity_stats_dict,
        snr_values,
        title: str = r"\textnormal{Porovnání metrik přesnosti pro různé úrovně SNR}",
        save_path: str = None,
        save_data_path: str = None,
    ):
        """
        Plot sensitivity and specificity metrics across different SNR values.

        Args:
            sensitivity_stats_dict (dict): Dictionary mapping SNR values to sensitivity stats.
            specificity_stats_dict (dict): Dictionary mapping SNR values to specificity stats.
            snr_values (list): List of SNR values.
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF.
            save_data_path (str): Path to save metrics statistics as a JSON file.

        Returns:
            None
        """
        # Convert dimensions to inches (13 cm x 9 cm)
        width_in_inches = 13.5 / 2.54
        height_in_inches = 10 / 2.54

        # Prepare the data
        combined_data = []
        for snr in snr_values:
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{Sensitivita}",
                        "Value": val,
                    }
                    for val in sensitivity_stats_dict[snr]
                ]
            )
            combined_data.extend(
                [
                    {
                        "SNR": r"\textnormal{" + f"{snr}" + r"}",
                        "Metric": r"\textnormal{Specificita}",
                        "Value": val,
                    }
                    for val in specificity_stats_dict[snr]
                ]
            )

        # Convert to DataFrame
        data = pd.DataFrame(combined_data)

        # Define specific colors
        deep_palette = sns.color_palette("deep")
        metric_colors = {
            r"\textnormal{Sensitivita}": deep_palette[4],  # Deep Purple
            r"\textnormal{Specificita}": deep_palette[0],  # Deep Blue
        }

        # Set Seaborn theme
        sns.set_theme(style="whitegrid")
        sns.set_context("paper")

        # Plot
        plt.figure(figsize=(width_in_inches, height_in_inches))
        ax = sns.boxplot(
            x="SNR",
            y="Value",
            hue="Metric",
            data=data,
            palette=metric_colors,
            showfliers=True,
        )

        # Customizations
        plt.title(title, fontsize=10)
        plt.ylabel(r"\textnormal{Hodnota metriky [-]}", fontsize=10)
        plt.xlabel(r"\textnormal{SNR}", fontsize=10)

        # Apply LaTeX-style x-axis tick labels
        ax.set_xticklabels(
            [
                r"$\textnormal{" + str(label.get_text()) + r"}$"
                for label in ax.get_xticklabels()
            ],
            fontsize=8,
        )
        plt.yticks(fontsize=8)

        sns.despine()

        # Adjust legend
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            title=None,
            frameon=False,
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save the plot if save_path is provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")

        # Show the plot
        plt.show()

        # Save metrics data if save_data_path is provided
        if save_data_path:
            metrics_data = {
                "SNR Values": snr_values,
                "Metrics": {
                    "Sensitivity": sensitivity_stats_dict,
                    "Specificity": specificity_stats_dict,
                },
            }
            with open(save_data_path, "w") as f:
                json.dump(metrics_data, f, indent=4)
            print(f"Data saved to {save_data_path}")

    @staticmethod
    def plot_mean_roc_curves_for_snrs(
        roc_results: Dict[int, Any],
        title: str = r"\textnormal{ROC křivky pro různé SNR úrovně}",
        save_path=None,
    ):
        """
        Plot mean ROC curves for multiple SNR levels in a single graph.

        Args:
            roc_results (dict): Dictionary where keys are SNR levels and values are
                                dicts containing "mean_fpr", "mean_tpr", "mean_auc", and "std_auc".
            title (str): Title of the plot.
            save_path (str): Path to save the plot as a PDF. Defaults to None.

        Returns:
            None
        """
        import seaborn as sns

        sns.set_theme(style="whitegrid")  # Apply whitegrid style
        sns.set_context("paper")

        # Convert dimensions to inches (13.5 cm width x 13 cm height)
        graph_width_in_inches = 13.5 / 2.54  # 5.31 inches
        graph_height_in_inches = 13 / 2.54  # 3.94 inches

        # Generate colors for the SNR levels
        deep_palette = sns.color_palette("deep", n_colors=len(roc_results))

        # Create the plot
        plt.figure(figsize=(graph_width_in_inches, graph_height_in_inches + 2))

        for idx, (snr, roc_data) in enumerate(roc_results.items()):
            mean_fpr = roc_data["mean_fpr"]
            mean_tpr = roc_data["mean_tpr"]
            mean_auc = roc_data["mean_auc"]
            std_auc = roc_data["std_auc"]

            plt.plot(
                mean_fpr,
                mean_tpr,
                lw=2.0,
                color=deep_palette[idx],
                label=rf"\textnormal{{SNR {snr} (AUC = {mean_auc:.4f})}}",
            )

        # Plot the random chance line
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            lw=1.0,
            label=r"\textnormal{Náhodný klasifikátor}",
        )

        # Customize the plot
        plt.title(title, fontsize=10)
        plt.xlabel(r"\textnormal{1 − Specificita}", fontsize=10)
        plt.ylabel(r"\textnormal{Sensitivita}", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        sns.despine()

        # Move legend below the plot and center it
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
            frameon=False,
            prop={"size": 8, "family": "serif"},
        )

        # Gridlines
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Adjust layout to account for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.85])

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")

        # Show the plot
        plt.show()
