from collections import defaultdict
from typing import List, Dict, Any
from src.patient.patient import Patient


class PatientStatistics:
    def __init__(self, patients: List[Patient]):
        """
        Initialize the PatientStatistics class with a list of patients.

        Args:
            patients (List[Patient]): A list of Patient objects.
        """
        self.patients = patients

    def _group_patients_by_type(self) -> Dict[str, List[Patient]]:
        """
        Group patients by their type.

        Returns:
            Dict[str, List[Patient]]: A dictionary where keys are patient types
            and values are lists of patients in each group.
        """
        patients_by_type = defaultdict(list)
        for patient in self.patients:
            patient_type = patient.patient_type.value
            patients_by_type[patient_type].append(patient)
        return dict(patients_by_type)

    def _calculate_summary_counts(
        self, patients_by_type: Dict[str, List[Patient]]
    ) -> Dict[str, int]:
        """
        Calculate the number of patients for each type.

        Args:
            patients_by_type (Dict[str, List[Patient]]): Patients grouped by type.

        Returns:
            Dict[str, int]: A dictionary where keys are patient types and values
            are the count of patients in each group.
        """
        return {
            patient_type: len(patients)
            for patient_type, patients in patients_by_type.items()
        }

    def _calculate_epoch_statistics(self) -> Dict[str, int]:
        """
        Calculate totals for raw and processed epochs across all patients.

        Returns:
            Dict[str, int]: A dictionary containing epoch statistics.
        """
        total_epochs = 0
        total_ied_epochs = 0
        total_non_ied_epochs = 0
        raw_ied_epochs = 0
        raw_non_ied_epochs = 0

        for patient in self.patients:
            if isinstance(patient.raw_epoch_count, dict):
                total_epochs += sum(patient.raw_epoch_count.values())
                raw_ied_epochs += patient.raw_epoch_count.get("IED_present", 0)
                raw_non_ied_epochs += patient.raw_epoch_count.get("IED_absent", 0)
            else:
                total_epochs += patient.raw_epoch_count
                if patient.patient_type.value == "epileptic_simulated":
                    raw_ied_epochs += patient.raw_epoch_count
                elif patient.patient_type.value in [
                    "healthy_real",
                    "healthy_simulated",
                ]:
                    raw_non_ied_epochs += patient.raw_epoch_count

            if patient.patient_type.value == "epileptic_real":
                total_ied_epochs += sum(
                    patient.processed_epoch_count_by_montage.get(
                        "IED_present", {}
                    ).values()
                    if isinstance(
                        patient.processed_epoch_count_by_montage.get("IED_present", 0),
                        dict,
                    )
                    else [
                        patient.processed_epoch_count_by_montage.get("IED_present", 0)
                    ]
                )
                total_non_ied_epochs += sum(
                    patient.processed_epoch_count_by_montage.get(
                        "IED_absent", {}
                    ).values()
                    if isinstance(
                        patient.processed_epoch_count_by_montage.get("IED_absent", 0),
                        dict,
                    )
                    else [patient.processed_epoch_count_by_montage.get("IED_absent", 0)]
                )
            elif patient.patient_type.value == "epileptic_simulated":
                total_ied_epochs += sum(
                    patient.processed_epoch_count_by_montage.values()
                    if isinstance(patient.processed_epoch_count_by_montage, dict)
                    else [patient.processed_epoch_count_by_montage]
                )
            else:
                total_non_ied_epochs += sum(
                    patient.processed_epoch_count_by_montage.values()
                    if isinstance(patient.processed_epoch_count_by_montage, dict)
                    else [patient.processed_epoch_count_by_montage]
                )

        return {
            "total_epochs": total_epochs,
            "total_ied_epochs": total_ied_epochs,
            "total_non_ied_epochs": total_non_ied_epochs,
            "raw_ied_epochs": raw_ied_epochs,
            "raw_non_ied_epochs": raw_non_ied_epochs,
        }

    def summarize(self) -> Dict[str, Any]:
        """
        Summarize patient data by grouping patients, counting them, and calculating epoch statistics.

        Returns:
            Dict[str, Any]: A dictionary containing the grouped patients, summary counts,
            and epoch statistics.
        """
        patients_by_type = self._group_patients_by_type()
        summary_counts = self._calculate_summary_counts(patients_by_type)
        epoch_stats = self._calculate_epoch_statistics()

        total_number_of_patients = sum(summary_counts.values())
        total_processed_epochs = (
            epoch_stats["total_ied_epochs"] + epoch_stats["total_non_ied_epochs"]
        )

        return {
            "patients_by_type": patients_by_type,
            "summary_counts": summary_counts,
            **epoch_stats,
            "total_number_of_patients": total_number_of_patients,
            "total_processed_epochs": total_processed_epochs,
        }
