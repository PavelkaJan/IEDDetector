"""
Functions only for diplomka. Used patients group heatlhy, full_of_spikes, sometimes_spikes, rare_spikes.
"""

import random
import logging
from src.constants import PatientType
from src.logging_config.logging_setup import setup_logging
from collections import defaultdict
from typing import List, Tuple

setup_logging()
logger = logging.getLogger(__name__)


class NNDiplomkaFunctions:
    @staticmethod
    def generate_full_coverage_cv_splits(patients_dict, n_splits=10):
        """
        Generates balanced cross-validation splits with full patient coverage.

        Args:
            patients_dict (dict): A dictionary mapping patient groups (keys) to patient objects (values).
            n_splits (int): Number of folds for cross-validation.

        Returns:
            list: A list of tuples, where each tuple contains a training set and a testing set.
        """
        healthy = patients_dict["healthy"]
        groups = {
            "full_of_spikes": patients_dict["full_of_spikes"],
            "sometimes_spikes": patients_dict["sometimes_spikes"],
            "rare_spikes": patients_dict["rare_spikes"],
        }
        splits = []
        random.shuffle(healthy)

        # Divide healthy patients into n_splits equal parts
        num_healthy_per_fold = len(healthy) // n_splits
        healthy_folds = [
            healthy[i * num_healthy_per_fold : (i + 1) * num_healthy_per_fold]
            for i in range(n_splits)
        ]
        # Handle remaining healthy patients
        remaining_healthy = healthy[n_splits * num_healthy_per_fold :]
        for idx, patient in enumerate(remaining_healthy):
            healthy_folds[idx].append(patient)

        # Prepare rotations for epileptic groups
        group_rotations = {}
        for group_name, group_patients in groups.items():
            rotation = group_patients * ((n_splits // len(group_patients)) + 1)
            random.shuffle(rotation)
            group_rotations[group_name] = rotation

        # Generate splits
        for i in range(n_splits):
            # Test set
            test_set = healthy_folds[i] + [
                group_rotations["full_of_spikes"][i],
                group_rotations["sometimes_spikes"][i],
                group_rotations["rare_spikes"][i],
            ]

            # Train set
            train_set = [
                patient for patient in healthy if patient not in healthy_folds[i]
            ]
            for group_name, group_patients in groups.items():
                train_set += [
                    patient
                    for patient in group_patients
                    if patient not in [group_rotations[group_name][i]]
                ]

            splits.append((train_set, test_set))

        return splits

    @staticmethod
    def categorize_patients_by_diplomka_category(patients):
        """
        Categorize patients into healthy and epileptic groups based on their IDs and type.
        """
        healthy_patients = []
        full_of_spikes = []
        sometimes_spikes = []
        rare_spikes = []

        for patient in patients:
            if patient.patient_type == PatientType.HEALTHY_REAL:
                healthy_patients.append(patient)
            elif patient.patient_type == PatientType.EPILEPTIC_REAL:
                if patient.id in ["P143", "P249", "P310", "P322"]:
                    full_of_spikes.append(patient)
                elif patient.id in ["P314", "P315", "P317"]:
                    sometimes_spikes.append(patient)
                elif patient.id in ["P311", "P312", "P323"]:
                    rare_spikes.append(patient)
                else:
                    logger.warning(
                        f"Epileptic patient {patient.id} not assigned to any group."
                    )
            else:
                logger.warning(
                    f"Patient {patient.id} has unknown patient type: {patient.patient_type}"
                )

        return {
            "healthy": healthy_patients,
            "full_of_spikes": full_of_spikes,
            "sometimes_spikes": sometimes_spikes,
            "rare_spikes": rare_spikes,
        }

    @staticmethod
    def create_cross_validation_splits_with_constraints(
        patients: List[object], n_splits: int = 10
    ) -> List[Tuple[List[object], List[object]]]:
        """
        Splits patients into n-fold cross-validation ensuring each SimulatedTemplate (all versions)
        appears in the test set only once across all folds.

        Args:
            patients (list): List of patient objects.
            n_splits (int): Number of folds (default is 10).

        Returns:
            list: A list of tuples (train_set, test_set) for each fold, where each set contains patient objects.
        """
        # Group SimulatedTemplates by their base identifier
        grouped_patients = defaultdict(list)
        healthy_patients = []

        for patient in patients:
            if patient.id.startswith("SimulatedTemplate"):
                base_id = patient.id.split("Ver")[
                    0
                ]  # Extract base identifier (e.g., "SimulatedTemplateP143")
                grouped_patients[base_id].append(patient)
            else:
                healthy_patients.append(patient)

        # Create a list of grouped patient keys
        simulated_template_keys = list(grouped_patients.keys())
        random.shuffle(simulated_template_keys)
        random.shuffle(healthy_patients)

        # Distribute simulated templates across folds
        test_folds = [[] for _ in range(n_splits)]
        for idx, key in enumerate(simulated_template_keys):
            test_folds[idx % n_splits].extend(grouped_patients[key])

        # Distribute healthy patients across folds
        for idx, patient in enumerate(healthy_patients):
            test_folds[idx % n_splits].append(patient)

        # Create train-test splits
        cross_validation_splits = []
        for i in range(n_splits):
            test_set = test_folds[i]
            train_set = [
                patient for fold in test_folds if fold != test_set for patient in fold
            ]
            cross_validation_splits.append((train_set, test_set))

        return cross_validation_splits
