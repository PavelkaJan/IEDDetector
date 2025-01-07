from src.logging_config.logging_setup import setup_logging
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import (
    process_patients,
    load_and_process_4s_epochs,
)
from pathlib import Path

setup_logging()


main_folder = Path("MINI_DATASET")
channels_from_BS_file_name = "original_channel_names_BS.mat"
original_fs = 128


patient_ids = [
    "P143",
    "P249",
    "P315",
]
patients = []

for patient_id in patient_ids:
    patient = Patient(
        id=patient_id,
        patient_type="epileptic_real",
        base_folder_path=main_folder,
        original_channel_names_CA=Patient.get_channel_names_from_mat(
            main_folder / patient_id / channels_from_BS_file_name
        ),
        original_fs=original_fs,
    )
    patients.append(patient)


for patient in patients:
    load_and_process_4s_epochs(
        patient.epochs_BS_4s_folder_path,
        patient.epochs_BS_folder_path["IED_present"],
        original_fs,
    )

# Epileptics need folder epochs_BS/IED_absent even if is empty

process_patients(patients)
