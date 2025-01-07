from src.logging_config.logging_setup import setup_logging
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import (
    process_patients,
    load_and_process_4s_epochs,
)
from pathlib import Path


setup_logging()

main_folder = Path("D:\\DIPLOMKA_DATASET\\")
channels_from_BS_file_name = "channel_10-20_19.mat"
original_fs = 256


patient_ids = [
    "SimulatedHealthySubject21Theta",
    "SimulatedHealthySubject22Theta",
    "SimulatedHealthySubject23Theta",
    "SimulatedHealthySubject24Theta",
    "SimulatedHealthySubject25Theta",
    "SimulatedHealthySubject26Theta",
    "SimulatedHealthySubject27Theta",
    "SimulatedHealthySubject28Theta",
    "SimulatedHealthySubject29Theta",
    "SimulatedHealthySubject30Theta",
    "SimulatedHealthySubject31Theta",
    "SimulatedHealthySubject32Theta",
    "SimulatedHealthySubject33Theta",
    "SimulatedHealthySubject34Theta",
    "SimulatedHealthySubject35Theta",
    "SimulatedHealthySubject36Theta",
    "SimulatedHealthySubject37Theta",
    "SimulatedHealthySubject38Theta",
    "SimulatedHealthySubject39Theta",
    "SimulatedHealthySubject40Theta",
    "SimulatedHealthySubject41Delta",
    "SimulatedHealthySubject42Delta",
    "SimulatedHealthySubject43Delta",
    "SimulatedHealthySubject44Delta",
    "SimulatedHealthySubject45Delta",
    "SimulatedHealthySubject46Delta",
    "SimulatedHealthySubject47Delta",
    "SimulatedHealthySubject48Delta",
    "SimulatedHealthySubject49Delta",
    "SimulatedHealthySubject50Delta",
    "SimulatedHealthySubject51Delta",
    "SimulatedHealthySubject52Delta",
    "SimulatedHealthySubject53Delta",
    "SimulatedHealthySubject54Delta",
    "SimulatedHealthySubject55Delta",
    "SimulatedHealthySubject56Delta",
    "SimulatedHealthySubject57Delta",
    "SimulatedHealthySubject58Delta",
    "SimulatedHealthySubject59Delta",
    "SimulatedHealthySubject60Delta",
]


base_path = "D:\\DIPLOMKA_DATASET\\"
patients = []


for patient_id in patient_ids:
    patient = Patient(
        id=patient_id,
        patient_type="healthy_simulated",
        base_folder_path=main_folder,
        original_channel_names_CA=Patient.get_channel_names_from_mat(
            main_folder / patient_id / channels_from_BS_file_name
        ),
        original_fs=original_fs,
    )
    patients.append(patient)


for patient in patients:
    load_and_process_4s_epochs(
        patient.epochs_BS_4s_folder_path, patient.epochs_BS_folder_path, original_fs
    )

process_patients(patients)
