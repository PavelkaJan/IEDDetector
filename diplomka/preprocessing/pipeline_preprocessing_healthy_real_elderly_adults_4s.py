from src.logging_config.logging_setup import setup_logging
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import (
    process_patients,
    load_and_process_4s_epochs,
)
from pathlib import Path

setup_logging()

main_folder = Path("D:\\DIPLOMKA_DATASET\\")
channels_from_BS_file_name = "channel.mat"
original_fs = 128


patient_ids = [
    "ElderlyAdultsSubject40",
    "ElderlyAdultsSubject41",
    "ElderlyAdultsSubject43",
    "ElderlyAdultsSubject44",
    "ElderlyAdultsSubject45",
    "ElderlyAdultsSubject46",
    "ElderlyAdultsSubject47",
    "ElderlyAdultsSubject48",
    "ElderlyAdultsSubject49",
    "ElderlyAdultsSubject50",
    "ElderlyAdultsSubject51",
    "ElderlyAdultsSubject54",
    "ElderlyAdultsSubject55",
    "ElderlyAdultsSubject57",
    "ElderlyAdultsSubject58",
    "ElderlyAdultsSubject59",
    "ElderlyAdultsSubject60",
    "ElderlyAdultsSubject62",
    "ElderlyAdultsSubject63",
    "ElderlyAdultsSubject64",
    "ElderlyAdultsSubject65",
    "ElderlyAdultsSubject66",
    "ElderlyAdultsSubject67",
    "ElderlyAdultsSubject68",
    "ElderlyAdultsSubject70",
    # "ElderlyAdultsSubject71",
    # "ElderlyAdultsSubject73",
]


base_path = "D:\\DIPLOMKA_DATASET\\"
patients = []


for patient_id in patient_ids:
    patient = Patient(
        id=patient_id,
        patient_type="healthy_real",
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
