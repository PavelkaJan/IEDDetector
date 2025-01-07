from src.logging_config.logging_setup import setup_logging
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import (
    process_patients,
    load_and_process_4s_epochs,
)
from pathlib import Path

setup_logging()

main_folder = Path("D:/DIPLOMKA_DATASET")
channels_from_BS_file_name = "channel.mat"
original_fs = 128


patient_ids = [
    "YoungAdultsSubject01",
    "YoungAdultsSubject02",
    "YoungAdultsSubject03",
    "YoungAdultsSubject04",
    "YoungAdultsSubject05",
    "YoungAdultsSubject06",
    "YoungAdultsSubject07",
    "YoungAdultsSubject08",
    "YoungAdultsSubject09",
    "YoungAdultsSubject10",
    "YoungAdultsSubject11",
    "YoungAdultsSubject12",
    "YoungAdultsSubject13",
    "YoungAdultsSubject14",
    "YoungAdultsSubject16",
    "YoungAdultsSubject17",
    "YoungAdultsSubject18",
    "YoungAdultsSubject19",
    "YoungAdultsSubject20",
    "YoungAdultsSubject21",
    "YoungAdultsSubject22",
    "YoungAdultsSubject23",
    "YoungAdultsSubject24",
    "YoungAdultsSubject25",
    "YoungAdultsSubject26",
    "YoungAdultsSubject27",
    "YoungAdultsSubject28",
    # "YoungAdultsSubject29",
    # "YoungAdultsSubject30",
    "YoungAdultsSubject31",
    # "YoungAdultsSubject32",
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
