from src.logging_config.logging_setup import setup_logging
from src.plotting.eeg_plotter import EEGPlotter
from pprint import pprint
from enum import Enum
import numpy as np
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import (
    process_patients,
    list_files_in_folder,
    process_4s_epochs,
    list_mat_files,
    load_and_process_4s_epochs,
)
from pathlib import Path

setup_logging()


main_folder = Path("D:/DIPLOMKA_DATASET")
channels_from_BS_file_name = "channel_10-20_19.mat"
original_fs = 256


TemplateP143 = [
    "SimulatedTemplateP143Ver1LocFL1SuperficialSNR0",
    "SimulatedTemplateP143Ver2LocFL2SuperficialSNR0",
    "SimulatedTemplateP143Ver3LocPL1SuperficialSNR0",
    "SimulatedTemplateP143Ver4LocOL10DeepSNR0",
    "SimulatedTemplateP143Ver5LocTL1SuperificialSNR0",
    "SimulatedTemplateP143Ver6LocIn10DeepSNR0",
]

TemplateP249 = [
    "SimulatedTemplateP249Ver1LocFL3SuperficialSNR0",
    "SimulatedTemplateP249Ver2LocFL15DeepSNR0",
    "SimulatedTemplateP249Ver3LocPL2SuperficialSNR0",
    "SimulatedTemplateP249Ver4LocOL1SuperficialSNR0",
    "SimulatedTemplateP249Ver5LocTL2SuperficialSNR0",
    "SimulatedTemplateP249Ver6LocIn1DeepSNR0",
]

TemplateP310 = [
    "SimulatedTemplateP310Ver1LocFL4SuperficialSNR0",
    "SimulatedTemplateP310Ver2LocFL5SuperficialSNR0",
    "SimulatedTemplateP310Ver3LocPL9DeepSNR0",
    "SimulatedTemplateP310Ver4LocOL2SuperficialSNR0",
    "SimulatedTemplateP310Ver5LocTL3SuperficialSNR0",
    "SimulatedTemplateP310Ver6LocIn2DeepSNR0",
]

TemplateP311 = [
    "SimulatedTemplateP311Ver1LocFL6SuperficialSNR0",
    "SimulatedTemplateP311Ver2LocFL7SuperficialSNR0",
    "SimulatedTemplateP311Ver3LocPL3SuperficialSNR0",
    "SimulatedTemplateP311Ver4LocOL9DeepSNR0",
    "SimulatedTemplateP311Ver5LocTL4SuperficialSNR0",
    "SimulatedTemplateP311Ver6LocIn3DeepSNR0",
]

TemplateP312 = [
    "SimulatedTemplateP312Ver1LocFL8SuperficialSNR0",
    "SimulatedTemplateP312Ver2LocFL16DeepSNR0",
    "SimulatedTemplateP312Ver3LocPL4SuperficialSNR0",
    "SimulatedTemplateP312Ver4LocOL3SuperficialSNR0",
    "SimulatedTemplateP312Ver5LocTL5SuperficialSNR0",
    "SimulatedTemplateP312Ver6LocIn4DeepSNR0",
]

TemplateP314 = [
    "SimulatedTemplateP314Ver1LocFL9SuperficialSNR0",
    "SimulatedTemplateP314Ver2LocFL17DeepSNR0",
    "SimulatedTemplateP314Ver3LocPL5SuperficialSNR0",
    "SimulatedTemplateP314Ver4LocOL4SuperficialSNR0",
    "SimulatedTemplateP314Ver5LocTL8DeepSNR0",
    "SimulatedTemplateP314Ver6LocIn5DeepSNR0",
]

TemplateP315 = [
    "SimulatedTemplateP315Ver1LocFL10SuperficialSNR0",
    "SimulatedTemplateP315Ver2LocFL18DeepSNR0",
    "SimulatedTemplateP315Ver3LocPL8DeepSNR0",
    "SimulatedTemplateP315Ver4LocOL5SuperficialSNR0",
    "SimulatedTemplateP315Ver5LocTL6SuperficialSNR0",
    "SimulatedTemplateP315Ver6LocIn6DeepSNR0",
]

TemplateP317 = [
    "SimulatedTemplateP317Ver1LocFL11SuperficialSNR0",
    "SimulatedTemplateP317Ver2LocFL12SuperficialSNR0",
    "SimulatedTemplateP317Ver3LocPL6SuperficialSNR0",
    "SimulatedTemplateP317Ver4LocOL6SuperficialSNR0",
    "SimulatedTemplateP317Ver5LocTL9DeepSNR0",
    "SimulatedTemplateP317Ver6LocIn7DeepSNR0",
]

TemplateP322 = [
    "SimulatedTemplateP322Ver1LocFL13SuperficialSNR0",
    "SimulatedTemplateP322Ver2LocFL19DeepSNR0",
    "SimulatedTemplateP322Ver3LocPL7SuperficialSNR0",
    "SimulatedTemplateP322Ver4LocOL7SuperficialSNR0",
    "SimulatedTemplateP322Ver5LocTL7SuperficialSNR0",
    "SimulatedTemplateP322Ver6LocInDeepSNR0",
]

TemplateP323 = [
    "SimulatedTemplateP323Ver1LocFL14SuperficialSNR0",
    "SimulatedTemplateP323Ver2LocFL20DeepSNR0",
    "SimulatedTemplateP323Ver3LocPL10DeepSNR0",
    "SimulatedTemplateP323Ver4LocOL8DeepSNR0",
    "SimulatedTemplateP323Ver5LocTL10DeepSNR0",
    "SimulatedTemplateP323Ver6LocIn9DeepSNR0",
]


patient_ids = (
    TemplateP143
    + TemplateP249
    + TemplateP310
    + TemplateP311
    + TemplateP312
    + TemplateP314
    + TemplateP315
    + TemplateP317
    + TemplateP322
    + TemplateP323
)

patients_ids_updated = [patient.replace("SNR0", "SNR3") for patient in patient_ids]

patients_ids_updated = ["SimulatedTemplateP314Ver6LocIn5DeepSNR0"]
patients = []
for patient_id in patients_ids_updated:
    patient = Patient(
        id=patient_id,
        patient_type="epileptic_simulated",
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
