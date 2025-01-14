from src.logging_config.logging_setup import setup_logging
from src.patient.patient import Patient
from src.pipeline_functions.pipeline_process_functions import process_patients
from pathlib import Path

setup_logging()


# Set the main folder where are the subfolders for new patients to evaluate
main_folder = Path("EVALUATE_NEW_PATIENT_DEMO_DATA")
# Channel file from Brainstrom
channels_from_BS_file_name = "channel.mat"
# Patient ID for the tested patient, has to be the same as the folder name
patient_id = "P314"

# Create patient object
P314 = Patient(
    id=patient_id,
    patient_type="epileptic_simulated",  # For evaluation without true labels is better to use epileptic_simulated, epileptic_real needs two folders IED_absent and IED_present
    base_folder_path=main_folder,
    original_channel_names_CA=Patient.get_channel_names_from_mat(
        main_folder / patient_id / channels_from_BS_file_name
    ),
    original_fs=128,
)


patients = [P314]


process_patients(patients)


# Now you have processed epochs in python files and also in the three montages ready for evaluation
