
# Project Setup Instructions

## 1. Create a Virtual Environment

### Windows
#### a. PowerShell:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### b. Command Prompt (CMD):
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Requirements

To install the required dependencies, run the following command after activating the virtual environment:
```bash
pip install -r requirements.txt
```

## 3. Install `src` Folder as a Package

To install a `src` folder as a package, use the following command:
```bash
pip install -e .
```

# How to run the project?

In the folder `example_mini_dataset` there are basic files for preprocessing and also simple pipeline how to run neural network.

More detailed pipelines used in diploma thesis are in the folder `diplomka`. There are also resuls for each dataset.

Other examples are in the folder DEMOs.

For faster computation you need to instal PyTorch with cuda if you have proper external graphic card. It is necessary to install it inside the venv.

https://pytorch.org/get-started/locally/

# How to run the example_mini_dataset?

All epochs from Brainstorm are stored in the MINI_DATASET folder. The provided `.pkl` files serve only as examples, the stored files are not actually uploaded. To generate proper `.pkl` files, you first need to run the preprocessing scripts located in `example_mini_dataset/preprocessing` for real epileptic and healthy patient data.

Once the preprocessing is complete, you can run the script `example_mini_dataset/simple_neural_network.py` to generate a sample report.

# How to run the example_evaluate_new_eeg?
To use the trained models on new data, you first need to prepare the EEG data in Brainstorm in a 2-second, non-overlapping format. Detailed instructions can be found in the file `cookbook_find_IED_in_new_EEG.pdf`. Example files are stored in the folder `EVALUATE_NEW_PATIENT_DEMO_DATA`.

After preparing the data in Brainstorm and setting up the required folder structure, use the script `example_evaluate_new_eeg/prepare_new_eeg_for_evaluation.py` to preprocess the data. Once preprocessing is complete, you can classify the epochs using the script `example_evaluate_new_eeg/evaluate_new_patient.py`.