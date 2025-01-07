
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

# How to Run the Project?

In the folder `example_mini_dataset` there are basic files for preprocessing and also simple pipeline how to run neural network.

More detailed pipelines used in diploma thesis are in the folder `diplomka`. There are also resuls for each dataset.

Other examples are in the folder DEMOs.

For faster computation you need to instal PyTorch with cuda if you have proper external graphic card. 
https://pytorch.org/get-started/locally/