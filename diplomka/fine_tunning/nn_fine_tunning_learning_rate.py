import torch
from src.logging_config.logging_setup import setup_logging
from src.neural_network.nn_control import NNControl
from src.neural_network.nn_fine_tunning import NNFineTunning
import os

# Configure logging
setup_logging()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Specify the main storage location
base_path = "D:\\DIPLOMKA_DATASET"

# Get patient file paths
patient_paths = NNControl.get_patient_file_paths(base_path)

# Load and split patient data
patients = NNControl.load_patients(patient_paths)
train_patients, test_patients = NNControl.split_data(patients)

# Define hyperparameters
batch_size = 64
num_epochs = 4
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Learning rates to evaluate

# Find the optimal learning rate
print("Starting learning rate evaluation...")
results = NNFineTunning.find_optimal_learning_rate(
    train_patients=train_patients,
    test_patients=test_patients,
    device=device,
    learning_rates=learning_rates,
    batch_size=batch_size,
    num_epochs=num_epochs
)

# Extract the optimal learning rate and its accuracy
optimal_learning_rate = max(results, key=results.get) 
optimal_accuracy = results[optimal_learning_rate]    

# Log the results
print(f"Optimal Learning Rate: {optimal_learning_rate}, Accuracy: {optimal_accuracy:.4f}")

# Plot the results
NNFineTunning.plot_learning_rate_results(results)
