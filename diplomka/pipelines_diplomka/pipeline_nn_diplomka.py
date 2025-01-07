"""
Simple pipeline for testing purposes.
"""

import torch
from src.logging_config.logging_setup import setup_logging
from src.neural_network.nn_control import NNControl
from src.report.report import Report
from src.neural_network.nn_evaluation_metrics import NNEvaluationMetrics
from sklearn.metrics import auc
import logging

setup_logging()

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 0.01
num_epochs = 2

# Specify the main storage location
base_path = "D:\\DIPLOMKA_DATASET"
model_save_path = "saved_model.pth"


patient_paths = NNControl.get_patient_file_paths(base_path)
patients = NNControl.load_patients(patient_paths)
train_patients, test_patients = NNControl.split_data(patients)
# train_patients, test_patients = NNControl.split_data_according_to_predefined_list(patients, prepared_training_dataset, prepared_testing_dataset)

train_loader, test_loader = NNControl.prepare_data_loaders(
    train_patients, test_patients, batch_size=batch_size
)

model, criterion, optimizer = NNControl.initialize_model(
    device=device, learning_rate=learning_rate
)

train_losses, train_accuracies, test_losses, test_accuracies = NNControl.train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
)

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

test_loss, all_test_labels, all_test_probs, patient_metrics = (
    NNControl.evaluate_with_patient_info(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
)

logger.info(f"Overall Test Loss: {test_loss:.4f}")
for patient_id, stats in patient_metrics.items():
    print(f"Patient {patient_id}: Accuracy = {stats['accuracy']:.2f}")

# Prepare NN inputs for the report
fpr, tpr, thresholds = NNEvaluationMetrics.compute_roc_curve(
    all_test_labels, all_test_probs
)
roc_auc = auc(fpr, tpr)

nn_report_inputs = {
    "all_test_labels": all_test_labels,
    "all_test_probs": all_test_probs,
    "train_losses": train_losses,
    "test_losses": test_losses,
    "train_accuracies": train_accuracies,
    "test_accuracies": test_accuracies,
    "roc_auc": roc_auc,
}

# Generate and save the HTML report
report = Report(patients, "patient_report.html")
report.generate_and_save(nn_report_inputs=nn_report_inputs)
