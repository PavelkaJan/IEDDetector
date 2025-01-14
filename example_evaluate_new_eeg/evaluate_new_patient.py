from src.logging_config.logging_setup import setup_logging
from src.neural_network.epoch_classifier import EpochClassifier
from src.neural_network.nn_control import NNControl


setup_logging()


# Specify the new epochs that were not included in the model's training. You can choose the montage CA, DB or SD
epoch_folder_path = "EVALUATE_NEW_PATIENT_DEMO_DATA/P314/epochs_PY/CA"
# Choose the model that you want to usu
model_save_path = "fold_2_best_model.pth"

# Optional if you have true labels
# label_file_path = "D:\DIPLOMKA_PART_2\ElderlyAdultsSubject71_SpikeRate10\labels\labels_CA.npy" # Optional: Path to true labels (e.g., "labels/labels_CA.npy")


classifier = EpochClassifier(model_path=model_save_path, device="cuda")
classifier.load_model(NNControl.initialize_model)
epochs_tensor, epoch_files = classifier.load_epochs(epoch_folder_path)
predicted_labels, probabilities = classifier.classify_epochs(epochs_tensor)

# Handle labels and calculate accuracy if provided
# true_labels, accuracy = classifier.handle_labels(label_file_path, predicted_labels)

results_file = "epoch_results_with_accuracy.csv"
classifier.save_results(results_file, epoch_files, predicted_labels, probabilities)
