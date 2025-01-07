import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import os
from src.patient.patient import Patient
from src.neural_network.vgg16 import VGG16
from src.neural_network.eeg_dataset import EEGDataset
from src.neural_network.nn_functions import validate_model, train_model
from src.constants import PatientType
from typing import List, Dict

logger = logging.getLogger(__name__)


class NNControl:
    def get_patient_file_paths(
        base_path: str, file_suffix: str = "_instance.pkl"
    ) -> List[str]:
        """
        Get a list of full file paths for directories in the base path with a specific file suffix.

        Args:
            base_path (str): The main directory containing folders with patient files.
            file_suffix (str): The suffix of the files to include. Default is "_instance.pkl".

        Returns:
            List[str]: A list of full file paths matching the specified suffix.
        """
        folders_with_files = []

        # Iterate through all directories in the base path
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)

            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, f"{folder_name}{file_suffix}")
                folders_with_files.append(file_path)

        return folders_with_files

    @staticmethod
    def load_patients(patient_paths: List[str]) -> List[Patient]:
        """Load patient instances from file paths."""
        logger.info("Loading patient instances from file paths.")
        return [Patient.load_patient_instance(path) for path in patient_paths]

    @staticmethod
    def split_data(
        patients: List[Patient], test_size: float = 0.3, random_state: int = 42
    ) -> Tuple[List[Patient], List[Patient]]:
        """Split patients into training and testing sets."""
        logger.info("Splitting data into training and testing sets.")
        return train_test_split(
            patients, test_size=test_size, random_state=random_state
        )

    @staticmethod
    def split_data_according_to_predefined_list(
        patients: List[Patient],
        train_patient_ids: List[str],
        test_patient_ids: List[str],
    ) -> Tuple[List[Patient], List[Patient]]:
        """
        Splits the list of patients into training and testing sets based on predefined lists of patient IDs.

        Args:
            patients (List[Patient]): List of all Patient instances.
            train_patient_ids (List[str]): List of patient IDs to include in the training set.
            test_patient_ids (List[str]): List of patient IDs to include in the testing set.

        Returns:
            Tuple[List[Patient], List[Patient]]:
                - List of Patient instances for training.
                - List of Patient instances for testing.
        """
        train_patients = [
            patient for patient in patients if patient.id in train_patient_ids
        ]
        test_patients = [
            patient for patient in patients if patient.id in test_patient_ids
        ]

        return train_patients, test_patients

    @staticmethod
    def prepare_data_loaders(
        train_patients: List[Patient],
        test_patients: List[Patient],
        batch_size: int = 32,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create DataLoader instances for training and testing."""
        logger.info("Preparing DataLoader instances for training and testing.")
        train_dataset = EEGDataset(train_patients)
        test_dataset = EEGDataset(test_patients)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    @staticmethod
    def initialize_model(
        device: torch.device, learning_rate: float = 0.001
    ) -> Tuple[nn.Module, nn.CrossEntropyLoss, optim.Optimizer]:
        """Initialize the model, criterion, and optimizer."""
        logger.info("Initializing model, criterion, and optimizer.")
        model = VGG16().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return model, criterion, optimizer

    @staticmethod
    def train(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int = 10,
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Train the model over multiple epochs and evaluate on the test set after each epoch.

        Returns:
            Tuple:
                - Training losses (List[float])
                - Training accuracies (List[float])
                - Testing losses (List[float])
                - Testing accuracies (List[float])
        """
        logger.info("Starting training process.")
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        for epoch in range(num_epochs):
            # Training Phase
            model.train()
            train_loss, train_acc = train_model(
                model, train_loader, criterion, optimizer
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Testing Phase
            model.eval()
            with torch.no_grad():
                test_loss, test_acc, _, _ = validate_model(
                    model, test_loader, criterion
                )
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)

            # Log metrics for this epoch in a single line
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

        logger.info("Training process completed.")
        return train_losses, train_accuracies, test_losses, test_accuracies

    @staticmethod
    def train_with_early_stopping(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int = 100,
        patience: int = 5,
        save_best_model: bool = True,
        best_model_path: str = "best_model.pth",
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Train the model over multiple epochs with early stopping and evaluate on the test set.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            criterion (nn.Module): Loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            device (torch.device): Device to run training (CPU or GPU).
            num_epochs (int): Maximum number of epochs to train.
            patience (int): Number of epochs to wait for improvement before stopping early.
            save_best_model (bool): Whether to save the best model during training.
            best_model_path (str): Path to save the best model.

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]:
                - Training losses per epoch
                - Training accuracies per epoch
                - Testing losses per epoch
                - Testing accuracies per epoch
        """
        logger.info("Starting training process with early stopping.")
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []

        best_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training Phase
            model.train()
            train_loss, train_acc = train_model(
                model, train_loader, criterion, optimizer
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Validation Phase
            model.eval()
            with torch.no_grad():
                test_loss, test_acc, _, _ = validate_model(
                    model, test_loader, criterion
                )
                test_losses.append(test_loss)
                test_accuracies.append(test_acc)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )

            # Early Stopping Check
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict()  # Save best model state
                if save_best_model:
                    torch.save(best_model_state, best_model_path)
                    logger.info(f"Best model saved to {best_model_path}")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement} epoch(s). "
                    f"Best Loss: {best_loss:.4f}"
                )

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered.")
                break

        # Restore the best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored the best model state.")

        logger.info("Training process completed.")
        return train_losses, train_accuracies, test_losses, test_accuracies

    @staticmethod
    def evaluate(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, List[int], List[float]]:
        """
        Evaluate the model on the test set and return average test loss,
        true labels, and predicted probabilities.

        Args:
            model (nn.Module): Trained model to evaluate.
            test_loader (DataLoader): DataLoader for the test dataset.
            criterion (nn.Module): Loss function.
            device (torch.device): Device to use for evaluation (CPU or GPU).

        Returns:
            Tuple[float, List[int], List[float]]: Test loss, true labels, and predicted probabilities.
        """
        logger.info("Starting evaluation on the test set.")
        model.eval()

        total_loss = 0.0
        all_test_probs = []
        all_test_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Collect predictions and true labels
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_test_probs.extend(probs.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        logger.info(f"Evaluation completed. Average test Loss: {avg_loss:.4f}")

        return avg_loss, all_test_labels, all_test_probs

    @staticmethod
    def evaluate_with_patient_info(
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, List[int], List[float], dict]:
        """
        Evaluate the model on the test set and calculate detailed metrics for each patient.

        Args:
            model (nn.Module): Trained model to evaluate.
            test_loader (DataLoader): DataLoader for the test dataset.
            criterion (nn.Module): Loss function.
            device (torch.device): Device to use for evaluation (CPU or GPU).

        Returns:
            Tuple:
                - Average test loss
                - True labels for all samples
                - Predicted probabilities for all samples
                - Dictionary of patient-wise metrics
        """
        model.eval()
        total_loss = 0.0
        all_test_probs = []
        all_test_labels = []
        patient_metrics = {}

        with torch.no_grad():
            for inputs, labels, patient_ids in test_loader:  # Includes patient IDs
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Convert outputs to probabilities and predictions
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, dim=1)

                all_test_probs.extend(probs.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

                # Calculate metrics for each patient
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_metrics:
                        patient_metrics[patient_id] = {
                            "TP": 0,
                            "FP": 0,
                            "TN": 0,
                            "FN": 0,
                            "correct": 0,
                            "total": 0,
                        }

                    patient_metrics[patient_id]["total"] += 1  # Increment total samples

                    # Update TP, FP, TN, FN counts
                    if labels[i] == 1 and predicted[i] == 1:  # True Positive
                        patient_metrics[patient_id]["TP"] += 1
                        patient_metrics[patient_id]["correct"] += 1
                    elif labels[i] == 0 and predicted[i] == 1:  # False Positive
                        patient_metrics[patient_id]["FP"] += 1
                    elif labels[i] == 0 and predicted[i] == 0:  # True Negative
                        patient_metrics[patient_id]["TN"] += 1
                        patient_metrics[patient_id]["correct"] += 1
                    elif labels[i] == 1 and predicted[i] == 0:  # False Negative
                        patient_metrics[patient_id]["FN"] += 1

            # Finalize metrics for each patient
            for patient_id, stats in patient_metrics.items():
                tp, fp, tn, fn, correct, total = (
                    stats["TP"],
                    stats["FP"],
                    stats["TN"],
                    stats["FN"],
                    stats["correct"],
                    stats["total"],
                )

                # Compute accuracy for each patient
                stats["accuracy"] = correct / total if total > 0 else 0

                # Metrics for epileptic patients
                stats["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                stats["false_negative_rate"] = fn / (tp + fn) if (tp + fn) > 0 else 0

                # Metrics for healthy patients
                stats["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
                stats["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0

        avg_loss = total_loss / len(test_loader)
        return avg_loss, all_test_labels, all_test_probs, patient_metrics

    @staticmethod
    def create_averaged_model(model_paths, device, save_path):
        """
        Aggregates (averages) the weights of multiple trained models to create a single consolidated model.

        Args:
            model_paths (list): List of file paths to the saved model weights from each fold.
            device (torch.device): Device to load and process the models (CPU or GPU).
            save_path (str): Path to save the averaged model.

        Returns:
            torch.nn.Module: The model with aggregated (averaged) weights.
        """

        if not model_paths:
            raise ValueError("No model paths provided for aggregation.")

        first_model_state = torch.load(
            model_paths[0], map_location=device, weights_only=True
        )

        aggregated_state_dict = {
            key: torch.zeros_like(val) for key, val in first_model_state.items()
        }

        for model_path in model_paths:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] += state_dict[key]

        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] /= len(model_paths)

        final_model, _, _ = NNControl.initialize_model(
            device=device, learning_rate=1e-5
        )
        final_model.load_state_dict(aggregated_state_dict)

        torch.save(final_model.state_dict(), save_path)
        logger.info(f"Aggregated model saved to {save_path}")

        return final_model

    @staticmethod
    def get_patient_ids_by_type(
        patient_types: Dict[str, PatientType], patient_type: PatientType
    ) -> List[str]:
        """
        Filter patient IDs by a specified patient type.

        Args:
            patient_types (dict): Dictionary mapping patient IDs to their types.
            patient_type (PatientType): The type of patient to filter (e.g., PatientType.HEALTHY_REAL).

        Returns:
            List[str]: List of patient IDs matching the specified type.
        """
        return [
            patient_id
            for patient_id, p_type in patient_types.items()
            if p_type == patient_type
        ]
