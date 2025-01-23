import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)


class EpochClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize the EpochClassifier.

        Args:
            model_path (str): Path to the saved model (.pth file).
            device (torch.device, optional): Device to use ('cuda' or 'cpu'). If None, it defaults to available hardware.
        """
        self.model_path = model_path
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None

    def load_model(self, model_initializer):
        """
        Load the saved model using a provided initializer function.

        Args:
            model_initializer (function): Function to initialize the model structure.
        """
        self.model, _, _ = model_initializer(device=self.device, learning_rate=0.0001)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def load_epochs(self, epoch_folder):
        """
        Load epoch data from the specified folder.

        Args:
            epoch_folder (str): Path to the folder containing epoch files.

        Returns:
            torch.Tensor: Tensor of loaded epochs.
            list: List of file names corresponding to the epochs.
        """
        epoch_files = sorted(
            [
                f
                for f in os.listdir(epoch_folder)
                if f.endswith(".npy")
            ]
        )

        if not epoch_files:
            logger.error(f"No epoch files found in folder: {epoch_folder}")
            raise FileNotFoundError(f"No epoch files found in folder: {epoch_folder}")

        epochs = []
        for file_name in epoch_files:
            file_path = os.path.join(epoch_folder, file_name)
            epochs.append(np.load(file_path))

        logger.info(f"Loaded {len(epochs)} epochs from {epoch_folder}.")
        return torch.tensor(np.stack(epochs), dtype=torch.float32), epoch_files

    def classify_epochs(self, epochs, batch_size=1):
        """
        Classify the epochs and compute probabilities.

        Args:
            epochs (torch.Tensor): Tensor of epochs to classify.
            batch_size (int): Batch size for DataLoader.

        Returns:
            np.ndarray: Predicted class labels.
            np.ndarray: Probabilities for each class.
        """
        dataset = TensorDataset(epochs)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

        all_preds_np = torch.cat(all_preds).numpy()
        all_probs_np = torch.cat(all_probs).numpy()

        # Count the number of IED and non-IED epochs
        num_ied = np.sum(all_preds_np == 1)
        num_non_ied = np.sum(all_preds_np == 0)

        logger.info(f"Classification complete. Processed {len(epochs)} epochs.")
        logger.info(f"Number of epochs classified as IED: {num_ied}")
        logger.info(f"Number of epochs classified as non-IED: {num_non_ied}")

        return all_preds_np, all_probs_np

    def evaluate(self, predicted_labels, true_labels):
        """
        Evaluate the predictions against the true labels.

        Args:
            predicted_labels (np.ndarray): Predicted class labels.
            true_labels (np.ndarray): Ground truth labels.

        Returns:
            float: Accuracy of the predictions.
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def handle_labels(self, label_file_path, predicted_labels):
        """
        Load and validate true labels, and optionally calculate accuracy.

        Args:
            label_file_path (str): Path to the numpy file containing true labels (optional).
            predicted_labels (np.ndarray): Predicted class labels.

        Returns:
            np.ndarray or None: Loaded true labels if provided, otherwise None.
            float or None: Calculated accuracy if true labels are provided, otherwise None.
        """
        if label_file_path:
            if not os.path.exists(label_file_path):
                logger.error(f"Labels file not found: {label_file_path}")
                raise FileNotFoundError(f"Labels file not found: {label_file_path}")

            true_labels = np.load(label_file_path)

            if len(true_labels) != len(predicted_labels):
                logger.error(
                    "Mismatch between the number of true labels and predicted labels!"
                )
                raise ValueError(
                    "Mismatch between the number of true labels and predicted labels!"
                )

            # Count actual IED and non-IED epochs
            num_true_ied = np.sum(true_labels == 1)
            num_true_non_ied = np.sum(true_labels == 0)

            # Count how many were correctly recognized as IED and non-IED
            num_correct_ied = np.sum((true_labels == 1) & (predicted_labels == 1))
            num_correct_non_ied = np.sum((true_labels == 0) & (predicted_labels == 0))

            # Calculate accuracy
            accuracy = self.evaluate(predicted_labels, true_labels)

            logger.info(f"Number of true IED epochs: {num_true_ied}")
            logger.info(f"Number of true non-IED epochs: {num_true_non_ied}")
            logger.info(f"Number of correctly recognized IED epochs: {num_correct_ied}")
            logger.info(
                f"Number of correctly recognized non-IED epochs: {num_correct_non_ied}"
            )

            return true_labels, accuracy
        else:
            logger.info("No true labels provided. Skipping evaluation.")
            return None, None

    def save_results(
        self,
        results_file,
        epoch_files,
        predicted_labels,
        probabilities,
        true_labels=None,
    ):
        """
        Save classification results to a CSV file.

        Args:
            results_file (str): Path to the output CSV file.
            epoch_files (list): List of epoch file names.
            predicted_labels (np.ndarray): Predicted class labels.
            probabilities (np.ndarray): Probabilities for each class.
            true_labels (np.ndarray, optional): True labels, if available.
        """
        with open(results_file, "w") as f:
            f.write(
                "Epoch,Predicted_Class,Probability_Class_0_non_IED,Probability_Class_1_IED"
            )
            if true_labels is not None:
                f.write(",True_Class\n")
            else:
                f.write("\n")

            for i, (file_name, pred, probs) in enumerate(
                zip(epoch_files, predicted_labels, probabilities)
            ):
                line = f"{file_name},{pred},{probs[0]},{probs[1]}"
                if true_labels is not None:
                    line += f",{true_labels[i]}"
                f.write(line + "\n")

        logger.info(f"Results saved to {results_file}.")
