import torch
from typing import List, Dict
import matplotlib.pyplot as plt
from src.neural_network.nn_control import NNControl


class NNFineTunning:
    @staticmethod
    def find_optimal_batch_size(
        train_patients: List,
        test_patients: List,
        device: torch.device,
        batch_sizes: List[int],
        learning_rate: float,
        num_epochs: int = 10,
    ) -> Dict[int, float]:
        """
        Perform grid search to find the optimal batch size.

        Args:
            train_patients: List of training patients.
            test_patients: List of testing patients.
            device: Device (CPU or GPU) for computation.
            batch_sizes: List of batch sizes to evaluate.
            learning_rate: Learning rate for optimizer initialization.
            num_epochs: Number of epochs to train each model.

        Returns:
            Dict with batch sizes and their corresponding accuracies.
        """
        results = {}

        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            train_loader, test_loader = NNControl.prepare_data_loaders(
                train_patients, test_patients, batch_size=batch_size
            )

            model, criterion, optimizer = NNControl.initialize_model(
                device, learning_rate=learning_rate
            )

            train_losses, train_accuracies = NNControl.train(
                model, train_loader, criterion, optimizer, num_epochs=num_epochs
            )

            test_loss, test_acc, _, _ = NNControl.evaluate(
                model, test_loader, criterion
            )
            results[batch_size] = test_acc
            print(f"Batch Size: {batch_size}, Test Accuracy: {test_acc:.4f}")

        return results

    @staticmethod
    def plot_batch_size_results(results: Dict[int, float]) -> None:
        """
        Plot the results of batch size evaluation.

        Args:
            results: Dictionary with batch sizes as keys and accuracies as values.
        """
        batch_sizes = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(10, 6))
        plt.plot(
            batch_sizes, accuracies, marker="o", linestyle="-", label="Test Accuracy"
        )
        plt.xscale("log")  # Log scale for batch sizes
        plt.xlabel("Batch Size")
        plt.ylabel("Test Accuracy")
        plt.title("Batch Size vs. Test Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def find_optimal_learning_rate(
        train_patients: List,
        test_patients: List,
        device: torch.device,
        learning_rates: List[float],
        batch_size: int,
        num_epochs: int = 10,
    ) -> Dict[float, float]:
        """
        Perform grid search to find the optimal learning rate.

        Args:
            train_patients: List of training patients.
            test_patients: List of testing patients.
            device: Device (CPU or GPU) for computation.
            learning_rates: List of learning rates to evaluate.
            batch_size: Batch size for data loaders.
            num_epochs: Number of epochs to train each model.

        Returns:
            Dict with learning rates and their corresponding accuracies.
        """
        results = {}

        for lr in learning_rates:
            print(f"Testing learning rate: {lr}")
            train_loader, test_loader = NNControl.prepare_data_loaders(
                train_patients, test_patients, batch_size=batch_size
            )
            model, criterion, optimizer = NNControl.initialize_model(
                device, learning_rate=lr
            )
            train_losses, train_accuracies = NNControl.train(
                model, train_loader, criterion, optimizer, num_epochs=num_epochs
            )
            test_loss, test_acc, _, _ = NNControl.evaluate(
                model, test_loader, criterion
            )
            results[lr] = test_acc
            print(f"Learning Rate: {lr}, Test Accuracy: {test_acc:.4f}")
        return results

    @staticmethod
    def plot_learning_rate_results(results: Dict[float, float]) -> None:
        """
        Plot the results of learning rate evaluation.

        Args:
            results: Dictionary with learning rates as keys and accuracies as values.
        """
        learning_rates = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(10, 6))
        plt.plot(
            learning_rates, accuracies, marker="o", linestyle="-", label="Test Accuracy"
        )
        plt.xscale("log")  # Log scale for learning rates
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Accuracy")
        plt.title("Learning Rate vs. Test Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()
