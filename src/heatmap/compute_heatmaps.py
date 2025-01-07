from pathlib import Path
import numpy as np
from src.logging_config.logging_setup import setup_logging
from src.neural_network.epoch_classifier import EpochClassifier
from src.neural_network.nn_control import NNControl
from captum.attr import Occlusion
import torch

setup_logging()

# Parameters
epoch_folder_path = "D:\DIPLOMKA_DATASET\P322\epochs_PY\IED_present\CA"
model_save_path = "saved_model.pth"
heatmap_file = "occlusion_heatmaps.npy"
label_file_path = "D:\DIPLOMKA_DATASET\P322\labels\IED_present\labels_CA.npy" # Optional: Path to true labels (e.g., "labels/labels_CA.npy")

# Initialize classifier
classifier = EpochClassifier(model_path=model_save_path, device="cuda")
classifier.load_model(NNControl.initialize_model)
epochs_tensor, epoch_files = classifier.load_epochs(epoch_folder_path)
predicted_labels, probabilities = classifier.classify_epochs(epochs_tensor)


# Handle labels and calculate accuracy if provided
true_labels, accuracy = classifier.handle_labels(label_file_path, predicted_labels)

results_file = "epoch_results_with_accuracy.csv"
classifier.save_results(
    results_file, epoch_files, predicted_labels, probabilities, true_labels
)

# Compute and save heatmaps
def compute_and_save_occlusion_heatmaps(classifier, epochs_tensor, predicted_labels, sliding_window_shapes, strides, save_path):
    import torch
    import numpy as np
    from scipy.ndimage import gaussian_filter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs_tensor = epochs_tensor.to(device)
    occlusion_heatmaps = []

    for i, epoch_tensor in enumerate(epochs_tensor):
        target_class = int(predicted_labels[i])
        input_tensor = epoch_tensor.unsqueeze(0)  # Shape: (1, 1, n_channels, n_samples)

        occlusion = Occlusion(classifier.model)
        heatmap = occlusion.attribute(
            input_tensor,
            strides=strides,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            baselines=torch.tensor(0, device=device)
        )

        # Normalize heatmap to range [0, 1]
        heatmap = heatmap.squeeze(0).cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

        # Smooth heatmap using Gaussian filter
        smoothed_heatmap = gaussian_filter(heatmap, sigma=1)

        occlusion_heatmaps.append(smoothed_heatmap)

    np.save(save_path, occlusion_heatmaps)
    print(f"Occlusion heatmaps saved to {save_path}")

# Run computation
compute_and_save_occlusion_heatmaps(
    classifier, epochs_tensor, predicted_labels, sliding_window_shapes=(1, 1, 256), strides=(1,1,16), save_path=heatmap_file
)

# Save the epochs tensor for visualization
np.save("epochs_tensor.npy", epochs_tensor.cpu().numpy()) 