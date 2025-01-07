import numpy as np
from src.eeg_montages.common_average_montage import CommonAverageMontage
from src.constants import DoubleBananaMontageChannels


class DoubleBananaMontage(CommonAverageMontage):
    """
    Class that represent the Double Banana montage.
    This montage is specific to the 10-20 system.
    Has one extra channel (mean) bcs of neural network input.

    As default EEG channels for computation uses EEGChannels1020System in constants.py.
    Channel names and order are specified in DoubleBananaMontageChannels in constants.py.
    """

    def __init__(self, eeg_signal: np.ndarray):
        super().__init__(eeg_signal)
        self.eeg_signal = CommonAverageMontage(eeg_signal).compute_montage()
        self.channel_names = DoubleBananaMontageChannels

    def compute_montage(self) -> np.ndarray:
        """
        Computes the double banana montage from the EEG data by applying differential referencing
        between predefined pairs of channels.
        This montage has 19 channels, and there is an extra one (mean) because of neural network input.

        Returns:
            np.ndarray: A 2D NumPy array representing the EEG data in the double banana montage format,
                        where each row corresponds to a specific montage pair and each column corresponds to time points.
        """
        num_channels = len(DoubleBananaMontageChannels)
        double_banana_montage = np.zeros((num_channels, self.eeg_signal.shape[1]))

        channel_pairs = [
            # Left temporal chain
            ("Fp1", "F7"),
            ("F7", "T3"),
            ("T3", "T5"),
            ("T5", "O1"),
            # Right temporal chain
            ("Fp2", "F8"),
            ("F8", "T4"),
            ("T4", "T6"),
            ("T6", "O2"),
            # Left parasagittal chain
            ("Fp1", "F3"),
            ("F3", "C3"),
            ("C3", "P3"),
            ("P3", "O1"),
            # Right parasagittal chain
            ("Fp2", "F4"),
            ("F4", "C4"),
            ("C4", "P4"),
            ("P4", "O2"),
            # Central chain
            ("Fz", "Cz"),
            ("Cz", "Pz"),
        ]

        for idx, (ch1, ch2) in enumerate(channel_pairs):
            double_banana_montage[idx, :] = getattr(self, ch1) - getattr(self, ch2)

        # Compute the mean channel for the last row using the computed montage values
        double_banana_montage[-1, :] = self._compute_mean_channel(double_banana_montage)
        
        return double_banana_montage
    
    def _compute_mean_channel(self, montage: np.ndarray) -> np.ndarray:
        """
        Computes the mean signal across all channels in the double banana montage.

        Args:
            montage (np.ndarray): The 2D array containing the montage values.

        Returns:
            np.ndarray: A 1D NumPy array representing the mean signal across all montage channels.
        """
        return np.mean(montage[:-1, :], axis=0)

