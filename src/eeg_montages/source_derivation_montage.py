import numpy as np
from src.constants import SourceDerivationMontageChannels
from src.eeg_montages.common_average_montage import CommonAverageMontage


class SourceDerivationMontage(CommonAverageMontage):
    """
    Class that represent the Source Derivation montage.
    This montage is specific to the 10-20 system.

    As default EEG channels for computation uses EEGChannels1020System in constants.py.
    Channel names and order are specified in SourceDerivationMontageChannels in constants.py.
    """

    def __init__(self, eeg_signal: np.ndarray) -> None:
        super().__init__(eeg_signal)
        self.eeg_signal = CommonAverageMontage(eeg_signal).compute_montage()
        self.channel_names = SourceDerivationMontageChannels

    def compute_montage(self) -> np.ndarray:
        """
        Computes the source derivation montage from the EEG data.

        Returns:
            np.ndarray: A 2D NumPy array representing the EEG data in the source derivation montage format,
                        where each row corresponds to a specific channel derivation and each column corresponds to time points.
        """
        num_channels = len(SourceDerivationMontageChannels)
        source_derivation_montage = np.zeros((num_channels, self.eeg_signal.shape[1]))

        # Define channel derivations
        channel_derivations = [
            (
                "Fp1",
                [
                    ("Fp2", 0.1683),
                    ("F8", 0.0891),
                    ("F7", 0.1683),
                    ("T3", 0.0891),
                    ("F4", 0.0990),
                    ("F3", 0.1683),
                    ("C3", 0.0891),
                    ("Fz", 0.1287),
                ],
            ),
            (
                "Fp2",
                [
                    ("F8", 0.1683),
                    ("T4", 0.0891),
                    ("Fp1", 0.1683),
                    ("F7", 0.0891),
                    ("F4", 0.1683),
                    ("C4", 0.0891),
                    ("F3", 0.0990),
                    ("Fz", 0.1287),
                ],
            ),
            (
                "F3",
                [
                    ("Fp1", 0.19),
                    ("F7", 0.19),
                    ("T3", 0.12),
                    ("C3", 0.19),
                    ("Fz", 0.19),
                    ("Cz", 0.12),
                ],
            ),
            (
                "F4",
                [
                    ("Fp2", 0.19),
                    ("F8", 0.19),
                    ("T4", 0.12),
                    ("C4", 0.19),
                    ("Fz", 0.19),
                    ("Cz", 0.12),
                ],
            ),
            (
                "C3",
                [
                    ("F7", 0.1),
                    ("T3", 0.15),
                    ("T5", 0.1),
                    ("F3", 0.15),
                    ("P3", 0.15),
                    ("Fz", 0.1),
                    ("Cz", 0.15),
                    ("Pz", 0.1),
                ],
            ),
            (
                "C4",
                [
                    ("F8", 0.1),
                    ("T4", 0.15),
                    ("T6", 0.1),
                    ("F4", 0.15),
                    ("P4", 0.15),
                    ("Fz", 0.1),
                    ("Cz", 0.15),
                    ("Pz", 0.1),
                ],
            ),
            (
                "P3",
                [
                    ("T3", 0.12),
                    ("T5", 0.19),
                    ("C3", 0.19),
                    ("O1", 0.19),
                    ("Cz", 0.12),
                    ("Pz", 0.19),
                ],
            ),
            (
                "P4",
                [
                    ("T4", 0.12),
                    ("T6", 0.19),
                    ("C4", 0.19),
                    ("O2", 0.19),
                    ("Cz", 0.12),
                    ("Pz", 0.19),
                ],
            ),
            (
                "O1",
                [
                    ("T6", 0.0891),
                    ("T3", 0.0891),
                    ("T5", 0.1683),
                    ("P4", 0.099),
                    ("O2", 0.1683),
                    ("C3", 0.0891),
                    ("P3", 0.1683),
                    ("Pz", 0.1287),
                ],
            ),
            (
                "O2",
                [
                    ("T4", 0.0891),
                    ("T6", 0.1683),
                    ("T5", 0.0891),
                    ("C4", 0.0891),
                    ("P4", 0.1683),
                    ("P3", 0.099),
                    ("O1", 0.1683),
                    ("Pz", 0.1287),
                ],
            ),
            (
                "F7",
                [
                    ("Fp2", 0.1),
                    ("Fp1", 0.19),
                    ("T3", 0.19),
                    ("T5", 0.1),
                    ("F3", 0.19),
                    ("C3", 0.13),
                    ("Fz", 0.1),
                ],
            ),
            (
                "F8",
                [
                    ("Fp2", 0.19),
                    ("Fp1", 0.1),
                    ("T4", 0.19),
                    ("T6", 0.1),
                    ("F4", 0.19),
                    ("C4", 0.13),
                    ("Fz", 0.1),
                ],
            ),
            (
                "T3",
                [
                    ("Fp1", 0.0808),
                    ("F7", 0.1717),
                    ("T5", 0.1717),
                    ("F3", 0.1212),
                    ("C3", 0.1717),
                    ("P3", 0.1212),
                    ("O1", 0.0808),
                    ("Cz", 0.0808),
                ],
            ),
            (
                "T4",
                [
                    ("Fp2", 0.0808),
                    ("F8", 0.1717),
                    ("T6", 0.1717),
                    ("F4", 0.1212),
                    ("C4", 0.1717),
                    ("P4", 0.1212),
                    ("O2", 0.0808),
                    ("Cz", 0.0808),
                ],
            ),
            (
                "T5",
                [
                    ("F7", 0.1),
                    ("T3", 0.19),
                    ("O2", 0.1),
                    ("C3", 0.13),
                    ("P3", 0.19),
                    ("O1", 0.19),
                    ("Pz", 0.1),
                ],
            ),
            (
                "T6",
                [
                    ("F8", 0.1),
                    ("T4", 0.19),
                    ("C4", 0.13),
                    ("P4", 0.19),
                    ("O2", 0.19),
                    ("O1", 0.1),
                    ("Pz", 0.1),
                ],
            ),
            (
                "Fz",
                [
                    ("Fp2", 0.12),
                    ("F8", 0.07),
                    ("Fp1", 0.12),
                    ("F7", 0.07),
                    ("F4", 0.15),
                    ("C4", 0.1),
                    ("F3", 0.15),
                    ("C3", 0.1),
                    ("Cz", 0.12),
                ],
            ),
            (
                "Cz",
                [
                    ("F4", 0.1),
                    ("C4", 0.15),
                    ("P4", 0.1),
                    ("F3", 0.1),
                    ("C3", 0.15),
                    ("P3", 0.1),
                    ("Fz", 0.15),
                    ("Pz", 0.15),
                ],
            ),
            (
                "Pz",
                [
                    ("T6", 0.07),
                    ("T5", 0.07),
                    ("C4", 0.1),
                    ("P4", 0.15),
                    ("O2", 0.12),
                    ("C3", 0.1),
                    ("P3", 0.15),
                    ("O1", 0.12),
                    ("Cz", 0.12),
                ],
            ),
        ]

        # Apply derivations to compute the montage
        for idx, (ch1, derivations) in enumerate(channel_derivations):
            source_derivation_montage[idx, :] = getattr(self, ch1)
            for ch2, coeff in derivations:
                source_derivation_montage[idx, :] -= coeff * getattr(self, ch2)

        return source_derivation_montage
