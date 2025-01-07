"""
Example file how to use EpochSimulator module.
"""

from src.epoch_simulator.epoch_simulator import EpochSimulator
from src.logging_config.logging_setup import setup_logging

setup_logging()


# After signal is preprocessed load patient instance. Random processed epochs in Common Average montage are chosed.
# Then the spike with the biggest value is chosen. You can specify how many spikes shoul be chosen.
# Then signal matrix to SIMMEG is create with the multiplier. It means that the randomly chose spikes are mutliplied.
# For now I use only one source in SIMMEG modelling but for simplicity the the same spikes are
# also copied into second and third source.

patient_path = "D:\DIPLOMKA_DATASET\P249\P249_instance.pkl"

simulation = EpochSimulator(patient_path)

# Choose number of epochs that should be selected:
num_epochs = 1
spike_matrix, metadata = simulation.create_spike_matrix(num_epochs)

# Save information about epoch, channel index, channel name etc.
simulation.save_spike_metadata_to_csv(metadata)

# Output matlab matrix that is ready to use in SIMMEEG
spike_multiplicator = 50
output_simmeeg_matrix = simulation.transfer_spike_matrix_into_simmmeeg_format(
    spike_matrix, spike_multiplicator
)
