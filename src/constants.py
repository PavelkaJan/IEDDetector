from enum import Enum

# ==========================================================
#                      SIGNAL CONSTANTS
# ==========================================================

SAMPLING_FREQUENCY = 128  # [Hz]
TIME = 2  # [s]
NUM_OF_CHANS_10_20 = 19


# ==========================================================
#                      EEG MONTAGES
# ==========================================================


# HDEEG channels could have different channel names
# -> need to translated according to EEGChannels1020System.
TRANSLATION_EEG_CHANS_MAP = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


class EEGChannels1020System(Enum):
    """
    Default order and names for 10-20 system.
    """

    Fp1 = 1
    Fp2 = 2
    F3 = 3
    F4 = 4
    C3 = 5
    C4 = 6
    P3 = 7
    P4 = 8
    O1 = 9
    O2 = 10
    F7 = 11
    F8 = 12
    T3 = 13
    T4 = 14
    T5 = 15
    T6 = 16
    Fz = 17
    Cz = 18
    Pz = 19


class DoubleBananaMontageChannels(Enum):
    """
    Channel names and their order for the Double Banana montage.
    Each entry is a tuple of (index, string_name).
    """

    Fp1_F7 = (1, "Fp1-F7")
    F7_T3 = (2, "F7-T3")
    T3_T5 = (3, "T3-T5")
    T5_O1 = (4, "T5-O1")

    Fp2_F8 = (5, "Fp2-F8")
    F8_T4 = (6, "F8-T4")
    T4_T6 = (7, "T4-T6")
    T6_O2 = (8, "T6-O2")

    Fp1_F3 = (9, "Fp1-F3")
    F3_C3 = (10, "F3-C3")
    C3_P3 = (11, "C3-P3")
    P3_O1 = (12, "P3-O1")

    Fp2_F4 = (13, "Fp2-F4")
    F4_C4 = (14, "F4-C4")
    C4_P4 = (15, "C4-P4")
    P4_O2 = (16, "P4-O2")

    Fz_Cz = (17, "Fz-Cz")
    Cz_Pz = (18, "Cz-Pz")

    # Extra channel bcs of neural network input
    Mean = (19, "Mean")

    @property
    def index(self):
        return self.value[0]

    @property
    def label(self):
        return self.value[1]


class SourceDerivationMontageChannels(Enum):
    sFp1 = 1
    sFp2 = 2
    sF3 = 3
    sF4 = 4
    sC3 = 5
    sC4 = 6
    sP3 = 7
    sP4 = 8
    sO1 = 9
    sO2 = 10
    sF7 = 11
    sF8 = 12
    sT3 = 13
    sT4 = 14
    sT5 = 15
    sT6 = 16
    sFz = 17
    sCz = 18
    sPz = 19


# ==========================================================
#                          MATLAB
# ==========================================================

# Mandatory variables for MATLAB files
MATLAB_FILE_MANDATATORY_VARS = [
    (("d",), "EEG signal"),
    (("fs",), "sampling frequency"),
    (("header",), "labels, startdate"),
    (("header", "label"), "channel names"),
    (("header", "startdate"), "start date"),
]


# ==========================================================
#                        PROCESSING
# ==========================================================

# Cutoff frequency for drift remove
CUTOFF_FREQUENCY = 0.5


# ==========================================================
#                      NEURAL NETWORK
# ==========================================================
NN_INPUT_DIMENSIONS = 3


# ==========================================================
#                         PATIENT
# ==========================================================

class PatientType(Enum):
    EPILEPTIC_REAL = "epileptic_real"
    EPILEPTIC_SIMULATED = "epileptic_simulated"
    HEALTHY_REAL = "healthy_real"
    HEALTHY_SIMULATED = "healthy_simulated"

# ==========================================================
#                     EPOCH SIMULATION
# ==========================================================
SAMPLING_FREQUENCY_INTERPOLATION = 256

# ==========================================================
#                     4s EPOCHS SHIFT
# ==========================================================
EPOCH_SHIFT_SPIKE_MID = (-1, 1)
EPOCH_SHIFT_SPIKE_LEFT = (-1.5, 0.5)
EPOCH_SHIFT_SPIKE_RIGHT = (-0.5, 1.5)

EPOCH_SHIFTS = [EPOCH_SHIFT_SPIKE_MID, EPOCH_SHIFT_SPIKE_LEFT, EPOCH_SHIFT_SPIKE_RIGHT]