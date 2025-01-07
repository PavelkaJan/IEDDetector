import numpy as np
import mne
from src.patient.patient import Patient
from src.constants import SAMPLING_FREQUENCY


def create_mne_info(patient: Patient) -> mne.Info:
    """
    Create MNE info object from patient data.

    Args:
        patient (Patient): A Patient object containing EEG metadata.
            The Patient object must have `channels` and `fs` attributes.
            `channels` should be a list of channel names.
            `fs` should be the sampling frequency.

    Returns:
        mne.Info: An MNE info object containing metadata for the EEG recording.

    Raises:
        ValueError: If `channels` or `fs` are not specified in the patient object.
    """
    # TODO: fix ch_names 
    if patient.channel_names_CA:
        ch_names = patient.channel_names_CA
        ch_names = ch_names[:19]
    else:
        raise ValueError('You need to specify channel names in the patient object.')
    
    # TODO: patient.fs
    # if not patient.fs:
    #     raise ValueError('You need to specify sampling frequency in the patient object.')
    
    # TODO: for now delete the last channel
    # ch_names = ch_names[:-1]
    ch_types = ['eeg'] * len(ch_names)
    
    return mne.create_info(ch_names=ch_names, sfreq=SAMPLING_FREQUENCY, ch_types=ch_types)


def create_mne_raw_array(data: np.ndarray, info: mne.Info) -> mne.io.RawArray:
    """
    Create an MNE RawArray object from data and info.

    Args:
        data (numpy.ndarray): The EEG data array.
        info (mne.Info): The MNE info object containing metadata for the EEG recording.

    Returns:
        mne.io.RawArray: The MNE RawArray object.
    """
    return mne.io.RawArray(data, info)