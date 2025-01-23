import numpy as np
import mne
from typing import List, Type, Union, Optional
from enum import Enum
import matplotlib.pyplot as plt
from src.constants import (
    EEGChannels1020System,
    DoubleBananaMontageChannels,
    SourceDerivationMontageChannels,
    SAMPLING_FREQUENCY,
)
from src.signal_preprocessing.loading.eeg_data_io import EEGDataIO
from src.signal_preprocessing.validating.eeg_validator import EEGValidator
import seaborn as sns


class EEGPlotter:
    """
    A utility class for plotting EEG data, supporting both multi-channel and single-channel visualizations.

    Methods:
        - plot_eeg: Plots signals from all EEG channels with standard or custom montages.
        - plot_single_channel: Plots the signal from a specific EEG channel.
    """

    @staticmethod
    def plot_eeg(
        eeg_signal: Union[str, np.ndarray],
        fs: int = SAMPLING_FREQUENCY,
        eeg_montage: str = "CA",
        channel_names: Optional[Union[Type[Enum], List[str]]] = None,
    ) -> None:
        """
        Plots the EEG signals from all channels using the specified montage or custom channel names.

        Args:
            eeg_signal (Union[str, np.ndarray]): The EEG signal data, either as a path to a .npy file or as a NumPy array.
            fs (int): The sampling frequency of the EEG data. Defaults to SAMPLING_FREQUENCY.
            eeg_montage (str): The montage used for channel naming. Defaults to "CA" (Common Average). Other options are "DB (Double Banana)" or "SD" (Source Derivation).
            channel_names (Optional[Union[Type[Enum], List[str]]]):
                - An Enum representing the EEG channels according to a standard montage.
                - A list of custom channel names.

        Returns:
            None
        """
        if isinstance(eeg_signal, str):
            eeg_signal = EEGValidator.validate_eeg_signal_dims(
                EEGDataIO.load_eeg_epoch(eeg_signal)
            )
        else:
            eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        num_channels = eeg_signal.shape[0]
        channel_names = EEGPlotter._get_channel_names(
            channel_names, num_channels, eeg_montage
        )

        info = mne.create_info(
            ch_names=channel_names, sfreq=fs, ch_types=["eeg"] * num_channels
        )

        eeg_signal = -1 * eeg_signal # To keep the same orientation as in Brainstorm
    
        raw = mne.io.RawArray(eeg_signal, info)
        raw.plot(block=True, scalings="auto")

    @staticmethod
    def plot_single_channel(
        eeg_signal: Union[str, np.ndarray],
        channel_index: int = 1,
        fs: int = SAMPLING_FREQUENCY,
        eeg_montage: str = "CA",
        channel_names: Optional[Union[Type[Enum], List[str]]] = None,
    ) -> None:
        """
        Plots the signal of a single EEG channel.

        Args:
            eeg_signal (Union[str, np.ndarray]): The EEG signal data, either as a path to a .npy file or as a NumPy array.
            channel_index (int): The index of the channel to plot. Defaults to 1 (1-based index).
            fs (int): The sampling frequency of the EEG data. Defaults to SAMPLING_FREQUENCY.
            eeg_montage (str): The montage used for channel naming. Defaults to "CA" (Common Average). Other options "DB" (Double Banana), "SD" (Source Derivation).
            channel_names (Optional[Union[Type[Enum], List[str]]]):
                - An Enum representing the EEG channels according to a standard montage.
                - A list of custom channel names.

        Returns:
            None
        """
        if isinstance(eeg_signal, str):
            eeg_signal = EEGValidator.validate_eeg_signal_dims(
                EEGDataIO.load_eeg_epoch(eeg_signal)
            )
        else:
            eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        num_channels = eeg_signal.shape[0]

        if not (1 <= channel_index <= num_channels):
            raise ValueError(f"Channel index must be between 1 and {num_channels}")

        channel_index -= 1  # Adjust index for zero-based Python indexing
        single_channel_data = eeg_signal[channel_index, :]
        channel_name = EEGPlotter._get_channel_name(
            channel_index, num_channels, eeg_montage, channel_names
        )
        time_vector = np.arange(single_channel_data.shape[0]) / fs

        plt.figure(figsize=(10, 4))
        plt.plot(time_vector, single_channel_data, color="blue")
        plt.title(f"Channel: {channel_name} ({channel_index + 1})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    @staticmethod
    def _get_channel_names(
        channel_names: Optional[Union[Type[Enum], List[str]]],
        num_channels: int,
        eeg_montage: str,
    ) -> List[str]:
        """
        Retrieves a list of channel names based on the provided montage or a custom list.

        Args:
            channel_names (Optional[Union[Type[Enum], List[str]]]):
                - An Enum representing the EEG channels according to a standard montage (e.g., 10-20 system).
                - A list of custom channel names.
                - If None, defaults to using the montage specified by `eeg_montage`.
            num_channels (int):
                The number of channels in the EEG data.
            eeg_montage (str):
                The montage to use for channel naming if `channel_names` is not provided.
                Options include "CA" (Common Average), "DB" (Double Banana), or "SD" (Source Derivation).

        Returns:
            List[str]: A list of channel names corresponding to the EEG data.

        Raises:
            ValueError:
                - If the number of channels in the EEG data exceeds the number of provided channel names.
                - If an unsupported channel name type is provided.
        """
        if not channel_names:
            montage_map = {
                "CA": EEGChannels1020System,
                "DB": DoubleBananaMontageChannels,
                "SD": SourceDerivationMontageChannels,
            }
            channel_names = montage_map.get(
                eeg_montage, [f"EEG {i+1}" for i in range(num_channels)]
            )

        if isinstance(channel_names, type(Enum)):
            return EEGPlotter._get_enum_channel_names(channel_names, num_channels)
        elif isinstance(channel_names, list):
            if num_channels > len(channel_names):
                raise ValueError(
                    "Number of channels in the EEG data exceeds provided channel names."
                )
            return channel_names[:num_channels]
        else:
            raise ValueError("Unsupported channel name type.")

    @staticmethod
    def _get_enum_channel_names(enum_type: Type[Enum], num_channels: int) -> List[str]:
        """
        Extracts channel names from an Enum class representing EEG channels.

        Args:
            enum_type (Type[Enum]): The Enum class representing the EEG channels.
            num_channels (int): The number of channels in the EEG data.

        Returns:
            List[str]: A list of channel names based on the Enum class.

        Raises:
            ValueError: If the provided Enum type is unsupported.
        """
        if issubclass(enum_type, EEGChannels1020System) or issubclass(
            enum_type, SourceDerivationMontageChannels
        ):
            return [label.name for label in enum_type if label.value <= num_channels]
        elif issubclass(enum_type, DoubleBananaMontageChannels):
            return [label.label for label in enum_type if label.index <= num_channels]
        else:
            raise ValueError("Unsupported channel name Enum type.")

    @staticmethod
    def _get_channel_name(
        channel_index: int,
        num_channels: int,
        eeg_montage: str,
        channel_names: Optional[Union[Type[Enum], List[str]]],
    ) -> str:
        """
        Retrieves the name of a specific EEG channel based on its index.

        Args:
            channel_index (int): The index of the channel (1-based index).
            num_channels (int): The total number of channels in the EEG data.
            eeg_montage (str): The montage used for channel naming.
            channel_names (Optional[Union[Type[Enum], List[str]]]):
                - An Enum representing the EEG channels according to a standard montage.
                - A list of custom channel names.

        Returns:
            str: The name of the specified channel.

        Raises:
            ValueError: If no matching channel is found or if the channel index is out of range.
        """
        if not channel_names:
            channel_names = EEGPlotter._get_channel_names(
                None, num_channels, eeg_montage
            )

        if isinstance(channel_names, type(Enum)):
            if issubclass(channel_names, DoubleBananaMontageChannels):
                for label in channel_names:
                    if label.index == channel_index + 1:
                        return label.label
            else:
                for label in channel_names:
                    if label.value == channel_index + 1:
                        return label.name
            raise ValueError(f"No matching channel found for index {channel_index + 1}")
        elif isinstance(channel_names, list):
            if 0 <= channel_index < len(channel_names):
                return channel_names[channel_index]
            else:
                raise ValueError(f"Channel index {channel_index + 1} is out of range.")

    @staticmethod
    def browse_eeg_interactively(eeg_data, sfreq, eeg_montage="CA"):
        """
        Interactive data browsing with MNE's Raw.plot(), using channel names from EEGPlotter._get_channel_names.

        Args:
            eeg_data (np.ndarray): EEG data as a 2D NumPy array (channels x timepoints).
            sfreq (float): Sampling frequency of the EEG signal in Hz.
            eeg_montage (str): Montage type to determine channel names (Default: "CA").

        Returns:
            None
        """
        # Retrieve the number of channels from the data
        num_channels = eeg_data.shape[0]

        # Retrieve channel names dynamically
        channel_names = EEGPlotter._get_channel_names(
            channel_names=None,  # Use default or dynamically fetch names
            num_channels=num_channels,
            eeg_montage=eeg_montage,
        )

        # Validate the shape of EEG data
        if num_channels != len(channel_names):
            raise ValueError(
                f"Number of channels in data ({num_channels}) does not match "
                f"the expected number of channels ({len(channel_names)})."
            )

        # Create MNE Info object
        info = mne.create_info(
            ch_names=channel_names, sfreq=sfreq, ch_types=["eeg"] * num_channels
        )

        # Create MNE Raw object
        raw = mne.io.RawArray(eeg_data, info)

        # Plot the Raw data interactively
        raw.plot(title="Interactive EEG Data Browsing", scalings="auto", show=True)
        plt.show()

    @staticmethod
    def plot_eeg_to_diplomka(
        eeg_signal: Union[str, np.ndarray],
        fs: int = SAMPLING_FREQUENCY,
        eeg_montage: str = "CA",
        channel_names: Optional[Union[Type[Enum], List[str]]] = None,
        title: Optional[str] = None,  # New argument for plot title
    ) -> None:
        if isinstance(eeg_signal, str):
            eeg_signal = EEGValidator.validate_eeg_signal_dims(
                EEGDataIO.load_eeg_epoch(eeg_signal)
            )
        else:
            eeg_signal = EEGValidator.validate_eeg_signal_dims(eeg_signal)

        num_channels = eeg_signal.shape[0]
        channel_names = EEGPlotter._get_channel_names(
            channel_names, num_channels, eeg_montage
        )

        info = mne.create_info(
            ch_names=channel_names, sfreq=fs, ch_types=["eeg"] * num_channels
        )
        raw = mne.io.RawArray(eeg_signal, info)

        sns.set(style="whitegrid")
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        # Plot the EEG signals
        fig = raw.plot(
            scalings=dict(eeg=20e-6),
            show_options=False,
            show_scrollbars=False,
            show_scalebars=False,
            splash=False,
            start=0,  # Plot starts at 0
            duration=2,  # Plot spans 2 seconds
            show=False,
        )

        # Rewrite x-axis ticks dynamically
        ax = fig.mne.ax_main

        def update_ticks():
            """Dynamically update x-axis tick labels."""
            ticks = ax.get_xticks()  # Get current tick positions
            if len(ticks) > 0:  # Ensure there are ticks
                # Set new ticks and labels
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    [f"{tick - 1:.1f}" for tick in ticks]
                )  # Shift labels by -1
            fig.canvas.draw_idle()

        # Connect dynamic updates to zoom or pan events
        fig.canvas.mpl_connect("draw_event", lambda event: update_ticks())

        # Customize labels and ticks
        ax.set_xlabel(r"Čas [s]", fontsize=10)
        ax.set_ylabel(r"[millivolts]", fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Add the title if provided
        if title:
            ax.set_title(title, fontsize=12)

        # Hide the grid
        ax.grid(False)  # Disable grid lines explicitly
        sns.despine()  # Remove spines if needed

        # Adjust font size for channel names
        for text_obj in ax.texts:
            text_obj.set_fontsize(10)

        # Dynamically print scaling information
        def on_key_release(event):
            if event.key in ["-", "="]:
                fig.canvas.draw_idle()
                current_scale = fig.mne.scalings["eeg"]
                formatted_scale = f"{current_scale:.6f}"
                print("Current EEG amplitude scale:", formatted_scale, "V")

        fig.canvas.mpl_connect("key_release_event", on_key_release)

        # Set the figure size for saving (width=4cm, height=6cm)
        fig.set_size_inches(4 / 2.54, 6 / 2.54)

        # Show the plot
        plt.show()

        # Save the final scaling with custom dimensions
        final_scale = fig.mne.scalings["eeg"]
        print(f"Final saved EEG amplitude scale: {final_scale:.10f} V")

        fig.savefig(
            "eeg_plot_final_scale_custom_size_no_grid.pdf",
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )
