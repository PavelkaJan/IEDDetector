import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Button


def plot_raw_with_rainbow_heatmap(raw: mne.io.Raw, heatmap: np.ndarray, **kwargs):
    """
    Plots raw EEG data with a rainbow gradient heatmap mapped directly onto the signal lines,
    enabling zooming and panning with MNE's interactive plot.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data to be plotted.
    heatmap : np.ndarray
        The heatmap values corresponding to the channels (n_channels x n_times).
    **kwargs : dict
        Additional keyword arguments to pass to the plot_raw function.

    Returns:
    None
    """
    # Normalize and smooth the heatmap
    def process_heatmap(heatmap):
        # Normalize to range [0, 1]
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
        
        # Apply Gaussian smoothing
        smoothed_heatmap = gaussian_filter(normalized_heatmap, sigma=1)
        return smoothed_heatmap

    # Process the heatmap
    processed_heatmap = process_heatmap(heatmap)

    # Helper function to extract line data
    def extract_data_lines(ax_main, n_times):
        return [line for line in ax_main.lines if len(line.get_ydata()) == n_times]

    # Helper function to create colored line segments
    def create_line_segments(times, data_line, heatmap_values, xlim):
        # Filter data based on visible x-limits
        mask = (times >= xlim[0]) & (times <= xlim[1])
        filtered_times = times[mask]
        filtered_data_line = data_line[mask]
        filtered_heatmap_values = heatmap_values[mask]

        if len(filtered_times) < 2:
            return [], []

        # Create segments for the line
        seg = np.column_stack([filtered_times, filtered_data_line])
        segments = [seg[i:i + 2] for i in range(len(seg) - 1)]

        # Map heatmap values to colors using the rainbow colormap
        rgba_colors = [scalar_mappable.to_rgba(val) for val in filtered_heatmap_values[:-1]]
        rgba_colors = [(r, g, b, 1.0) for r, g, b, _ in rgba_colors]  # Ensure full opacity

        return segments, rgba_colors

    # Create a line collection from heatmap-mapped colors
    def create_line_collection(xlim):
        segments, colors = [], []
        for ch_idx, line in enumerate(data_lines):
            if ch_idx < processed_heatmap.shape[0]:
                heatmap_values = processed_heatmap[ch_idx]
                times = np.array(line.get_xdata())
                data_line = np.array(line.get_ydata())

                if len(heatmap_values) != len(times):
                    continue

                segs, rgba_colors = create_line_segments(times, data_line, heatmap_values, xlim)
                segments.extend(segs)
                colors.extend(rgba_colors)

        return LineCollection(segments, colors=colors, linewidth=2.5)

    # Update line colors dynamically during zoom/pan
    def update_line_colors():
        nonlocal line_collection
        if line_collection:
            line_collection.remove()
        line_collection = create_line_collection(ax_main.get_xlim())
        ax_main.add_collection(line_collection)
        fig.canvas.draw_idle()

    # Remove line colors
    def remove_line_colors():
        nonlocal line_collection
        if line_collection:
            line_collection.remove()
            line_collection = None
        fig.canvas.draw_idle()

    # Toggle line color mapping
    def toggle_colors(event):
        nonlocal color_on
        color_on = not color_on
        if color_on:
            update_line_colors()
        else:
            remove_line_colors()

    # Toggle colorbar visibility
    def toggle_colorbar(event):
        nonlocal colorbar_on
        colorbar_on = not colorbar_on
        cbar_ax.set_visible(colorbar_on)
        fig.canvas.draw_idle()

    # Handle x-axis limit changes (e.g., during zoom/pan)
    def on_xlim_changed(event_ax):
        if color_on:
            update_line_colors()

    # Initialize state variables
    color_on, colorbar_on = True, True

    # Create a colormap and normalize values
    norm = Normalize(vmin=0, vmax=1)
    scalar_mappable = ScalarMappable(norm=norm, cmap='rainbow')  # Use the rainbow colormap

    # Plot raw EEG data
    fig = mne.viz.plot_raw(raw, show=False, **kwargs)
    fig.canvas.draw()
    ax_main = fig.mne.ax_main
    data_lines = extract_data_lines(ax_main, raw.n_times)

    # Initialize the line collection for heatmap colors
    line_collection = None
    update_line_colors()

    # Add buttons for toggling heatmap and colorbar
    ax_button_color = plt.axes([0.81, 0.05, 0.1, 0.075])
    button_color = Button(ax_button_color, 'Toggle Colors')
    button_color.on_clicked(toggle_colors)

    cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label('Normalized Heatmap Values')

    ax_button_colorbar = plt.axes([0.81, 0.85, 0.1, 0.075])
    button_colorbar = Button(ax_button_colorbar, 'Toggle Colorbar')
    button_colorbar.on_clicked(toggle_colorbar)

    fig.mne.ax_main.callbacks.connect('xlim_changed', on_xlim_changed)

    plt.show()
