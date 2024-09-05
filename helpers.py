import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional
import streamlit as st

 
def create_plot(main_image: np.ndarray, overlays: List[np.ndarray], last_x: int, last_y: int, kernel_size: int, titles: List[str], cmap: str = "viridis", search_window: Optional[int] = None, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    n_plots = 1 + len(overlays)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    axs = [axs] if n_plots == 1 else axs

    # Plot main image with overlays
    vmin, vmax = np.min(main_image), np.max(main_image)
    axs[0].imshow(main_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(titles[0])
    axs[0].axis('off')
    
    # Add kernel rectangle
    kernel_start_x = last_x - kernel_size // 2
    kernel_start_y = last_y - kernel_size // 2
    kernel_rect = plt.Rectangle((kernel_start_x - 0.5, kernel_start_y - 0.5), kernel_size, kernel_size, edgecolor="r", linewidth=2, facecolor="none")
    axs[0].add_patch(kernel_rect)
    
    # Add kernel grid
    kernel_lines = ([[(kernel_start_x - 0.5 + i, kernel_start_y - 0.5), (kernel_start_x - 0.5 + i, kernel_start_y + kernel_size - 0.5)] for i in range(1, kernel_size)] +
                    [[(kernel_start_x - 0.5, kernel_start_y - 0.5 + i), (kernel_start_x + kernel_size - 0.5, kernel_start_y - 0.5 + i)] for i in range(1, kernel_size)])
    axs[0].add_collection(LineCollection(kernel_lines, colors='gray', linestyles=':', linewidths=0.5))
    
    if search_window is not None:
        if search_window == "full":  # Full image search window
            search_rect = plt.Rectangle((-0.5, -0.5), main_image.shape[1], main_image.shape[0], 
                                        edgecolor="blue", linewidth=2, facecolor="none")
            axs[0].add_patch(search_rect)
        elif isinstance(search_window, int):  # Integer search window size
            half_window = search_window // 2
            search_x_start = last_x - half_window
            search_y_start = last_y - half_window
            search_rect = plt.Rectangle((search_x_start - 0.5, search_y_start - 0.5), 
                                        search_window, search_window,
                                        edgecolor="blue", linewidth=2, facecolor="none")
            axs[0].add_patch(search_rect)

    # Plot overlays
    for ax, overlay, title in zip(axs[1:], overlays, titles[1:]):
        ax.imshow(overlay, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')

    fig.tight_layout(pad=2)
    return fig

def display_kernel_view(kernel_data: np.ndarray, full_image_data: np.ndarray, title: str, placeholder: st.empty, cmap: str = "viridis", fontsize: int = 10, text_color: str = "red") -> None:
    vmin, vmax = np.min(full_image_data), np.max(full_image_data)
    fig, ax = plt.subplots()
    
    ax.imshow(kernel_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    
    for i, row in enumerate(kernel_data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    
    fig.tight_layout(pad=2)
    placeholder.pyplot(fig)
