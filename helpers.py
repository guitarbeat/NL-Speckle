import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional, Union
import streamlit as st

def add_kernel_overlay(ax: plt.Axes, last_x: int, last_y: int, kernel_size: int) -> None:
    kernel_start_x = last_x - kernel_size // 2
    kernel_start_y = last_y - kernel_size // 2
    
    # Add kernel rectangle
    kernel_rect = plt.Rectangle((kernel_start_x - 0.5, kernel_start_y - 0.5), kernel_size, kernel_size, 
                                edgecolor="r", linewidth=2, facecolor="none")
    ax.add_patch(kernel_rect)
    
    # Add kernel grid
    kernel_lines = ([[(kernel_start_x - 0.5 + i, kernel_start_y - 0.5), (kernel_start_x - 0.5 + i, kernel_start_y + kernel_size - 0.5)] for i in range(1, kernel_size)] +
                    [[(kernel_start_x - 0.5, kernel_start_y - 0.5 + i), (kernel_start_x + kernel_size - 0.5, kernel_start_y - 0.5 + i)] for i in range(1, kernel_size)])
    ax.add_collection(LineCollection(kernel_lines, colors='gray', linestyles=':', linewidths=0.5))

def add_search_window(ax: plt.Axes, search_window: Union[str, int], last_x: int, last_y: int, image_shape: Tuple[int, int]) -> None:
    if search_window == "full":
        search_rect = plt.Rectangle((-0.5, -0.5), image_shape[1], image_shape[0], 
                                    edgecolor="blue", linewidth=2, facecolor="none")
    elif isinstance(search_window, int):
        half_window = search_window // 2
        search_x_start = last_x - half_window
        search_y_start = last_y - half_window
        search_rect = plt.Rectangle((search_x_start - 0.5, search_y_start - 0.5), 
                                    search_window, search_window,
                                    edgecolor="blue", linewidth=2, facecolor="none")
    ax.add_patch(search_rect)

def create_plot(main_image: np.ndarray, overlays: List[np.ndarray], last_x: int, last_y: int, kernel_size: int, titles: List[str], cmap: str = "viridis", search_window: Optional[Union[str, int]] = None, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    n_plots = 1 + len(overlays)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    axs = [axs] if n_plots == 1 else axs

    # Plot main image with overlays
    vmin, vmax = np.min(main_image), np.max(main_image)
    axs[0].imshow(main_image, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(titles[0])
    axs[0].axis('off')
    
    add_kernel_overlay(axs[0], last_x, last_y, kernel_size)
    
    if search_window is not None:
        add_search_window(axs[0], search_window, last_x, last_y, main_image.shape)

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
