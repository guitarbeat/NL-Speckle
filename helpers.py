import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from typing import Tuple, List, Optional
from matplotlib.colors import Colormap
import streamlit as st

def create_search_window(x: int, y: int, window_size: int | str, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    if window_size == "full":
        return 0, 0, image_shape[1], image_shape[0]
    else:
        half_window = window_size // 2
        return (max(0, x - half_window), max(0, y - half_window),
                min(image_shape[1], x + half_window), min(image_shape[0], y + half_window))

def apply_colormap_to_images(img1: np.ndarray, img2: np.ndarray, cmap: Colormap) -> Tuple[np.ndarray, np.ndarray]:
    normalize = lambda img: (img - np.min(img)) / (np.max(img) - np.min(img))
    apply_cmap = lambda img: (cmap(normalize(img))[:, :, :3] * 255).astype(np.uint8)
    return apply_cmap(img1), apply_cmap(img2)

def configure_axes(ax: plt.Axes, title: str, image: np.ndarray, cmap: str = "viridis", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = np.min(image), np.max(image)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

def add_rectangle(ax: plt.Axes, x: int, y: int, width: int, height: int, show_grid: bool = True, **kwargs) -> None:
    rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, **kwargs)
    ax.add_patch(rect)
    if show_grid:
        lines = ([[(x - 0.5 + i, y - 0.5), (x - 0.5 + i, y + height - 0.5)] for i in range(1, width)] +
                 [[(x - 0.5, y - 0.5 + i), (x + width - 0.5, y - 0.5 + i)] for i in range(1, height)])
        ax.add_collection(LineCollection(lines, colors='gray', linestyles=':', linewidths=0.5))

def plot_image_with_overlays(ax: plt.Axes, image: np.ndarray, last_x: int, last_y: int, kernel_size: int, title: str, cmap: str = "viridis", search_window: Optional[int] = None) -> None:
    configure_axes(ax, title, image, cmap=cmap, vmin=np.min(image), vmax=np.max(image))
    add_rectangle(ax, last_x, last_y, kernel_size, kernel_size, edgecolor="r", linewidth=2, facecolor="none")
    if search_window is not None:
        if isinstance(search_window, int) or search_window == "full":
            search_x_start, search_y_start, search_x_end, search_y_end = create_search_window(last_x, last_y, search_window, image.shape)
            add_rectangle(ax, search_x_start, search_y_start, search_x_end - search_x_start, search_y_end - search_y_start,
                          edgecolor="g", linewidth=2, facecolor="none", show_grid=False)

def update_plot(main_image: np.ndarray, overlays: List[np.ndarray], last_x: int, last_y: int, kernel_size: int, titles: List[str], cmap: str = "viridis", search_window: Optional[int] = None, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    n_plots = 1 + len(overlays)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    axs = [axs] if n_plots == 1 else axs
    plot_image_with_overlays(axs[0], main_image, last_x, last_y, kernel_size, titles[0], cmap, search_window)
    for ax, overlay, title in zip(axs[1:], overlays, titles[1:]):
        configure_axes(ax, title, overlay, cmap=cmap)
    fig.tight_layout(pad=2)
    return fig

def display_data_and_zoomed_view(data: np.ndarray, full_data: np.ndarray, last_x: int, last_y: int, stride: int, title: str, data_placeholder: st.empty, zoomed_placeholder: st.empty, cmap: str = "viridis", zoom_size: int = 1, fontsize: int = 10, text_color: str = "red") -> None:
    fig_full, _ = plt.subplots()
    configure_axes(fig_full.gca(), title, data, cmap)
    data_placeholder.pyplot(fig_full)
    
    zoomed_data = data[last_y // stride : last_y // stride + zoom_size, last_x // stride : last_x // stride + zoom_size]
    fig_zoom, ax_zoom = plt.subplots()
    configure_axes(ax_zoom, f"Zoomed-In {title}", zoomed_data, cmap)
    for i, row in enumerate(zoomed_data):
        for j, val in enumerate(row):
            ax_zoom.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    fig_zoom.tight_layout(pad=2)
    zoomed_placeholder.pyplot(fig_zoom)

def display_kernel_view(kernel_data: np.ndarray, full_image_data: np.ndarray, title: str, placeholder: st.empty, cmap: str = "viridis", fontsize: int = 10, text_color: str = "red") -> None:
    vmin, vmax = np.min(full_image_data), np.max(full_image_data)
    fig, ax = plt.subplots()
    configure_axes(ax, title, kernel_data, cmap, vmin, vmax)
    for i, row in enumerate(kernel_data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=fontsize)
    fig.tight_layout(pad=2)
    placeholder.pyplot(fig)
