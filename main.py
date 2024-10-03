"""
Main entry point for the Speckle Contrast Visualization Streamlit application.
"""

import streamlit as st
import numpy as np
from src.sidebar import setup_ui
from src.images import ImageProcessor
from src.render import (
    create_technique_config,
    display_filters,
    get_zoomed_image_section,
)
from src.draw.formula import display_formula_details
import src.session_state as session_state
import matplotlib.pyplot as plt
from src.draw.overlay import add_overlays
import traceback

# App Configuration
APP_CONFIG = {
    "page_title": "Speckle Contrast Visualization",
    "layout": "wide",
    "page_icon": "favicon.png",
    "initial_sidebar_state": "expanded",
}


def main():
    """
    Main function to set up and run the Streamlit application.
    """
    try:
        st.set_page_config(**APP_CONFIG)
        session_state.initialize_session_state()
        setup_ui()

        # Get image array and name from session state
        image_array = session_state.get_image_array()
        image_name = session_state.get_session_state("image_file", "test")

        if image_array is None:
            st.warning("Please load an image first.")
        else:
            # Extract image dimensions
            height, width = image_array.shape

            # Calculate kernel size and processable pixel limits
            current_kernel_size = st.session_state.get(
                "kernel_size", session_state.DEFAULT_KERNEL_SIZE
            )
            half_kernel = current_kernel_size // 2
            processable_height = height - 2 * half_kernel
            processable_width = width - 2 * half_kernel
            total_processable_pixels = processable_height * processable_width

            # Layout the columns for the UI
            col1, col2, col3, col4, col5 = st.columns(5)

            # Per-pixel processing toggle and pixel limit slider
            with col1:
                show_per_pixel = st.toggle(
                    "Show Per-Pixel Processing",
                    value=st.session_state.get("show_per_pixel", False),
                    key="show_per_pixel_toggle",
                    on_change=session_state.update_show_per_pixel,
                )

                if show_per_pixel:
                    max_pixels = st.slider(
                        "Maximum Pixels to Process",
                        min_value=1,
                        max_value=total_processable_pixels,
                        value=min(
                            st.session_state.get("pixels_to_process", 1),
                            total_processable_pixels,
                        ),
                        step=1,
                        key="max_pixels_slider",
                        on_change=session_state.update_pixels_to_process,
                        help="Set the maximum number of pixels to process for per-pixel visualization",
                    )
                else:
                    max_pixels = total_processable_pixels

            # Kernel size adjustment
            with col2:
                st.number_input(
                    "Kernel Size",
                    min_value=1,
                    max_value=min(height, width),
                    value=current_kernel_size,
                    step=2,
                    key="kernel_size_input",
                    on_change=session_state.update_kernel_size,
                    help="Size of the kernel used for processing",
                )

            # Non-local means (NLM) options
            with col3:
                nlm_opts = st.session_state.get(
                    "nlm_options", session_state.DEFAULT_VALUES["nlm_options"]
                )
                use_whole_image = st.toggle(
                    "Use full image for search",
                    value=nlm_opts["use_whole_image"],
                    key="use_whole_image_checkbox",
                    on_change=session_state.update_nlm_use_whole_image,
                    help="Use the entire image as the search window for NLM",
                )

                max_search_window = min(101, height, width)
                min_search_window = min(21, max_search_window)

                search_window_size = (
                    max_search_window
                    if use_whole_image
                    else st.slider(
                        "Search Window Size",
                        min_value=min_search_window,
                        max_value=max_search_window,
                        value=nlm_opts["search_window_size"],
                        step=2,
                        key="search_window_size_slider",
                        on_change=session_state.update_nlm_search_window_size,
                        help="Size of the search window for NLM (must be odd)",
                    )
                )

                # Ensure search window size is odd
                if search_window_size % 2 == 0:
                    search_window_size += 1

            with col4:
                # Filter strength adjustment
                st.slider(
                    "Filter Strength",
                    min_value=0.1,
                    max_value=100.0,
                    value=nlm_opts["filter_strength"],
                    step=0.1,
                    format="%.1f",
                    key="filter_strength_slider",
                    on_change=session_state.update_nlm_filter_strength,
                    help="Filter strength for NLM (higher values mean more smoothing)",
                )

            # Display information about pixel processing
            with col5:
                if show_per_pixel:
                    st.info(
                        f"Processing {max_pixels:,} out of {total_processable_pixels:,} pixels ({max_pixels / total_processable_pixels:.2%})"
                    )
                else:
                    st.info(f"Processing all {total_processable_pixels:,} pixels")

            # Update session state with processable area and pixels to process
            processable_area = {
                "top": half_kernel,
                "bottom": height - half_kernel,
                "left": half_kernel,
                "right": width - half_kernel,
            }
            st.session_state["processable_area"] = processable_area
            st.session_state["pixels_to_process"] = max_pixels

        # Create tabs for LSCI and NL-Means
        tab_speckle, tab_nlm = st.tabs(["LSCI", "NL-Means"])

        for technique_tab in [("lsci", tab_speckle), ("nlm", tab_nlm)]:
            technique, tab = technique_tab
            with tab:
                if session_state.get_session_state("image") is None:
                    st.warning("Please load an image before processing.")
                    continue

                # Handle filter selection UI
                filter_options = session_state.get_filter_options(technique)
                if f"{technique}_filters" not in st.session_state:
                    st.session_state[f"{technique}_filters"] = (
                        session_state.get_filter_selection(technique)
                    )

                selected_filters = st.multiselect(
                    f"Select {technique.upper()} filters to display",
                    options=filter_options,
                    default=st.session_state[f"{technique}_filters"],
                    key=f"{technique}_filter_selection",
                )

                if selected_filters != st.session_state[f"{technique}_filters"]:
                    st.session_state[f"{technique}_filters"] = selected_filters
                    session_state.set_session_state(
                        f"{technique}_filters", selected_filters
                    )

                # Process the technique if needed
                if session_state.needs_processing(technique):
                    image = session_state.get_image_array()
                    params = session_state.get_technique_params(technique)
                    if image is None or image.size == 0 or params is None:
                        error_message = f"{'No image data found' if image is None or image.size == 0 else 'No parameters found for ' + technique}. Please check your input."
                        st.error(error_message)
                        continue

                    pixels_to_process = session_state.get_session_state(
                        "pixels_to_process", image.size
                    )
                    kernel_size = params.get(
                        "kernel_size", 3
                    )  # Default to 3 if not specified
                    params.update(
                        {
                            "pixels_to_process": pixels_to_process,
                            "kernel_size": kernel_size,
                        }
                    )

                    try:
                        processor = ImageProcessor(
                            image=image.astype(np.float32),
                            technique=technique,
                            params=params,
                            image_name=image_name,
                        )
                        result = processor.run_sequential_processing()
                        if result is not None:
                            session_state.set_session_state(
                                f"{technique}_result", result
                            )
                            session_state.set_last_processed(
                                technique, pixels_to_process
                            )
                    except Exception as e:
                        error_message = (
                            f"An error occurred during image processing: {str(e)}"
                        )
                        st.error(error_message)
                        traceback_info = traceback.format_exc()
                        print(f"Traceback:\n{traceback_info}")
                        raise RuntimeError(
                            f"{error_message}\nDebug info:\n{traceback_info}"
                        )

                # Retrieve and display results
                result = session_state.get_technique_result(technique)
                if result is None:
                    st.warning(
                        f"No results available for {technique}. Processing may have failed."
                    )
                    continue

                config = create_technique_config(technique, tab)
                if config is not None:
                    display_data = display_filters(config)
                    for plot_config, placeholder, zoomed in display_data:
                        # Display image logic (previously in display_image function)
                        if zoomed:
                            zoom_data, center = get_zoomed_image_section(
                                plot_config["filter_data"],
                                plot_config["last_processed_pixel"][1],  # x-coordinate
                                plot_config["last_processed_pixel"][0],  # y-coordinate
                                plot_config["kernel"]["size"],
                            )
                        else:
                            zoom_data = plot_config["filter_data"]
                            center = None

                        fig, ax = plt.subplots(
                            1, 1, figsize=(4 if zoomed else 8, 4 if zoomed else 8)
                        )
                        ax.set_title(
                            f"{'Zoomed ' if zoomed else ''}{plot_config['title']}"
                        )
                        ax.imshow(
                            zoom_data,
                            vmin=plot_config["vmin"],
                            vmax=plot_config["vmax"],
                            cmap=session_state.get_color_map(),
                        )
                        ax.axis("off")

                        if zoomed:
                            plot_config["zoom"] = True
                            plot_config["last_processed_pixel"] = center

                        add_overlays(ax, zoom_data, plot_config)
                        fig.tight_layout(pad=2)

                        if zoomed:
                            placeholder_key = f"zoomed_{plot_config['title'].lower().replace(' ', '_')}"
                            if placeholder_key in placeholder:
                                placeholder[placeholder_key].pyplot(fig)
                        else:
                            placeholder.pyplot(fig)

                        plt.close(fig)

                    display_formula_details(config)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
