import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_total_densities_to_ax(
    ax: Axes,
    all_td_x: list[float],
    all_td_y: list[float],
    x_axis_label_td: str,
    y_axis_label_td: str,
    lines: list[float] = [],
) -> Axes:
    """
    Plot all total density profiles to ax

    :param ax: Axes object to plot on.
    :param all_td_x: List of lists: x coordinates for each total density profile
    :param all_td_y: List of lists: y coordinates for each total density profile
    :param x_axis_label_td: y axis label for total density
    :param y_axis_label_td: x axis label for total density
    :param lines: List of x values to plot as vertical lines

    :return: ax object with total densities (and lines) plotted
    """
    if isinstance(all_td_y, list):
        for x_vector, y_vector in zip(all_td_x, all_td_y):
            ax.plot(x_vector, y_vector)
    elif isinstance(all_td_y, pd.DataFrame):
        for _, row in all_td_y.iterrows():
            ax.plot(all_td_x, row.to_list())
    for value in lines:
        ax.axvline(value, color="k", linestyle="solid")
    ax.set_xlabel(x_axis_label_td)
    ax.set_ylabel(y_axis_label_td)
    return ax


def plot_form_factors_to_ax(
    ax: Axes, sim_FF_df: pd.DataFrame, x_axis_label_ff: str, y_axis_label_ff: str
) -> Axes:
    """
    Plot all form factor profiles to ax

    :param ax: Axes object to plot on
    :param sim_FF_df: pd.DataFrame with form factors as rows
    :param x_axis_label_ff: y axis label for form factor
    :param y_axis_label_ff: x axis label for form factor

    :return: ax object with form factors plotted
    """
    for index, row in sim_FF_df.iterrows():
        ax.plot(row.to_list(), linewidth=0.5)
    ax.set_xlabel(x_axis_label_ff)
    ax.set_ylabel(y_axis_label_ff)
    return ax


def plot_training_trajectory(ax: Axes, history: object) -> Axes:
    """
    Plot the training trajectory

    :param ax: Axes object to plot on
    :param history: Training history object output from the model fit procedure

    :return: ax object with training history plotted
    """
    ax.plot(history["loss"], color="green", label="Training loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], color="orange", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History (Final Re-Fit)")
    ax.legend()
    return ax


def plot_absolute_deviation_to_ax(
    fig: Figure,
    ax: Axes,
    x_values: np.ndarray,
    absolute_deviation: np.ndarray,
    x_axis_label_td: str,
    y_axis_label_td: str,
    title: str = "",
    height: float = 6,
    width: float = 8,
    label: str = "",
) -> Axes:
    """
    Plot absolute deviation to ax

    :param ax: Axes object to plot on
    :param absolute_deviation: Absolute deviation in total density

    :return: ax object with absolute deviations plotted
    """
    if len(absolute_deviation.shape) == 1:
        ax.plot(x_values, absolute_deviation, linestyle="-", label=label)
        ax.legend()
    else:
        for residuals in absolute_deviation:
            ax.plot(x_values, residuals, color="k", linestyle="-", alpha=0.2)
    ax.set_xlabel(x_axis_label_td)
    ax.set_ylabel(y_axis_label_td)
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax
