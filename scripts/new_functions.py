import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib.axes import Axes
from typing import Any, Optional
from pathlib import Path


def getFormFactorAndTotalDensityPair(
    system: dict[str, Any], databankPath: str
) -> tuple[Optional[list[Any]], Optional[list[Any]]]:
    """
    Returns form factor and total density profiles of the simulation

    :param system: NMRlipids databank dictionary describing the simulation
    :param databankPath: Path to the databank

    :return: form factor (FFsim) and total density (TDsim) of the simulation
    """
    databankPath = Path(databankPath)
    FFpathSIM = (
        databankPath / "Data" / "Simulations" / system["path"] / "FormFactor.json"
    )
    TDpathSIM = (
        databankPath / "Data" / "Simulations" / system["path"] / "TotalDensity.json"
    )

    # Load form factor and total density
    try:
        with open(FFpathSIM, "r") as json_file:
            FFsim = json.load(json_file)
        with open(TDpathSIM, "r") as json_file:
            TDsim = json.load(json_file)
    except Exception:
        FFsim = None
        TDsim = None

    return FFsim, TDsim


def getPOPC(
    system: dict[str, Any], databankPath: str
) -> tuple[Optional[list[Any]], Optional[list[Any]]]:
    """
    Returns POPC of the simulation

    :param system: NMRlipids databank dictionary describing the simulation
    :param databankPath: Path to the databank

    :return: POPC of the simulation
    """
    databankPath = Path(databankPath)
    eq_times_path = (
        databankPath / "Data" / "Simulations" / system["path"] / "eq_times.json"
    )

    # Load POPC file
    try:
        with open(eq_times_path, "r") as json_file:
            eq_times = json.load(json_file)
            POPC = eq_times["POPC"]
    except Exception:
        POPC = None

    return POPC


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

    :return: ax object with form factors plotted
    """
    for index, row in sim_FF_df.iterrows():
        ax.plot(row.to_list())
    ax.set_xlabel(x_axis_label_ff)
    ax.set_ylabel(y_axis_label_ff)
    return ax


def extrapolate_X(
    x_vector: np.ndarray,
    desired_length_of_padded_data: int,
    x_interval_start: float,
    x_interval_end: float,
) -> np.ndarray:
    """
    Extrapolates total density x values to match desired x range and dimensionality

    :param x_vector: Original total density x values
    :param desired_length_of_padded_data: Desired length of padded data
    :param x_interval_start: Lower end of range for the homogenized data
    :param x_interval_end: Lower end of range for the homogenized data

    :return: padded x vector
    """
    padding_length = max(0, desired_length_of_padded_data - len(x_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    x_min = min(x_vector)
    x_max = max(x_vector)

    # Check if the range of the x values is smaller than the required range:
    if x_min > x_interval_start and x_max < x_interval_end:
        # If narrower, extrapolate in the x direction by replicating the y values at the ends
        padding_start = np.linspace(
            x_interval_start, x_min, num=max(0, first_padding_length), endpoint=False
        )
        padding_end = np.linspace(
            x_max, x_interval_end, num=max(0, last_padding_length), endpoint=False
        )
    elif x_min < x_interval_start and x_max > x_interval_end:
        # If wider, pad at the ends without extrapolating to make dimensions equal
        padding_start = np.repeat(x_min, first_padding_length)
        padding_end = np.repeat(x_max, last_padding_length)
    else:
        raise NotImplementedError
    return np.concatenate([padding_start, x_vector, padding_end])


def extrapolate_Y(
    y_vector: np.ndarray, desired_length_of_padded_data: int
) -> np.ndarray:
    """
    Extrapolates total density y values by repeating the y values at the ends of the observation window

    :param y_vector: Original total density y values
    :param desired_length_of_padded_data: Desired length of padded data

    :return: padded y vector
    """
    padding_length = max(0, desired_length_of_padded_data - len(y_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    first_pad_value = y_vector[0]
    last_pad_value = y_vector[-1]

    padding_start = [first_pad_value] * first_padding_length
    padding_end = [last_pad_value] * last_padding_length
    return np.concatenate([padding_start, y_vector, padding_end])


def rescale_to_zero_centered_unit_range(values, min_value, max_value):
    """
    Rescale values to [-0.5, 0.5] range consistent with provided global min and max values

    :param values: values to rescale
    :param min_value: minimum of original range
    :param max_value: maximum of original range

    :return: Values rescaled to range of [-0.5, 0.5]
    """
    return (values - min_value) / (max_value - min_value) - 0.5


def rescale_back_to_true_range(values, min_value, max_value):
    """
    Rescale values back to original range

    :param values: values in range [-0.5, 0.5] to rescale
    :param min_value: minimum of range
    :param max_value: maximum of range

    :return: Values rescaled to original range
    """
    return (values + 0.5) * (max_value - min_value) + min_value


def interpolate_with_GPR(
    all_td_x: list[np.ndarray],
    all_td_y: list[np.ndarray],
    uniform_x_range: np.ndarray,
) -> list[np.ndarray]:
    """
    Fits a Gaussian process regression to the observed points, then predicts values on the  uniform grid between the observations.
    To ensure the GPR hyperparameters are well suited to all cases, the densities are scaled to a range of [-0.5, 0.5], then scaled back again after interpolation

    :param rescaled_all_x: List of total density x values (np.ndarray) for all cases
    :param rescaled_all_y: List of total density y values (np.ndarray) for all cases
    :param uniform_x_range: X coordinates on which the y values will be predicted for all patients

    :return: List of np.ndarrays, where each np.ndarray contains the total density y values predicted by on uniform x range for one patient
    """
    # Before interpolation, rescale all x and y values to a range of [-0.5, 0.5]
    global_min_td_x = min(min(td_x) for td_x in all_td_x)
    global_min_td_y = min(min(td_y) for td_y in all_td_y)

    global_max_td_x = max(max(td_x) for td_x in all_td_x)
    global_max_td_y = max(max(td_y) for td_y in all_td_y)

    rescaled_all_td_x = [
        rescale_to_zero_centered_unit_range(td_x, global_min_td_x, global_max_td_x)
        for td_x in all_td_x
    ]
    rescaled_all_td_y = [
        rescale_to_zero_centered_unit_range(td_y, global_min_td_y, global_max_td_y)
        for td_y in all_td_y
    ]

    rescaled_start = rescale_to_zero_centered_unit_range(
        uniform_x_range[0], global_min_td_x, global_max_td_x
    )
    rescaled_end = rescale_to_zero_centered_unit_range(
        uniform_x_range[-1], global_min_td_x, global_max_td_x
    )
    rescaled_uniform_x_range = np.linspace(
        rescaled_start, rescaled_end, len(uniform_x_range)
    ).reshape(-1, 1)

    # Create and fit Gaussian process regressor
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    interpolated_y = [
        gp.fit(x_vector.reshape(-1, 1), y_vector).predict(rescaled_uniform_x_range)
        for x_vector, y_vector in zip(rescaled_all_td_x, rescaled_all_td_y)
    ]

    # Scale back to original range
    return [
        rescale_back_to_true_range(y_vector, global_min_td_y, global_max_td_y)
        for y_vector in interpolated_y
    ]


def plot_training_trajectory(ax: Axes, history: object) -> Axes:
    """
    Plot the training trajectory

    :param ax: Axes object to plot on
    :param history: Training history object output from the model fit procedure

    :return: ax object with training history plotted
    """
    ax.plot(history.history["loss"], color="green", label="Training loss")
    ax.plot(history.history["val_loss"], color="orange", label="Validation loss")
    ax.legend()
    return ax
