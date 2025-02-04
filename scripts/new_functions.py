import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Optional
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers, models


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


def getEq_times(
    system: dict[str, Any], databankPath: str
) -> tuple[Optional[list[Any]], Optional[list[Any]]]:
    """
    Returns eq_times of the simulation

    :param system: NMRlipids databank dictionary describing the simulation
    :param databankPath: Path to the databank

    :return: eq_times of the simulation
    """
    databankPath = Path(databankPath)
    eq_times_path = (
        databankPath / "Data" / "Simulations" / system["path"] / "eq_times.json"
    )

    # Load eq_times
    try:
        with open(eq_times_path, "r") as json_file:
            eq_times = json.load(json_file)
    except Exception:
        eq_times = None

    return eq_times


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
        ax.plot(row.to_list(), linewidth=0.5)
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


def rescale_to_zero_centered_unit_range(
    values: np.ndarray, min_value: float, max_value: float
):
    """
    Rescale values to [-0.5, 0.5] range consistent with provided global min and max values

    :param values: values to rescale
    :param min_value: minimum of original range
    :param max_value: maximum of original range

    :return: Values rescaled to range of [-0.5, 0.5]
    """
    return (values - min_value) / (max_value - min_value) - 0.5


def rescale_back_to_true_range(values: np.ndarray, min_value: float, max_value: float):
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


def add_noise_to_form_factor(
    form_factor_df: pd.DataFrame,
    steepness: float = 10.0,
    max_noise_magnitude: float = 1.0,
    probability_of_noise: float = 1.0,
) -> pd.DataFrame:
    """
    Adds random noise to each row of the input DataFrame, with noise amplitude determined by a pure sigmoid function (scaled by `max_noise_magnitude`) across the columns. The amplitude starts near 0 in the first column and approaches `max_noise_magnitude` in the last column.

    The amplitude for column i is calculated as:
        amplitude(i) = max_noise_magnitude * [ 1 / (1 + exp(-steepness * (fraction - 0.5))) ]
    where fraction = i / (num_cols - 1).

    :param form_factor_df: A pandas DataFrame where each row is a form factor
    :param steepness: Controls how quickly the sigmoid transitions from 0 to max_noise_magnitude. Higher values make the transition sharper around the midpoint.
    :param max_noise_magnitude: The maximum amplitude of the noise (occurs at the highest q value).
    :param probability_of_noise: The chance that a single form factor will be noisy.

    :return: A new pandas DataFrame of the same shape, with column-wise noise added.
    """
    num_rows, num_cols = form_factor_df.shape

    # Array of values from 0 to 1 in the dimension of q, to make the sigmoid
    fraction_array = np.linspace(0.0, 1.0, num_cols)

    # Noise amplitudes follow the logistic (sigmoid) function, scaled by max_noise_magnitude
    # At fraction=0, amplitude ~ 0; at fraction=1, amplitude ~ max_noise_magnitude
    noise_amplitudes = max_noise_magnitude * (
        1.0 / (1.0 + np.exp(-steepness * (fraction_array - 0.5)))
    )

    # Replicate amplitudes for each row. The shape becomes (num_rows, num_cols)
    scale_array = np.tile(noise_amplitudes, (num_rows, 1))

    # Generate a mask indicating which rows should receive noise (80% probability)
    noise_mask = np.random.rand(num_rows) < probability_of_noise

    # Generate random noise with per-column standard deviation according to noise_amplitudes
    noise = np.random.normal(loc=0.0, scale=scale_array)

    # Add noise to the original data
    noisy_values = form_factor_df.values.copy()
    noisy_values[noise_mask] += noise[noise_mask]

    # Return as a DataFrame (preserving index and columns)
    return pd.DataFrame(
        noisy_values, index=form_factor_df.index, columns=form_factor_df.columns
    )


def build_fully_connected_model(
    input_dim: int,
    hidden_layer_dims: tuple[int, ...],
    output_dim: int,
) -> tf.keras.Model:
    """
    Builds and compiles a fully connected neural network Keras model.

    :param input_dim: Dimension of the input to the neural network
    :param hidden_layer_dims: A tuple specifying the number of nodes in each hidden layer
    :param output_dim: Dimension of the output layer

    :return: A compiled TensorFlow Keras Model ready for training
    """
    keras_input = keras.Input(shape=(input_dim,))
    x = keras_input

    # Create as many hidden layers as specified in hidden_layer_dims
    for dim in hidden_layer_dims:
        x = layers.Dense(dim, activation="relu")(x)

    # Final output layer
    output_layer = layers.Dense(output_dim, activation="linear")(x)

    model = tf.keras.Model(inputs=keras_input, outputs=output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(amsgrad=True),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def build_convolution_model(
    input_dim: int,
    hidden_layer_filters: tuple[int, ...],
    output_dim: int,
    kernel_size: int = 3,
    activation_function: str = "relu",
) -> tf.keras.Model:
    """
    Builds and compiles a 1D convolutional model for regression.

    :param input_dim: The length of the input sequence.
    :param output_dim: The dimension of the output layer.
    :param kernel_size: The size of the convolution kernel.
    :param activation_function: The activation function to use in the Conv1D layers.

    :return: A compiled 1D convolutional tf.keras.Model neural network model
    """
    keras_input = tf.keras.Input(shape=(input_dim, 1))
    x = keras_input

    # Create as many hidden layers as specified in hidden_layer_filters
    for n_filters in hidden_layer_filters:
        x = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation_function,
        )(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # Flatten and final dense layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_dim, activation=None)(x)

    model = models.Model(inputs=keras_input, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
