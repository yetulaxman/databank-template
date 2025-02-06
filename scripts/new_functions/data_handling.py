import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


def rescale_to_zero_centered_unit_range(
    values: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """
    Rescale values to [-0.5, 0.5] range consistent with provided global min and max values

    :param values: values to rescale
    :param min_value: minimum of original range
    :param max_value: maximum of original range

    :return: Values rescaled to range of [-0.5, 0.5]
    """
    if abs(max_value - min_value) < 1e-9:
        raise ValueError(
            "min_value and max_value must be sufficiently different to avoid division errors."
        )
    return (values - min_value) / (max_value - min_value) - 0.5


def rescale_back_to_true_range(
    values: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    """
    Rescale values back to original range

    :param values: values in range [-0.5, 0.5] to rescale
    :param min_value: minimum of range
    :param max_value: maximum of range

    :return: Values rescaled to original range
    """
    return (values + 0.5) * (max_value - min_value) + min_value


def extrapolate_X(
    x_vector: np.ndarray,
    length_of_padded_data: int,
    x_interval_start: float,
    x_interval_end: float,
) -> np.ndarray:
    """
    If x_vector is more narrow than the x interval, this function extrapolates it in both direction so the resulting x vector spans the desired x range and has dimensionality equal to length_of_padded_data
    Otherwise, if x_vector is wider than the x interval, the values at each end of x vector are repeated to make the dimensionality equal to length_of_padded_data

    :param x_vector: Original total density x values
    :param length_of_padded_data: Desired length of padded data
    :param x_interval_start: Lower end of range for the homogenized data
    :param x_interval_end: Lower end of range for the homogenized data

    :return: padded x vector
    """
    padding_length = max(0, length_of_padded_data - len(x_vector))
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


def extrapolate_Y(y_vector: np.ndarray, length_of_padded_data: int) -> np.ndarray:
    """
    Extrapolates total density y values by repeating the y values at both ends of the observation window until the dimension of y vector is equal to length_of_padded_data

    :param y_vector: Original total density y values
    :param length_of_padded_data: Desired length of padded data

    :return: padded y vector
    """
    padding_length = max(0, length_of_padded_data - len(y_vector))
    first_padding_length = padding_length // 2
    last_padding_length = padding_length - first_padding_length

    first_pad_value = y_vector[0]
    last_pad_value = y_vector[-1]

    padding_start = [first_pad_value] * first_padding_length
    padding_end = [last_pad_value] * last_padding_length
    return np.concatenate([padding_start, y_vector, padding_end])


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
    global_min_td_x = np.min(all_td_x)
    global_max_td_x = np.max(all_td_x)

    global_min_td_y = np.min(all_td_y)
    global_max_td_y = np.max(all_td_y)

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
    gp_regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
    interpolated_y = [
        gp_regressor.fit(x_vector.reshape(-1, 1), y_vector).predict(
            rescaled_uniform_x_range
        )
        for x_vector, y_vector in zip(rescaled_all_td_x, rescaled_all_td_y)
    ]

    # Scale back to original range
    return [
        rescale_back_to_true_range(y_vector, global_min_td_y, global_max_td_y)
        for y_vector in interpolated_y
    ]


def add_noise_to_form_factor(
    form_factor_df: pd.DataFrame,
    steepness: float = 10.0,
    max_noise_magnitude: float = 1.0,
    probability_of_noise: float = 1.0,
) -> pd.DataFrame:
    """
    Adds simulated noise that emulates noise seen in experimental data to each row of the input DataFrame.
    The noise amplitude is determined by a pure sigmoid function (scaled by `max_noise_magnitude`) across the columns.
    The amplitude starts near 0 in the first column and approaches `max_noise_magnitude` in the last column.

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

    # Noise amplitudes follow the logistic (sigmoid) function, scaled by max_noise_magnitude
    # At fraction=0, amplitude ~ 0; at fraction=1, amplitude ~ max_noise_magnitude
    fraction_array = np.linspace(0.0, 1.0, num_cols)
    noise_amplitudes = max_noise_magnitude * (
        1.0 / (1.0 + np.exp(-steepness * (fraction_array - 0.5)))
    )

    # # Replicate amplitudes for each row. The shape becomes (num_rows, num_cols)
    # scale_array = np.tile(noise_amplitudes, (num_rows, 1))

    # Generate a mask indicating which rows should receive noise
    noise_mask = np.random.rand(num_rows) < probability_of_noise

    # Generate random noise with per-column standard deviation according to noise_amplitudes
    noise = np.random.normal(loc=0.0, scale=noise_amplitudes, size=(num_rows, num_cols))

    # Add noise to the selected rows
    noisy_values = form_factor_df.values.copy()
    noisy_values[noise_mask] += noise[noise_mask]

    # Return as a DataFrame (preserving index and columns)
    return pd.DataFrame(
        noisy_values, index=form_factor_df.index, columns=form_factor_df.columns
    )
