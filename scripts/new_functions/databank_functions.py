import json
from typing import Any, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

FormFactorType = list[tuple[float, float]]
TotalDensityType = list[tuple[float, float]]

ErrorHandlingOptions = Literal["ignore", "log", "raise"]


def handle_error(
    error: Exception, error_strategy: ErrorHandlingOptions = "log", error_text: str = ""
) -> None:
    """
    Handles an error according to the specified error handling option.

    :param error: The exception instance that was caught.
    :param error_strategy: Decides what should be done with errors.
    :param error_text: Additional text to include when logging or raising the error.

    :return: None
    """
    match error_strategy:
        case "ignore":
            return
        case "log":
            logger.error("%s: %s", error_text, type(error), exc_info=True)
        case "raise":
            raise type(error)(error_text) from error
        case _:
            # Default for unrecognized error strategies
            logger.error(
                "Unknown error handling option '%s'. Logging the error instead: %s",
                error_strategy,
                error,
                exc_info=True,
            )


def get_form_factor_and_total_density_pair(
    system: dict[str, Any],
    databank_path: str | Path,
    error_strategy: ErrorHandlingOptions = "log",
) -> tuple[FormFactorType | None, TotalDensityType | None]:
    """
    Returns form factor and total density profiles of the simulation.

    :param system: NMRlipids databank dictionary describing the simulation.
    :param databank_path: Path to the databank.
    :param error_strategy: Decides what should be done with errors.

    :return: Form factor and total density of the simulation.
    """
    databank_path = Path(databank_path)
    simulations_path = databank_path / "Data" / "Simulations" / system["path"]
    form_factor_path = simulations_path / "FormFactor.json"
    total_density_path = simulations_path / "TotalDensity.json"

    # Load form factor and total density
    try:
        with form_factor_path.open("r") as json_file:
            form_factor_simulation = json.load(json_file)
        with total_density_path.open("r") as json_file:
            total_density_simulation = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        handle_error(e, error_strategy, "Error loading simulation data")
        return None, None

    return form_factor_simulation, total_density_simulation


def get_equilibration_times(
    system: dict[str, Any],
    databank_path: str | Path,
    error_strategy: ErrorHandlingOptions = "log",
) -> dict[str, float] | None:
    """
    Returns equilibration times of all the molecules in the simulation.

    :param system: NMRlipids databank dictionary describing the simulation.
    :param databank_path: Path to the databank.
    :param error_strategy: Decides what should be done with errors.

    :return: Equilibration times of the simulation as a dict, or None if an error occurred.
    """
    databank_path = Path(databank_path)
    equilibration_times_path = (
        databank_path / "Data" / "Simulations" / system["path"] / "eq_times.json"
    )

    # Load equilibration_times
    try:
        with equilibration_times_path.open("r") as json_file:
            equilibration_times = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError) as file_not_found:
        handle_error(
            file_not_found, error_strategy, "Error loading equilibration times data:"
        )
        return None

    # Validate that equilibration_times is a dict with the right type of content
    if not isinstance(equilibration_times, dict):
        handle_error(
            TypeError(
                "Invalid format: expected a dict but got %s" % type(equilibration_times)
            ),
            error_strategy,
            "Invalid format: expected a dict",
        )
        return None

    validated_times: dict[str, float] = {}
    for key, value in equilibration_times.items():
        if not isinstance(key, str):
            handle_error(
                TypeError("Invalid key type: expected str but got %s" % type(key)),
                error_strategy,
                "Invalid key type",
            )
            return None
        if not isinstance(value, (int, float)):
            handle_error(
                TypeError(
                    "Invalid value type for key '%s': expected int or float but got %s"
                    % (key, type(value))
                ),
                error_strategy,
                "Invalid value type",
            )
            return None
        validated_times[key] = float(value)  # Convert int to float if necessary

    return validated_times
