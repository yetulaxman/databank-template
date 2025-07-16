import json
import logging
from typing import Any, Literal
import os

import DatabankLib as dlb

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
    error_strategy: ErrorHandlingOptions = "log",
) -> tuple[FormFactorType | None, TotalDensityType | None]:
    """
    Returns form factor and total density profiles of the simulation.

    :param system: NMRlipids databank dictionary describing the simulation.
    :param databank_path: Path to the databank.
    :param error_strategy: Decides what should be done with errors.

    :return: Form factor and total density of the simulation.
    """
    system_endpath = os.path.join(dlb.NMLDB_SIMU_PATH, system["path"])
    form_factor_path = os.path.join(system_endpath, "FormFactor.json")
    total_density_path = os.path.join(system_endpath, "TotalDensity.json")

    # Load form factor and total density
    try:
        with open(form_factor_path, "r") as json_file:
            form_factor_simulation = json.load(json_file)
        with open(total_density_path, "r") as json_file:
            total_density_simulation = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        handle_error(e, error_strategy, "Error loading simulation data")
        return None, None

    return form_factor_simulation, total_density_simulation
