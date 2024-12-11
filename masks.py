from typing import List
import numpy as np


def generate_decaying_intervals_matrix(
    interval_end_times: List[int] | np.ndarray, Tk: int, decay_factor: float = 0.5
) -> np.ndarray:
    """
    Generate a matrix where each row introduces a new interval set to 1.0 from
    the previous interval boundary (or zero) to the current interval end time,
    while all previously introduced intervals decay by the specified factor
    with each new row.

    Parameters
    ----------
    interval_end_times : List[int] | np.ndarray
        A sorted list or array of integer time indices that mark the boundaries
        where intervals end.
    Tk : int
        The total length of each resulting vector (number of columns).
    decay_factor : float, optional
        The factor by which previously introduced intervals decay in each
        subsequent row. By default 0.5.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row corresponds to an interval setup.
        The newly introduced interval is set to 1.0, and previously introduced
        intervals are multiplied by decay_factor for each additional row
        introduced after them.
    """
    # Convert to a list if a numpy array is provided
    if isinstance(interval_end_times, np.ndarray):
        interval_end_times = interval_end_times.tolist()

    # Ensure interval_end_times is sorted
    interval_end_times = sorted(interval_end_times)

    num_rows = len(interval_end_times)

    # Initialize the result array with zeros
    result = np.zeros((num_rows, Tk), dtype=float)

    for i in range(num_rows):
        # Determine the start and end of the current interval
        current_start = 0 if i == 0 else interval_end_times[i - 1]
        current_end = interval_end_times[i]

        # Set previously introduced intervals with decayed values
        for j in range(i):
            prev_start = 0 if j == 0 else interval_end_times[j - 1]
            prev_end = interval_end_times[j]
            interval_value = decay_factor ** (i - j)
            result[i, prev_start:prev_end] = interval_value

        # The current interval for row i is always set to 1.0
        result[i, current_start:current_end] = 1.0

    return result


# Example usage:
# interval_end_times = np.array([3, 7, 10])
# Tk = 15
# decay_factor = 0.7
# matrix = generate_decaying_intervals_matrix(interval_end_times, Tk, decay_factor)
# print(matrix)


def make_basic_xpos_mask(
    nonzero_times,
    Tk,
):
    mask = generate_decaying_intervals_matrix(nonzero_times, Tk, 0.5)
    return mask


def make_basic_qpos_target_and_mask(
    tks,
    qpos_targs,
    Tk,
):
    nq = len(qpos_targs[0])
    target = np.zeros((Tk, nq))
    mask = np.zeros((Tk, nq))
    for tk in tks:
        target[tk] = qpos_targs[tk]
        mask[tk] = 1
    return target, mask
