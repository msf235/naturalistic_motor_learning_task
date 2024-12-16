from typing import List
import numpy as np


def generate_decaying_intervals(
    interval_end_times: List[int] | np.ndarray, decay_factor: float = 0.5
) -> List[List[float]]:
    """
    Generate a matrix where each row introduces a new interval set to 1.0 from
    the previous interval boundary (or zero) to the current interval end time,
    while all previously introduced intervals decay by the specified factor
    with each new row. After processing all given interval_end_times, one more
    row is added from interval_end_times[-1] to Tk set to 1.0, applying the
    same decay logic to previously introduced intervals.

    Parameters
    ----------
    interval_end_times : List[int] | np.ndarray
        A sorted list or array of integer time indices that mark the boundaries
        where intervals end.
    decay_factor : float, optional
        The factor by which previously introduced intervals decay in each
        subsequent row. By default 0.5.

    Returns
    -------
    List[List[float]]
        A matrix (list of lists) where each row corresponds to an interval
        setup. The newly introduced interval is set to 1.0, and previously
        introduced intervals are multiplied by decay_factor for each additional
        row introduced after them.
    """
    # Convert to a list if a numpy array is provided
    if isinstance(interval_end_times, np.ndarray):
        interval_end_times = interval_end_times.tolist()

    # Ensure interval_end_times is sorted
    interval_end_times = sorted(interval_end_times)

    num_rows = len(interval_end_times) - 1

    # Initialize the result array with zeros
    # We'll have one extra row beyond the number of interval boundaries
    Tk = interval_end_times[-1]
    result: List[List[float]] = [[0.0] * Tk for _ in range(num_rows + 1)]

    # Main loop for the given intervals
    for i in range(num_rows + 1):
        # Determine the start and end of the current interval
        current_start = 0 if i == 0 else interval_end_times[i - 1]
        current_end = interval_end_times[i]

        # Set previously introduced intervals with decayed values
        for j in range(i):
            prev_start = 0 if j == 0 else interval_end_times[j - 1]
            prev_end = interval_end_times[j]
            interval_value = decay_factor ** (i - j)
            for idx in range(prev_start, prev_end):
                result[i][idx] = interval_value

        # The current interval for row i is always set to 1.0
        for idx in range(current_start, current_end):
            result[i][idx] = 1.0

    return result


# Example usage:
# interval_end_times = [3, 7, 10]
# Tk = 15
# decay_factor = 0.7
# matrix = generate_decaying_intervals_matrix(interval_end_times, Tk, decay_factor)
# for row in matrix:
#     print(row)


def make_basic_xpos_masks(
    interval_end_tks: list | np.ndarray,
):
    """We assume that the first interval starts at tk=0."""
    # if isinstance(interval_start_tks, np.ndarray):
    #     interval_start_tks = interval_start_tks.tolist()
    # interval_end_tks = interval_start_tks[1:].copy()
    mask_list = generate_decaying_intervals(interval_end_tks, 0.5)
    return mask_list


def make_basic_qpos_masks(
    target_data_exists_tks,
    q_opt_ids,
    interval_end_tks,
    nq,
):
    Tk = interval_end_tks[-1]
    masks = np.zeros((len(interval_end_tks), Tk, nq))
    for k, tek in enumerate(interval_end_tks):
        for tk in target_data_exists_tks:
            if tk <= tek:
                for id in q_opt_ids:
                    masks[k][tk][id] = 1
    return masks
