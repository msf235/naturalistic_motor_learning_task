from typing import List, Union
import numpy as np


def generate_decaying_intervals(
    interval_end_times: Union[List[int], np.ndarray], Tk: int, decay_factor: float = 0.5
) -> List[List[float]]:
    """
    Generate a matrix where each row introduces a new interval set to 1.0 from
    the previous interval boundary (or zero) to the current interval end time,
    while all previously introduced intervals decay by the specified factor with
    each new row.

    Parameters
    ----------
    interval_end_times : Union[List[int], np.ndarray]
        A sorted list or array of integer time indices that mark the boundaries
        where intervals end.
    Tk : int
        The total length of each resulting vector (number of columns).
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

    num_rows = len(interval_end_times)

    # Initialize the result array with zeros
    result: List[List[float]] = [[0.0] * Tk for _ in range(num_rows)]

    for i in range(num_rows):
        # Determine the start and end of the current interval
        if i == 0:
            current_start = 0
        else:
            current_start = interval_end_times[i - 1]
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


def make_basic_xpos_masks(
    interval_end_tks,
    mask_incr_its,
    Tk,
):
    mask_list = generate_decaying_intervals(interval_end_tks, Tk, 0.5)
    mask_dict = {incr_time: mask_list[k] for k, incr_time in enumerate(mask_incr_its)}
    return mask_dict


def make_basic_qpos_target_and_mask(
    tks,
    qpos_targs,
    Tk,
    interval_end_tks,
):
    nq = len(qpos_targs[0])
    target = np.zeros((Tk, nq))
    for tk in tks:
        target[tk] = qpos_targs[tk]
    masks = np.zeros((len(interval_end_tks), Tk, nq))
    for k, tek in enumerate(interval_end_tks):
        for tk in tks:
            if tk <= tek:
                masks[k][tk] = 1
    return target, masks
