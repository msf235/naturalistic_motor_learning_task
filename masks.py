from typing import List
import numpy as np


def generate_decaying_intervals(
    interval_end_times: List[int] | np.ndarray, Tk: int, decay_factor: float = 0.5
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
        row introduced after them. The last row has an interval from the final
        boundary to Tk.
    """
    # Convert to a list if a numpy array is provided
    if isinstance(interval_end_times, np.ndarray):
        interval_end_times = interval_end_times.tolist()

    # Ensure interval_end_times is sorted
    interval_end_times = sorted(interval_end_times)

    num_rows = len(interval_end_times)

    # Initialize the result array with zeros
    # We'll have one extra row beyond the number of interval boundaries
    result: List[List[float]] = [[0.0] * Tk for _ in range(num_rows + 1)]

    # Main loop for the given intervals
    for i in range(num_rows):
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

    # Now add the extra row
    # This row goes from interval_end_times[-1] to Tk and sets it to 1.0
    extra_row_index = num_rows
    last_boundary = interval_end_times[-1]

    # Decay previously introduced intervals
    for j in range(num_rows):
        prev_start = 0 if j == 0 else interval_end_times[j - 1]
        prev_end = interval_end_times[j]
        interval_value = decay_factor ** (extra_row_index - j)
        for idx in range(prev_start, prev_end):
            result[extra_row_index][idx] = interval_value

    # Set the final interval to 1.0
    for idx in range(last_boundary, Tk):
        result[extra_row_index][idx] = 1.0

    return result


# Example usage:
# interval_end_times = [3, 7, 10]
# Tk = 15
# decay_factor = 0.7
# matrix = generate_decaying_intervals_matrix(interval_end_times, Tk, decay_factor)
# for row in matrix:
#     print(row)


def make_basic_xpos_masks(
    interval_end_tks,
    # mask_incr_its,
    Tk,
):
    mask_list = generate_decaying_intervals(interval_end_tks, Tk, 0.5)
    # mask_dict = {incr_it: mask_list[k] for k, incr_it in enumerate(mask_incr_its)}
    # return mask_dict
    return mask_list


def make_basic_qpos_masks(
    target_data_exists_tks,
    q_opt_ids,
    interval_end_its,
    nq,
    Tk,
):
    masks = np.zeros((len(interval_end_its), Tk, nq))
    for k, tek in enumerate(interval_end_its):
        for tk in target_data_exists_tks:
            if tk <= tek:
                for id in q_opt_ids:
                    masks[k][tk][id] = 1
    breakpoint()
    return masks
