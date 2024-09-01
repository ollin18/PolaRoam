import numpy as np

def compute_intervals(labels, times, max_time_between=86400):
    """
    Compute stop and move intervals from the list of labels.

    Parameters
    ----------
    labels : 1d np.array of integers
        Array of labels for each time point.
    times : 1d np.array of integers
        Array of time points corresponding to each label.
    max_time_between : int, optional
        Maximum allowed time between consecutive points within the same interval.

    Returns
    -------
    intervals : list of lists
        List containing intervals in the format [label, start_time, end_time].
    """
    assert len(labels) == len(times), "`labels` and `times` must match in length"

    # Combine labels and times into one array for easy iteration
    trajectory = np.column_stack((labels, times))

    final_trajectory = []
    loc_prev, t_start = trajectory[0]
    t_end = t_start

    for loc, time in trajectory[1:]:
        if is_same_interval(loc, loc_prev, time, t_end, max_time_between):
            t_end = time
        else:
            final_trajectory.append(create_interval(loc_prev, t_start, t_end))
            loc_prev, t_start, t_end = loc, time, time

    # Handle the final interval
    final_trajectory.append(create_interval(loc_prev, t_start, t_end))

    return final_trajectory

def is_same_interval(loc, loc_prev, time, t_end, max_time_between):
    """
    Check if the current location is part of the same interval.

    Parameters
    ----------
    loc : int
        Current location label.
    loc_prev : int
        Previous location label.
    time : int
        Current time point.
    t_end : int
        End time of the previous interval.
    max_time_between : int
        Maximum allowed time between points within the same interval.

    Returns
    -------
    bool
        True if the current point is part of the same interval, False otherwise.
    """
    return loc == loc_prev and (time - t_end) < max_time_between

def create_interval(loc, t_start, t_end):
    """
    Create an interval entry.

    Parameters
    ----------
    loc : int
        Location label for the interval.
    t_start : int
        Start time of the interval.
    t_end : int
        End time of the interval.

    Returns
    -------
    list
        A list representing the interval [loc, t_start, t_end].
    """
    return [loc, t_start, t_end]

