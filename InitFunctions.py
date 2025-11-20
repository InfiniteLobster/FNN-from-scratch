import numpy as np
import random

#
def randomIni(array, lower_bound=0.0, upper_bound=1.0):
    """
    Initialize given weight array.

    By default (lower_bound=0.0, upper_bound=1.0), Xavier/Glorot uniform
    initialization is used based on array shape:
        W ~ U(-limit, limit),  limit = sqrt(6 / (fan_in + fan_out))

    If different bounds are provided, standard uniform(lower_bound, upper_bound)
    is used.
    """
    array_init = array
    n_rows, n_cols = array_init.shape  # n_rows = fan_out, n_cols = fan_in + 1 (bias)

    # If default bounds are used, switch to Xavier/Glorot
    if lower_bound == 0.0 and upper_bound == 1.0:
        fan_out = n_rows
        fan_in = max(n_cols - 1, 1)  # subtract bias column; guard against zero
        if fan_in + fan_out > 0:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
        else:
            limit = 0.01
        low, high = -limit, limit
    else:
        # Use user-provided bounds directly
        low, high = lower_bound, upper_bound

    for iRow in range(n_rows):
        for iCol in range(n_cols):
            value_ran = random.uniform(low, high)
            array_init[iRow, iCol] = value_ran

    return array_init
