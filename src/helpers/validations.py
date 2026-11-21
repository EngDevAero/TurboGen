"""Contains all validation functions."""

from .constants import Array1D, Array1DList


def is_arrays_resolution_match(
    *arrays: Array1D | Array1DList, res: int | None = None, raise_err: bool = False
) -> bool:
    """
    Raise error or return False if resolution not match.

    Parameters
    ----------
    *arrays:   Any number of arrays to check for matching resolution.
    res:       Correct resolution.
    raise_err: Raise error or not.
    """
    if not res:
        res = len(arrays[0])

    for array in arrays:
        if len(array) != res:
            if raise_err:
                raise ValueError("Array resolution does not match")
            return False
    return True
