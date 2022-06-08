"""Useful functions for the project."""
import numpy as np
import argparse


def amplitude_using_all_modes(mode_dict):
    """Get the amplitude using all the available modes.

    Parameters:
    ----------
    mode_dict:
        Dictionary containing waveform modes.

    Returns:
        Square root of the qudrature sum of the amplitudes of all the
        available modes in mode_dict.
    """
    amp = 0
    for mode in mode_dict.keys():
        amp += np.abs(mode_dict[mode])**2
    return np.sqrt(amp)


def peak_time_via_quadratic_fit(t, func):
    """
    Find the peak time of a function quadratically.

    Fits the function to a quadratic over the 5 points closest to the argmax
    func.
    t : an array of times
    func : array of function values
    Returns: tpeak, fpeak
    """
    # Find the time closest to the peak, making sure we have room on either
    # side
    index = np.argmax(func)
    index = max(2, min(len(t) - 3, index))

    # Do a quadratic fit to 5 points,
    # subtracting t[index] to make the matrix inversion nice
    testTimes = t[index-2:index+3] - t[index]
    testFuncs = func[index-2:index+3]
    xVecs = np.array([np.ones(5), testTimes, testTimes**2.])
    invMat = np.linalg.inv(np.array([[v1.dot(v2) for v1 in xVecs]
                                     for v2 in xVecs]))

    yVec = np.array([testFuncs.dot(v1) for v1 in xVecs])
    coefs = np.array([yVec.dot(v1) for v1 in invMat])
    return t[index] - coefs[1]/(2.*coefs[2]), (coefs[0]
                                               - coefs[1]**2./4/coefs[2])


def check_kwargs_and_set_defaults(user_kwargs=None,
                                  default_kwargs=None,
                                  name="user given kwargs"):
    """Sanity check user given dicionary of kwargs and set default values.

    parameters:
    user_kwargs: Dictionary of kwargs by user
    default_kwargs: Dictionary of default kwargs
    name: string to represnt the dictionary

    returns:
    updated user_kwargs
    """
    # make user_kwargs iterable
    if user_kwargs is None:
        user_kwargs = {}

    for kw in user_kwargs.keys():
        if kw not in default_kwargs:
            raise ValueError(f"Invalid key {kw} in {name}."
                             " Should be one of "
                             f"{list(default_kwargs.keys())}")

    for kw in default_kwargs.keys():
        if kw not in user_kwargs:
            user_kwargs[kw] = default_kwargs[kw]

    return user_kwargs


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Stolen from https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text."""

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def time_deriv_4thOrder(y, dt):
    """
    Fourth order accurate time derivative.

    Assuming constant time step.
    Tested for convergence up to 1e-12 level.
    """
    # Use a 5 point stencil
    res = 0*y
    # First do the interior
    res[2:-2] = (y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:])/12.

    # Next do the edges
    res[1] = y[:5].dot(np.array([-3, -10, 18, -6, 1]) / 12.)
    res[0] = y[:5].dot(np.array([-25, 48, -36, 16, -3]) / 12.)
    res[-2] = y[-5:].dot(np.array([-1, 6, -18, 10, 3]) / 12.)
    res[-1] = y[-5:].dot(np.array([3, -16, 36, -48, 25]) / 12.)
    return res / dt
