"""Useful functions for the project."""
import numpy as np
import argparse
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import PchipInterpolator
import warnings


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

    Parameters:
    -----------
    t:
        An array of times.
    func:
        Array of function values.

    Returns:
    --------
    tpeak:
        Time at peak of the function func.
    fpeak:
        Value of function func at tpeak.
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
                                  name="user given kwargs",
                                  location=None):
    """Sanity check user given dicionary of kwargs and set default values.

    Parameters:
    ----------
    user_kwargs:
        Dictionary of kwargs by user.
    default_kwargs:
        Dictionary of default kwargs.
    name:
        string to represent the dictionary
    location:
        string pointing to where the defaults are defined

    Returns:
    --------
    updated user_kwargs
    """
    # make user_kwargs iterable
    if user_kwargs is None:
        user_kwargs = {}

    for kw in user_kwargs.keys():
        if kw not in default_kwargs:
            raise ValueError(f"Invalid key {kw} in {name}."
                             " Should be one of "
                             f"{list(default_kwargs.keys())}\n"
                             f"To add a new keyword, please modify {location}")

    for kw in default_kwargs.keys():
        if kw not in user_kwargs:
            user_kwargs[kw] = default_kwargs[kw]

    return user_kwargs


def raise_exception_if_none(kwargs, keys_to_check, name, location):
    """Raise exception if any key from `keys_to_check` has value None."""
    for kw in keys_to_check:
        if kwargs[kw] is None:
            raise Exception(f"kw {kw} for {name} can not be None."
                            f" Check documentation of {location} for more"
                            " details.")


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Smart Formatter for argparse helper strings.

    Stolen from https://stackoverflow.com/questions/3853722/how-to-insert-newlines-on-argparse-help-text.
    """

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def time_deriv_4thOrder(y, dt):
    """Fourth order accurate time derivative.

    Assuming constant time step.
    Tested for convergence up to 1e-12 level.

    Parameters:
    -----------
    y:
        1d array to take time derivative of.
    dt:
        Time step.

    Returns:
    --------
    dydt:
        Fourth order time derivative of y.
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
    dydt = res / dt
    return dydt


def get_default_spline_kwargs():
    """Defaults for spline settings."""
    default_spline_kwargs = {
        "w": None,
        "bbox": [None, None],
        "k": 3,
        "ext": 2,
        "check_finite": False}
    return default_spline_kwargs


def interpolate(newX,
                oldX,
                oldY,
                allowExtrapolation=False,
                interpolator="spline",
                spline_kwargs=None,
                check_kwargs=True):
    """Interpolate.

    Parameters:
    -----------
    newX:
        Points where interpolant is to be evaluated.
    oldX:
        1d array of monotonically increasing real values.
    oldY:
        1d array of monotonically increasing real values.
    allowExtrapolation:
        Bool. If True returns extrapolated values. Default is False.
        If False, an exception is raised if trying to extrapolate.
    interpolator:
        String to choose an interpolator to interpolate oldX, oldY.
        Could be one of the following:
        "spline": Uses scipy.interpolate.InterpolatedUnivariateSpline.
        "monotonic_spline":  Uses scipy.interpolate.PchipInterpolator.
        Default is "spline".
    spline_kwargs:
        See under get_interpolant.
    check_kwargs:
        Check spline_kwargs if check_kwargs is True. Default is True.

    Returns:
    --------
    newY:
        Intepolated values at newX.
    """
    if len(oldY) != len(oldX):
        raise Exception("Lengths dont match.")

    if not allowExtrapolation:
        if np.min(newX) < np.min(oldX) - 1e-10 \
           or np.max(newX) > np.max(oldX) + 1e-10:
            print(f"Min of newX = {np.min(newX)}, "
                  f"Min of oldX = {np.min(oldX)}")
            print(f"Max of newX = {np.max(newX)}, "
                  f"Max of oldX = {np.max(oldX)}")
            print("newX has vlaues below oldX? "
                  f"{np.min(newX) < np.min(oldX)}")
            print("newX has values above oldX? "
                  f"{np.max(newX) > np.max(oldX)}")
            raise Exception("Trying to extrapolate, "
                            "but allowExtrapolation=False")
    newY = get_interpolant(oldX, oldY, allowExtrapolation, interpolator,
                           spline_kwargs, check_kwargs)(newX)
    return newY


def get_interpolant(oldX,
                    oldY,
                    allowExtrapolation=False,
                    interpolator="spline",
                    spline_kwargs=None,
                    check_kwargs=True):
    """Create Interpolant.

    Parameters:
    -----------
    oldX:
        1d array of monotonically increasing real values.
    oldY:
        1d array of monotonically increasing real values.
    allowExtrapolation:
        Bool. If True returns extrapolated values. Default is False.
        If False, an exception is raised if trying to extrapolate.
    interpolator:
        String to choose an interpolator to interpolate oldX, oldY.
        Could be one of the following:
        "spline": Uses scipy.interpolate.InterpolatedUnivariateSpline.
        "monotonic_spline":  Uses scipy.interpolate.PchipInterpolator.
        Default is "spline".
    spline_kwargs:
        Dictionary of kwargs to be provided to the interpolator if
        "interpolator"="spline". The allowed kwargs are the same as that of the
        spline function scipy.interpolate.InterpolatedUnivariateSpline and the
        defaults are set using gw_eccentricity.utils.get_default_spline_kwargs.
        Since we use allowExtraplotion arg separately, value of "ext" in
        extra_kwargs will be overridden by allowExtrapolation.
    check_kwargs:
        Check spline_kwargs if check_kwargs is True. Default is True.

    Returns:
    --------
    Intepolatnt.
    """
    if not np.all(np.diff(oldX) > 0):
        raise Exception("oldX must have increasing values")

    if interpolator == "spline":
        if check_kwargs:
            spline_kwargs = check_kwargs_and_set_defaults(
                spline_kwargs, get_default_spline_kwargs(), "spline kwargs",
                "utils.get_default_spline_kwargs")
        # check that num of data points > order of spline
        if len(oldX) >= 2:
            if len(oldX) <= spline_kwargs["k"]:
                warnings.warn(f"No of data points is {len(oldX)} but "
                              f"spline order k = {spline_kwargs['k']}. "
                              f"Decreasing k to {len(oldX) - 1}.")
                # make a copy so that the original spline_kwargs remains
                # unmodified.
                kwargs = spline_kwargs.copy()
                kwargs["k"] = len(oldX) - 1
            else:
                kwargs = spline_kwargs
        else:
            raise Exception("Number of data points is {len(oldX)}."
                            " Cannot build an interpolant.")
        # If allowExtrapolation is True but ext=2 then raise a warning
        # and override ext to 0
        if allowExtrapolation and kwargs["ext"] == 2:
            # ext = 0, returns extraploted values
            kwargs["ext"] = 0
        # If allowExtraplotion is False but ext != 2 then raise a warning
        # and override ext to 2
        if not allowExtrapolation and kwargs["ext"] != 2:
            # ext = 2, raises exception if extrapolation is attempted.
            kwargs["ext"] = 2
        interpolant = InterpolatedUnivariateSpline(oldX, oldY, **kwargs)
    elif interpolator == "monotonic_spline":
        if spline_kwargs is not None:
            warnings.warn(f"Interpolator is {interpolator} but spline_kwargs "
                          "are passed. spline_kwargs will be ignored.")
        interpolant = PchipInterpolator(oldX, oldY,
                                        extrapolate=allowExtrapolation)
    else:
        raise ValueError(f"Unknown interpolator {interpolator}. Must be one"
                         " of ['spline', 'monotonic_spline']")
    return interpolant


def debug_message(message, debug_level, important=True,
                  point_to_verbose_output=False):
    """Show message based on debug_level.

    parameters:
    -----------
    message: str
        Message to display.

    debug_level: int
        Indicator for level of debug message. Based on it, one of the
        following actions if performed:
        -1: No action is performed and hence no message is displayed.
        0: Warning is issued with the input message only if important=True
        1: Warning is issued with the input message.
        2: Exception is raised with the input message.

    important: bool
        Only if True, the message gets printed when debug_level=0. For
        other debug_levels, this does nothing.
        Default is True.

    point_to_verbose_output: bool
        When True, if debug_level is 0 and important is True, points to
        debug_level = 1 for more verbose output.
        Default is False
    """
    debug_levels = [-1, 0, 1, 2]
    if debug_level not in debug_levels:
        raise ValueError(
            f"Unknown debug_level {debug_level}. Should one "
            f"of {debug_levels}. See "
            "`gw_eccentricity.utils.debug_message` for action"
            "performed with each debug level.")
    if debug_level == -1:
        # Do nothing
        return
    if (debug_level == 0 and important) or debug_level == 1:
        if debug_level == 0 and point_to_verbose_output:
            message += "\nFor more verbose output use `debug_level=1`."
        # Issue warning. Use stacklevel=2 to point to actual line number
        # causing this warning instead of pointing to here.
        warnings.warn(message, stacklevel=2)
    if debug_level == 2:
        # raise Exception
        raise Exception(message)
