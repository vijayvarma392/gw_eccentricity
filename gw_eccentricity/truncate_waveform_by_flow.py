"""Truncate waveform by flow."""
from gw_eccentricity import get_available_methods
import numpy as np
import copy


def truncate_waveform_by_flow(dataDict=None,
                              flow=None,
                              method="Amplitude",
                              spline_kwargs=None,
                              extra_kwargs=None):
    """Truncate waveform by flow.

    Eccentric waveforms have a non-monotonic instantaneous frequency.
    Therefore, truncating waveform by demanding that the frequency of the
    truncated waveform should be greater than a given minimum frequency, say
    flow, must be done carefully since the instantaneous frequency can be equal
    to the given flow at multiple points in time.

    By demanding that the truncated waveform has frequencies that are
    greater than or equal to the flow, we mean that at all times in the
    truncated waveform the frequencies are >= flow. This could be done by using
    the frequency interpolant omega22_a(t) through the apocenters because
    1. It is monotonic function of time.
    2. If at a time tlow, omega22_a(tlow) = 2*pi*flow, then for any t > tlow,
       frequencies would be > flow.

    Thus, we find tlow such that omega22_a(tlow) = 2*pi*flow and truncate the
    waveform by retaing only the part where t >= tlow.

    Paramerers:
    -----------
    datadict: dict
        Dictionary containing waveform data in the following format:
        dataDict = {"t": t,
                    "hlm": hlm},
        where t is the time array and hlm is a dictionary of waveform modes
        of the following format:
        hlm = {(l, m): lm_mode}
    flow: float
        Lower cutoff frequency to truncate the given waveform modes.
        The truncated waveform would have frequencies >= flow.
    method: str
        Method to find the locations of the apocenters.
        See gw_eccentricity.get_available_modes for available modes.
    spline_kwargs: dict
        Dictionary of arguments to be provided the the
        scipy.interpolate.InterpolatedUnivariatespline interpolant
        to create an interpolant of omega22 at the apocenters.
        Default values are set using eccDefinition.get_default_spline_kwargs.
    extra_kwargs: dict
        Dictionary of arguments that might be used for extrema finding routine.
        Default values are set using eccDefinition.get_default_extra_kwargs.

    Returns:
    --------
    truncatedDict: dict
        Dictionary containing the truncated waveform.
        Has the same type as dataDict.
    gwecc_object: obj
        Object used to truncate the dataDict.
    """
    if dataDict is None:
        raise ValueError("dataDict can not be None.")
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](dataDict,
                                                 spline_kwargs=spline_kwargs,
                                                 extra_kwargs=extra_kwargs)
        omega22_apocenters_interp, apocenters_locations\
            = gwecc_object.interp_extrema("apocenters")
    # Find time where omega22_apocenter_interp(tlow) = 2 * pi * flow
    tmin = gwecc_object.t[apocenters_locations[0]]
    tmax = gwecc_object.t[apocenters_locations[-1]]
    tref = gwecc_object.t[np.logical_and(gwecc_object.t >= tmin,
                                         gwecc_object.t <= tmax)]
    fref = omega22_apocenters_interp(tref)/2/np.pi

    idx_low = np.where(fref >= flow)[0][0]
    tlow = tref[idx_low]
    # Since the instantaneous frequency is not monotonic, there might be some
    # part of the waveform that has f22 >= flow at t < tlow.  Therefore we
    # need to refine this tlow further.  We obtain the time of the apocenter
    # just before tlow and then use a root finding between
    # t_previous_apocenter and tlow to see where exactly frequency crosses
    # f_low.
    idx_of_previous_apocenter = np.where(
        gwecc_object.t[apocenters_locations] <= tlow)[0][-1]
    t_previous_apocenter = gwecc_object.t[
        apocenters_locations[idx_of_previous_apocenter]]
    # Refine only if t_previous_apocenter is earlier than tlow
    if t_previous_apocenter < tlow:
        # Take slice of frequency and time between t_previous_apocenter
        # and tlow
        f22 = gwecc_object.omega22[
            np.logical_and(gwecc_object.t >= t_previous_apocenter,
                           gwecc_object.t <= tlow)]/2/np.pi
        t = gwecc_object.t[
            np.logical_and(gwecc_object.t >= t_previous_apocenter,
                           gwecc_object.t <= tlow)]
        tlow = t[np.argmin(np.abs(f22 - flow))]

    truncatedDict = copy.deepcopy(dataDict)
    for mode in truncatedDict["hlm"]:
        truncatedDict["hlm"][mode] \
            = truncatedDict["hlm"][mode][truncatedDict["t"] >= tlow]
    truncatedDict["t"] = truncatedDict["t"][truncatedDict["t"] >= tlow]

    gwecc_object.method = method
    gwecc_object.tlow_for_trucating = tlow
    gwecc_object.truncatedDict = truncatedDict
    gwecc_object.f_low_for_truncating = flow
    gwecc_object.f22_apocenters_interp = fref
    gwecc_object.t_apocenters_interp = tref

    return truncatedDict, gwecc_object
