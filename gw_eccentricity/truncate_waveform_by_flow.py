"""Truncate waveform by flow."""
from gw_eccentricity import get_available_methods
import numpy as np
import copy


def truncate_waveform_by_flow(dataDict=None,
                              flow=None,
                              m_max=None,
                              method="Amplitude",
                              extra_kwargs=None):
    """Truncate waveform by flow.

    Eccentric waveforms have a non-monotonic instantaneous frequency.
    Therefore, truncating the waveform by demanding that the truncated waveform
    should contain all frequencies that are greater than or equal to a given
    minimum frequency, say flow, must be done carefully since the instantaneous
    frequency can be equal to the given flow at multiple points in time.

    We need to find the time tlow, such that all the frequencies at t < tlow
    are < flow and therefore the t >= tlow part of the waveform would
    retain all the frequencies that are >= flow. Note that the t >= tlow part
    could contain some frequencies < flow but that is fine, all we need is not
    to lose any frequencies >= flow.

    This could be done by using the frequency interpolant omega22_p(t) through
    the pericenters because
    1. It is a monotonic function of time.
    2. If at a time tlow, omega22_p(tlow) * (m_max/2) = 2*pi*flow, then all
    frequencies >= flow will be included in the waveform truncated at
    t=tlow. The m_max/2 factor ensures that this statement is true for all
    modes, as the frequency of the h_{l, m} mode scales approximately as m/2 *
    omega_22/(2*pi).

    Thus, we find tlow such that omega22_a(tlow) * (m_max/2) = 2*pi*flow and
    truncate the waveform by keeping only the part where t >= tlow.

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
        The truncated waveform would have all the frequencies that are >= flow.
    m_max: int
        Maximum m (index of h_{l, m}) to account for while setting the tlow
        for truncation.  If None, then it is set using the highest available
        m from the modes in the dataDict.
        Default is None.
    method: str
        Method to find the locations of the apocenters. Default is "Amplitude".
        See gw_eccentricity.get_available_modes for available modes.
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
                                                 extra_kwargs=extra_kwargs)
        # Get the pericenters
        pericenters = gwecc_object.find_extrema("pericenters")
        original_pericenters = pericenters.copy()
        gwecc_object.check_num_extrema(pericenters, "pericenters")
        # Get the good pericenters
        pericenters = gwecc_object.drop_extrema_if_extrema_jumps(
            pericenters, 1.5, "pericenters")
        gwecc_object.pericenters_location = gwecc_object.drop_extrema_if_too_close(
            pericenters, extrema_type="pericenters")
        # Build the interpolants of omega22 at the extrema
        gwecc_object.omega22_pericenters_interp = gwecc_object.interp_extrema("pericenters")
    # If m_max is not provided, get the highest available m from the dataDict
    if m_max is None:
        modes = gwecc_object.dataDict["hlm"].keys()
        m_max = max([m for (l, m) in modes])
    # Find time where omega22_apocenter_interp(tlow) = 2 * pi * flow
    tmin = gwecc_object.t[gwecc_object.pericenters_location[0]]
    tmax = gwecc_object.t[gwecc_object.pericenters_location[-1]]
    gwecc_object.t_pericenters_interp = gwecc_object.t[
        np.logical_and(gwecc_object.t >= tmin,
                       gwecc_object.t <= tmax)]
    gwecc_object.f22_pericenters_interp \
        = gwecc_object.omega22_pericenters_interp(gwecc_object.t_pericenters_interp)/2/np.pi

    idx_low = np.where(
        gwecc_object.f22_pericenters_interp * (m_max/2) >= flow)[0][0]
    tlow = gwecc_object.t_pericenters_interp[idx_low]

    truncatedDict = copy.deepcopy(dataDict)
    for mode in truncatedDict["hlm"]:
        truncatedDict["hlm"][mode] \
            = truncatedDict["hlm"][mode][truncatedDict["t"] >= tlow]
    truncatedDict["t"] = truncatedDict["t"][truncatedDict["t"] >= tlow]

    gwecc_object.method = method
    gwecc_object.m_max = m_max
    gwecc_object.tlow_for_trucating = tlow
    gwecc_object.truncatedDict = truncatedDict
    gwecc_object.f_low_for_truncating = flow

    return truncatedDict, gwecc_object
