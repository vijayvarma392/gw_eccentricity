"""Truncate waveform by flow.

Eccentric waveforms have non-monotonic instantaneous frequency.
Therefore, to truncate waveform by demanding the frequency of the
waveform is greater than a fiven minimum frequency, say flow, should
be done carefully since the instantaneous frequency can be equal to
the given flow at multiple point in time.

Also by demanding that the truncated waveform has frequencies that are
greater than or equal to the flow, we mean that all times in the truncated
waveform the frequencies are >= flow. This could be done by using the
frequency interpolant f_a(t) trough the apocenters because
1. It is monotonic.
2. If at any time tref, f_a(tref) >= flow, then for t >= tref, frequencies
would be > flow.
"""
from gw_eccentricity import get_available_methods
import numpy as np


def truncate_waveform_by_flow(dataDict=None,
                              method="Amplitude",
                              flow=None,
                              spline_kwargs=None,
                              extra_kwargs=None):
    """Truncate waveform by flow."""
    if dataDict is None:
        raise ValueError("dataDict can not be None.")
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](dataDict,
                                                 spline_kwargs=spline_kwargs,
                                                 extra_kwargs=extra_kwargs)
        omega22_apocenters_interp, apocenters_locations\
            = gwecc_object.interp_extrema("apocenters")
    # Find time where omega22_apocenter_interp(t_low) = 2 * pi * flow
    tmin = gwecc_object.t[apocenters_locations[0]]
    tmax = gwecc_object.t[apocenters_locations[-1]]
    tref = gwecc_object.t[np.logical_and(gwecc_object.t >= tmin,
                                         gwecc_object.t <= tmax)]
    fref = omega22_apocenters_interp(tref)/2/np.pi

    idx_low = np.where(fref >= flow)[0][0]
    t_low = tref[idx_low]

    truncatedDict = dataDict.copy()
    for mode in truncatedDict["hlm"]:
        truncatedDict[mode] = truncatedDict[mode][truncatedDict["t"] >= t_low]
    truncatedDict["t"] = truncatedDict["t"][truncatedDict["t"] >= t_low]

    return truncatedDict
