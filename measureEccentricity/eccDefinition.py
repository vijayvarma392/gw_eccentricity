"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 29, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .utils import get_peak_via_quadratic_fit


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict: dictionary containing waveform modes dict, time etc
        should follow the format {"t": time, "hlm": modeDict, ..}
        and modeDict = {(l, m): hlm_mode_data}
        for ResidualAmplitude method, provide "t_zeroecc" and "hlm_zeroecc"
        as well in the dataDict.
        """
        self.dataDict = dataDict
        self.t = self.dataDict["t"]
        self.hlm = self.dataDict["hlm"]
        self.h22 = self.hlm[(2, 2)]
        self.amp22 = np.abs(self.h22)
        # shift the time axis to make t = 0 at merger
        # t_ref would be then negative. This helps
        # when subtracting quasi circular amplitude from
        # eccentric amplitude in residual amplitude method
        self.t = self.t - get_peak_via_quadratic_fit(
            self.t, self.amp22)[0]
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = np.gradient(self.phase22, self.t)

    def find_extrema(self, which="maxima", extrema_finding_keywords=None):
        """Find the extrema in the data.

        parameters:
        -----------
        which: either maxima, peaks, minima or troughs
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function.

        returns:
        ------
        array of positions of extrema.
        """
        raise NotImplementedError("Please override me.")

    def interp_extrema(self, which="maxima", extrema_finding_keywords=None,
                       spline_keywords=None):
        """Interpolator through extrema.

        parameters:
        -----------
        which: either maxima, peaks, minima or troughs
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function.
        spline_keywords: arguments to be passed to InterpolatedUnivariateSpline

        returns:
        ------
        spline through extrema, positions of extrema
        """
        extrema_idx = self.find_extrema(which, extrema_finding_keywords)
        if len(extrema_idx) >= 2:
            return InterpolatedUnivariateSpline(self.t[extrema_idx],
                                                self.omega22[extrema_idx],
                                                **spline_keywords), extrema_idx
        else:
            raise Exception(
                f"Sufficient number of {which} are not found."
                " Can not create an interpolator.")

    def measure_ecc(self, t_ref, extrema_finding_keywords=None,
                    spline_keywords=None):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        t_ref: reference time to measure eccentricity and mean anomaly.
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function.
        spline_keywords: arguments to be passed to InterpolatedUnivariateSpline

        returns:
        --------
        ecc_ref: measured eccentricity at t_ref
        mean_ano_ref: measured mean anomaly at t_ref
        """
        t_ref = np.atleast_1d(t_ref)
        if any(t_ref >= 0):
            raise Exception("Reference time must be negative. Merger being"
                            " at t = 0.")
        default_spline_keywords = {"w": None,
                                   "bbox": [None, None],
                                   "k": 3,
                                   "ext": 0,
                                   "check_finite": False}
        if spline_keywords is None:
            spline_keywords = {}

        # Sanity check for spline keywords
        for keyword in spline_keywords:
            if keyword not in default_spline_keywords:
                raise ValueError(f"Invalid key {keyword} in spline_keywords."
                                 " Should be one of "
                                 f"{default_spline_keywords.keys()}")
        # Update spline keyword if given by user
        for keyword in default_spline_keywords.keys():
            if keyword in spline_keywords:
                default_spline_keywords[keyword] = spline_keywords[keyword]

        self.spline_keywords = default_spline_keywords
        omega_peaks_interp, self.peaks_location = self.interp_extrema(
            "maxima", extrema_finding_keywords, default_spline_keywords)
        omega_troughs_interp, self.troughs_location = self.interp_extrema(
            "minima", extrema_finding_keywords, default_spline_keywords)

        # check if the t_ref has a peak before and after
        # This required to define mean anomaly.
        t_peaks = self.t[self.peaks_location]
        if t_ref[0] < t_peaks[0] or t_ref[-1] >= t_peaks[-1]:
            raise Exception("Reference time must be within two peaks.")

        # compute eccentricty from the value of omega_peaks_interp
        # and omega_troughs_interp at t_ref using the fromula in
        # ref. arXiv:2101.11798 eq. 4
        omega_peak_at_t_ref = omega_peaks_interp(t_ref)
        omega_trough_at_t_ref = omega_troughs_interp(t_ref)
        ecc_ref = ((np.sqrt(omega_peak_at_t_ref)
                    - np.sqrt(omega_trough_at_t_ref))
                   / (np.sqrt(omega_peak_at_t_ref)
                      + np.sqrt(omega_trough_at_t_ref)))
        # and compute the mean anomaly using ref. arXiv:2101.11798 eq. 7
        # mean anomaly grows linearly from 0 to 2 pi over
        # the range [t_at_last_peak, t_at_next_peak]
        mean_ano_ref = np.zeros(len(t_ref))
        for idx, time in enumerate(t_ref):
            idx_at_last_peak = np.where(t_peaks <= time)[0][-1]
            t_at_last_peak = t_peaks[idx_at_last_peak]
            t_at_next_peak = t_peaks[idx_at_last_peak + 1]
            t_since_last_peak = time - t_at_last_peak
            current_period = (t_at_next_peak - t_at_last_peak)
            mean_ano_ref[idx] = (2 * np.pi
                                 * t_since_last_peak
                                 / current_period)

        if len(t_ref) == 1:
            mean_ano_ref = mean_ano_ref[0]
            ecc_ref = ecc_ref[0]

        return ecc_ref, mean_ano_ref
