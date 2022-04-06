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
                       spline_keywords=None,
                       exclude_num_orbits_before_merger=1):
        """Interpolator through extrema.

        parameters:
        -----------
        which: either maxima, peaks, minima or troughs
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function.
        spline_keywords: arguments to be passed to InterpolatedUnivariateSpline
        exclude_num_orbits_before_merger:
              could be either None or non negative real number. If None, then
              the full data even after merger is used but this might cause
              issues with he interpolaion trough exrema. For non negative real
              number, that many orbits prior to merger is exculded.
              Default is 1.

        returns:
        ------
        spline through extrema, positions of extrema
        """
        extrema_idx = self.find_extrema(which, extrema_finding_keywords)
        # experimenting wih throwing away peaks too close to merger
        # This helps in avoiding over fitting issue in the spline
        # thorugh the extrema
        if exclude_num_orbits_before_merger is not None:
            merger_idx = np.argmax(self.amp22)
            phase22_at_merger = self.phase22[merger_idx]
            phase_one_orbit_earlier_than_merger = (
                phase22_at_merger
                - 2 * np.pi
                * exclude_num_orbits_before_merger)
            idx_one_orbit_earlier_than_merger = np.where(
                self.phase22 >= phase_one_orbit_earlier_than_merger
            )[0][0]
            extrema_idx = extrema_idx[extrema_idx
                                      <= idx_one_orbit_earlier_than_merger]
        spline = InterpolatedUnivariateSpline(self.t[extrema_idx],
                                              self.omega22[extrema_idx],
                                              **spline_keywords)
        if len(extrema_idx) >= 2:
            return spline, extrema_idx
        else:
            raise Exception(
                f"Sufficient number of {which} are not found."
                " Can not create an interpolator.")

    def measure_ecc(self, tref_in, extrema_finding_keywords=None,
                    spline_keywords=None, extra_keywords=None):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        tref_in: Input reference time to measure eccentricity and mean anomaly.
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function.
        spline_keywords: arguments to be passed to InterpolatedUnivariateSpline
        extra_keywords: any extra keywords to be passed. Allowed keywords are
            exclude_num_orbits_before_merger:
              could be either None or non negative real number. If None, then
              the full data even after merger is used but this might cause
              issues with he interpolaion trough exrema. For non negative real
              number, that many orbits prior to merger is exculded.
              Default is 1.

        returns:
        --------
        tref_out: array of reference time where eccenricity and mean anomaly is
              measured. This would be different than tref_in if
              exclude_num_obrits_before_merger in the extra_keyword is not None

        ecc_ref: measured eccentricity at tref_out
        mean_ano_ref: measured mean anomaly at tref_out
        """
        tref_in = np.atleast_1d(tref_in)
        if any(tref_in >= 0):
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

        if extra_keywords is None:
            extra_keywords = {}
        default_extra_keywords = {"exclude_num_orbits_before_merger": 1}
        # sanity check for extra keywords
        for keyword in extra_keywords:
            if keyword not in default_extra_keywords:
                raise ValueError(f"Invalid key {keyword} in extra_keywords."
                                 " Should be one of "
                                 f"{default_extra_keywords.keys()}")
        # update extra keyword if given by the user
        for keyword in default_extra_keywords.keys():
            if keyword in extra_keywords:
                default_extra_keywords[keyword] = extra_keywords[keyword]

        omega_peaks_interp, self.peaks_location = self.interp_extrema(
            "maxima", extrema_finding_keywords, default_spline_keywords,
            default_extra_keywords["exclude_num_orbits_before_merger"])
        omega_troughs_interp, self.troughs_location = self.interp_extrema(
            "minima", extrema_finding_keywords, default_spline_keywords,
            default_extra_keywords["exclude_num_orbits_before_merger"])

        t_peaks = self.t[self.peaks_location]
        if default_extra_keywords["exclude_num_orbits_before_merger"] is not None:
            t_troughs = self.t[self.troughs_location]
            t_max = min(t_peaks[-1], t_troughs[-1])
            tref_out = tref_in[tref_in <= t_max]
        else:
            tref_out = tref_in
        # check if the tref_out has a peak before and after
        # This required to define mean anomaly.
        if tref_out[0] < t_peaks[0] or tref_out[-1] >= t_peaks[-1]:
            raise Exception("Reference time must be within two peaks.")

        # compute eccentricty from the value of omega_peaks_interp
        # and omega_troughs_interp at tref_out using the fromula in
        # ref. arXiv:2101.11798 eq. 4
        self.omega_peak_at_tref_out = omega_peaks_interp(tref_out)
        self.omega_trough_at_tref_out = omega_troughs_interp(tref_out)
        ecc_ref = ((np.sqrt(self.omega_peak_at_tref_out)
                    - np.sqrt(self.omega_trough_at_tref_out))
                   / (np.sqrt(self.omega_peak_at_tref_out)
                      + np.sqrt(self.omega_trough_at_tref_out)))

        @np.vectorize
        def compute_mean_ano(time):
            """
            Compute the mean anomaly using Eq.7 of arXiv:2101.11798.
            Mean anomaly grows linearly in time from 0 to 2 pi over
            the range [t_at_last_peak, t_at_next_peak], where t_at_last_peak
            is the time at the previous periastron, and t_at_next_peak is
            the time at the next periastron.
            """
            idx_at_last_peak = np.where(t_peaks <= time)[0][-1]
            t_at_last_peak = t_peaks[idx_at_last_peak]
            t_at_next_peak = t_peaks[idx_at_last_peak + 1]
            t_since_last_peak = time - t_at_last_peak
            current_period = t_at_next_peak - t_at_last_peak
            mean_ano_ref = 2 * np.pi * t_since_last_peak / current_period
            return mean_ano_ref

        # Compute mean anomaly at tref_out
        mean_ano_ref = compute_mean_ano(tref_out)

        if len(tref_out) == 1:
            mean_ano_ref = mean_ano_ref[0]
            ecc_ref = ecc_ref[0]

        return tref_out, ecc_ref, mean_ano_ref
