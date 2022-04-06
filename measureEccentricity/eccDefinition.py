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
            merger_idx = np.argmin(np.abs(self.t))
            phase22_at_merger = self.phase22[merger_idx]
            # one orbit changes the 22 mode phase by 4 pi since
            # omega22 = 2 omega_orb
            phase22_num_orbits_earlier_than_merger = (
                phase22_at_merger
                - 4 * np.pi
                * exclude_num_orbits_before_merger)
            idx_num_orbit_earlier_than_merger = np.argmin(np.abs(
                self.phase22 - phase22_num_orbits_earlier_than_merger))
            # use only the extrema those are atleast num_orbits away from the
            # merger to avoid overfitting in the spline through the exrema
            extrema_idx = extrema_idx[extrema_idx
                                      <= idx_num_orbit_earlier_than_merger]
        if len(extrema_idx) >= 2:
            spline = InterpolatedUnivariateSpline(self.t[extrema_idx],
                                                  self.omega22[extrema_idx],
                                                  **spline_keywords)
            return spline, extrema_idx
        else:
            raise Exception(
                f"Sufficient number of {which} are not found."
                " Can not create an interpolator.")

    def do_sanity_check(name, user_keywords, default_keywords):
        """Sanity check for user given dicionary of keywords.

        parameters:
        name: string to represnt the dictionary
        user_keywords: Dictionary of keywords by user
        default_keywords: Dictionary of default keywords
        """
        for keyword in user_keywords.keys():
            if keyword not in default_keywords:
                raise ValueError(f"Invalid key {keyword} in {name}."
                                 " Should be one of "
                                 f"{default_keywords.keys()}")

    def update_user_keywords_dict(user_keywords, default_keywords):
        """Update user given dictionary of keywords by adding missing keys.

        parameters:
        user_keywords: Dictionary of keywords by user
        default_keywords: Dictionary of default keywords
        """
        for keyword in default_keywords.keys():
            if keyword not in user_keywords:
                user_keywords[keyword] = default_keywords[keyword]

    def measure_ecc(self, tref_in, extrema_finding_keywords=None,
                    spline_keywords=None, extra_keywords=None):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        tref_in:
              Input reference time to measure eccentricity and mean anomaly.
              This is the input array provided by the user to evaluate
              eccenricity and mean anomaly at. However, if
              exclude_num_orbits_before_merger is not None, then the
              interpolator used to measure eccentricty is constructed using
              extrema only upto exclude_num_orbits_before_merger and accorindly
              a tmax is set by chosing the min of time of last peak/trough.
              Thus the eccentricity and mean anomaly are computed only upto
              tmax and a new time array tref_out is returned with
              max(tref_out) = tmax

        extrema_finding_keywords:
             Dictionary of arguments to be passed to the
             peak finding function.

        spline_keywords:
             arguments to be passed to InterpolatedUnivariateSpline

        extra_keywords:
            any extra keywords to be passed. Allowed keywords are

            exclude_num_orbits_before_merger:
              could be either None or non negative real number. If None, then
              the full data even after merger is used but this might cause
              issues with he interpolaion trough exrema. For non negative real
              number, that many orbits prior to merger is exculded.
              Default is 1.

        returns:
        --------
        tref_out: array of reference time where eccenricity and mean anomaly is
              measured. This would be different from tref_in if
              exclude_num_obrits_before_merger in the extra_keywords
              is not None

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
        # make it iterable
        if spline_keywords is None:
            spline_keywords = {}

        # Sanity check for spline keywords
        self.do_sanity_check("spline_keywords", spline_keywords,
                             default_spline_keywords)

        # Add default value to keyword if not passed by user
        self.update_user_keywords_dict(spline_keywords,
                                       default_spline_keywords)

        self.spline_keywords = spline_keywords

        if extra_keywords is None:
            extra_keywords = {}
        default_extra_keywords = {"exclude_num_orbits_before_merger": 1}
        # sanity check for extra keywords
        self.do_sanity_check("extra_keywords", extra_keywords,
                             default_extra_keywords)
        # Add default value to keyword if not passed by user
        self.update_user_keywords_dict(extra_keywords, default_extra_keywords)

        omega_peaks_interp, self.peaks_location = self.interp_extrema(
            "maxima", extrema_finding_keywords, spline_keywords,
            default_extra_keywords["exclude_num_orbits_before_merger"])
        omega_troughs_interp, self.troughs_location = self.interp_extrema(
            "minima", extrema_finding_keywords, spline_keywords,
            default_extra_keywords["exclude_num_orbits_before_merger"])

        t_peaks = self.t[self.peaks_location]
        if extra_keywords["exclude_num_orbits_before_merger"] is not None:
            t_troughs = self.t[self.troughs_location]
            t_max = min(t_peaks[-1], t_troughs[-1])
            # measure eccentricty and mean anomaly only upto t_max
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
