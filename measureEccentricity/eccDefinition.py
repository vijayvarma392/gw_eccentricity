"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 29, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .utils import get_peak_via_quadratic_fit, check_kwargs_and_set_defaults
import warnings


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict, spline_kwargs=None, extra_kwargs=None):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict:
            Dictionary containing waveform modes dict, time etc should follow
            the format {"t": time, "hlm": modeDict, ..}, with
            modeDict = {(l, m): hlm_mode_data}.
            For ResidualAmplitude method, also provide "t_zeroecc" and
            "hlm_zeroecc", for the quasicircular counterpart.

        spline_kwargs:
             Arguments to be passed to InterpolatedUnivariateSpline.

        extra_kwargs:
            Any extra kwargs to be passed. Allowed kwargs are
                num_orbits_to_exclude_before_merger:
                    Could be either None or non negative real number. If None,
                    then the full data even after merger is used but this might
                    cause issues with he interpolaion trough exrema. For non
                    negative real number, that many orbits prior to merger is
                    exculded.
                    Default: 1.
                extrema_finding_kwargs:
                    Dictionary of arguments to be passed to the peak finding
                    function (typically scipy.signal.find_peaks).
               debug:
                    Run additional sanity checks if debug is True.
                    Default: True.
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

        # Sanity check various kwargs and set default values
        self.spline_kwargs = check_kwargs_and_set_defaults(
            spline_kwargs, self.get_default_spline_kwargs(),
            "spline_kwargs")
        self.extra_kwargs = check_kwargs_and_set_defaults(
            extra_kwargs, self.get_default_extra_kwargs(),
            "extra_kwargs")
        if self.extra_kwargs["num_orbits_to_exclude_before_merger"] \
           is not None and \
           self.extra_kwargs["num_orbits_to_exclude_before_merger"] < 0:
            raise ValueError(
                "num_orbits_to_exclude_before_merger must be non-negative. "
                "Given value was "
                f"{self.extra_kwargs['num_orbits_to_exclude_before_merger']}")

    def get_default_spline_kwargs(self):
        """Defaults for spline settings."""
        default_spline_kwargs = {
            "w": None,
            "bbox": [None, None],
            "k": 3,
            "ext": 2,
            "check_finite": False}
        return default_spline_kwargs

    def get_default_extra_kwargs(self):
        """Defaults for additional kwargs."""
        default_extra_kwargs = {
            "num_orbits_to_exclude_before_merger": 1,
            "extrema_finding_kwargs": {},   # Gets overriden in methods like
                                            # eccDefinitionUsingAmplitude
            "debug": True
            }
        return default_extra_kwargs

    def find_extrema(self, extrema_type="maxima"):
        """Find the extrema in the data.

        parameters:
        -----------
        extrema_type:
            One of 'maxima', 'peaks', 'minima' or 'troughs'.

        returns:
        ------
        array of positions of extrema.
        """
        raise NotImplementedError("Please override me.")

    def interp_extrema(self, extrema_type="maxima"):
        """Interpolator through extrema.

        parameters:
        -----------
        extrema_type:
            One of 'maxima', 'peaks', 'minima' or 'troughs'.

        returns:
        ------
        spline through extrema, positions of extrema
        """
        extrema_idx = self.find_extrema(extrema_type)
        # experimenting wih throwing away peaks too close to merger
        # This helps in avoiding unwanted feature in the spline
        # thorugh the extrema
        if self.extra_kwargs["num_orbits_to_exclude_before_merger"] is not None:
            merger_idx = np.argmin(np.abs(self.t))
            phase22_at_merger = self.phase22[merger_idx]
            # one orbit changes the 22 mode phase by 4 pi since
            # omega22 = 2 omega_orb
            phase22_num_orbits_earlier_than_merger = (
                phase22_at_merger
                - 4 * np.pi
                * self.extra_kwargs["num_orbits_to_exclude_before_merger"])
            idx_num_orbit_earlier_than_merger = np.argmin(np.abs(
                self.phase22 - phase22_num_orbits_earlier_than_merger))
            # use only the extrema those are atleast num_orbits away from the
            # merger to avoid unphysical features like nonmonotonic
            # eccentricity near the merger
            extrema_idx = extrema_idx[extrema_idx
                                      <= idx_num_orbit_earlier_than_merger]
        if len(extrema_idx) >= 2:
            spline = InterpolatedUnivariateSpline(self.t[extrema_idx],
                                                  self.omega22[extrema_idx],
                                                  **self.spline_kwargs)
            return spline, extrema_idx
        else:
            raise Exception(
                f"Sufficient number of {extrema_type} are not found."
                " Can not create an interpolator.")

    def measure_ecc(self, tref_in):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        tref_in:
              Input reference time to measure eccentricity and mean anomaly.
              This is the input array provided by the user to evaluate
              eccenricity and mean anomaly at. However, if
              num_orbits_to_exclude_before_merger in extra_kwargs is not None,
              the interpolator used to measure eccentricty is constructed using
              extrema only upto num_orbits_to_exclude_before_merger and
              accorindly a tmax is set by chosing the min of time of last
              peak/trough. Thus the eccentricity and mean anomaly are computed
              only upto tmax and a new time array tref_out is returned with
              max(tref_out) = tmax.

        returns:
        --------
        tref_out:
            Array of reference times where eccenricity and mean anomaly are
            measured. This would be different from tref_in if
            exclude_num_obrits_before_merger in the extra_kwargs is not None.

        ecc_ref:
            Measured eccentricity at tref_out.

        mean_ano_ref:
            Measured mean anomaly at tref_out.
        """
        tref_in = np.atleast_1d(tref_in)
        if any(tref_in >= 0):
            raise Exception("Reference time must be negative. Merger being"
                            " at t = 0.")
        omega_peaks_interp, self.peaks_location = self.interp_extrema("maxima")
        omega_troughs_interp, self.troughs_location = self.interp_extrema("minima")

        t_peaks = self.t[self.peaks_location]
        if self.extra_kwargs["num_orbits_to_exclude_before_merger"] is not None:
            t_troughs = self.t[self.troughs_location]
            t_max = min(t_peaks[-1], t_troughs[-1])
            # measure eccentricty and mean anomaly only upto t_max
            tref_out = tref_in[tref_in <= t_max]
        else:
            tref_out = tref_in

        # check separation between extrema
        self.orb_phase_diff_at_peaks = self.check_extrema_separation(
            self.peaks_location, "peaks")
        self.orb_phase_diff_at_troughs = self.check_extrema_separation(
            self.troughs_location, "troughs")

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
            Compute mean anomaly.

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

        # check if eccenricity is monotonic and convex
        if len(tref_out) > 1 and self.extra_kwargs["debug"]:
            self.check_monotonicity_and_convexity(tref_out, ecc_ref)

        return tref_out, ecc_ref, mean_ano_ref

    def check_extrema_separation(self, extrema_location,
                                 extrema_type="extrema",
                                 max_orb_phase_diff=3 * np.pi,
                                 min_orb_phase_diff=np.pi):
        """Check if two extrema are too close or too far."""
        orb_phase_at_extrema = self.phase22[extrema_location] / 2
        orb_phase_diff = np.diff(orb_phase_at_extrema)
        # This might suggest that the data is noisy, for example, and a
        # spurious peak got picked up.
        if any(orb_phase_diff < min_orb_phase_diff):
            warnings.warn(f"At least a pair of {extrema_type} are too close."
                          " Minimum orbital phase diff is "
                          f"{min(orb_phase_diff)}")
        if any(np.abs(orb_phase_diff - np.pi)
               < np.abs(orb_phase_diff - 2 * np.pi)):
            warnings.warn("Phase shift closer to pi than 2 pi detected.")
        # This might suggest that the peak finding method missed an extrema.
        if any(orb_phase_diff > max_orb_phase_diff):
            warnings.warn(f"At least a pair of {extrema_type} are too far."
                          " Maximum orbital phase diff is "
                          f"{max(orb_phase_diff)}")
        return orb_phase_diff

    def check_monotonicity_and_convexity(self, tref_out, ecc_ref,
                                         check_convexity=False,
                                         t_for_ecc_test=None):
        """Check if measured eccentricity is a monotonic function of time.

        parameters:
        tref_out: Output reference time from eccentricty measurement
        ecc_ref: measured eccentricity at tref_out
        check_convexity: In addition to monotonicity, it will check for
        convexity as well. Default is False.
        t_for_ecc_test: Time array to build a spline. If None, then uses
        a new time array with delta_t = 0.1 for same range as in tref_out
        Default is None.
        """
        spline = InterpolatedUnivariateSpline(tref_out, ecc_ref)
        if t_for_ecc_test is None:
            t_for_ecc_test = np.arange(tref_out[0], tref_out[-1], 0.1)
            len_t_for_ecc_test = len(t_for_ecc_test)
            if len_t_for_ecc_test > 100000:
                warnings.warn("time array t_for_ecc_test is too long."
                              f" Length is {len_t_for_ecc_test}")

        # Get derivative of ecc(t) using cubic splines.
        self.decc_dt = spline.derivative(n=1)(t_for_ecc_test)
        self.t_for_ecc_test = t_for_ecc_test
        self.decc_dt = self.decc_dt

        # Is ecc(t) a monotoniccally decreasing function?
        if any(self.decc_dt > 0):
            warnings.warn("Ecc(t) is non monotonic.")

        # Is ecc(t) a convex function? That is, is the second
        # derivative always positive?
        if check_convexity:
            self.d2ecc_dt = spline.derivative(n=2)(t_for_ecc_test)
            self.d2ecc_dt = self.d2ecc_dt
            if any(self.d2ecc_dt > 0):
                warnings.warn("Ecc(t) is concave.")
