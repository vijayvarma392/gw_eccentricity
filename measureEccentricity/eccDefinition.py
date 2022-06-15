"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 29, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .utils import peak_time_via_quadratic_fit, check_kwargs_and_set_defaults
from .utils import amplitude_using_all_modes
from .utils import time_deriv_4thOrder
from .plot_settings import use_fancy_plotsettings, colorsDict
import matplotlib.pyplot as plt
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
                    Can be None or a non negative real number.
                    If None, the full waveform data (even post-merger) is used
                    to measure eccentricity, but this might cause issues when
                    interpolating trough extrema.
                    For a non negative real
                    num_orbits_to_exclude_before_merger, that many orbits prior
                    to merger are excluded when finding extrema.
                    Default: 1.
                extrema_finding_kwargs:
                    Dictionary of arguments to be passed to the peak finding
                    function (typically scipy.signal.find_peaks).
                debug:
                    Run additional sanity checks if True.
                    Default: True.
                treat_mid_points_between_peaks_as_troughs:
                    If True, instead of trying to find local minima in the
                    data, we simply find the midpoints between local maxima
                    and treat them as apastron locations. This is helpful for
                    eccentricities ~1 where periastrons are easy to find but
                    apastrons are not.
                    Default: False
        """
        self.dataDict = dataDict
        self.t = self.dataDict["t"]
        # check if the time steps are equal, the derivative function
        # requires uniform time steps
        self.t_diff = np.diff(self.t)
        if not np.allclose(self.t_diff, self.t_diff[0]):
            raise Exception("Input time array must have uniform time steps.\n"
                            f"Time steps are {self.t_diff}")
        self.hlm = self.dataDict["hlm"]
        self.h22 = self.hlm[(2, 2)]
        self.amp22 = np.abs(self.h22)
        # We need to know the merger time of eccentric waveform.
        # This is useful, for example, to substract the quasi circular
        # amplitude from eccentric amplitude in residual amplitude method
        self.t_merger = peak_time_via_quadratic_fit(
            self.t,
            amplitude_using_all_modes(self.dataDict["hlm"]))[0]
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = time_deriv_4thOrder(self.phase22,
                                           self.t[1] - self.t[0])

        if "hlm_zeroecc" in dataDict:
            self.compute_res_amp_and_omega22()

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
        self.extrema_finding_kwargs = check_kwargs_and_set_defaults(
            self.extra_kwargs['extrema_finding_kwargs'],
            self.get_default_extrema_finding_kwargs(),
            "extrema_finding_kwargs")

    def get_default_spline_kwargs(self):
        """Defaults for spline settings."""
        default_spline_kwargs = {
            "w": None,
            "bbox": [None, None],
            "k": 3,
            "ext": 2,
            "check_finite": False}
        return default_spline_kwargs

    def get_default_extrema_finding_kwargs(self):
        """Defaults for extrema_finding_kwargs."""
        default_extrema_finding_kwargs = {
            "height": None,
            "threshold": None,
            "distance": None,
            "prominence": None,
            "width": self.get_width_for_peak_finder_from_phase22(),
            "wlen": None,
            "rel_height": 0.5,
            "plateau_size": None}
        return default_extrema_finding_kwargs

    def get_default_extra_kwargs(self):
        """Defaults for additional kwargs."""
        default_extra_kwargs = {
            "num_orbits_to_exclude_before_merger": 1,
            "extrema_finding_kwargs": {},   # Gets overriden in methods like
                                            # eccDefinitionUsingAmplitude
            "debug": True,
            "omega22_averaging_method": "average_between_extrema",
            "treat_mid_points_between_peaks_as_troughs": False
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
            merger_idx = np.argmin(np.abs(self.t - self.t_merger))
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

    def measure_ecc(self, tref_in=None, fref_in=None):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        tref_in:
            Input reference time at which to measure eccentricity and mean
            anomaly.
            Can be a single float or an array.
            NOTE: eccentricity/mean_ano are
            returned on a different time array tref_out, described below.

        fref_in:
            Input reference frequency at which to measure the eccentricity and
            mean anomaly. It can be a single float or an array.
            NOTE: eccentricity/mean anomaly are returned on a different freq
            array fref_out, described below.

            Given an fref_in, we find the corresponding tref_in such that,
            omega22_average(tref_in) = 2 * pi * fref_in.
            Here, omega22_average(t) is a monotonically increasing average
            frequency that is computed from the instantaneous omega22(t).
            Note that this is not a moving average; depending on which averaging
            method is used (see the omega22_averaging_method option below),
            it means slightly different things.

            Currently, following options are implemented to calculate the
            omega22_average
            - "average_between_extrema": Mean of the omega22 given by the
              spline through the peaks and the spline through the troughs.
            - "orbital_average_at_extrema": A spline through the orbital
              averaged omega22 evaluated at all available extrema.
            - "omega22_zeroecc": omega22 of the zero eccentricity waveform
            The default is "average_between_extrema". A method could be passed
            through the "extra_kwargs" option with the key
            "omega22_averaging_method".

        returns:
        --------
        tref_out/fref_out:
            tref_out is the output reference time, while fref_out is the
            output reference frequency, at which eccentricity and mean anomaly
            are measured.

            NOTE: Only one of these is returned depending on whether tref_in or
            fref_in is provided. If tref_in is provided then tref_out is
            returned and if fref_in provided then fref_out is returned.

            tref_out is set as tref_out = tref_in[tref_in >= tmin && tref_in < tmax],
            where tmax = min(t_peaks[-1], t_troughs[-1]),
            and tmin = max(t_peaks[0], t_troughs[0]). This is necessary because
            eccentricity is computed using interpolants of omega22_peaks and
            omega22_troughs. The above cutoffs ensure that we are not
            extrapolating in omega22_peaks/omega22_troughs.
            In addition, if num_orbits_to_exclude_before_merger in extra_kwargs
            is not None, only the data up to that many orbits before merger is
            included when finding the t_peaks/t_troughs. This helps avoid
            unphysical features like nonmonotonic eccentricity near the merger.

            fref_out is set as fref_out = fref_in[fref_in >= fmin && fref_in < fmax].
            where fmin = omega22_average(tmin)/2/pi, and
            fmax = omega22_average(tmax)/2/pi. tmin/tmax are defined above.

        ecc_ref:
            Measured eccentricity at tref_out/fref_out.

        mean_ano_ref:
            Measured mean anomaly at tref_out/fref_out.
        """
        self.omega22_peaks_interp, self.peaks_location = self.interp_extrema("maxima")
        # In some cases it is easier to find the peaks than finding the
        # troughs. For such cases, one can only find the peaks and use the
        # mid points between two consecutive peaks as the location of the
        # troughs.
        if self.extra_kwargs["treat_mid_points_between_peaks_as_troughs"]:
            self.omega22_troughs_interp, self.troughs_location = self.get_troughs_from_peaks()
        else:
            self.omega22_troughs_interp, self.troughs_location = self.interp_extrema("minima")

        # check that peaks and troughs are appearing alternatively
        self.check_peaks_and_troughs_appear_alternatingly()

        t_peaks = self.t[self.peaks_location]
        t_troughs = self.t[self.troughs_location]
        self.t_max = min(t_peaks[-1], t_troughs[-1])
        self.t_min = max(t_peaks[0], t_troughs[0])
        # check that only one of tref_in or fref_in is provided
        if (tref_in is not None) + (fref_in is not None) != 1:
            raise KeyError("Exactly one of tref_in and fref_in"
                           " should be specified.")
        elif tref_in is not None:
            self.tref_in = np.atleast_1d(tref_in)
        else:
            fref_in = np.atleast_1d(fref_in)
            # get the tref_in and fref_out from fref_in
            self.tref_in, self.fref_out = self.compute_tref_in_and_fref_out_from_fref_in(fref_in)
        # We measure eccentricity and mean anomaly from t_min to t_max.
        # Note that here we do not include the t_max. This is because
        # the mean anomaly computation requires to looking
        # for a peak before and after the ref time to calculate the current
        # period.
        # If ref time is t_max, which could be equal to the last peak, then
        # there is no next peak and that would cause a problem.
        self.tref_out = self.tref_in[np.logical_and(self.tref_in < self.t_max,
                                                    self.tref_in >= self.t_min)]

        # Sanity checks
        # check that fref_out and tref_out are of the same length
        if fref_in is not None:
            if len(self.fref_out) != len(self.tref_out):
                raise Exception(f"Length of fref_out {len(self.fref_out)}"
                                " is different from "
                                f"Length of tref_out {len(self.tref_out)}")
        # Check if tref_out is reasonable
        if len(self.tref_out) == 0:
            if self.tref_in[-1] > self.t_max:
                raise Exception(f"tref_in {self.tref_in} is later than t_max="
                                f"{self.t_max}, "
                                "which corresponds to min(last periastron "
                                "time, last apastron time).")
            if self.tref_in[0] < self.t_min:
                raise Exception(f"tref_in {self.tref_in} is earlier than t_min="
                                f"{self.t_min}, "
                                "which corresponds to max(first periastron "
                                "time, first apastron time).")
            raise Exception("tref_out is empty. This can happen if the "
                            "waveform has insufficient identifiable "
                            "periastrons/apastrons.")

        # check separation between extrema
        self.orb_phase_diff_at_peaks, \
            self.orb_phase_diff_ratio_at_peaks \
            = self.check_extrema_separation(self.peaks_location, "peaks")
        self.orb_phase_diff_at_troughs, \
            self.orb_phase_diff_ratio_at_troughs \
            = self.check_extrema_separation(self.troughs_location, "troughs")

        # Check if tref_out has a peak before and after.
        # This is required to define mean anomaly.
        # See explaination on why we do not include the last peak above.
        if self.tref_out[0] < t_peaks[0] or self.tref_out[-1] >= t_peaks[-1]:
            raise Exception("Reference time must be within two peaks.")

        # compute eccentricty from the value of omega22_peaks_interp
        # and omega22_troughs_interp at tref_out using the fromula in
        # ref. arXiv:2101.11798 eq. 4
        self.omega22_peak_at_tref_out = self.omega22_peaks_interp(self.tref_out)
        self.omega22_trough_at_tref_out = self.omega22_troughs_interp(self.tref_out)
        self.ecc_ref = ((np.sqrt(self.omega22_peak_at_tref_out)
                         - np.sqrt(self.omega22_trough_at_tref_out))
                        / (np.sqrt(self.omega22_peak_at_tref_out)
                           + np.sqrt(self.omega22_trough_at_tref_out)))

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
        self.mean_ano_ref = compute_mean_ano(self.tref_out)

        # check if eccenricity is positive
        if any(self.ecc_ref < 0):
            warnings.warn("Encountered negative eccentricity.")

        # check if eccenricity is monotonic and convex
        if len(self.tref_out) > 1:
            self.check_monotonicity_and_convexity(
                self.tref_out, self.ecc_ref,
                debug=self.extra_kwargs["debug"])

        if len(self.tref_out) == 1:
            self.mean_ano_ref = self.mean_ano_ref[0]
            self.ecc_ref = self.ecc_ref[0]
            self.tref_out = self.tref_out[0]
        if fref_in is not None and len(self.fref_out) == 1:
            self.fref_out = self.fref_out[0]

        return_array = self.fref_out if fref_in is not None else self.tref_out
        return return_array, self.ecc_ref, self.mean_ano_ref

    def check_extrema_separation(self, extrema_location,
                                 extrema_type="extrema",
                                 max_orb_phase_diff_factor=1.5,
                                 min_orb_phase_diff=np.pi):
        """Check if two extrema are too close or too far."""
        orb_phase_at_extrema = self.phase22[extrema_location] / 2
        orb_phase_diff = np.diff(orb_phase_at_extrema)
        # This might suggest that the data is noisy, for example, and a
        # spurious peak got picked up.
        t_at_extrema = self.t[extrema_location][1:]
        if any(orb_phase_diff < min_orb_phase_diff):
            too_close_idx = np.where(orb_phase_diff < min_orb_phase_diff)[0]
            too_close_times = t_at_extrema[too_close_idx]
            warnings.warn(f"At least a pair of {extrema_type} are too close."
                          " Minimum orbital phase diff is "
                          f"{min(orb_phase_diff)}. Times of occurances are"
                          f" {too_close_times}")
        if any(np.abs(orb_phase_diff - np.pi)
               < np.abs(orb_phase_diff - 2 * np.pi)):
            warnings.warn("Phase shift closer to pi than 2 pi detected.")
        # This might suggest that the peak finding method missed an extrema.
        # We will check if the phase diff at an extrema is greater than
        # max_orb_phase_diff_factor times the orb_phase_diff at the
        # previous peak
        orb_phase_diff_ratio = orb_phase_diff[1:]/orb_phase_diff[:-1]
        # make it of same length as orb_phase_diff by prepending 0
        orb_phase_diff_ratio = np.append([0], orb_phase_diff_ratio)
        if any(orb_phase_diff_ratio > max_orb_phase_diff_factor):
            too_far_idx = np.where(orb_phase_diff_ratio
                                   > max_orb_phase_diff_factor)[0]
            too_far_times = t_at_extrema[too_far_idx]
            warnings.warn(f"At least a pair of {extrema_type} are too far."
                          " Maximum orbital phase diff is "
                          f"{max(orb_phase_diff)}. Times of occurances are"
                          f" {too_far_times}")
        return orb_phase_diff, orb_phase_diff_ratio

    def check_monotonicity_and_convexity(self, tref_out, ecc_ref,
                                         check_convexity=False,
                                         debug=False,
                                         t_for_ecc_test=None):
        """Check if measured eccentricity is a monotonic function of time.

        parameters:
        tref_out:
            Output reference time from eccentricty measurement
        ecc_ref:
            measured eccentricity at tref_out
        check_convexity:
            In addition to monotonicity, it will check for
            convexity as well. Default is False.
        debug:
            If True then warning is generated when length for interpolation
            is greater than 100000. Default is False.
        t_for_ecc_test:
            Time array to build a spline. If None, then uses
            a new time array with delta_t = 0.1 for same range as in tref_out
            Default is None.
        """
        spline = InterpolatedUnivariateSpline(tref_out, ecc_ref)
        if t_for_ecc_test is None:
            t_for_ecc_test = np.arange(tref_out[0], tref_out[-1], 0.1)
            len_t_for_ecc_test = len(t_for_ecc_test)
            if debug and len_t_for_ecc_test > 1e6:
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

    def check_peaks_and_troughs_appear_alternatingly(self):
        """Check that peaks and troughs appear alternatingly."""
        # if peaks and troughs appear alternatingly, then the number
        # of peaks and troughs should differ by one.
        if abs(len(self.peaks_location) - len(self.troughs_location)) >= 2:
            warnings.warn(
                "Number of peaks and number of troughs differ by "
                f"{abs(len(self.peaks_location) - len(self.troughs_location))}"
                ". This implies that peaks and troughs are not appearing"
                " alternatingly.")
        else:
            # If the number of peaks and troughs differ by zero or one then we
            # do the following:
            if len(self.peaks_location) == len(self.troughs_location):
                # Check the time of the first peak and the first trough
                # whichever comes first is assigned as arr1 and the other one
                # as arr2
                if self.t[self.peaks_location][0] < self.t[self.troughs_location][0]:
                    arr1 = self.peaks_location
                    arr2 = self.troughs_location
                else:
                    arr2 = self.peaks_location
                    arr1 = self.troughs_location
            else:
                # Check the number of peaks and troughs
                # whichever is larger is assigned as arr1 and the other one as
                # arr2
                if len(self.peaks_location) > len(self.troughs_location):
                    arr1 = self.peaks_location
                    arr2 = self.troughs_location
                else:
                    arr2 = self.peaks_location
                    arr1 = self.troughs_location
            # create a new array which takes elements from arr1 and arr2
            # alternatingly
            arr = np.zeros(arr1.shape[0] + arr2.shape[0], dtype=arr1.dtype)
            # assign every other element to values from arr1 starting from
            # index = 0
            arr[::2] = arr1
            # assign every other element to values from arr2 starting from
            # index = 1
            arr[1::2] = arr2
            # get the time difference between consecutive locations in arr
            t_diff = np.diff(self.t[arr])
            # If peaks and troughs appear alternatingly then all the time
            # differences in t_diff should be positive
            if any(t_diff < 0):
                warnings.warn(
                    "There is at least one instance where "
                    "peaks and troughs do not appear alternatingly.")

    def compute_res_amp_and_omega22(self):
        """Compute residual amp22 and omega22."""
        self.hlm_zeroecc = self.dataDict["hlm_zeroecc"]
        self.t_zeroecc = self.dataDict["t_zeroecc"]
        # check that the time steps are equal
        self.t_zeroecc_diff = np.diff(self.t_zeroecc)
        if not np.allclose(self.t_zeroecc_diff, self.t_zeroecc_diff[0]):
            raise Exception(
                "Input time array t_zeroecc must have uniform time steps\n"
                f"Time steps are {self.t_zeroecc_diff}")
        self.h22_zeroecc = self.hlm_zeroecc[(2, 2)]
        # to get the residual amplitude and omega, we need to shift the
        # zeroecc time axis such that the merger of the zeroecc is at the
        # same time as that of the eccentric waveform
        self.t_merger_zeroecc = peak_time_via_quadratic_fit(
            self.t_zeroecc,
            amplitude_using_all_modes(self.dataDict["hlm_zeroecc"]))[0]
        self.t_zeroecc_shifted = (self.t_zeroecc
                                  - self.t_merger_zeroecc
                                  + self.t_merger)
        # check that the length of zeroecc waveform is greater than equal
        # to the length of the eccentric waveform.
        # This is important to satisfy. Otherwise there will be extrapolation
        # when we interpolate zeroecc ecc waveform data on the eccentric
        # waveform time.
        if (self.t_zeroecc_shifted[0] > self.t[0]):
            raise Exception("Length of zeroecc waveform must be >= the length "
                            "of the eccentric waveform. Eccentric waveform "
                            f"starts at {self.t[0]} whereas zeroecc waveform "
                            f"starts at {self.t_zeroecc_shifted[0]}. Try "
                            "starting the zeroecc waveform at lower Momega0.")
        self.amp22_zeroecc_interp = InterpolatedUnivariateSpline(
            self.t_zeroecc_shifted, np.abs(self.h22_zeroecc))(self.t)
        self.res_amp22 = self.amp22 - self.amp22_zeroecc_interp

        self.phase22_zeroecc = - np.unwrap(np.angle(self.h22_zeroecc))
        self.omega22_zeroecc = time_deriv_4thOrder(
            self.phase22_zeroecc,
            self.t_zeroecc[1] - self.t_zeroecc[0])
        self.omega22_zeroecc_interp = InterpolatedUnivariateSpline(
            self.t_zeroecc_shifted, self.omega22_zeroecc)(self.t)
        self.res_omega22 = (self.omega22
                            - self.omega22_zeroecc_interp)

    def compute_orbital_averaged_omega22_at_extrema(self, t):
        """Compute reference frequency by orbital averaging at extrema.

        We compute the orbital average of omega22 at the periastrons
        and the apastrons following:
        omega22_avg((t[i]+ t[i+1])/2) = int_t[i]^t[i+1] omega22(t)dt
                                        / (t[i+1] - t[i])
        where t[i] is the time of ith extrema.
        We do this for peaks and troughs and combine the results
        """
        extrema_locations = {"peaks": self.peaks_location,
                             "troughs": self.troughs_location}
        @np.vectorize
        def orbital_averaged_omega22_at_extrema(n, extrema_type="peaks"):
            """Compute orbital averaged omega22 between n and n+1 extrema."""
            # integrate omega22 between n and n+1 extrema
            # We do not need to do the integration here since
            # we already have phase22 available to us which is
            # nothing but the integration of omega22 over time.
            # We want to integrate from nth extrema to n+1 extrema
            # which is equivalent to phase difference between
            # these two extrema
            integ = (self.phase22[extrema_locations[extrema_type][n+1]]
                     - self.phase22[extrema_locations[extrema_type][n]])
            period = (self.t[extrema_locations[extrema_type][n+1]]
                      - self.t[extrema_locations[extrema_type][n]])
            return integ / period
        # get the mid points between the peaks as avg time for peaks
        t_average_peaks = (self.t[self.peaks_location][1:]
                           + self.t[self.peaks_location][:-1]) / 2
        omega22_average_peaks = orbital_averaged_omega22_at_extrema(
            np.arange(len(self.peaks_location) - 1), "peaks")
        # get the mid points between the troughs as avg time for troughs
        t_average_troughs = (self.t[self.troughs_location][1:]
                             + self.t[self.troughs_location][:-1]) / 2
        omega22_average_troughs = orbital_averaged_omega22_at_extrema(
            np.arange(len(self.troughs_location) - 1), "troughs")
        # combine results from avergae at peaks and toughs
        t_average = np.append(t_average_troughs, t_average_peaks)
        # sort the times
        sorted_idx = np.argsort(t_average)
        t_average = t_average[sorted_idx]
        # check if the average omega22 are monotonically increasing
        if any(np.diff(omega22_average_peaks) <= 0):
            raise Exception("Omega22 average at peaks are not strictly "
                            "monotonically increaing")
        if any(np.diff(omega22_average_troughs) <= 0):
            raise Exception("Omega22 average at troughs are not strictly "
                            "monotonically increaing")
        omega22_average = np.append(omega22_average_troughs,
                                    omega22_average_peaks)
        # sort omega22
        omega22_average = omega22_average[sorted_idx]
        return InterpolatedUnivariateSpline(t_average, omega22_average)(t)

    def compute_omega22_average_between_extrema(self, t):
        """Find omega22 average between extrema".

        Take mean of omega22 using spline through omega22 peaks
        and spline through omega22 troughs.
        """
        return ((self.omega22_peaks_interp(t)
                 + self.omega22_troughs_interp(t)) / 2)

    def compute_omega22_zeroecc(self, t):
        """Find omega22 from zeroecc data."""
        return InterpolatedUnivariateSpline(
            self.t_zeroecc_shifted, self.omega22_zeroecc)(t)

    def get_availabe_omega22_averaging_methods(self):
        """Return available omega22 averaging methods."""
        available_methods = {
            "average_between_extrema": self.compute_omega22_average_between_extrema,
            "orbital_average_at_extrema": self.compute_orbital_averaged_omega22_at_extrema,
            "omega22_zeroecc": self.compute_omega22_zeroecc
        }
        return available_methods

    def compute_tref_in_and_fref_out_from_fref_in(self, fref_in):
        """Compute tref_in and fref_out from fref_in.

        Using chosen omega22 average method we get the tref_in and fref_out
        for the given fref_in.

        When the input is frequencies where eccentricity/mean anomaly is to be
        measured, we internally want to map the input frequencies to a tref_in
        and then we proceed to calculate the eccentricity and mean anomaly for
        these tref_in in the same way as we do when the input array was time
        instead of frequencies.

        We first compute omega22_average(t) using the instantaneous omega22(t),
        which can be done in different ways as described below. Then, we keep
        only the allowed frequencies in fref_in by doing
        fref_out = fref_in[fref_in >= omega22_average(tmin) / (2 pi) &&
                           fref_in < omega22_average(tmax) / (2 pi)]
        Finally, we find the times where omega22_average(t) = 2 * pi * fref_out,
        and set those to tref_in.

        omega22_average(t) could be calculated in the following ways
        - Mean of the omega22 given by the spline through the peaks and the
          spline through the troughs, we call this "average_between_extrema"
        - Orbital average at the extrema, we call this "orbital_average_at_extrema"
        - omega22 of the zero eccentricity waveform, called "omega22_zeroecc"

        User can provide a method through the "extra_kwargs" option with the key
        "omega22_averaging_method". Default is "average_between_extrema"

        Once we get the reference frequencies, we create a spline to get time
        as function of these reference frequencies. This should work if the
        refrence frequency is monotonic which it should be.

        Finally we evaluate this spine on the fref_in to get the tref_in.
        """
        self.available_averaging_methods = self.get_availabe_omega22_averaging_methods()
        method = self.extra_kwargs["omega22_averaging_method"]
        if method in self.available_averaging_methods:
            # The fref_in array could have frequencies that is outside the range
            # of frequencies in omega22 average. Therefore, we want to create
            # a separate array of frequencies fref_out which is created by
            # taking on those frequencies that falls within the omega22 average
            # Then proceed to evaluate the tref_in based on these fref_out
            fref_out = self.get_fref_out(fref_in, method)
            # get omega22_average by evaluating the omega22_average(t)
            # on t, from tmin to tmax
            self.t_for_omega22_average = self.t[
                np.logical_and(self.t >= self.t_min, self.t < self.t_max)]
            self.omega22_average = self.available_averaging_methods[
                method](self.t_for_omega22_average)
            # check if average omega22 is monotonically increasing
            if any(np.diff(self.omega22_average) <= 0):
                warnings.warn(f"Omega22 average from method {method} is not "
                              "monotonically increasing.")
            t_of_fref_out = InterpolatedUnivariateSpline(
                self.omega22_average / (2 * np.pi),
                self.t_for_omega22_average)
            tref_in = t_of_fref_out(fref_out)
            # check if tref_in is monotonically increasing
            if any(np.diff(tref_in) <= 0):
                warnings.warn(f"tref_in from fref_in using method {method} is"
                              " not monotonically increasing.")
            return tref_in, fref_out
        else:
            raise KeyError(f"Omega22 averaging method {method} does not exist."
                           " Must be one of "
                           f"{list(self.available_averaging_methods.keys())}")

    def get_fref_out(self, fref_in, method):
        """Get fref_out from fref_in that falls within the valid average f22 range.

        Parameters:
        ----------
        fref_in:
            Input 22 mode reference frequency array.

        method:
            method for getting average omega22

        Returns:
        -------
        fref_out:
            Slice of fref_in that satisfies
            fref_in >= omega22_average(t_min) / 2 pi and
            fref_in < omega22_average(t_max) / 2 pi
        """
        # get min an max value f22_average from omega22_average
        self.omega22_average_min = self.available_averaging_methods[
            method](self.t_min)
        self.omega22_average_max = self.available_averaging_methods[
            method](self.t_max)
        self.f22_average_min = self.omega22_average_min / (2 * np.pi)
        self.f22_average_max = self.omega22_average_max / (2 * np.pi)
        fref_out = fref_in[
            np.logical_and(fref_in >= self.f22_average_min,
                           fref_in < self.f22_average_max)]
        if len(fref_out) == 0:
            if fref_in[0] < self.f22_average_min:
                raise Exception("fref_in is earlier than minimum available "
                                "frequency "
                                f"{self.f22_average_min}")
            if fref_in[-1] > self.f22_average_max:
                raise Exception("fref_in is later than maximum available "
                                "frequency "
                                f"{self.f22_average_max}")
            else:
                raise Exception("fref_out is empty. This can happen if the "
                                "waveform has insufficient identifiable "
                                "periastrons/apastrons.")
        return fref_out

    def make_diagnostic_plots(self, usetex=True, **kwargs):
        """Make dignostic plots for the eccDefinition method.

        We plot differenct quantities to asses how well our eccentricity
        measurment method is working. This could be seen as a diagnostic tool
        to check an implemented method.

        We plot the following quantities
        - The eccentricity vs vs time
        - decc/dt vs time, this is to test the monotonicity of eccentricity as
          a function of time
        - mean anomaly vs time
        - omega_22 vs time with the peaks and troughs shown. This would show
          if the method is missing any peaks/troughs or selecting one which is
          not a peak/trough
        - deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
          change in orbital phase from the previous extrema to the ith extrema.
          This helps to look for missing extrema, as there will be a drastic
          (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
          extrema, and the ratio will go from ~1 to ~2.

        Additionally, we plot the following if data for zero eccentricity is
        provided
        - residual amp22 vs time with the location of peaks and troughs shown.
        - residual omega22 vs time with the location of peaks and troughs
          shown.
        These two plots with further help in understanding any unwanted feature
        in the measured eccentricity vs time plot. For example, non smoothness
        the residual omega22 would indicate that the data in omega22 is not
        good which might be causing glitches in the measured eccentricity plot.
        """
        nrows = 7 if "hlm_zeroecc" in self.dataDict else 5
        figsize = (12, 4 * nrows)
        default_kwargs = {"nrows": nrows,
                          "figsize": figsize}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        use_fancy_plotsettings(usetex=usetex)
        fig, ax = plt.subplots(**kwargs)
        self.plot_measured_ecc(fig, ax[0])
        self.plot_decc_dt(fig, ax[1])
        self.plot_mean_ano(fig, ax[2])
        self.plot_extrema_in_omega22(fig, ax[3])
        self.plot_phase_diff_ratio_between_peaks(fig, ax[4])
        if "hlm_zeroecc" in self.dataDict:
            self.plot_residual_omega22(fig, ax[5])
            self.plot_residual_amp22(fig, ax[6])
        fig.tight_layout()
        return fig, ax

    def plot_measured_ecc(self, fig=None, ax=None, **kwargs):
        """Plot measured ecc as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        axNew.plot(self.tref_out, self.ecc_ref, **kwargs)
        axNew.set_xlabel(r"$t$")
        axNew.set_ylabel(r"Eccentricity $e$")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_decc_dt(self, fig=None, ax=None, **kwargs):
        """Plot decc_dt as function of time to check monotonicity.

        If decc_dt becomes positive, ecc(t) is not monotonically decreasing.
        """
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        axNew.plot(self.t_for_ecc_test, self.decc_dt, **kwargs)
        axNew.set_xlabel("$t$")
        axNew.set_ylabel(r"$de/dt$")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_mean_ano(self, fig=None, ax=None, **kwargs):
        """Plot measured mean anomaly as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        axNew.plot(self.tref_out, self.mean_ano_ref, **kwargs)
        axNew.set_xlabel("$t$")
        axNew.set_ylabel("mean anomaly")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_extrema_in_omega22(self, fig=None, ax=None, **kwargs):
        """Plot omega22, the locations of the apastrons and periastrons, and their corresponding interpolants.

        This would show if the method is missing any peaks/troughs or
        selecting one which is not a peak/trough.
        """
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.tref_out, self.omega22_peak_at_tref_out,
                   c=colorsDict["periastron"], label=r"$\omega_{p}$",
                   **kwargs)
        axNew.plot(self.tref_out, self.omega22_trough_at_tref_out,
                   c=colorsDict["apastron"], label=r"$\omega_{a}$",
                   **kwargs)
        # plot only upto merger to make the plot readable
        end = np.argmin(np.abs(self.t - self.t_merger))
        axNew.plot(self.t[: end], self.omega22[: end],
                   c=colorsDict["default"], label=r"$\omega_{22}$")
        axNew.plot(self.t[self.peaks_location],
                   self.omega22[self.peaks_location],
                   c=colorsDict["periastron"],
                   marker=".", ls="")
        axNew.plot(self.t[self.troughs_location],
                   self.omega22[self.troughs_location],
                   c=colorsDict["apastron"],
                   marker=".", ls="")
        axNew.set_xlabel(r"$t$")
        axNew.grid()
        axNew.set_ylabel(r"$\omega_{22}$")
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_extrema_in_amp22(self, fig=None, ax=None, **kwargs):
        """Plot amp22, the locations of the apastrons and periastrons.

        This would show if the method is missing any peaks/troughs or
        selecting one which is not a peak/trough.
        """
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        # plot only upto merger to make the plot readable
        end = np.argmin(np.abs(self.t - self.t_merger))
        axNew.plot(self.t[: end], self.amp22[: end],
                   c=colorsDict["default"], label=r"$A_{22}$")
        axNew.plot(self.t[self.peaks_location],
                   self.amp22[self.peaks_location],
                   c=colorsDict["periastron"],
                   marker=".", ls="", label="Pericenters")
        axNew.plot(self.t[self.troughs_location],
                   self.amp22[self.troughs_location],
                   c=colorsDict["apastron"],
                   marker=".", ls="", label="Apocenters")
        axNew.set_xlabel(r"$t$")
        axNew.grid()
        axNew.set_ylabel(r"$A_{22}$")
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_phase_diff_ratio_between_peaks(self, fig=None, ax=None, **kwargs):
        """Plot phase diff ratio between consecutive as function of time.

        Plots deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
        change in orbital phase from the previous extrema to the ith extrema.
        This helps to look for missing extrema, as there will be a drastic
        (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
        extrema, and the ratio will go from ~1 to ~2.
        """
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        tpeaks = self.t[self.peaks_location[1:]]
        axNew.plot(tpeaks[1:], self.orb_phase_diff_ratio_at_peaks[1:],
                   c=colorsDict["periastron"],
                   marker=".", label="Periastron phase diff ratio")
        ttroughs = self.t[self.troughs_location[1:]]
        axNew.plot(ttroughs[1:], self.orb_phase_diff_ratio_at_troughs[1:],
                   c=colorsDict["apastron"],
                   marker=".", label="Apastron phase diff ratio")
        axNew.set_xlabel(r"$t$")
        axNew.set_ylabel(r"$\Delta \Phi_{orb}[i] / \Delta \Phi_{orb}[i-1]$")
        axNew.grid()
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_residual_omega22(self, fig=None, ax=None, **kwargs):
        """Plot residual omega22, the locations of the apastrons and periastrons, and their corresponding interpolants.

        Useful to look for bad omega22 data near merger.
        We also throw away post merger before since it makes the plot
        unreadble.
        """
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        # plot only upto merger to make the plot readable
        end = np.argmin(np.abs(self.t - self.t_merger))
        axNew.plot(self.t[: end], self.res_omega22[:end], c=colorsDict["default"])
        axNew.plot(self.t[self.peaks_location],
                   self.res_omega22[self.peaks_location],
                   marker=".", ls="", label="Periastron",
                   c=colorsDict["periastron"])
        axNew.plot(self.t[self.troughs_location],
                   self.res_omega22[self.troughs_location],
                   marker=".", ls="", label="Apastron",
                   c=colorsDict["apastron"])
        axNew.set_xlabel(r"$t$")
        axNew.grid()
        axNew.set_ylabel(r"$\Delta\omega_{22}$")
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_residual_amp22(self, fig=None, ax=None, **kwargs):
        """Plot residual amp22, the locations of the apastrons and periastrons, and their corresponding interpolants."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.t, self.res_amp22, c=colorsDict["default"])
        axNew.plot(self.t[self.peaks_location],
                   self.res_amp22[self.peaks_location],
                   c=colorsDict["periastron"],
                   marker=".", ls="", label="Periastron")
        axNew.plot(self.t[self.troughs_location],
                   self.res_amp22[self.troughs_location],
                   c=colorsDict["apastron"],
                   marker=".", ls="", label="Apastron")
        axNew.set_xlabel(r"$t$")
        axNew.grid()
        axNew.set_ylabel(r"$\Delta A_{22}$")
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def get_troughs_from_peaks(self):
        """Get Interpolator through troughs and their locations.

        This function treats the mid points between two successive peaks
        as the location of the trough in between the same two peaks. Thus
        it does not find the locations of the troughs using peak
        finder at all. It is useful in situation where finding peaks
        is easy but finding the troughs in between is difficult. This is
        the case for highly eccentric systems where eccentricity approaches
        1. For such systems the amp22/omega22 data between the peaks is almost
        flat and hard to find the local minima.

        returns:
        ------
        spline through troughs, positions of troughs
        """
        # NOTE: Assuming uniform time steps.
        # TODO: Make it work for non uniform time steps
        # In the following we get the location of mid point between ith peak
        # and (i+1)th peak as (loc[i] + loc[i+1])/2 where loc is the array
        # that contains the peak locations. This works because time steps are
        # assumed to be uniform and hence proportional to the time itself.
        troughs_idx = (self.peaks_location[:-1] + self.peaks_location[1:]) / 2
        troughs_idx = troughs_idx.astype(int)  # convert to ints
        if len(troughs_idx) >= 2:
            spline = InterpolatedUnivariateSpline(self.t[troughs_idx],
                                                  self.omega22[troughs_idx],
                                                  **self.spline_kwargs)
            return spline, troughs_idx
        else:
            raise Exception(
                "Sufficient number of troughs are not found."
                " Can not create an interpolator.")

    def get_width_for_peak_finder_for_dimless_units(
            self,
            width_for_unit_timestep=50):
        """Get the minimal value of width parameter for extrema finding.

        The extrema finding method, i.e., find_peaks from scipy.signal
        needs a "width" parameter that is used to determine the minimal
        separation between consecutive extrema. If the "width" is too small
        then some noisy features in the signal might be mistaken for extrema
        and on the other hand if the "width" is too large then we might miss
        an extrema.

        This function gets an appropriate width by scaling it with the
        time steps in the time array of the waveform data.
        NOTE: As the function name mentions, this should be used only for
        dimensionless units. This is because the `width_for_unit_timestep`
        parameter refers to unit timestep in units of M. It is the fiducial
        width to use if the time step is 1M. If using time in seconds, this
        would depend on the total mass.

        Parameters:
        ----------
        width_for_unit_timestep:
            Width to use when the time step in the wavefrom data is 1.

        Returns:
        -------
        width:
            Minimal width to separate consecutive peaks.
        """
        return int(width_for_unit_timestep / (self.t[1] - self.t[0]))

    def get_width_for_peak_finder_from_phase22(self,
                                               num_orbits_before_merger=2):
        """Get the minimal value of width parameter for extrema finding.

        The extrema finding method, i.e., find_peaks from scipy.signal
        needs a "width" parameter that is used to determine the minimal
        separation between consecutive extrema. If the "width" is too small
        then some noisy features in the signal might be mistaken for extrema
        and on the other hand if the "width" is too large then we might miss
        an extrema.

        This function tries to use the phase22 to get a reasonable value of
        "width" by looking at the time scale over which the phase22 changes by
        about 4pi because the change in phase22 over one orbit would be
        approximately twice the change in the orbital phase which is about 2pi.
        Finally we divide this by 4 so that the width is always smaller than
        the two consecutive extrema otherwise we risk of missing a few extrema
        very close to the merger.

        Parameters:
        ----------
        num_orbits_before_merger:
            Number of orbits before merger to get the time at which the width
            parameter is determined. We want to do this near the merger as this
            is where the time between extrema is the smallest, and the width
            parameter sets the minimal separation between extrema.
            Default is 2.

        Returns:
        -------
        width:
            Minimal width to separate consecutive peaks.
        """
        # get the phase22 at merger.
        phase22_merger = self.phase22[np.argmin(np.abs(self.t - self.t_merger))]
        # get the time for getting width at num orbits before merger.
        # for 22 mode phase changes about 2 * 2pi for each orbit.
        t_at_num_orbits_before_merger = self.t[
            np.argmin(
                np.abs(
                    self.phase22
                    - (phase22_merger
                       - (num_orbits_before_merger * 4 * np.pi))))]
        t_at_num_minus_one_orbits_before_merger = self.t[
            np.argmin(
                np.abs(
                    self.phase22
                    - (phase22_merger
                       - ((num_orbits_before_merger - 1) * 4 * np.pi))))]
        # change in time over which phase22 change by 4 pi
        # between num_orbits_before_merger and num_orbits_before_merger - 1
        dt = (t_at_num_minus_one_orbits_before_merger
              - t_at_num_orbits_before_merger)
        # get the width using dt and the time step
        width = dt / (self.t[1] - self.t[0])
        # we want to use a width that is always smaller than the separation
        # between extrema, otherwise we might miss a few peaks near merger
        return int(width / 4)
