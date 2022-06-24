"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .utils import peak_time_via_quadratic_fit, check_kwargs_and_set_defaults
from .utils import amplitude_using_all_modes
from .utils import time_deriv_4thOrder
from .plot_settings import use_fancy_plotsettings, colorsDict, figWidthsTwoColDict
import matplotlib.pyplot as plt
import warnings


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict, spline_kwargs=None, extra_kwargs=None):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict:
            Dictionary containing waveform modes dict, time etc.
            Should follow the format:
                dataDict = {"t": time,
                            "hlm": modeDict,
                            "t_zeroecc": time,
                            "hlm_zeroecc": modeDict, ...
                           },
            where time is an array with the same convention as tref_in, and
            modeDict should have the format:
                modeDict = {(l1, m1): h_{l1, m1},
                           (l2, m2): h_{l2, m2}, ...
                           }.

            "t_zeroecc" and "hlm_zeroecc" are only required for ResidualAmplitude
            and ResidualFrequency methods, but if they are provided, they will be
            used to produce additional diagnostic plots, which can be helpful for
            all methods. "t_zeroecc" and "hlm_zeroecc" should include the time and
            modeDict for the quasi-circular limit of the eccentric waveform in
            "hlm". For a waveform model, "hlm_zeroecc" can be obtained by
            evaluating the model by keeping the rest of the binary parameters fixed
            but setting the eccentricity to zero. For NR, if such a quasi-circular
            counterpart is not available, we recommend using quasi-circular
            waveforms like NRHybSur3dq8 or PhenomT, depending on the mass ratio and
            spins. We require that "hlm_zeroecc" be at least as long as "hlm" so
            that residual amplitude/frequency can be computed.

            For dataDict, we currently only allow time-domain, nonprecessing
            waveforms with a uniform time array. Please make sure that the time
            step is small enough that omega22(t) can be accurately computed; we use
            a 4th-order finite difference scheme. In dimensionless units, we
            recommend a time step of dtM = 0.1M to be conservative, but you may be
            able to get away with larger time steps like dtM = 1M. The
            corresponding time step in seconds would be dtM * M * lal.MTSUN_SI,
            where M is the total mass in Solar masses.

            The (2,2) mode is always required in "hlm"/"hlm_zeroecc". If additional
            modes are included, they will be used in determining the pericenter time
            following Eq.(5) of arxiv:1905.09300. The pericenter time is used to
            time-align the two waveforms before computing the residual
            amplitude/frequency.

        spline_kwargs:
            Dictionary of arguments to be passed to the spline interpolation
            routine (scipy.interpolate.InterpolatedUnivariateSpline) used to
            compute omega22_pericenters(t) and omega22_apocenters(t).
            Defaults are the same as those of InterpolatedUnivariateSpline.

        extra_kwargs: A dict of any extra kwargs to be passed. Allowed kwargs are:
            num_orbits_to_exclude_before_merger:
                Can be None or a non negative number.
                If None, the full waveform data (even post-merger) is used for
                finding extrema, but this might cause interpolation issues.
                For a non negative num_orbits_to_exclude_before_merger, that
                many orbits prior to merger are excluded when finding extrema.
                Default: 1.
            extrema_finding_kwargs:
                Dictionary of arguments to be passed to the extrema finder,
                scipy.signal.find_peaks.
                The Defaults are the same as those of scipy.signal.find_peaks,
                except for the "width" parameter. "width" denotes the minimum
                separation between two consecutive pericenters/apocenters. Setting
                this can help avoid false extrema in noisy data (for example, due
                to junk radiation in NR). The default for "width" is set using
                phi22(t) near the merger. Starting from 4 cycles of the (2,2) mode
                before merger, we find the number of time steps taken to cover 2
                cycles, let's call this "the gap". Note that 2 cycles of the (2,2)
                mode is approximately one orbit, so this allows us to approximate
                the smallest gap between two pericenters/apocenters. However, to be
                conservative, we divide this gap by 4 and set it as the width
                parameter for find_peaks.
            debug:
                Run additional sanity checks if debug is True.
                Default: True.
            omega22_averaging_method:
                Options for obtaining omega22_average(t) from the instantaneous
                omega22(t).
                - "mean_of_extrema_interpolants": The mean of omega22_pericenters(t) and
                  omega22_apocenters(t) is used as a proxy for the average frequency.
                - "interpolate_orbit_averages_at_extrema": First, orbit averages
                  are obtained at each pericenter by averaging omega22(t) over the
                  time from the current pericenter to the next one. This average value
                  is associated with the time at mid point between the current and the
                  next pericenter. Similarly orbit averages are computed at apocenters.
                  Finally, a spline interpolant is constructed between all of these orbit
                  averages at extrema locations. Due to the nature of the averaging,
                  the final time over which the spline is constructed always starts
                  half and orbit after the first extrema and ends half an orbit before
                  the last extrema.
                - "omega22_zeroecc": omega22(t) of the quasi-circular counterpart
                  is used as a proxy for the average frequency. This can only be
                  used if "t_zeroecc" and "hlm_zeroecc" are provided in dataDict.
                Default is "mean_of_extrema_interpolants".
            treat_mid_points_between_pericenters_as_apocenters:
                If True, instead of trying to find apocenter locations by looking
                for local minima in the data, we simply find the midpoints between
                pericenter locations and treat them as apocenters. This is helpful
                for eccentricities ~1 where pericenters are easy to find but
                apocenters are not.
                Default: False.
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
        # This is useful, for example, to subtract the quasi circular
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
            "extrema_finding_kwargs": {},   # Gets overridden in methods like
                                            # eccDefinitionUsingAmplitude
            "debug": True,
            "omega22_averaging_method": "mean_of_extrema_interpolants",
            "treat_mid_points_between_pericenters_as_apocenters": False
            }
        return default_extra_kwargs

    def find_extrema(self, extrema_type="maxima"):
        """Find the extrema in the data.

        parameters:
        -----------
        extrema_type:
            One of 'maxima', 'pericenters', 'minima' or 'apocenters'.

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
            One of 'maxima', 'pericenters', 'minima' or 'apocenters'.

        returns:
        ------
        spline through extrema, positions of extrema
        """
        extrema_idx = self.find_extrema(extrema_type)
        # experimenting with throwing away pericenters too close to merger
        # This helps in avoiding unwanted feature in the spline
        # through the extrema
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
            # use only the extrema those are at least num_orbits away from the
            # merger to avoid nonphysical features like non-monotonic
            # eccentricity near the merger
            self.latest_time_used_for_extrema_finding = self.t[idx_num_orbit_earlier_than_merger]
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
        """Measure eccentricity and mean anomaly from a gravitational waveform.

        Eccentricity is measured using the GW frequency omega22(t) = dphi22(t)/dt,
        where phi22(t) is the phase of the (2,2) waveform mode. We evaluate
        omega22(t) at pericenter times, t_pericenters, and build a spline interpolant
        omega22_pericenters(t) using those points. Similarly, we build omega22_apocenters(t)
        using the apocenter times, t_apocenters. To find the pericenter/apocenter
        locations, one can look for extrema in different waveform data, like
        omega22(t) or Amp22(t), the amplitude of the (2,2) mode. Pericenters
        correspond to pericenters, while apocenters correspond to apocenters in the data.
        The method option (described below) lets you pick which waveform data to
        use to find extrema.

        The eccentricity is defined using omega22_pericenters(t) and omega22_apocenters(t),
        as described in Eq.(1) of arxiv:xxxx.xxxx. Similarly, the mean anomaly is
        defined using the pericenter locations as described in Eq.(2) of
        arxiv:xxxx.xxxx.

        FIXME ARIF: Fill in arxiv number when available. Make sure the above Eq
        numbers are right, once the paper is finalized.
        QUESTION FOR ARIF: Maybe we want to avoid saying pericenters/apocenters and just say
        pericenters/apocenters here and in the code? Also, get rid of all "astrons"
        throughout the code/documentation for consistency?

        parameters:
        ----------
        tref_in:
            Input reference time at which to measure eccentricity and mean anomaly.
            Can be a single float or an array.

        fref_in:
            Input reference GW frequency at which to measure the eccentricity and
            mean anomaly. Can be a single float or an array. Only one of
            tref_in/fref_in should be provided.

            Given an fref_in, we find the corresponding tref_in such that
            omega22_average(tref_in) = 2 * pi * fref_in. Here, omega22_average(t)
            is a monotonically increasing average frequency that is computed from
            the instantaneous omega22(t). omega22_average(t) is not a moving
            average; depending on which averaging method is used (see the
            omega22_averaging_method option below), it means slightly different
            things.

            Eccentricity and mean anomaly measurements are returned on a subset of
            tref_in/fref_in, called tref_out/fref_out, which are described below.
            If dataDict is provided in dimensionless units, tref_in should be in
            units of M and fref_in should be in units of cycles/M. If dataDict is
            provided in MKS units, t_ref should be in seconds and fref_in should be
            in Hz.

        returns:
        --------
        tref_out/fref_out:
            tref_out/fref_out is the output reference time/frequency at which
            eccentricity and mean anomaly are measured. If tref_in is provided,
            tref_out is returned, and if fref_in provided, fref_out is returned.
            Units of tref_out/fref_out are the same as those of tref_in/fref_in.

            tref_out is set as tref_out = tref_in[tref_in >= tmin & tref_in < tmax],
            where tmax = min(t_pericenters[-1], t_apocenters[-1]) and
                  tmin = max(t_pericenters[0], t_apocenters[0]),
            As eccentricity measurement relies on the interpolants omega22_pericenters(t)
            and omega22_apocenters(t), the above cutoffs ensure that we only compute
            the eccentricity where both omega22_pericenters(t) and omega22_apocenters(t) are
            within their bounds.

            fref_out is set as fref_out = fref_in[fref_in >= fmin & fref_in < fmax],
            where fmin = omega22_average(tmin)/2/pi and
                  fmax = omega22_average(tmax)/2/pi, with tmin/tmax same as above.

        ecc_ref:
            Measured eccentricity at tref_out/fref_out. Same type as
            tref_out/fref_out.

        mean_ano_ref:
            Measured mean anomaly at tref_out/fref_out. Same type as
            tref_out/fref_out.
        """
        self.omega22_pericenters_interp, self.pericenters_location = self.interp_extrema("maxima")
        # In some cases it is easier to find the pericenters than finding the
        # apocenters. For such cases, one can only find the pericenters and use the
        # mid points between two consecutive pericenters as the location of the
        # apocenters.
        if self.extra_kwargs["treat_mid_points_between_pericenters_as_apocenters"]:
            self.omega22_apocenters_interp, self.apocenters_location = self.get_apocenters_from_pericenters()
        else:
            self.omega22_apocenters_interp, self.apocenters_location = self.interp_extrema("minima")

        # check that pericenters and apocenters are appearing alternately
        self.check_pericenters_and_apocenters_appear_alternately()

        self.t_pericenters = self.t[self.pericenters_location]
        self.t_apocenters = self.t[self.apocenters_location]
        self.t_max = min(self.t_pericenters[-1], self.t_apocenters[-1])
        self.t_min = max(self.t_pericenters[0], self.t_apocenters[0])
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
        # for a pericenter before and after the ref time to calculate the current
        # period.
        # If ref time is t_max, which could be equal to the last pericenter, then
        # there is no next pericenter and that would cause a problem.
        self.tref_out = self.tref_in[np.logical_and(self.tref_in < self.t_max,
                                                    self.tref_in >= self.t_min)]
        # set time for checks and diagnostics
        self.t_for_checks = self.dataDict["t"][
            np.logical_and(self.dataDict["t"] >= self.t_min,
                           self.dataDict["t"] < self.t_max)]

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
                                "which corresponds to min(last pericenter "
                                "time, last apocenter time).")
            if self.tref_in[0] < self.t_min:
                raise Exception(f"tref_in {self.tref_in} is earlier than t_min="
                                f"{self.t_min}, "
                                "which corresponds to max(first pericenter "
                                "time, first apocenter time).")
            raise Exception("tref_out is empty. This can happen if the "
                            "waveform has insufficient identifiable "
                            "pericenters/apocenters.")

        # check separation between extrema
        self.orb_phase_diff_at_pericenters, \
            self.orb_phase_diff_ratio_at_pericenters \
            = self.check_extrema_separation(self.pericenters_location, "pericenters")
        self.orb_phase_diff_at_apocenters, \
            self.orb_phase_diff_ratio_at_apocenters \
            = self.check_extrema_separation(self.apocenters_location, "apocenters")

        # Check if tref_out has a pericenter before and after.
        # This is required to define mean anomaly.
        # See explanation on why we do not include the last pericenter above.
        if self.tref_out[0] < self.t_pericenters[0] or self.tref_out[-1] >= self.t_pericenters[-1]:
            raise Exception("Reference time must be within two pericenters.")

        # compute eccentricity at self.tref_out
        self.ecc_ref = self.compute_eccentricity(self.tref_out)
        # Compute mean anomaly at tref_out
        self.mean_ano_ref = self.compute_mean_ano(self.tref_out)

        # check if eccentricity is positive
        if any(self.ecc_ref < 0):
            warnings.warn("Encountered negative eccentricity.")

        # check if eccentricity is monotonic and convex
        self.check_monotonicity_and_convexity()

        if len(self.tref_out) == 1:
            self.mean_ano_ref = self.mean_ano_ref[0]
            self.ecc_ref = self.ecc_ref[0]
            self.tref_out = self.tref_out[0]
        if fref_in is not None and len(self.fref_out) == 1:
            self.fref_out = self.fref_out[0]

        return_array = self.fref_out if fref_in is not None else self.tref_out
        return return_array, self.ecc_ref, self.mean_ano_ref

    def compute_eccentricity(self, t):
        """
        Computer eccentricity at time t.

        Compute eccentricity from the value of omega22_pericenters_interpolant
        and omega22_apocenters_interpolant at t using the formula in
        ref. arXiv:2101.11798 Eq. (4).

        #FIXME: ARIF change the above reference when gw eccentricity paper is out

        Paramerers:
        -----------
        t:
            Time to compute the eccentricity at. Could be scalar or an array.

        Returns:
        --------
        Eccentricity at t.
        """
        omega22_pericenter_at_t = self.omega22_pericenters_interp(t)
        omega22_apocenter_at_t = self.omega22_apocenters_interp(t)
        return ((np.sqrt(omega22_pericenter_at_t)
                 - np.sqrt(omega22_apocenter_at_t))
                / (np.sqrt(omega22_pericenter_at_t)
                   + np.sqrt(omega22_apocenter_at_t)))

    def derivative_of_eccentricity(self, n=1):
        """Get derivative of eccentricity.

        Parameters:
        -----------
        n: int
            Order of derivative. Should be 1 or 2, since it uses
            cubic spine to get the derivatives.

        Returns:
        --------
            nth order derivative of eccentricity.
        """
        if not hasattr(self, "ecc_for_checks"):
            self.ecc_for_checks = self.compute_eccentricity(
                self.t_for_checks)

        spline = InterpolatedUnivariateSpline(self.t_for_checks,
                                              self.ecc_for_checks)
        # Get derivative of ecc(t) using cubic splines.
        return spline.derivative(n=1)(self.t_for_checks)
        
    def compute_mean_ano(self, t):
        """
        Compute mean anomaly at time t.
        
        Compute the mean anomaly using Eq.7 of arXiv:2101.11798.
        Mean anomaly grows linearly in time from 0 to 2 pi over
        the range [t_at_last_pericenter, t_at_next_pericenter], where t_at_last_pericenter
        is the time at the previous pericenter, and t_at_next_pericenter is
        the time at the next pericenter.

        #FIXME: ARIF Change the above reference when gw eccentricity paper is out

        Parameters:
        -----------
        t:
            Time to compute mean anomaly at. Could be scalar or an array.

        Returns:
        --------
        Mean anomaly at t.
        """
        @np.vectorize
        def mean_ano(t):
            idx_at_last_pericenter = np.where(self.t_pericenters <= t)[0][-1]
            t_at_last_pericenter = self.t_pericenters[idx_at_last_pericenter]
            t_at_next_pericenter = self.t_pericenters[idx_at_last_pericenter + 1]
            t_since_last_pericenter = t - t_at_last_pericenter
            current_period = t_at_next_pericenter - t_at_last_pericenter
            mean_ano_ref = 2 * np.pi * t_since_last_pericenter / current_period
            return mean_ano_ref

        return mean_ano(t)

    def check_extrema_separation(self, extrema_location,
                                 extrema_type="extrema",
                                 max_orb_phase_diff_factor=1.5,
                                 min_orb_phase_diff=np.pi):
        """Check if two extrema are too close or too far."""
        orb_phase_at_extrema = self.phase22[extrema_location] / 2
        orb_phase_diff = np.diff(orb_phase_at_extrema)
        # This might suggest that the data is noisy, for example, and a
        # spurious pericenter got picked up.
        t_at_extrema = self.t[extrema_location][1:]
        if any(orb_phase_diff < min_orb_phase_diff):
            too_close_idx = np.where(orb_phase_diff < min_orb_phase_diff)[0]
            too_close_times = t_at_extrema[too_close_idx]
            warnings.warn(f"At least a pair of {extrema_type} are too close."
                          " Minimum orbital phase diff is "
                          f"{min(orb_phase_diff)}. Times of occurrences are"
                          f" {too_close_times}")
        if any(np.abs(orb_phase_diff - np.pi)
               < np.abs(orb_phase_diff - 2 * np.pi)):
            warnings.warn("Phase shift closer to pi than 2 pi detected.")
        # This might suggest that the extrema finding method missed an extrema.
        # We will check if the phase diff at an extrema is greater than
        # max_orb_phase_diff_factor times the orb_phase_diff at the
        # previous extrema
        orb_phase_diff_ratio = orb_phase_diff[1:]/orb_phase_diff[:-1]
        # make it of same length as orb_phase_diff by prepending 0
        orb_phase_diff_ratio = np.append([0], orb_phase_diff_ratio)
        if any(orb_phase_diff_ratio > max_orb_phase_diff_factor):
            too_far_idx = np.where(orb_phase_diff_ratio
                                   > max_orb_phase_diff_factor)[0]
            too_far_times = t_at_extrema[too_far_idx]
            warnings.warn(f"At least a pair of {extrema_type} are too far."
                          " Maximum orbital phase diff is "
                          f"{max(orb_phase_diff)}. Times of occurrences are"
                          f" {too_far_times}")
        return orb_phase_diff, orb_phase_diff_ratio

    def check_monotonicity_and_convexity(self,
                                         check_convexity=False):
        """Check if measured eccentricity is a monotonic function of time.

        parameters:
        check_convexity:
            In addition to monotonicity, it will check for
            convexity as well. Default is False.
        """
        self.decc_dt = self.derivative_of_eccentricity(n=1)

        # Is ecc(t) a monotonically decreasing function?
        if any(self.decc_dt > 0):
            warnings.warn("Ecc(t) is non monotonic.")

        # Is ecc(t) a convex function? That is, is the second
        # derivative always positive?
        if check_convexity:
            self.d2ecc_dt = self.derivative_of_eccentricity(n=2)
            self.d2ecc_dt = self.d2ecc_dt
            if any(self.d2ecc_dt > 0):
                warnings.warn("Ecc(t) is concave.")

    def check_pericenters_and_apocenters_appear_alternately(self):
        """Check that pericenters and apocenters appear alternately."""
        # if pericenters and apocenters appear alternately, then the number
        # of pericenters and apocenters should differ by one.
        if abs(len(self.pericenters_location) - len(self.apocenters_location)) >= 2:
            warnings.warn(
                "Number of pericenters and number of apocenters differ by "
                f"{abs(len(self.pericenters_location) - len(self.apocenters_location))}"
                ". This implies that pericenters and apocenters are not appearing"
                " alternately.")
        else:
            # If the number of pericenters and apocenters differ by zero or one then we
            # do the following:
            if len(self.pericenters_location) == len(self.apocenters_location):
                # Check the time of the first pericenter and the first apocenter
                # whichever comes first is assigned as arr1 and the other one
                # as arr2
                if self.t[self.pericenters_location][0] < self.t[self.apocenters_location][0]:
                    arr1 = self.pericenters_location
                    arr2 = self.apocenters_location
                else:
                    arr2 = self.pericenters_location
                    arr1 = self.apocenters_location
            else:
                # Check the number of pericenters and apocenters
                # whichever is larger is assigned as arr1 and the other one as
                # arr2
                if len(self.pericenters_location) > len(self.apocenters_location):
                    arr1 = self.pericenters_location
                    arr2 = self.apocenters_location
                else:
                    arr2 = self.pericenters_location
                    arr1 = self.apocenters_location
            # create a new array which takes elements from arr1 and arr2
            # alternately
            arr = np.zeros(arr1.shape[0] + arr2.shape[0], dtype=arr1.dtype)
            # assign every other element to values from arr1 starting from
            # index = 0
            arr[::2] = arr1
            # assign every other element to values from arr2 starting from
            # index = 1
            arr[1::2] = arr2
            # get the time difference between consecutive locations in arr
            t_diff = np.diff(self.t[arr])
            # If pericenters and apocenters appear alternately then all the time
            # differences in t_diff should be positive
            if any(t_diff < 0):
                warnings.warn(
                    "There is at least one instance where "
                    "pericenters and apocenters do not appear alternately.")

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

        We compute the orbital average of omega22 at the pericenters
        and the apocenters following:
        omega22_avg((t[i]+ t[i+1])/2) = int_t[i]^t[i+1] omega22(t)dt
                                        / (t[i+1] - t[i])
        where t[i] is the time of ith extrema.
        We do this for pericenters and apocenters and combine the results
        """
        extrema_locations = {"pericenters": self.pericenters_location,
                             "apocenters": self.apocenters_location}
        @np.vectorize
        def orbital_averaged_omega22_at_extrema(n, extrema_type="pericenters"):
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
        # get the mid points between the pericenters as avg time for pericenters
        t_average_pericenters = (self.t[self.pericenters_location][1:]
                           + self.t[self.pericenters_location][:-1]) / 2
        omega22_average_pericenters = orbital_averaged_omega22_at_extrema(
            np.arange(len(self.pericenters_location) - 1), "pericenters")
        # get the mid points between the apocenters as avg time for apocenters
        t_average_apocenters = (self.t[self.apocenters_location][1:]
                             + self.t[self.apocenters_location][:-1]) / 2
        omega22_average_apocenters = orbital_averaged_omega22_at_extrema(
            np.arange(len(self.apocenters_location) - 1), "apocenters")
        # combine results from average at pericenters and toughs
        t_average = np.append(t_average_apocenters, t_average_pericenters)
        # sort the times
        sorted_idx = np.argsort(t_average)
        t_average = t_average[sorted_idx]
        # check if the average omega22 are monotonically increasing
        if any(np.diff(omega22_average_pericenters) <= 0):
            raise Exception("Omega22 average at pericenters are not strictly "
                            "monotonically increaing")
        if any(np.diff(omega22_average_apocenters) <= 0):
            raise Exception("Omega22 average at apocenters are not strictly "
                            "monotonically increasing")
        omega22_average = np.append(omega22_average_apocenters,
                                    omega22_average_pericenters)
        # sort omega22
        omega22_average = omega22_average[sorted_idx]
        return InterpolatedUnivariateSpline(t_average, omega22_average)(t)

    def compute_omega22_average_between_extrema(self, t):
        """Find omega22 average between extrema".

        Take mean of omega22 using spline through omega22 pericenters
        and spline through omega22 apocenters.
        """
        return ((self.omega22_pericenters_interp(t)
                 + self.omega22_apocenters_interp(t)) / 2)

    def compute_omega22_zeroecc(self, t):
        """Find omega22 from zeroecc data."""
        return InterpolatedUnivariateSpline(
            self.t_zeroecc_shifted, self.omega22_zeroecc)(t)

    def get_available_omega22_averaging_methods(self):
        """Return available omega22 averaging methods."""
        available_methods = {
            "mean_of_extrema_interpolants": self.compute_omega22_average_between_extrema,
            "interpolate_orbit_averages_at_extrema": self.compute_orbital_averaged_omega22_at_extrema,
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
        - Mean of the omega22 given by the spline through the pericenters and the
          spline through the apocenters, we call this "mean_of_extrema_interpolants"
        - Orbital average at the extrema, we call this "interpolate_orbit_averages_at_extrema"
        - omega22 of the zero eccentricity waveform, called "omega22_zeroecc"

        User can provide a method through the "extra_kwargs" option with the key
        "omega22_averaging_method". Default is "mean_of_extrema_interpolants"

        Once we get the reference frequencies, we create a spline to get time
        as function of these reference frequencies. This should work if the
        reference frequency is monotonic which it should be.

        Finally we evaluate this spine on the fref_in to get the tref_in.
        """
        self.available_averaging_methods = self.get_available_omega22_averaging_methods()
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
                                "pericenters/apocenters.")
        return fref_out

    def make_diagnostic_plots(
            self,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Make diagnostic plots for the eccDefinition method.

        We plot different quantities to asses how well our eccentricity
        measurement method is working. This could be seen as a diagnostic tool
        to check an implemented method.

        We plot the following quantities
        - The eccentricity vs vs time
        - decc/dt vs time, this is to test the monotonicity of eccentricity as
          a function of time
        - mean anomaly vs time
        - omega_22 vs time with the pericenters and apocenters shown. This would show
          if the method is missing any pericenters/apocenters or selecting one which is
          not a pericenter/apocenter
        - deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
          change in orbital phase from the previous extrema to the ith extrema.
          This helps to look for missing extrema, as there will be a drastic
          (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
          extrema, and the ratio will go from ~1 to ~2.

        Additionally, we plot the following if data for zero eccentricity is
        provided and method is not residual method
        - residual amp22 vs time with the location of pericenters and apocenters shown.
        - residual omega22 vs time with the location of pericenters and apocenters
          shown.
        If the method itself uses residual data, then add one plot for
        - data that is not being used for finding extrema.
        For example, if method is ResidualAmplitude
        then plot residual omega and vice versa.
        These two plots further help in understanding any unwanted feature
        in the measured eccentricity vs time plot. For example, non smoothness
        in the residual omega22 would indicate that the data in omega22 is not
        good which might be causing glitches in the measured eccentricity plot.

        Finally, plot
        - data that is being used for finding extrema.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        fig, ax
        """
        # Make a list of plots we want to add
        list_of_plots = [self.plot_measured_ecc,
                         self.plot_decc_dt,
                         self.plot_mean_ano,
                         self.plot_omega22,
                         self.plot_phase_diff_ratio_between_pericenters]
        if "hlm_zeroecc" in self.dataDict:
            # add residual amp22 plot
            if "Delta A" not in self.label_for_data_for_finding_extrema:
                list_of_plots.append(self.plot_residual_amp22)
            # add residual omega22 plot
            if "Delta\omega" not in self.label_for_data_for_finding_extrema:
                list_of_plots.append(self.plot_residual_omega22)
        # add plot of data used for finding extrema
        list_of_plots.append(self.plot_data_used_for_finding_extrema)

        # Initiate figure, axis
        figsize = (12, 4 * len(list_of_plots))
        default_kwargs = {"nrows": len(list_of_plots),
                          "figsize": figsize,
                          "sharex": True}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        fig, ax = plt.subplots(**kwargs)

        # populate figure, axis
        for idx, plot in enumerate(list_of_plots):
            plot(
                fig,
                ax[idx],
                add_help_text=add_help_text,
                usetex=usetex,
                use_fancy_settings=False)
            ax[idx].tick_params(labelbottom=True)
        fig.tight_layout()
        return fig, ax

    def plot_measured_ecc(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot measured ecc as function of time.

                Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        ax.plot(self.t_for_checks, self.ecc_for_checks, **kwargs)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"Eccentricity $e$")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_decc_dt(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot decc_dt as function of time to check monotonicity.

        If decc_dt becomes positive, ecc(t) is not monotonically decreasing.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        if not hasattr(self, "decc_dt"):
            self.decc_dt = self.derivative_of_eccentricity(n=1)
        ax.plot(self.t_for_checks, self.decc_dt, **kwargs)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$de/dt$")
        if add_help_text:
            ax.text(
                0.5,
                0.98,
                ("We expect decc/dt to be always negative"),
                ha="center",
                va="top",
                transform=ax.transAxes)
        # add line to indicate y = 0
        ax.axhline(0, ls="--")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_mean_ano(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot measured mean anomaly as function of time.

                Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        ax.plot(self.t_for_checks,
                self.compute_mean_ano(self.t_for_checks),
                **kwargs)
        ax.set_xlabel("$t$")
        ax.set_ylabel("mean anomaly")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_omega22(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot omega22, the locations of the apocenters and pericenters.

        Also plots their corresponding interpolants.
        This would show if the method is missing any pericenters/apocenters or
        selecting one which is not a pericenter/apocenter.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        ax.plot(self.t_for_checks,
                self.omega22_pericenters_interp(self.t_for_checks),
                c=colorsDict["pericenter"], label=r"$\omega_{p}$",
                **kwargs)
        ax.plot(self.t_for_checks, self.omega22_apocenters_interp(self.t_for_checks),
                c=colorsDict["apocenter"], label=r"$\omega_{a}$",
                **kwargs)
        ax.plot(self.t, self.omega22,
                c=colorsDict["default"], label=r"$\omega_{22}$")
        ax.plot(self.t[self.pericenters_location],
                self.omega22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="")
        ax.plot(self.t[self.apocenters_location],
                self.omega22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="")
        # set reasonable ylims
        ymin = min(min(self.omega22[self.pericenters_location]),
                   min(self.omega22[self.apocenters_location]))
        ymax = max(max(self.omega22[self.pericenters_location]),
                   max(self.omega22[self.apocenters_location]))
        pad = 0.05 * ymax # 5 % buffer for better visibility
        ax.set_ylim(ymin - pad, ymax + pad)
        # add help text
        backslashchar = "\\"
        if add_help_text:
            ax.text(
                0.5,
                0.98,
                (f"tref{backslashchar if usetex else ''}_out excludes the first and last extrema\n"
                 " to avoid extrapolation when computing ecc(t)"),
                ha="center",
                va="top",
                transform=ax.transAxes)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\omega_{22}$")
        ax.legend(frameon=True, loc="upper left")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_amp22(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot amp22, the locations of the apocenters and pericenters.

        This would show if the method is missing any pericenters/apocenters or
        selecting one which is not a pericenter/apocenter.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        ax.plot(self.t, self.amp22,
                c=colorsDict["default"], label=r"$A_{22}$")
        ax.plot(self.t[self.pericenters_location],
                self.amp22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label="Pericenters")
        ax.plot(self.t[self.apocenters_location],
                self.amp22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label="Apocenters")
        # set reasonable ylims
        ymin = min(min(self.amp22[self.pericenters_location]),
                   min(self.amp22[self.apocenters_location]))
        ymax = max(max(self.amp22[self.pericenters_location]),
                   max(self.amp22[self.apocenters_location]))
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$A_{22}$")
        ax.legend()
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_phase_diff_ratio_between_pericenters(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot phase diff ratio between consecutive as function of time.

        Plots deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
        change in orbital phase from the previous extrema to the ith extrema.
        This helps to look for missing extrema, as there will be a drastic
        (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
        extrema, and the ratio will go from ~1 to ~2.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        tpericenters = self.t[self.pericenters_location[1:]]
        ax.plot(tpericenters[1:], self.orb_phase_diff_ratio_at_pericenters[1:],
                c=colorsDict["pericenter"],
                marker=".", label="Pericenter phase diff ratio")
        tapocenters = self.t[self.apocenters_location[1:]]
        ax.plot(tapocenters[1:], self.orb_phase_diff_ratio_at_apocenters[1:],
                c=colorsDict["apocenter"],
                marker=".", label="Apocenter phase diff ratio")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Delta \Phi_{orb}[i] / \Delta \Phi_{orb}[i-1]$")
        if add_help_text:
            ax.text(
                0.5,
                0.98,
                ("If phase difference exceeds 1.5,\n"
                 "There may be missing extrema."),
                ha="center",
                va="top",
                transform=ax.transAxes)
        ax.set_title("Ratio of phase difference between consecutive extrema")
        ax.legend(frameon=True)
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_residual_omega22(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual omega22, the locations of the apocenters and pericenters.

        Useful to look for bad omega22 data near merger.
        We also throw away post merger before since it makes the plot
        unreadble.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        ax.plot(self.t, self.res_omega22, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_omega22[self.pericenters_location],
                marker=".", ls="", label="Pericenter",
                c=colorsDict["pericenter"])
        ax.plot(self.t[self.apocenters_location],
                self.res_omega22[self.apocenters_location],
                marker=".", ls="", label="Apocenter",
                c=colorsDict["apocenter"])
        # set reasonable ylims
        ymin = min(min(self.res_omega22[self.pericenters_location]),
                   min(self.res_omega22[self.apocenters_location]))
        ymax = max(max(self.res_omega22[self.pericenters_location]),
                   max(self.res_omega22[self.apocenters_location]))
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Delta\omega_{22}$")
        ax.legend(frameon=True, loc="center left")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_residual_amp22(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual amp22, the locations of the apocenters and pericenters.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        ax.plot(self.t, self.res_amp22, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_amp22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label="Pericenter")
        ax.plot(self.t[self.apocenters_location],
                self.res_amp22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label="Apocenter")
        # set reasonable ylims
        ymin = min(min(self.res_amp22[self.pericenters_location]),
                   min(self.res_amp22[self.apocenters_location]))
        ymax = max(max(self.res_amp22[self.pericenters_location]),
                   max(self.res_amp22[self.apocenters_location]))
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\Delta A_{22}$")
        ax.legend(frameon=True, loc="center left")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_data_used_for_finding_extrema(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            journal="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot the data that is being used.

        Also the locations of the apocenters and pericenters.
        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure object.
            Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis object.
            Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        journal:
            Set font size, figure size suitable for particular use case. For example,
            to generate plot for "APS" journals, use journal="APS".
            For showing plots in a jupyter notebook, use "Notebook" so that
            plots are bigger and fonts are appropriately larger and so on.
            See plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize = (figWidthsTwoColDict[journal], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, journal=journal)
        # To make it work for FrequencyFits
        # FIXME: Harald, Arif: Think about how to make this better.
        if hasattr(self, "t_analyse"):
            t_for_finding_extrema = self.t_analyse
            self.latest_time_used_for_extrema_finding = self.t_analyse[-1]
        else:
            t_for_finding_extrema = self.t
        ax.plot(t_for_finding_extrema, self.data_for_finding_extrema, c=colorsDict["default"])
        pericenters, = ax.plot(
            t_for_finding_extrema[self.pericenters_location],
            self.data_for_finding_extrema[self.pericenters_location],
            c=colorsDict["pericenter"],
            marker=".", ls="")
        apocenters, = ax.plot(
            t_for_finding_extrema[self.apocenters_location],
            self.data_for_finding_extrema[self.apocenters_location],
            c=colorsDict["apocenter"],
            marker=".", ls="")
        # set reasonable ylims
        ymin = min(min(self.data_for_finding_extrema[self.pericenters_location]),
                   min(self.data_for_finding_extrema[self.apocenters_location]))
        ymax = max(max(self.data_for_finding_extrema[self.pericenters_location]),
                   max(self.data_for_finding_extrema[self.apocenters_location]))
        # we want to make the ylims symmetric about y=0 when Residual data is used
        if "Delta" in self.label_for_data_for_finding_extrema:
            ylim = max(ymax, -ymin)
            pad = 0.05 * ylim # 5 % buffer for better visibility
            ax.set_ylim(-ylim - pad, ylim + pad)
        else:
            pad = 0.05 * ymax
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(self.label_for_data_for_finding_extrema)
        # Add vertical line to indicate the latest time used for extrema finding
        latest_time_vline = ax.axvline(
            self.latest_time_used_for_extrema_finding,
            c=colorsDict["vline"], ls="--")
        # add legends
        ax.legend(handles=[pericenters, apocenters, latest_time_vline],
                  labels=["Pericenters", "Apocenters", "Latest time used for finding extrema."],
                  loc="center left",
                  frameon=True)
        ax.set_title(
            "Data being used for finding the extrema.",
            ha="center")
        if add_help_text:
            if isinstance(self.tref_out, (float, int)):
                ax.axvline(self.tref_out, c=colorsDict["pericentersvline"])
                ax.text(
                    self.tref_out,
                    ymin,
                    (r"$t_{\mathrm{ref}}$"),
                    ha="right",
                    va="bottom")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def get_apocenters_from_pericenters(self):
        """Get Interpolator through apocenters and their locations.

        This function treats the mid points between two successive pericenters
        as the location of the apocenter in between the same two pericenters. Thus
        it does not find the locations of the apocenters using pericenter
        finder at all. It is useful in situation where finding pericenters
        is easy but finding the apocenters in between is difficult. This is
        the case for highly eccentric systems where eccentricity approaches
        1. For such systems the amp22/omega22 data between the pericenters is almost
        flat and hard to find the local minima.

        returns:
        ------
        spline through apocenters, positions of apocenters
        """
        # NOTE: Assuming uniform time steps.
        # TODO: Make it work for non uniform time steps
        # In the following we get the location of mid point between ith pericenter
        # and (i+1)th pericenter as (loc[i] + loc[i+1])/2 where loc is the array
        # that contains the pericenter locations. This works because time steps are
        # assumed to be uniform and hence proportional to the time itself.
        apocenters_idx = (self.pericenters_location[:-1] + self.pericenters_location[1:]) / 2
        apocenters_idx = apocenters_idx.astype(int)  # convert to ints
        if len(apocenters_idx) >= 2:
            spline = InterpolatedUnivariateSpline(self.t[apocenters_idx],
                                                  self.omega22[apocenters_idx],
                                                  **self.spline_kwargs)
            return spline, apocenters_idx
        else:
            raise Exception(
                "Sufficient number of apocenters are not found."
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
            parameter sets the minimal separation between peaks.
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
        # between extrema, otherwise we might miss a few extrema near merger
        return int(width / 4)
