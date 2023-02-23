"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Classes for eccentricity definitions should be derived from `eccDefinition`
(or one of it's derived classes. See the following wiki
https://github.com/vijayvarma392/gw_eccentricity/wiki/Adding-new-eccentricity-definitions)
"""

import numpy as np
from .utils import peak_time_via_quadratic_fit, check_kwargs_and_set_defaults
from .utils import amplitude_using_all_modes
from .utils import time_deriv_4thOrder
from .utils import interpolate
from .utils import get_interpolant
from .utils import get_default_spline_kwargs
from .utils import debug_message
from .plot_settings import use_fancy_plotsettings, colorsDict, labelsDict
from .plot_settings import figWidthsTwoColDict, figHeightsDict
import matplotlib.pyplot as plt
import copy


class eccDefinition:
    """Base class to define eccentricity for given waveform data dictionary."""

    def __init__(self, dataDict, num_orbits_to_exclude_before_merger=2,
                 extra_kwargs=None):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict: dict
            Dictionary containing waveform modes dict, time etc. Should follow
            the format:
            dataDict = {"t": time,
                        "hlm": modeDict,
                        "t_zeroecc": time,
                        "hlm_zeroecc": modeDict,
                       },
            "t" and "hlm" are mandatory. "t_zeroecc" and "hlm_zeroecc"
            are only required for `ResidualAmplitude` and
            `ResidualFrequency` methods, but if provided, they are
            used for additional diagnostic plots, which can be helpful
            for all methods. Any other keys in dataDict will be
            ignored, with a warning.

            The recognized keys are:
            - "t": 1d array of times.
                - Should be uniformly sampled, with a small enough time step
                  so that omega22(t) can be accurately computed. We use a
                  4th-order finite difference scheme. In dimensionless units,
                  we recommend a time step of dtM = 0.1M to be conservative,
                  but one may be able to get away with larger time steps like
                  dtM = 1M. The corresponding time step in seconds would be dtM
                  * M * lal.MTSUN_SI, where M is the total mass in Solar
                  masses.
                - We do not require the waveform peak amplitude to occur at any
                  specific time, but tref_in should follow the same convention
                  for peak time as "t".
            - "hlm": Dictionary of waveform modes associated with "t".
                - Should have the format:
                    modeDict = {(l1, m1): h_{l1, m1},
                                (l2, m2): h_{l2, m2},
                                ...
                               },
                    where h_{l, m} is a 1d complex array representing the (l,
                    m) waveform mode. Should contain at least the (2, 2) mode,
                    but more modes can be included, as indicated by the
                    ellipsis '...'  above.
            - "t_zeroecc" and "hlm_zeroecc":
                - Same as above, but for the quasicircular counterpart to the
                  eccentric waveform. The quasicircular counterpart can be
                  obtained by evaluating a waveform model by keeping the rest
                  of the binary parameters fixed (same as the ones used to
                  generate "hlm") but setting the eccentricity to zero. For NR,
                  if such a quasicircular counterpart is not available, we
                  recommend using quasicircular models like NRHybSur3dq8 or
                  IMRPhenomT, depending on the mass ratio and spins.
                - "t_zeroecc" should be uniformly spaced, but does not have to
                  follow the same time step as that of "t", as long as the step
                  size is small enough to compute the frequency. Similarly,
                  peak time does not have to match that of "t".
                - We require that "hlm_zeroecc" be at least as long as "hlm" so
                  that residual amplitude/frequency can be computed.

        num_orbits_to_exclude_before_merger:
                Can be None or a non negative number.  If None, the full
                waveform data (even post-merger) is used for finding extrema,
                but this might cause interpolation issues.  For a non negative
                num_orbits_to_exclude_before_merger, that many orbits before
                the merger are excluded when finding extrema.  If your waveform
                does not have a merger (e.g. PN/EMRI), use
                num_orbits_to_exclude_before_merger = None.

                The default value is chosen via an investigation on a set of NR
                waveforms. See the following wiki page for more details,
                https://github.com/vijayvarma392/gw_eccentricity/wiki/NR-investigation-to-set-default-number-of-orbits-to-exclude-before-merger
                Default: 2.

        extra_kwargs: dict
            A dictionary of any extra kwargs to be passed. Allowed kwargs
            are:
            spline_kwargs: dict
                Dictionary of arguments to be passed to the spline
                interpolation routine
                (scipy.interpolate.InterpolatedUnivariateSpline) used to
                compute quantities like omega22_pericenters(t) and
                omega22_apocenters(t).
                Defaults are set using utils.get_default_spline_kwargs

            extrema_finding_kwargs: dict
                Dictionary of arguments to be passed to the extrema finder,
                scipy.signal.find_peaks.
                The Defaults are the same as those of scipy.signal.find_peaks,
                except for the "width", which sets the minimum allowed "full
                width at half maximum" for the extrema. Setting this can help
                avoid false extrema in noisy data (for example, due to junk
                radiation in NR). The default for "width" is set using phi22(t)
                near the merger. Starting from 4 cycles of the (2, 2) mode
                before the merger, we find the number of time steps taken to
                cover 2 cycles, let's call this "the gap". Note that 2 cycles
                of the (2, 2) mode are approximately one orbit, so this allows
                us to approximate the smallest gap between two
                pericenters/apocenters. However, to be conservative, we divide
                this gap by 4 and set it as the width parameter for
                find_peaks. See
                eccDefinition.get_width_for_peak_finder_from_phase22 for more
                details.

            debug_level: int
                Debug settings for warnings/errors:
                -1: All warnings are suppressed. NOTE: Use at your own risk!
                0: Only important warnings are issued.
                1: All warnings are issued. Use when investigating.
                2: All warnings become exceptions.
                Default: 0.

            debug_plots: bool
                If True, diagnostic plots are generated. This can be
                computationally expensive and should only be used when
                debugging. When True, look for figures saved as
                `gwecc_{method}_*.pdf`.

            omega22_averaging_method:
                Options for obtaining omega22_average(t) from the instantaneous
                omega22(t).
                - "orbit_averaged_omega22": First, orbit averages are obtained
                  at each pericenter by averaging omega22(t) over the time from
                  the current pericenter to the next one. This average value is
                  associated with the time at mid point between the current and
                  the next pericenter. Similarly orbit averages are computed at
                  apocenters.  Finally, a spline interpolant is constructed
                  between all of these orbit averages at extrema
                  locations. However, the final time over which the spline is
                  constructed is constrained to be between tmin_for_fref and
                  tmax_for_fref which are close to tmin and tmax,
                  respectively. See eccDefinition.get_fref_bounds() for
                  details.
                - "mean_of_extrema_interpolants": The mean of
                  omega22_pericenters(t) and omega22_apocenters(t) is used as a
                  proxy for the average frequency.
                - "omega22_zeroecc": omega22(t) of the quasicircular
                  counterpart is used as a proxy for the average
                  frequency. This can only be used if "t_zeroecc" and
                  "hlm_zeroecc" are provided in dataDict.
                See Sec. IID of arXiv:2302.11257 for more detail description of
                average omega22.
                Default is "orbit_averaged_omega22".

            treat_mid_points_between_pericenters_as_apocenters:
                If True, instead of trying to find apocenter locations by
                looking for local minima in the data, we simply find the
                midpoints between pericenter locations and treat them as
                apocenters. This is helpful for eccentricities ~1 where
                pericenters are easy to find but apocenters are not.
                Default: False.

            kwargs_for_fits_methods:
                Extra kwargs to be passed to FrequencyFits and AmplitudeFits
                methods. See
                eccDefinitionUsingFrequencyFits.get_default_kwargs_for_fits_methods
                for allowed keys.
        """
        # Truncate dataDict if num_orbits_to_exclude_before_merger is not None
        self.dataDict, self.t_merger, self.amp22_merger, min_width_for_extrema \
            = self.truncate_dataDict_if_necessary(
                dataDict, num_orbits_to_exclude_before_merger, extra_kwargs)
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
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = time_deriv_4thOrder(self.phase22,
                                           self.t[1] - self.t[0])
        # Sanity check various kwargs and set default values
        self.extra_kwargs = check_kwargs_and_set_defaults(
            extra_kwargs, self.get_default_extra_kwargs(),
            "extra_kwargs",
            "eccDefinition.get_default_extra_kwargs()")
        self.extrema_finding_kwargs = check_kwargs_and_set_defaults(
            self.extra_kwargs['extrema_finding_kwargs'],
            self.get_default_extrema_finding_kwargs(min_width_for_extrema),
            "extrema_finding_kwargs",
            "eccDefinition.get_default_extrema_finding_kwargs()")
        self.spline_kwargs = check_kwargs_and_set_defaults(
            self.extra_kwargs["spline_kwargs"],
            get_default_spline_kwargs(),
            "spline_kwargs",
            "utils.get_default_spline_kwargs()")
        self.available_averaging_methods \
            = self.get_available_omega22_averaging_methods()
        self.debug_level = self.extra_kwargs["debug_level"]
        self.debug_plots = self.extra_kwargs["debug_plots"]
        # check if there are unrecognized keys in the dataDict
        self.recognized_dataDict_keys = self.get_recognized_dataDict_keys()
        for kw in dataDict.keys():
            if kw not in self.recognized_dataDict_keys:
                debug_message(
                    f"kw {kw} is not a recognized key word in dataDict.",
                    debug_level=self.debug_level)
        # Measured values of eccentricities to perform diagnostic checks.  For
        # example to plot ecc vs time plot, or checking monotonicity of
        # eccentricity as a function of time. These are values of
        # eccentricities measured at t_for_checks where t_for_checks is the
        # time array in dataDict lying between tmin and tmax.  tmin is
        # max(t_pericenters, t_apocenters) and tmax is min(t_pericenters,
        # t_apocenters) Initially set to None, but will get computed when
        # necessary, in either derivative_of_eccentricity or plot_eccentricity.
        self.ecc_for_checks = None
        # Spline interpolant of measured eccentricity as function of time built
        # using ecc_for_checks at t_for_checks. This is used to get
        # first/second derivative of eccentricity with respect to time.
        # Initially set to None, but will get computed when necessary, in
        # derivative_of_eccentricity.
        self.ecc_interp = None
        # First derivative of eccentricity with respect to time at
        # t_for_checks. Will be used to check monotonicity, plot decc_dt
        # Initially set to None, but will get computed when necessary, either
        # in check_monotonicity_and_convexity or plot_decc_dt.
        self.decc_dt_for_checks = None
        # omega22_average and the associated time array. omega22_average is
        # used to convert a given fref to a tref. If fref is not specified,
        # these will remain as None. However, if get_omega22_average() is
        # called, these get set in that function.
        self.t_for_omega22_average = None
        self.omega22_average = None

        if "hlm_zeroecc" in self.dataDict:
            self.compute_res_amp_and_omega22()

    def get_recognized_dataDict_keys(self):
        """Get the list of recognized keys in dataDict."""
        list_of_keys = [
            "t",                # time array of waveform modes
            "hlm",              # Dict of eccentric waveform modes
            "t_zeroecc",        # time array of quasicircular waveform
            "hlm_zeroecc",      # Dict of quasicircular waveform modes
        ]
        return list_of_keys

    def truncate_dataDict_if_necessary(self,
                                       dataDict,
                                       num_orbits_to_exclude_before_merger,
                                       extra_kwargs):
        """Truncate dataDict if "num_orbits_to_exclude_before_merger" is not None.

        parameters:
        ----------
        dataDict:
            Dictionary containing modes and times.
        num_orbits_to_exclude_before_merger: Number of orbits to exclude before
            merger to get the truncated dataDict.
        extra_kwargs:
            Extra kwargs passed to the measure eccentricity.

        returns:
        --------
        dataDict:
            Truncated if num_orbits_to_exclude_before_merger is not None
            else the unchanged dataDict.
        t_merger:
            Merger time evaluated as the time of the global maximum of
            amplitude_using_all_modes. This is computed before the truncation.
        amp22_merger:
            Amplitude of the (2, 2) mode at t_merger. This is computed before
            the truncation.
        min_width_for_extrema:
            Minimum width for find_peaks function. This is computed before the
            truncation.
        """
        t = dataDict["t"]
        phase22 = - np.unwrap(np.angle(dataDict["hlm"][(2, 2)]))
        # We need to know the merger time of eccentric waveform.
        # This is useful, for example, to subtract the quasi circular
        # amplitude from eccentric amplitude in residual amplitude method
        # We also compute amp22 and phase22 at the merger which are needed
        # to compute location at certain number orbits earlier than merger
        # and to rescale amp22 by it's value at the merger (in AmplitudeFits)
        # respectively.
        t_merger = peak_time_via_quadratic_fit(
            t,
            amplitude_using_all_modes(dataDict["hlm"]))[0]
        merger_idx = np.argmin(np.abs(t - t_merger))
        amp22_merger = np.abs(dataDict["hlm"][(2, 2)])[merger_idx]
        phase22_merger = phase22[merger_idx]
        # Minimum width for peak finding function
        min_width_for_extrema = self.get_width_for_peak_finder_from_phase22(
            t, phase22, phase22_merger)
        if num_orbits_to_exclude_before_merger is not None:
            # Truncate the last num_orbits_to_exclude_before_merger number of
            # orbits before merger.
            # This helps in avoiding non-physical features in the omega22
            # interpolants through the pericenters and the apocenters due
            # to the data being too close to the merger.
            if num_orbits_to_exclude_before_merger < 0:
                raise ValueError(
                    "num_orbits_to_exclude_before_merger must be non-negative."
                    " Given value was {num_orbits}")
            index_num_orbits_earlier_than_merger \
                = self.get_index_at_num_orbits_earlier_than_merger(
                    phase22, phase22_merger,
                    num_orbits_to_exclude_before_merger)
            dataDict = copy.deepcopy(dataDict)
            for mode in dataDict["hlm"]:
                dataDict["hlm"][mode] \
                    = dataDict["hlm"][mode][
                        :index_num_orbits_earlier_than_merger]
            dataDict["t"] \
                = dataDict["t"][:index_num_orbits_earlier_than_merger]
        return dataDict, t_merger, amp22_merger, min_width_for_extrema

    def get_width_for_peak_finder_from_phase22(self,
                                               t,
                                               phase22,
                                               phase22_merger,
                                               num_orbits_before_merger=2):
        """Get the minimal value of `width` parameter for extrema finding.

        The extrema finding method, i.e., find_peaks from scipy.signal uses the
        `width` parameter to filter the array of peak locations using the
        condition that each peak in the filtered array has a width >= the value
        of `width`. Finally, the filtered array of peak locations is returned.

        The widths of the peaks in the initial array of peak locations are
        computed internally using scipy.signal.peak_widths. By default, the
        width is calculated at `rel_height=0.5` which gives the so-called Full
        Width at Half Maximum (FWHM). `rel_height` is provided as a percentage
        of the `prominence`. For details see the documentation of
        scipy.signal.peak_widths.

        If the `width` is too small then some noisy features in
        the signal might be mistaken for extrema and on the other hand if the
        `width` is too large then we might miss an extremum.

        This function uses phase22 (phase of the (2, 2) mode) to get a
        reasonable value of `width` by looking at the time scale over which the
        phase22 changes by about 4pi because the change in phase22 over one
        orbit would be approximately twice the change in the orbital phase
        which is about 2pi.  Finally, we divide this by 4 so that the `width`
        is always smaller than the separation between the two troughs
        surrounding the current peak. Otherwise, we risk missing a
        few extrema very close to the merger.

        Parameters:
        -----------
        t:
            Time array.
        phase22:
            Phase of the (2, 2) mode.
        phase22_merger:
            Phase of the (2, 2) mode at the merger.
        num_orbits_before_merger:
            Number of orbits before merger to get the time at which the `width`
            parameter is determined. We want to do this near the merger as this
            is where the time between extrema is the smallest, and the `width`
            parameter sets the minimal width of a peak in the signal.
            Default is 2.

        Returns:
        -------
        width:
            Minimal `width` to filter out noisy extrema.
        """
        # get the time for getting width at num orbits before merger.
        # for 22 mode phase changes about 2 * 2pi for each orbit.
        t_at_num_orbits_before_merger = t[
            self.get_index_at_num_orbits_earlier_than_merger(
                phase22, phase22_merger, num_orbits_before_merger)]
        t_at_num_minus_one_orbits_before_merger = t[
            self.get_index_at_num_orbits_earlier_than_merger(
                phase22, phase22_merger, num_orbits_before_merger-1)]
        # change in time over which phase22 change by 4 pi
        # between num_orbits_before_merger and num_orbits_before_merger - 1
        dt = (t_at_num_minus_one_orbits_before_merger
              - t_at_num_orbits_before_merger)
        # get the width using dt and the time step
        width = dt / (t[1] - t[0])
        # we want to use a width to select a peak/trough such that it is always
        # smaller than the separation between the troughs/peaks surrounding the
        # given peak/trough, otherwise we might miss a few extrema near merger
        return int(width / 4)

    def get_index_at_num_orbits_earlier_than_merger(self,
                                                    phase22,
                                                    phase22_merger,
                                                    num_orbits):
        """Get the index of time num orbits earlier than merger.

        parameters:
        -----------
        phase22:
            1d array of phase of (2, 2) mode of the full waveform.
        phase22_merger:
            Phase of (2, 2) mode at the merger.
        num_orbits:
            Number of orbits earlier than merger to use for computing
            the index of time.
        """
        # one orbit changes the 22 mode phase by 4 pi since
        # omega22 = 2 omega_orb
        phase22_num_orbits_earlier_than_merger = (phase22_merger
                                                  - 4 * np.pi
                                                  * num_orbits)
        # check if the waveform is longer than num_orbits
        if phase22_num_orbits_earlier_than_merger < phase22[0]:
            raise Exception(f"Trying to find index at {num_orbits}"
                            " orbits earlier than the merger but the waveform"
                            f" has less than {num_orbits} orbits of data.")
        return np.argmin(np.abs(
            phase22 - phase22_num_orbits_earlier_than_merger))

    def get_default_extrema_finding_kwargs(self, width):
        """Defaults for extrema_finding_kwargs."""
        default_extrema_finding_kwargs = {
            "height": None,
            "threshold": None,
            "distance": None,
            "prominence": None,
            "width": width,
            "wlen": None,
            "rel_height": 0.5,
            "plateau_size": None}
        return default_extrema_finding_kwargs

    def get_default_extra_kwargs(self):
        """Defaults for additional kwargs."""
        default_extra_kwargs = {
            "spline_kwargs": {},
            "extrema_finding_kwargs": {},   # Gets overridden in methods like
                                            # eccDefinitionUsingAmplitude
            "debug_level": 0,
            "debug_plots": False,
            "omega22_averaging_method": "orbit_averaged_omega22",
            "treat_mid_points_between_pericenters_as_apocenters": False,
            "refine_extrema": False,
            "kwargs_for_fits_methods": {},  # Gets overriden in fits methods
        }
        return default_extra_kwargs

    def find_extrema(self, extrema_type="pericenters"):
        """Find the extrema in the data.

        parameters:
        -----------
        extrema_type:
            Either "pericenters" or "apocenters".

        returns:
        ------
        array of positions of extrema.
        """
        raise NotImplementedError("Please override me.")

    def drop_extra_extrema_at_ends(self, pericenters, apocenters):
        """Drop extra extrema at the start and the end of the data.

        Drop extrema at the end: If there are more than one
        apocenters/pericenters after the last pericenters/pericenters then drop
        the extra apocenters/pericenters.

        Drop extrema at the start of the data. If there are more than one
        apocenters/pericenters before the first pericenters/apocenters then
        drop the extra apocenters/pericenters.
        """
        # At the end of the data
        pericenters_at_end = pericenters[pericenters > apocenters[-1]]
        if len(pericenters_at_end) > 1:
            debug_message(
                f"Found {len(pericenters_at_end) - 1} extra pericenters at the"
                " end. The extra pericenters are not used when building the "
                "spline.", self.debug_level, important=False)
            pericenters = pericenters[pericenters <= pericenters_at_end[0]]
        apocenters_at_end = apocenters[apocenters > pericenters[-1]]
        if len(apocenters_at_end) > 1:
            debug_message(
                f"Found {len(apocenters_at_end) - 1} extra apocenters at the "
                "end. The extra apocenters are not used when building the "
                "spline.", self.debug_level, important=False)
            apocenters = apocenters[apocenters <= apocenters_at_end[0]]

        # At the start of the data
        pericenters_at_start = pericenters[pericenters < apocenters[0]]
        if len(pericenters_at_start) > 1:
            debug_message(
                f"Found {len(pericenters_at_start) - 1} extra pericenters at "
                "the start. The extra pericenters are not used when building "
                "the spline.", self.debug_level, important=False)
            pericenters = pericenters[pericenters >= pericenters_at_start[-1]]
        apocenters_at_start = apocenters[apocenters < pericenters[0]]
        if len(apocenters_at_start) > 1:
            debug_message(
                f"Found {len(apocenters_at_start) - 1} extra apocenters at the"
                " start. The extra apocenters are not used when building the "
                "spline.", self.debug_level, important=False)
            apocenters = apocenters[apocenters >= apocenters_at_start[-1]]

        return pericenters, apocenters

    def drop_extrema_if_extrema_jumps(self, extrema_location,
                                      max_r_delta_phase22_extrema=1.5,
                                      extrema_type="extrema"):
        """Drop the extrema if jump in extrema is detected.

        It might happen that an extremum between two successive extrema is
        missed by the extrema finder, This would result in the two extrema
        being too far from each other and therefore a jump in extrema will be
        introduced.

        To detect if an extremum has been missed we do the following:
        - Compute the phase22 difference between i-th and (i+1)-th extrema:
          delta_phase22_extrema[i] = phase22_extrema[i+1] - phase22_extrema[i]
        - Compute the ratio of delta_phase22: r_delta_phase22_extrema[i] =
          delta_phase22_extrema[i+1]/delta_phase22_extrema[i]
        For correctly separated extrema, the ratio r_delta_phase22_extrema
        should be close to 1.

        Therefore if anywhere r_delta_phase22_extrema[i] >
        max_r_delta_phase22_extrema, where max_r_delta_phase22_extrema = 1.5 by
        default, then delta_phase22_extrema[i+1] is too large and implies that
        phase22 difference between (i+2)-th and (i+1)-th extrema is too large
        and therefore an extrema is missing between (i+1)-th and (i+2)-th
        extrema. We therefore keep extrema only upto (i+1)-th extremum.

        It might also be that an extremum is missed at the start of the
        data. In such case, the phase22 difference would drop from large value
        due to missing extremum to normal value. Therefore, in this case, if
        anywhere r_delta_phase22_extrema[i] < 1 / max_r_delta_phase22_extrema
        then delta_phase22_extrema[i] is too large compared to
        delta_phase22_extrema[i+1] and therefore an extremum is missed between
        i-th and (i+1)-th extrema. Therefore, we keep only extrema starting
        from (i+1)-th extremum.
        """
        # Look for extrema jumps at the end of the data.
        phase22_extrema = self.phase22[extrema_location]
        delta_phase22_extrema = np.diff(phase22_extrema)
        r_delta_phase22_extrema = (delta_phase22_extrema[1:] /
                                   delta_phase22_extrema[:-1])
        idx_too_large_ratio = np.where(r_delta_phase22_extrema >
                                       max_r_delta_phase22_extrema)[0]
        mid_index = int(len(r_delta_phase22_extrema)/2)
        # Check if ratio is too large near the end of the data. Check also
        # that this occurs within the second half of the extrema locations
        if len(idx_too_large_ratio) > 0 and (idx_too_large_ratio[0]
                                             > mid_index):
            first_idx = idx_too_large_ratio[0]
            first_pair_indices = [extrema_location[first_idx+1],
                                  extrema_location[first_idx+2]]
            first_pair_times = [self.t[first_pair_indices[0]],
                                self.t[first_pair_indices[1]]]
            phase_diff_current = delta_phase22_extrema[first_idx+1]
            phase_diff_previous = delta_phase22_extrema[first_idx]
            debug_message(
                f"At least a pair of {extrema_type} are too widely separated"
                " from each other near the end of the data.\n"
                f"This implies that a {extrema_type[:-1]} might be missing.\n"
                f"First pair of such {extrema_type} are {first_pair_indices}"
                f" at t={first_pair_times}.\n"
                f"phase22 difference between this pair of {extrema_type}="
                f"{phase_diff_current/(4*np.pi):.2f}*4pi\n"
                "phase22 difference between the previous pair of "
                f"{extrema_type}={phase_diff_previous/(4*np.pi):.2f}*4pi\n"
                f"{extrema_type} after idx={first_pair_indices[0]}, i.e.,"
                f"t > {first_pair_times[0]} are therefore dropped.",
                self.debug_level, important=False)
            extrema_location = extrema_location[extrema_location <=
                                                extrema_location[first_idx+1]]
        # Check if ratio is too small
        idx_too_small_ratio = np.where(r_delta_phase22_extrema <
                                       (1 / max_r_delta_phase22_extrema))[0]
        # We want to detect extrema jump near the start of the data.
        # Check that the location where such jump is found falls within the
        # first half of the extrema locations.
        if len(idx_too_small_ratio) > 0 and (idx_too_small_ratio[-1]
                                             < mid_index):
            last_idx = idx_too_small_ratio[-1]
            last_pair_indices = [extrema_location[last_idx+1],
                                 extrema_location[last_idx+2]]
            last_pair_times = [self.t[last_pair_indices[0]],
                               self.t[last_pair_indices[1]]]
            phase_diff_current = delta_phase22_extrema[last_idx+1]
            phase_diff_previous = delta_phase22_extrema[last_idx]
            debug_message(
                f"At least a pair of {extrema_type} are too widely separated"
                " from each other near the start of the data.\n"
                f"This implies that a {extrema_type[:-1]} might be missing.\n"
                f"Last pair of such {extrema_type} are {last_pair_indices} at "
                f"t={last_pair_times}.\n"
                f"phase22 difference between this pair of {extrema_type}="
                f"{phase_diff_previous/(4*np.pi):.2f}*4pi\n"
                f"phase22 difference between the next pair of {extrema_type}="
                f"{phase_diff_current/(4*np.pi):.2f}*4pi\n"
                f"{extrema_type} before {last_pair_indices[1]}, i.e., t < t="
                f"{last_pair_times[-1]} are therefore dropped.",
                self.debug_level, important=False)
            extrema_location = extrema_location[extrema_location >=
                                                extrema_location[last_idx]]
        return extrema_location

    def drop_extrema_if_too_close(self, extrema_location,
                                  min_phase22_difference=4*np.pi,
                                  extrema_type="extrema"):
        """Check if a pair of extrema is too close to each other.

        If a pair of extrema is found to be too close to each other, then drop
        the extrema as necessary. If it happens at the start of the data, then
        drop the extrema before such a pair and if it happens at the end of the
        data then drop extrema after such a pair.

        Example: Assuming the real extrema have a separation of 2 indices
        extrema = [1, 3, 5, 7, 9, 11, 12, 14]
        Here the pair {11, 12} is too close. So we drop {12, 14}
        extrema = [1, 3, 4, 6, 8, 10, 12]
        Here the pair {3, 4} is too close. So we drop {1, 3}.

        For an example with EOB waveform, see here
        https://github.com/vijayvarma392/gw_eccentricity/wiki/debug-examples#drop-too-close-extrema
        """
        phase22_extrema = self.phase22[extrema_location]
        phase22_diff_extrema = np.diff(phase22_extrema)
        idx_too_close = np.where(phase22_diff_extrema
                                 < min_phase22_difference)[0]
        mid_index = int(len(phase22_diff_extrema)/2)
        if len(idx_too_close) > 0:
            # Look for too close pairs in the second half
            if idx_too_close[0] > mid_index:
                first_index = idx_too_close[0]
                first_pair = [extrema_location[first_index],
                              extrema_location[first_index+1]]
                first_pair_times = self.t[first_pair]
                debug_message(
                    f"At least a pair of {extrema_type} are too close to "
                    "each other with phase22 difference = "
                    f"{phase22_diff_extrema[first_index]/(4*np.pi):.2f}*4pi.\n"
                    " First pair of such extrema is located in the second half"
                    f" of the {extrema_type} locations between {first_pair},"
                    f"i.e., t={first_pair_times}.\n"
                    f"{extrema_type} after {extrema_location[first_index]}"
                    f" i.e., t > {self.t[extrema_location[first_index]]} "
                    "are dropped.",
                    self.debug_level, important=False)
                extrema_location = extrema_location[
                    extrema_location <= extrema_location[first_index]]
            # Look for too close pairs in the first half
            if idx_too_close[-1] < mid_index:
                last_index = idx_too_close[-1]
                last_pair = [extrema_location[last_index],
                             extrema_location[last_index-1]]
                last_pair_times = self.t[last_pair]
                debug_message(
                    f"At least a pair of {extrema_type} are too close to "
                    "each other with phase22 difference = "
                    f"{phase22_diff_extrema[last_index]/(4*np.pi):.2f}*4pi.\n"
                    " Last pair of such extrema is located in the first half"
                    f" of the {extrema_type} locations between {last_pair},"
                    f"i.e., t={last_pair_times}.\n"
                    f" {extrema_type} before {extrema_location[last_index]}"
                    f" i.e., t < {self.t[extrema_location[last_index]]} "
                    "are dropped.",
                    self.debug_level, important=False)
                extrema_location = extrema_location[
                    extrema_location >= extrema_location[last_index]]
        return extrema_location

    def get_good_extrema(self, pericenters, apocenters,
                         max_r_delta_phase22_extrema=1.5):
        """Retain only the good extrema if there are extra extrema or missing extrema.

        If the number of pericenters/apocenters, n, after the last
        apoceneters/pericenters is more than one, then the extra (n-1)
        pericenters/apocenters are discarded. Similarly, we discard the extra
        pericenters/apocenters before the first apocenters/pericenters.

        We also discard extrema before and after a jump (due to an extremum
        being missed) in the detected extrema.

        To retain only the good extrema, we first remove the extrema
        before/after jumps and then remove any extra extrema at the ends. This
        order is important because if we remove the extrema at the ends first
        and then remove the extrema after/before jumps, then we may still end
        up with extra extrema at the ends, and have to do the first step again.
        Example of doing it in the wrong order, assuming the real extrema have
        a separation of 2 indices:
        Let pericenters p = [1, 3, 5, 7, 11]
        and apoceneters a = [2, 4, 6, 8, 10, 12, 14]
        Here, we have an extra apoceneter and a jump in pericenter between 7
        and 11. Removing extra apoceneter gives
        p = [1, 3, 5, 7, 11]
        a = [2, 4, 6, 8, 10, 12]
        Now removing pericenter after jump gives
        p = [1, 3, 5, 7]. We still end up with extra apocenters on the right.
        However, if we do it in the correct order,
        Removing the pericenter after jump gives p = [1, 3, 5, 7]
        Then removing extra apoceneters gives a = [2, 4, 6, 8]
        Now we end up with extrema without jumps and without extra ones.

        parameters:
        -----------
        pericenters:
            1d array of locations of pericenters.
        apocenters:
            1d array of locations of apocenters.
        max_r_delta_phase22_extrema:
            Maximum value for ratio of successive phase22 difference between
            consecutive extrema. If the ratio is greater than
            max_r_delta_phase22 or less than 1/max_r_delta_phase22 then
            an extremum is considered to be missing.
        returns:
        --------
        pericenters:
            1d array of pericenters after dropping pericenters as necessary.
        apocenters:
            1d array of apocenters after dropping apocenters as necessary.
        """
        pericenters = self.drop_extrema_if_extrema_jumps(
            pericenters, max_r_delta_phase22_extrema, "pericenters")
        apocenters = self.drop_extrema_if_extrema_jumps(
            apocenters, max_r_delta_phase22_extrema, "apocenters")
        pericenters = self.drop_extrema_if_too_close(
            pericenters, extrema_type="pericenters")
        apocenters = self.drop_extrema_if_too_close(
            apocenters, extrema_type="apocenters")
        pericenters, apocenters = self.drop_extra_extrema_at_ends(
            pericenters, apocenters)
        return pericenters, apocenters

    def get_interp(self, oldX, oldY, allowExtrapolation=False,
                   interpolator="spline"):
        """Get interpolant.

        A wrapper of utils.get_interpolant with check_kwargs=False.
        This is to make sure that the checking of kwargs is not performed
        everytime the interpolation function is called. Instead, the kwargs
        are checked once in the init and passed to the interpolation
        function without repeating checks.
        """
        return get_interpolant(oldX, oldY, allowExtrapolation, interpolator,
                               spline_kwargs=self.spline_kwargs,
                               check_kwargs=False)

    def interp(self, newX, oldX, oldY, allowExtrapolation=False,
               interpolator="spline"):
        """Get interpolated values.

        A wrapper of utils.interpolate with check_kwargs=False for
        reasons explained in the documentation of get_interp function.
        """
        return interpolate(newX, oldX, oldY, allowExtrapolation, interpolator,
                           spline_kwargs=self.spline_kwargs,
                           check_kwargs=False)

    def interp_extrema(self, extrema_type="pericenters"):
        """Build interpolant through extrema.

        parameters:
        -----------
        extrema_type:
            Either "pericenters" or "apocenters".

        returns:
        ------
        Interpolant through extrema
        """
        if extrema_type == "pericenters":
            extrema = self.pericenters_location
        elif extrema_type == "apocenters":
            extrema = self.apocenters_location
        else:
            raise Exception("extrema_type must be either "
                            "'pericenrers' or 'apocenters'.")
        if len(extrema) >= 2:
            return self.get_interp(self.t[extrema],
                                   self.omega22[extrema])
        else:
            raise Exception(
                f"Sufficient number of {extrema_type} are not found."
                " Can not create an interpolant.")

    def check_num_extrema(self, extrema, extrema_type="extrema"):
        """Check number of extrema."""
        num_extrema = len(extrema)
        if num_extrema < 2:
            recommended_methods = ["ResidualAmplitude", "AmplitudeFits"]
            if self.method not in recommended_methods:
                method_message = ("It's possible that the eccentricity is too "
                                  f"low for the {self.method} method to detect"
                                  f" the {extrema_type}. Try one of "
                                  f"{recommended_methods}.")
            else:
                method_message = ""
            raise Exception(
                f"Number of {extrema_type} found = {num_extrema}.\n"
                f"Can not build frequency interpolant through the {extrema_type}.\n"
                f"{method_message}")

    def check_if_dropped_too_many_extrema(self, original_extrema, new_extrema,
                                          extrema_type="extrema",
                                          threshold_fraction=0.5):
        """Check if too many extrema was dropped.

        This function only has checks with the flag important=False, which
        means that warnings are suppressed when debug_level < 1. To avoid
        unnecessary computations, this function will therefore return without
        executing the body if debug_level < 1.
        If calling this function externally from an instance of eccDefinition,
        you need to change self.debug_level to be >= 1 if you want to
        un-suppress the warnings.

        Parameters:
        -----------
        original_extrema:
            1d array of original extrema locations.
        new_extrema:
            1d array of new extrema location after dropping extrema.
        extrema_type:
            String to describe the extrema. For example,
            "pericenrers"/"apocenters".
        threshold_fraction:
            Fraction of the original extrema.
            When num_dropped_extrema > threshold_fraction * len(original_extrema),
            an warning is raised.
        """
        # This function only has checks with the flag important=False, which
        # means that warnings are suppressed when debug_level < 1.
        # We return without running the rest of the body to avoid unnecessary
        # computations.
        if self.debug_level < 1:
            return

        num_dropped_extrema = len(original_extrema) - len(new_extrema)
        if num_dropped_extrema > (threshold_fraction * len(original_extrema)):
            debug_message(f"More than {threshold_fraction * 100}% of the "
                          f"original {extrema_type} was dropped.",
                          self.debug_level, important=False)

    def measure_ecc(self, tref_in=None, fref_in=None):
        """Measure eccentricity and mean anomaly from a gravitational waveform.

        Eccentricity is measured using the GW frequency omega22(t) =
        dphi22(t)/dt, where phi22(t) is the phase of the (2, 2) waveform
        mode. We currently only allow time-domain, nonprecessing waveforms. We
        evaluate omega22(t) at pericenter times, t_pericenters, and build a
        spline interpolant omega22_pericenters(t) using those data
        points. Similarly, we build omega22_apocenters(t) using omega22(t) at
        the apocenter times, t_apocenters.

        Using omega22_pericenters(t) and omega22_apocenters(t), we first
        compute e_omega22(t), as described in Eq.(4) of arXiv:2302.11257. We
        then use e_omega22(t) to compute the eccentricity egw(t) using Eq.(8)
        of arXiv:2302.11257. Mean anomaly is defined using t_pericenters, as
        described in Eq.(10) of arXiv:2302.11257.

        To find t_pericenters/t_apocenters, one can look for extrema in
        different waveform data, like omega22(t) or Amp22(t), the amplitude of
        the (2, 2) mode. Pericenters correspond to the local maxima, while
        apocenters correspond to the local minima in the data. The method
        option (described below) lets the user pick which waveform data to use
        to find t_pericenters/t_apocenters.

        parameters:
        ----------
        tref_in:
            Input reference time at which to measure eccentricity and mean
            anomaly.  Can be a single float or an array.

        fref_in:
            Input reference GW frequency at which to measure the eccentricity
            and mean anomaly. Can be a single float or an array. Only one of
            tref_in/fref_in should be provided.

            Given an fref_in, we find the corresponding tref_in such that
            omega22_average(tref_in) = 2 * pi * fref_in. Here,
            omega22_average(t) is a monotonically increasing average frequency
            obtained from the instantaneous omega22(t). omega22_average(t)
            defaults to the orbit averaged omega22, but other options are
            available (see omega22_averaging_method below).

            Eccentricity and mean anomaly measurements are returned on a subset
            of tref_in/fref_in, called tref_out/fref_out, which are described
            below.  If dataDict is provided in dimensionless units, tref_in
            should be in units of M and fref_in should be in units of
            cycles/M. If dataDict is provided in MKS units, t_ref should be in
            seconds and fref_in should be in Hz.

        returns:
        --------
        A dictionary containing the following keys
        tref_out:
            tref_out is the output reference time at which eccentricity and
            mean anomaly are measured.
            tref_out is included in the returned dictionary only when tref_in
            is provided.
            Units of tref_out are the same as that of tref_in.

            tref_out is set as
            tref_out = tref_in[tref_in >= tmin & tref_in <= tmax],
            where tmax = min(t_pericenters[-1], t_apocenters[-1]) and
                  tmin = max(t_pericenters[0], t_apocenters[0]),
            As eccentricity measurement relies on the interpolants
            omega22_pericenters(t) and omega22_apocenters(t), the above cutoffs
            ensure that we only compute the eccentricity where both
            omega22_pericenters(t) and omega22_apocenters(t) are within their
            bounds.

        fref_out:
            fref_out is the output reference frequency at which eccentricity
            and mean anomaly are measured.
            fref_out is included in the returned dictionary only when fref_in
            is provided.
            Units of fref_out are the same as that of fref_in.

            fref_out is set as
            fref_out = fref_in[fref_in >= fref_min && fref_in <= fref_max],
            where fref_min/fref_max are minimum/maximum allowed reference
            frequency, with fref_min = omega22_average(tmin_for_fref)/2/pi
            and fref_max = omega22_average(tmax_for_fref)/2/pi.
            tmin_for_fref/tmax_for_fref are close to tmin/tmax, see
            eccDefinition.get_fref_bounds() for details.

        eccentricity:
            Measured eccentricity at tref_out/fref_out. Same type as
            tref_out/fref_out.

        mean_anomaly:
            Measured mean anomaly at tref_out/fref_out. Same type as
            tref_out/fref_out.
        """
        # Get the pericenters and apocenters
        pericenters = self.find_extrema("pericenters")
        original_pericenters = pericenters.copy()
        self.check_num_extrema(pericenters, "pericenters")
        # In some cases it is easier to find the pericenters than finding the
        # apocenters. For such cases, one can only find the pericenters and use
        # the mid points between two consecutive pericenters as the location of
        # the apocenters.
        if self.extra_kwargs[
                "treat_mid_points_between_pericenters_as_apocenters"]:
            apocenters = self.get_apocenters_from_pericenters(pericenters)
        else:
            apocenters = self.find_extrema("apocenters")
        original_apocenters = apocenters.copy()
        self.check_num_extrema(apocenters, "apocenters")
        # Choose good extrema
        self.pericenters_location, self.apocenters_location \
            = self.get_good_extrema(pericenters, apocenters)

        # Check if we dropped too many extrema.
        self.check_if_dropped_too_many_extrema(original_pericenters,
                                               self.pericenters_location,
                                               "pericenters", 0.5)
        self.check_if_dropped_too_many_extrema(original_apocenters,
                                               self.apocenters_location,
                                               "apocenters", 0.5)
        # check that pericenters and apocenters are appearing alternately
        self.check_pericenters_and_apocenters_appear_alternately()
        # check extrema separation
        self.orb_phase_diff_at_pericenters, \
            self.orb_phase_diff_ratio_at_pericenters \
            = self.check_extrema_separation(self.pericenters_location,
                                            "pericenters")
        self.orb_phase_diff_at_apocenters, \
            self.orb_phase_diff_ratio_at_apocenters \
            = self.check_extrema_separation(self.apocenters_location,
                                            "apocenters")

        # Build the interpolants of omega22 at the extrema
        self.omega22_pericenters_interp = self.interp_extrema("pericenters")
        self.omega22_apocenters_interp = self.interp_extrema("apocenters")

        self.t_pericenters = self.t[self.pericenters_location]
        self.t_apocenters = self.t[self.apocenters_location]
        self.tmax = min(self.t_pericenters[-1], self.t_apocenters[-1])
        self.tmin = max(self.t_pericenters[0], self.t_apocenters[0])
        # check that only one of tref_in or fref_in is provided
        if (tref_in is not None) + (fref_in is not None) != 1:
            raise KeyError("Exactly one of tref_in and fref_in"
                           " should be specified.")
        elif tref_in is not None:
            tref_in_ndim = np.ndim(tref_in)
            self.tref_in = np.atleast_1d(tref_in)
        else:
            fref_in_ndim = np.ndim(fref_in)
            tref_in_ndim = fref_in_ndim
            fref_in = np.atleast_1d(fref_in)
            # get the tref_in and fref_out from fref_in
            self.tref_in, self.fref_out \
                = self.compute_tref_in_and_fref_out_from_fref_in(fref_in)
        # We measure eccentricity and mean anomaly from tmin to tmax.
        self.tref_out = self.tref_in[
            np.logical_and(self.tref_in <= self.tmax,
                           self.tref_in >= self.tmin)]
        # set time for checks and diagnostics
        self.t_for_checks = self.dataDict["t"][
            np.logical_and(self.dataDict["t"] >= self.tmin,
                           self.dataDict["t"] <= self.tmax)]

        # Sanity checks
        # check that fref_out and tref_out are of the same length
        if fref_in is not None:
            if len(self.fref_out) != len(self.tref_out):
                raise Exception(
                    "length of fref_out and tref_out do not match."
                    f"fref_out has length {len(self.fref_out)} and "
                    f"tref_out has length {len(self.tref_out)}.")

        # Check if tref_out is reasonable
        if len(self.tref_out) == 0:
            if self.tref_in[-1] > self.tmax:
                raise Exception(
                    f"tref_in {self.tref_in} is later than tmax="
                    f"{self.tmax}, "
                    "which corresponds to min(last pericenter "
                    "time, last apocenter time).")
            if self.tref_in[0] < self.tmin:
                raise Exception(
                    f"tref_in {self.tref_in} is earlier than tmin="
                    f"{self.tmin}, "
                    "which corresponds to max(first pericenter "
                    "time, first apocenter time).")
            raise Exception(
                "tref_out is empty. This can happen if the "
                "waveform has insufficient identifiable "
                "pericenters/apocenters.")
        # check that tref_out is within t_zeroecc_shifted to make sure that
        # the output is not in the extrapolated region.
        if "hlm_zeroecc" in self.dataDict and (self.tref_out[-1]
                                               > self.t_zeroecc_shifted[-1]):
            raise Exception("tref_out is in extrapolated region.\n"
                            f"Last element in tref_out = {self.tref_out[-1]}\n"
                            "Last element in t_zeroecc = "
                            f"{self.t_zeroecc_shifted[-1]}.\nThis might happen"
                            " when 'num_orbits_to_exclude_before_merger' is "
                            "set to None and part of zeroecc waveform is "
                            "shorter than that of the ecc waveform requiring "
                            "extrapolation to compute residual data.")

        # Check if tref_out has a pericenter before and after.
        # This is required to define mean anomaly.
        if self.tref_out[0] < self.t_pericenters[0] \
           or self.tref_out[-1] > self.t_pericenters[-1]:
            raise Exception("Reference time must be within two pericenters.")

        # compute eccentricity at self.tref_out
        self.eccentricity = self.compute_eccentricity(self.tref_out)
        # Compute mean anomaly at tref_out
        self.mean_anomaly = self.compute_mean_anomaly(self.tref_out)

        # check if eccentricity is positive
        if any(self.eccentricity < 0):
            debug_message("Encountered negative eccentricity.",
                          self.debug_level, point_to_verbose_output=True)

        # check if eccentricity is monotonic and convex
        self.check_monotonicity_and_convexity()

        # If tref_in is a scalar, return a scalar
        if tref_in_ndim == 0:
            self.mean_anomaly = self.mean_anomaly[0]
            self.eccentricity = self.eccentricity[0]
            self.tref_out = self.tref_out[0]

        if fref_in is not None and fref_in_ndim == 0:
            self.fref_out = self.fref_out[0]

        if self.debug_plots:
            # make a plot for diagnostics
            fig, axes = self.make_diagnostic_plots()
            self.save_debug_fig(fig, f"gwecc_{self.method}_diagnostics.pdf")
            plt.close(fig)
        return_dict = {"eccentricity": self.eccentricity,
                       "mean_anomaly": self.mean_anomaly}
        if fref_in is not None:
            return_dict.update({"fref_out": self.fref_out})
        else:
            return_dict.update({"tref_out": self.tref_out})
        return return_dict

    def et_from_ew22_0pn(self, ew22):
        """Get temporal eccentricity at Newtonian order.

        Parameters:
        -----------
        ew22:
            eccentricity measured from the 22-mode frequency.

        Returns:
        --------
        et:
            Temporal eccentricity at Newtonian order.
        """
        psi = np.arctan2(1. - ew22*ew22, 2.*ew22)
        et = np.cos(psi/3.) - np.sqrt(3) * np.sin(psi/3.)

        return et

    def compute_eccentricity(self, t):
        """
        Compute eccentricity at time t.

        Compute e_omega22 from the value of omega22_pericenters_interpolant and
        omega22_apocenters_interpolant at t using Eq.(4) in arXiv:2302.11257
        and then use Eq.(8) in arXiv:2302.11257 to compute e_gw from e_omega22.

        Paramerers:
        -----------
        t:
            Time to compute the eccentricity at. Could be scalar or an array.

        Returns:
        --------
        Eccentricity at t.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_time_limits(t)

        omega22_pericenter_at_t = self.omega22_pericenters_interp(t)
        omega22_apocenter_at_t = self.omega22_apocenters_interp(t)
        self.e_omega22 = ((np.sqrt(omega22_pericenter_at_t)
                           - np.sqrt(omega22_apocenter_at_t))
                          / (np.sqrt(omega22_pericenter_at_t)
                             + np.sqrt(omega22_apocenter_at_t)))
        # get the  temporal eccentricity from e_omega22
        return self.et_from_ew22_0pn(self.e_omega22)

    def derivative_of_eccentricity(self, t, n=1):
        """Get time derivative of eccentricity.

        Parameters:
        -----------
        t:
            Times to get the derivative at.
        n: int
            Order of derivative. Should be 1 or 2, since it uses
            cubic spine to get the derivatives.

        Returns:
        --------
            nth order time derivative of eccentricity.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_time_limits(t)

        if self.ecc_for_checks is None:
            self.ecc_for_checks = self.compute_eccentricity(
                self.t_for_checks)

        if self.ecc_interp is None:
            self.ecc_interp = self.get_interp(self.t_for_checks,
                                              self.ecc_for_checks)
        # Get derivative of ecc(t) using spline
        return self.ecc_interp.derivative(n=n)(t)

    def compute_mean_anomaly(self, t):
        """Compute mean anomlay for given t.

        Compute the mean anomaly using Eq.(10) of arXiv:2302.11257.  Mean
        anomaly grows linearly in time from 0 to 2 pi over the range
        [time_at_last_pericenter, time_at_next_pericenter], where
        time_at_last_pericenter is the time at the previous pericenter, and
        time_at_next_pericenter is the time at the next pericenter.

        Mean anomaly goes linearly from [2*pi*n to 2*pi*(n+1)] from the nth to
        (n+1)th pericenter. Therefore, if we have N+1 pericenters, we collect,
        xVals = [tp_0, tp_1, tp_2, ..., tp_N]
        yVals = [0, 2pi, 4pi, ..., 2*pi*N]
        where tp_n is the time at the nth pericenter.
        Finally, we build a linear interpolant for y using these xVals and
        yVals.

        Parameters:
        -----------
        t:
            Time to compute mean anomaly at. Could be scalar or an array.

        Returns:
        --------
        Mean anomaly at t.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_time_limits(t)

        # Get the mean anomaly at the pericenters
        mean_ano_pericenters = np.arange(len(self.t_pericenters)) * 2 * np.pi
        # Using linear interpolation since mean anomaly is a linear function of
        # time.
        mean_ano = np.interp(t, self.t_pericenters, mean_ano_pericenters)
        # Modulo 2pi to make the mean anomaly vary between 0 and 2pi
        return mean_ano % (2 * np.pi)

    def check_time_limits(self, t):
        """Check that time t is within tmin and tmax.

        To avoid any extrapolation, check that the times t are
        always greater than or equal to tmin and always less than tmax.
        """
        t = np.atleast_1d(t)
        if any(t > self.tmax):
            raise Exception(f"Found times later than tmax={self.tmax}, "
                            "which corresponds to min(last pericenter "
                            "time, last apocenter time).")
        if any(t < self.tmin):
            raise Exception(f"Found times earlier than tmin= {self.tmin}, "
                            "which corresponds to max(first pericenter "
                            "time, first apocenter time).")

    def check_extrema_separation(self, extrema_location,
                                 extrema_type="extrema",
                                 max_orb_phase_diff_factor=1.5,
                                 min_orb_phase_diff=np.pi,
                                 always_return=False):
        """Check if two extrema are too close or too far.

        This function only has checks with the flag important=False, which
        means that warnings are suppressed when debug_level < 1. To avoid
        unnecessary computations, this function will therefore return without
        executing the body if debug_level < 1, unless always_return=True.

        If calling this function externally from an instance of eccDefinition,
        you need to change self.debug_level to be >= 1 if you want to
        un-suppress the warnings. always_return=True is not sufficient to
        un-suppress the warnings.

        parameters:
        always_return:
            The return values of this function are used by some plotting
            functions, so if always_return=True, we execute the body and
            return values regardless of debug_level. However, the warnings
            will still be suppressed for debug_level < 1.
            Default is False.
        """

        # This function only has checks with the flag important=False, which
        # means that warnings are suppressed when debug_level < 1.
        # We return without running the rest of the body to avoid unnecessary
        # computations, unless always_return=True.
        if self.debug_level < 1 and always_return is False:
            return None, None

        orb_phase_at_extrema = self.phase22[extrema_location] / 2
        orb_phase_diff = np.diff(orb_phase_at_extrema)
        # This might suggest that the data is noisy, for example, and a
        # spurious pericenter got picked up.
        t_at_extrema = self.t[extrema_location][1:]
        if any(orb_phase_diff < min_orb_phase_diff):
            too_close_idx = np.where(orb_phase_diff < min_orb_phase_diff)[0]
            too_close_times = t_at_extrema[too_close_idx]
            debug_message(f"At least a pair of {extrema_type} are too close."
                          " Minimum orbital phase diff is "
                          f"{min(orb_phase_diff)}. Times of occurrences are"
                          f" {too_close_times}",
                          self.debug_level, important=False)
        if any(np.abs(orb_phase_diff - np.pi)
               < np.abs(orb_phase_diff - 2 * np.pi)):
            debug_message("Phase shift closer to pi than 2 pi detected.",
                          self.debug_level, important=False)
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
            debug_message(f"At least a pair of {extrema_type} are too far."
                          " Maximum orbital phase diff is "
                          f"{max(orb_phase_diff)}. Times of occurrences are"
                          f" {too_far_times}",
                          self.debug_level, important=False)
        return orb_phase_diff, orb_phase_diff_ratio

    def check_monotonicity_and_convexity(self,
                                         check_convexity=False):
        """Check if measured eccentricity is a monotonic function of time.

        parameters:
        check_convexity:
            In addition to monotonicity, it will check for
            convexity as well. Default is False.
        """
        if self.decc_dt_for_checks is None:
            self.decc_dt_for_checks = self.derivative_of_eccentricity(
                self.t_for_checks, n=1)

        # Is ecc(t) a monotonically decreasing function?
        if any(self.decc_dt_for_checks > 0):
            idx = np.where(self.decc_dt_for_checks > 0)[0]
            range = self.get_range_from_indices(idx, self.t_for_checks)
            message = ("egw(t) is nonmonotonic "
                       f"{'at' if len(idx) == 1 else 'in the range'} {range}")
            debug_message(message, self.debug_level,
                          point_to_verbose_output=True)

        # Is ecc(t) a convex function? That is, is the second
        # derivative always negative?
        if check_convexity:
            self.d2ecc_dt_for_checks = self.derivative_of_eccentricity(
                self.t_for_checks, n=2)
            if any(self.d2ecc_dt_for_checks > 0):
                idx = np.where(self.d2ecc_dt_for_checks > 0)[0]
                range = self.get_range_from_indices(idx, self.t_for_checks)
                message = ("Second derivative of egw(t) is positive "
                           f"{'at' if len(idx) == 1 else 'in the range'} "
                           f"{range}")
                debug_message(f"{message} expected to be always negative",
                              self.debug_level,
                              point_to_verbose_output=True)

    def get_range_from_indices(self, indices, times):
        """Get the range of time from indices for gives times array."""
        if len(indices) == 1:
            return times[indices[0]]
        else:
            return [times[indices[0]],
                    times[indices[-1]]]

    def check_pericenters_and_apocenters_appear_alternately(self):
        """Check that pericenters and apocenters appear alternately.

        This function only has checks with the flag important=False, which
        means that warnings are suppressed when debug_level < 1. To avoid
        unnecessary computations, this function will therefore return without
        executing the body if debug_level < 1.
        If calling this function externally from an instance of eccDefinition,
        you need to change self.debug_level to be >= 1 if you want to
        un-suppress the warnings.
        """
        # This function only has checks with the flag important=False, which
        # means that warnings are suppressed when debug_level < 1.
        # We return without running the rest of the body to avoid unnecessary
        # computations.
        if self.debug_level < 1:
            return

        # if pericenters and apocenters appear alternately, then the number
        # of pericenters and apocenters should differ by one or zero.
        if abs(len(self.pericenters_location)
               - len(self.apocenters_location)) >= 2:
            debug_message(
                "Number of pericenters and number of apocenters differ by "
                f"{abs(len(self.pericenters_location) - len(self.apocenters_location))}"
                ". This implies that pericenters and apocenters are not "
                "appearing alternately.",
                self.debug_level, important=False)
        else:
            # If the number of pericenters and apocenters differ by zero or one
            # then we do the following:
            if len(self.pericenters_location) == len(self.apocenters_location):
                # Check the time of the first pericenter and the first
                # apocenter whichever comes first is assigned as arr1 and the
                # other one as arr2
                if self.t[self.pericenters_location][0] < self.t[
                        self.apocenters_location][0]:
                    arr1 = self.pericenters_location
                    arr2 = self.apocenters_location
                else:
                    arr2 = self.pericenters_location
                    arr1 = self.apocenters_location
            else:
                # Check the number of pericenters and apocenters
                # whichever is larger is assigned as arr1 and the other one as
                # arr2
                if len(self.pericenters_location) > len(
                        self.apocenters_location):
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
            # If pericenters and apocenters appear alternately then all the
            # time differences in t_diff should be positive
            if any(t_diff < 0):
                debug_message(
                    "There is at least one instance where "
                    "pericenters and apocenters do not appear alternately.",
                    self.debug_level, important=False)

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
        if self.t_zeroecc_shifted[0] > self.t[0]:
            raise Exception("Length of zeroecc waveform must be >= the length "
                            "of the eccentric waveform. Eccentric waveform "
                            f"starts at {self.t[0]} whereas zeroecc waveform "
                            f"starts at {self.t_zeroecc_shifted[0]}. Try "
                            "starting the zeroecc waveform at lower Momega0.")
        # check if extrapolation happens in the pre-merger region.
        if self.t_zeroecc_shifted[-1] < self.t_merger:
            raise Exception("Trying to extrapolate zeroecc waveform in "
                            "pre-merger region.\n"
                            f"Merger time={self.t_merger}, last available "
                            f"zeroecc time={self.t_zeroecc_shifted[-1]}")
        # In case the post-merger part of the zeroecc waveform is shorter than
        # that of the the ecc waveform, we allow extrapolation so that the
        # residual quantities can be computed. Above, we check that this
        # extrapolation does not happen before t_merger, which is where
        # eccentricity is normally measured.
        self.amp22_zeroecc_interp = self.interp(
            self.t, self.t_zeroecc_shifted, np.abs(self.h22_zeroecc),
            allowExtrapolation=True)
        self.res_amp22 = self.amp22 - self.amp22_zeroecc_interp

        self.phase22_zeroecc = - np.unwrap(np.angle(self.h22_zeroecc))
        self.omega22_zeroecc = time_deriv_4thOrder(
            self.phase22_zeroecc, self.t_zeroecc[1] - self.t_zeroecc[0])
        self.omega22_zeroecc_interp = self.interp(
            self.t, self.t_zeroecc_shifted, self.omega22_zeroecc,
            allowExtrapolation=True)
        self.res_omega22 = (self.omega22 - self.omega22_zeroecc_interp)

    def get_t_average_for_orbit_averaged_omega22(self):
        """Get the times associated with the fref for orbit averaged omega22.

        t_average_pericenters are the times at midpoints between consecutive
        pericenters. We associate time (t[i] + t[i+1]) / 2 with the orbit
        averaged omega22 calculated between ith and (i+1)th pericenter. That
        is, omega22_average((t[i] + t[i+1])/2) = int_t[i]^t[i+1] omega22(t) dt
        / (t[i+1] - t[i]), where t[i] is the time at the ith pericenter.  And
        similarly, we calculate the t_average_apocenters. We combine
        t_average_pericenters and t_average_apocenters, and sort them to obtain
        t_average.

        Returns:
        --------
        t_for_orbit_averaged_omega22:
            Times associated with orbit averaged omega22
        sorted_idx_for_orbit_averaged_omega22:
            Indices used to sort the times associated with orbit averaged
            omega22
        """
        # get the mid points between the pericenters as avg time for
        # pericenters
        self.t_average_pericenters \
            = 0.5 * (self.t[self.pericenters_location][:-1]
                     + self.t[self.pericenters_location][1:])
        # get the mid points between the apocenters as avg time for
        # apocenters
        self.t_average_apocenters \
            = 0.5 * (self.t[self.apocenters_location][:-1]
                     + self.t[self.apocenters_location][1:])
        t_for_orbit_averaged_omega22 = np.append(
            self.t_average_apocenters,
            self.t_average_pericenters)
        sorted_idx_for_orbit_averaged_omega22 = np.argsort(
            t_for_orbit_averaged_omega22)
        t_for_orbit_averaged_omega22 = t_for_orbit_averaged_omega22[
            sorted_idx_for_orbit_averaged_omega22]
        return [t_for_orbit_averaged_omega22,
                sorted_idx_for_orbit_averaged_omega22]

    def get_orbit_averaged_omega22_at_pericenters(self):
        """Get orbital average of omega22 between two consecutive pericenters.

        Orbital average of omega22 between two consecutive pericenters
        i-th and (i+1)-th is given by
        <omega22>_i = (int_t[i]^t[i+1] omega22(t) dt)/(t[i+1] - t[i])
        t[i] is the time at the i-th extrema.
        Integration of omega22(t) from t[i] to t[i+1] is the same
        as taking the difference of phase22(t) between t[i] and t[i+1]
        <omega22>_i = (phase22[t[i+1]] - phase22[t[i]])/(t[i+1] - t[i])
        """
        return (np.diff(self.phase22[self.pericenters_location]) /
                np.diff(self.t[self.pericenters_location]))

    def get_orbit_averaged_omega22_at_apocenters(self):
        """Get orbital average of omega22 between two consecutive apocenters.

        The procedure to get the orbital average of omega22 at apocenters is
        the same as that at pericenters. See documentation of
        `get_orbit_averaged_omega22_at_pericenters` for details.
        """
        return (np.diff(self.phase22[self.apocenters_location]) /
                np.diff(self.t[self.apocenters_location]))

    def compute_orbit_averaged_omega22_at_extrema(self, t):
        """Compute reference frequency by orbital averaging omega22 at extrema.

        We compute the orbital average of omega22 at the pericenters
        and the apocenters following:
        <omega22>_i = (int_t[i]^t[i+1] omega22(t) dt) / (t[i+1] - t[i])
        where t[i] is the time of ith extrema and the suffix `i` stands for the
        i-th orbit between i-th and (i+1)-th extrema
        <omega22>_i is associated with the temporal midpoint between the i-th
        and (i+1)-th extrema,
        <t>_i = (t[i] + t[i+1]) / 2

        We do this averaging for pericenters and apocenters using the functions
        `get_orbit_averaged_omega22_at_pericenters` and
        `get_orbit_averaged_omega22_at_apocenters` and combine the results.
        The combined array is then sorted using the sorting indices from
        `get_t_average_for_orbit_averaged_omega22`.

        Finally we interpolate the data {<t>_i, <omega22>_i} and evaluate the
        interpolant at the input times `t`.
        """
        # get orbit averaged omega22 between consecutive pericenrers
        # and apoceneters
        self.orbit_averaged_omega22_pericenters \
            = self.get_orbit_averaged_omega22_at_pericenters()
        self.orbit_averaged_omega22_apocenters \
            = self.get_orbit_averaged_omega22_at_apocenters()
        # check monotonicity of the omega22 average
        self.check_monotonicity_of_omega22_average(
            self.orbit_averaged_omega22_pericenters,
            "omega22 averaged [pericenter to pericenter]")
        self.check_monotonicity_of_omega22_average(
            self.orbit_averaged_omega22_apocenters,
            "omega22 averaged [apocenter to apocenter]")
        # combine the average omega22 at pericenters and apocenters
        orbit_averaged_omega22 = np.append(
            self.orbit_averaged_omega22_apocenters,
            self.orbit_averaged_omega22_pericenters)

        # get the times associated to the orbit averaged omega22
        if not hasattr(self, "t_for_orbit_averaged_omega22"):
            self.t_for_orbit_averaged_omega22,\
                self.sorted_idx_for_orbit_averaged_omega22\
                = self.get_t_average_for_orbit_averaged_omega22()
        # We now sort omega22_average using
        # `sorted_idx_for_orbit_averaged_omega22`, the same array of indices
        # that was used to obtain the `t_for_orbit_averaged_omega22` in the
        # function `eccDefinition.get_t_average_for_orbit_averaged_omega22`.
        orbit_averaged_omega22 = orbit_averaged_omega22[
            self.sorted_idx_for_orbit_averaged_omega22]
        # check that omega22_average in strictly monotonic
        self.check_monotonicity_of_omega22_average(
            orbit_averaged_omega22,
            "omega22 averaged [apocenter to apocenter] and "
            "[pericenter to pericenter]")
        return self.interp(
            t, self.t_for_orbit_averaged_omega22, orbit_averaged_omega22)

    def check_monotonicity_of_omega22_average(self,
                                              omega22_average,
                                              description="omega22 average"):
        """Check that omega average is monotonically increasing.

        omega22_average:
            1d array of omega22 averages to check for monotonicity.
        description:
            String to describe what the the which omega22 average we are
            looking at.
        """
        idx_non_monotonic = np.where(
            np.diff(omega22_average) <= 0)[0]
        if len(idx_non_monotonic) > 0:
            first_idx = idx_non_monotonic[0]
            change_at_first_idx = (
                omega22_average[first_idx+1]
                - omega22_average[first_idx])
            if self.debug_plots:
                style = "APS"
                use_fancy_plotsettings(style=style)
                nrows = 4
                fig, axes = plt.subplots(
                    nrows=nrows,
                    figsize=(figWidthsTwoColDict[style],
                             nrows * figHeightsDict[style]))
                axes[0].plot(omega22_average, marker=".",
                             c=colorsDict["default"])
                axes[1].plot(np.diff(omega22_average), marker=".",
                             c=colorsDict["default"])
                axes[2].plot(self.t_average_pericenters,
                             self.orbit_averaged_omega22_pericenters,
                             label=labelsDict["pericenters"],
                             c=colorsDict["pericenter"],
                             marker=".")
                axes[2].plot(self.t_average_apocenters,
                             self.orbit_averaged_omega22_apocenters,
                             label=labelsDict["apocenters"],
                             c=colorsDict["apocenter"],
                             marker=".")
                axes[3].plot(self.t, self.omega22, c=colorsDict["default"])
                axes[3].plot(self.t_pericenters,
                             self.omega22[self.pericenters_location],
                             c=colorsDict["pericenter"],
                             label=labelsDict["pericenters"],
                             marker=".")
                axes[3].plot(self.t_apocenters,
                             self.omega22[self.apocenters_location],
                             c=colorsDict["apocenter"],
                             label=labelsDict["apocenters"],
                             marker=".")
                axes[2].legend()
                axes[2].set_ylabel(labelsDict["omega22_average"])
                axes[3].legend()
                axes[3].set_ylim(0,)
                axes[3].set_ylabel(labelsDict["omega22"])
                axes[1].axhline(0, c=colorsDict["vline"])
                axes[0].set_ylabel(labelsDict["omega22_average"])
                axes[1].set_ylabel(
                    fr"$\Delta$ {labelsDict['omega22_average']}")
                axes[0].set_title(
                    self.extra_kwargs["omega22_averaging_method"])
                fig.tight_layout()
                figName = (
                    "./gwecc_"
                    f"{self.method}_{description.replace(' ', '_')}.pdf")
                # fig.savefig(figName)
                self.save_debug_fig(fig, figName)
                plt.close(fig)
                plot_info = f"See the plot saved as {figName}."
            else:
                plot_info = ("For more verbose output use `debug_level=1` and "
                             "for diagnostic plot use `debug_plots=True` in "
                             "extra_kwargs")
            raise Exception(
                f"{description} are non-monotonic.\n"
                f"First non-monotonicity occurs at peak number {first_idx},"
                f" where omega22 drops from {omega22_average[first_idx]} to"
                f" {omega22_average[first_idx+1]}, a decrease by"
                f" {abs(change_at_first_idx)}.\nTotal number of places of"
                f" non-monotonicity is {len(idx_non_monotonic)}.\n"
                f"Last one occurs at peak number {idx_non_monotonic[-1]}.\n"
                f"{plot_info}\n"
                "Possible fixes: \n"
                "   - Increase sampling rate of data\n"
                "   - Add to extra_kwargs the option 'treat_mid_points_between"
                "_pericenters_as_apocenters': True")

    def compute_mean_of_extrema_interpolants(self, t):
        """Find omega22 average by taking mean of the extrema interpolants".

        Take mean of omega22 spline through omega22 pericenters
        and apocenters to get
        omega22_average = 0.5 * (omega22_p(t) + omega22_a(t))
        """
        return 0.5 * (self.omega22_pericenters_interp(t) +
                      self.omega22_apocenters_interp(t))

    def compute_omega22_zeroecc(self, t):
        """Find omega22 from zeroecc data."""
        return self.interp(
            t, self.t_zeroecc_shifted, self.omega22_zeroecc)

    def get_available_omega22_averaging_methods(self):
        """Return available omega22 averaging methods."""
        available_methods = {
            "mean_of_extrema_interpolants": self.compute_mean_of_extrema_interpolants,
            "orbit_averaged_omega22": self.compute_orbit_averaged_omega22_at_extrema,
            "omega22_zeroecc": self.compute_omega22_zeroecc
        }
        return available_methods

    def get_omega22_average(self, method=None):
        """Get times and corresponding values of omega22 average.

        Parameters:
        -----------
        method: str
            omega22 averaging method. Must be one of the following:
            - "mean_of_extrema_interpolants"
            - "orbit_averaged_omega22"
            - "omega22_zeroecc"
            See get_available_omega22_averaging_methods for available averaging
            methods and Sec.IID of arXiv:2302.11257 for more details.
            Default is None which uses the method provided in
            `self.extra_kwargs["omega22_averaging_method"]`

        Returns:
        --------
        t_for_omega22_average:
            Times associated with omega22_average.
        omega22_average:
            omega22 average using given "method".
            These are data interpolated on the times t_for_omega22_average,
            where t_for_omega22_average is a subset of tref_in passed to the
            eccentricity measurement function.

            For the "orbit_averaged_omega22" method, the original
            omega22_average data points <omega22>_i are obtained by averaging
            the omega22 over the ith orbit between ith to i+1-th extrema. The
            associated <t>_i are obtained by taking the times at the midpoints
            between i-th and i+1-the extrema, i.e., <t>_i = (t_i + t_(i+1))/2.

            These original orbit averaged omega22 data points can be accessed
            using the gwecc_object with the following variables

            - orbit_averaged_omega22_apocenters: orbit averaged omega22 between
              apocenters This is available when measuring eccentricity at
              reference frequency.  If it is not available, it can be computed
              using `get_orbit_averaged_omega22_at_apocenters`
            - t_average_apocenters: temporal midpoints between
              apocenters. These are associated with
              `orbit_averaged_omega22_apocenters`
            - orbit_averaged_omega22_pericenters: orbit averaged omega22
              between pericenters This is available when measuring eccentricity
              at reference frequency.  If it is not available, it can be
              computed using `get_orbit_averaged_omega22_at_pericenters`
            - t_average_pericenters: temporal midpoints between
              pericenters. These are associated with
              `orbit_averaged_omega22_pericenters`
        """
        if method is None:
            method = self.extra_kwargs["omega22_averaging_method"]
        if method != "orbit_averaged_omega22":
            # the average frequencies are using interpolants that use omega22
            # values between tmin and tmax, therefore the min and max time for
            # which omega22 average are the same as tmin and tmax, respectively.
            self.tmin_for_fref = self.tmin
            self.tmax_for_fref = self.tmax
        else:
            self.t_for_orbit_averaged_omega22, self.sorted_idx_for_orbit_averaged_omega22 = \
                self.get_t_average_for_orbit_averaged_omega22()
            # for orbit averaged omega22, the associated times are obtained
            # using the temporal midpoints of the extrema, therefore we need to
            # make sure that we use only those times that fall within tmin and
            # tmax.
            self.tmin_for_fref = max(self.tmin,
                                     min(self.t_for_orbit_averaged_omega22))
            self.tmax_for_fref = min(self.tmax,
                                     max(self.t_for_orbit_averaged_omega22))
        t_for_omega22_average = self.t[
            np.logical_and(self.t >= self.tmin_for_fref,
                           self.t <= self.tmax_for_fref)]
        omega22_average = self.available_averaging_methods[
            method](t_for_omega22_average)
        return t_for_omega22_average, omega22_average

    def compute_tref_in_and_fref_out_from_fref_in(self, fref_in):
        """Compute tref_in and fref_out from fref_in.

        Using chosen omega22 average method we get the tref_in and fref_out
        for the given fref_in.

        When the input is frequencies where eccentricity/mean anomaly is to be
        measured, we internally want to map the input frequencies to a tref_in
        and then we proceed to calculate the eccentricity and mean anomaly for
        this tref_in in the same way as we do when the input array was time
        instead of frequencies.

        We first compute omega22_average(t) using the instantaneous omega22(t),
        which can be done in different ways as described below. Then, we keep
        only the allowed frequencies in fref_in by doing
        fref_out = fref_in[fref_in >= fref_min && fref_in < fref_max],
        Where fref_min/fref_max is the minimum/maximum allowed reference
        frequency for the given omega22 averaging method. See get_fref_bounds
        for more details.
        Finally, we find the times where omega22_average(t) = 2*pi*fref_out,
        and set those to tref_in.

        omega22_average(t) could be calculated in the following ways
        - Mean of the omega22 given by the spline through the pericenters and
          the spline through the apocenters, we call this
          "mean_of_extrema_interpolants"
        - Orbital average at the extrema, we call this
          "orbit_averaged_omega22"
        - omega22 of the zero eccentricity waveform, called "omega22_zeroecc"

        Users can provide a method through the "extra_kwargs" option with the
        key "omega22_averaging_method".
        Default is "orbit_averaged_omega22"

        Once we get the reference frequencies, we create a spline to get time
        as a function of these reference frequencies. This should work if the
        reference frequency is monotonic which it should be.
        Finally, we evaluate this spline on the fref_in to get the tref_in.
        """
        method = self.extra_kwargs["omega22_averaging_method"]
        if method in self.available_averaging_methods:
            # The fref_in array could have frequencies that is outside the
            # range of frequencies in omega22 average. Therefore, we want to
            # create a separate array of frequencies fref_out which is created
            # by taking on those frequencies that falls within the omega22
            # average. Then proceed to evaluate the tref_in based on these
            # fref_out
            fref_out = self.get_fref_out(fref_in, method)

            # Now that we have fref_out, we want to know the corresponding
            # tref_in such that omega22_average(tref_in) = fref_out * 2 * pi
            # This is done by first creating an interpolant of time as function
            # of omega22_average.
            # We get omega22_average by evaluating the omega22_average(t)
            # on t, from tmin_for_fref to tmax_for_fref
            self.t_for_omega22_average, self.omega22_average = self.get_omega22_average(method)

            # check that omega22_average is monotonically increasing
            self.check_monotonicity_of_omega22_average(
                self.omega22_average, "Interpolated omega22_average")

            # Get tref_in using interpolation
            tref_in = self.interp(fref_out,
                                  self.omega22_average/(2 * np.pi),
                                  self.t_for_omega22_average)
            # check if tref_in is monotonically increasing
            if any(np.diff(tref_in) <= 0):
                debug_message(f"tref_in from fref_in using method {method} is"
                              " not monotonically increasing.",
                              self.debug_level, important=False)
            return tref_in, fref_out
        else:
            raise KeyError(f"Omega22 averaging method {method} does not exist."
                           " Must be one of "
                           f"{list(self.available_averaging_methods.keys())}")

    def get_fref_bounds(self, method=None):
        """Get the allowed min and max reference frequency of 22 mode.

        Depending on the omega22 averaging method, this function returns the
        minimum and maximum allowed reference frequency of 22 mode.

        Note: If omega22_average is already computed using a `method` and
        therefore is not None, then it returns the minimum and maximum of that
        omega22_average and does not recompute the omega22_average using the
        input `method`. In other words, if omega22_average is already not None
        then input `method` is ignored and the existing omega22_average is
        used.  To force recomputation of omega22_average, for example, with a
        new method one need to set it to None first.

        Parameters:
        -----------
        method:
            Omega22 averaging method.  See
            get_available_omega22_averaging_methods for available methods.
            Default is None which will use the default method for omega22
            averaging using `extra_kwargs["omega22_averaging_method"]`

        Returns:
            Minimum allowed reference frequency, Maximum allowed reference
            frequency.
        --------
        """
        if self.omega22_average is None:
            self.t_for_omega22_average, self.omega22_average = self.get_omega22_average(method)
        return [min(self.omega22_average)/2/np.pi,
                max(self.omega22_average)/2/np.pi]

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
            Slice of fref_in that satisfies:
            fref_in >= fref_min && fref_in < fref_max
        """
        fref_min, fref_max = self.get_fref_bounds(method)
        fref_out = fref_in[
            np.logical_and(fref_in >= fref_min,
                           fref_in < fref_max)]
        if len(fref_out) == 0:
            if fref_in[0] < fref_min:
                raise Exception("fref_in is earlier than minimum available "
                                f"frequency {fref_min}")
            if fref_in[-1] > fref_max:
                raise Exception("fref_in is later than maximum available "
                                f"frequency {fref_max}")
            else:
                raise Exception("fref_out is empty. This can happen if the "
                                "waveform has insufficient identifiable "
                                "pericenters/apocenters.")
        return fref_out

    def make_diagnostic_plots(
            self,
            add_help_text=True,
            usetex=True,
            style=None,
            use_fancy_settings=True,
            twocol=False,
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
        - omega_22 vs time with the pericenters and apocenters shown. This
          would show if the method is missing any pericenters/apocenters or
          selecting one which is not a pericenter/apocenter
        - deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
          change in orbital phase from the previous extrema to the ith extrema.
          This helps to look for missing extrema, as there will be a drastic
          (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
          extrema, and the ratio will go from ~1 to ~2.
        - omega22_average, where the omega22 average computed using the
          `omega22_averaging_method` is plotted as a function of time.
          omega22_average is used to get the reference time for a given
          reference frequency. Therefore, it should be a strictly monotonic
          function of time.

        Additionally, we plot the following if data for zero eccentricity is
        provided and method is not residual method
        - residual amp22 vs time with the location of pericenters and
          apocenters shown.
        - residual omega22 vs time with the location of pericenters and
          apocenters shown.
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
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.  If None, then uses "Notebook"
            when twocol is False and uses "APS" if twocol is True.
            Default is None.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.
        twocol:
            Use a two column grid layout. Default is False.
        **kwargs:
            kwargs to be passed to plt.subplots()

        Returns:
        fig:
            Figure object.
        axarr:
            Axes object.
        """
        # Make a list of plots we want to add
        list_of_plots = [self.plot_eccentricity,
                         self.plot_mean_anomaly,
                         self.plot_omega22,
                         self.plot_data_used_for_finding_extrema,
                         self.plot_decc_dt,
                         self.plot_phase_diff_ratio_between_extrema,
                         self.plot_omega22_average]
        if "hlm_zeroecc" in self.dataDict:
            # add residual amp22 plot
            if self.method != "ResidualAmplitude":
                list_of_plots.append(self.plot_residual_amp22)
            # add residual omega22 plot
            if self.method != "ResidualFrequency":
                list_of_plots.append(self.plot_residual_omega22)

        # Set style if None
        if style is None:
            style = "APS" if twocol else "Notebook"
        # Initiate figure, axis
        nrows = int(np.ceil(len(list_of_plots) / 2)) if twocol else len(
            list_of_plots)
        figsize = (figWidthsTwoColDict[style],
                   figHeightsDict[style] * nrows)
        default_kwargs = {"nrows": nrows,
                          "ncols": 2 if twocol else 1,
                          "figsize": figsize,
                          "sharex": True}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        fig, axarr = plt.subplots(**kwargs)
        axarr = np.reshape(axarr, -1, "C")

        # populate figure, axis
        for idx, plot in enumerate(list_of_plots):
            plot(
                fig,
                axarr[idx],
                add_help_text=add_help_text,
                usetex=usetex,
                use_fancy_settings=False)
            axarr[idx].tick_params(labelbottom=True)
            axarr[idx].set_xlabel("")
        # set xlabel in the last row
        axarr[-1].set_xlabel(labelsDict["t"])
        if twocol:
            axarr[-2].set_xlabel(labelsDict["t"])
        # delete empty subplots
        for idx, ax in enumerate(axarr[len(list_of_plots):]):
            fig.delaxes(ax)
        fig.tight_layout()
        return fig, axarr

    def plot_eccentricity(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            style="Notebook",
            use_fancy_settings=True,
            add_vline_at_tref=True,
            **kwargs):
        """Plot measured ecc as function of time.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.
        add_vline_at_tref:
            If tref_out is scalar and add_vline_at_tref is True then add a
            vertical line to indicate the location of tref_out on the time
            axis.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        if self.ecc_for_checks is None:
            self.ecc_for_checks = self.compute_eccentricity(
                self.t_for_checks)
        ax.plot(self.t_for_checks, self.ecc_for_checks, **kwargs)
        # add a vertical line in case of scalar tref_out/fref_out indicating
        # the corresponding reference time
        if self.tref_out.size == 1 and add_vline_at_tref:
            ax.axvline(self.tref_out, c=colorsDict["pericentersvline"], ls=":",
                       label=labelsDict["t_ref"])
            ax.plot(self.tref_out, self.eccentricity, ls="", marker=".")
            ax.legend(frameon=True, handlelength=1, labelspacing=0.2,
                      columnspacing=1)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["eccentricity"])
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
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot decc_dt as function of time to check monotonicity.

        If decc_dt becomes positive, ecc(t) is not monotonically decreasing.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        if self.decc_dt_for_checks is None:
            self.decc_dt_for_checks = self.derivative_of_eccentricity(
                self.t_for_checks, n=1)
        ax.plot(self.t_for_checks, self.decc_dt_for_checks, **kwargs)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["dedt"])
        if add_help_text:
            ax.text(
                0.05,
                0.05,
                ("We expect decc/dt to be always negative"),
                ha="left",
                va="bottom",
                transform=ax.transAxes)
        # change ticks to scientific notation
        ax.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        # add line to indicate y = 0
        ax.axhline(0, ls="--")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_mean_anomaly(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            style="Notebook",
            use_fancy_settings=True,
            add_vline_at_tref=True,
            **kwargs):
        """Plot measured mean anomaly as function of time.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.
        add_vline_at_tref:
            If tref_out is scalar and add_vline_at_tref is True then add a
            vertical line to indicate the location of tref_out on the time
            axis.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        default_kwargs = {"c": colorsDict["default"]}
        for key in default_kwargs:
            if key not in kwargs:
                kwargs.update({key: default_kwargs[key]})
        ax.plot(self.t_for_checks,
                self.compute_mean_anomaly(self.t_for_checks),
                **kwargs)
        # add a vertical line in case of scalar tref_out/fref_out indicating the
        # corresponding reference time
        if self.tref_out.size == 1 and add_vline_at_tref:
            ax.axvline(self.tref_out, c=colorsDict["pericentersvline"], ls=":",
                       label=labelsDict["t_ref"])
            ax.plot(self.tref_out, self.mean_anomaly, ls="", marker=".")
            ax.legend(frameon=True, handlelength=1, labelspacing=0.2,
                      columnspacing=1)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["mean_anomaly"])
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
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot omega22, the locations of the apocenters and pericenters.

        Also plots their corresponding interpolants.
        This would show if the method is missing any pericenters/apocenters or
        selecting one which is not a pericenter/apocenter.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
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
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t_for_checks,
                self.omega22_pericenters_interp(self.t_for_checks),
                c=colorsDict["pericenter"],
                label=labelsDict["omega22_pericenters"],
                **kwargs)
        ax.plot(self.t_for_checks, self.omega22_apocenters_interp(
            self.t_for_checks),
                c=colorsDict["apocenter"],
                label=labelsDict["omega22_apocenters"],
                **kwargs)
        ax.plot(self.t, self.omega22,
                c=colorsDict["default"], label=labelsDict["omega22"])
        ax.plot(self.t[self.pericenters_location],
                self.omega22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="")
        ax.plot(self.t[self.apocenters_location],
                self.omega22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="")
        # set reasonable ylims
        ymin = min(self.omega22)
        ymax = max(self.omega22)
        pad = 0.05 * ymax  # 5 % buffer for better visibility
        ax.set_ylim(ymin - pad, ymax + pad)
        # add help text
        if add_help_text:
            ax.text(
                0.22,
                0.98,
                (r"\noindent To avoid extrapolation, first and last\\"
                 r"extrema are excluded when\\"
                 r"evaluating $\omega_{a}$/$\omega_{p}$ interpolants"),
                ha="left",
                va="top",
                transform=ax.transAxes)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(labelsDict["omega22"])
        ax.legend(frameon=True,
                  handlelength=1, labelspacing=0.2, columnspacing=1)
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_omega22_average(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            style="Notebook",
            use_fancy_settings=True,
            plot_omega22=True,
            plot_orbit_averaged_omega22_at_extrema=False,
            **kwargs):
        """Plot omega22_average.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.
        plot_omega22: bool
            If True, plot omega22 also. Default is True.
        plot_orbit_averaged_omega22_at_extrema: bool
            If True and method is orbit_averaged_omega22, plot the the orbit
            averaged omega22 at the extrema as well. Default is False.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        # check if omega22_average is already available. If not
        # available, compute it.
        if self.omega22_average is None:
            self.t_for_omega22_average, self.omega22_average = self.get_omega22_average()
        ax.plot(self.t_for_omega22_average,
                self.omega22_average,
                c=colorsDict["default"],
                label="omega22_average",
                **kwargs)
        if plot_omega22:
            ax.plot(self.t, self.omega22,
                    c='k',
                    alpha=0.4,
                    lw=0.5,
                    label=labelsDict["omega22"])
        if (self.extra_kwargs["omega22_averaging_method"] == "orbit_averaged_omega22" and
            plot_orbit_averaged_omega22_at_extrema):
            ax.plot(self.t_average_apocenters,
                    self.orbit_averaged_omega22_apocenters,
                    c=colorsDict["apocenter"],
                    marker=".", ls="",
                    label=labelsDict["orbit_averaged_omega22_apocenters"])
            ax.plot(self.t_average_pericenters,
                    self.orbit_averaged_omega22_pericenters,
                    c=colorsDict["pericenter"],
                    marker=".", ls="",
                    label=labelsDict["orbit_averaged_omega22_pericenters"])
        # set reasonable ylims
        ymin = min(self.omega22)
        ymax = max(self.omega22)
        pad = 0.05 * ymax  # 5 % buffer for better visibility
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Averaged frequency")
        # add help text
        if add_help_text:
            ax.text(
                0.35,
                0.98,
                (r"\noindent omega22_average should be "
                 "monotonically increasing."),
                ha="left",
                va="top",
                transform=ax.transAxes)
        ax.legend(frameon=True,
                  handlelength=1, labelspacing=0.2, columnspacing=1)
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
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot amp22, the locations of the apocenters and pericenters.

        This would show if the method is missing any pericenters/apocenters or
        selecting one which is not a pericenter/apocenter.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
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
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t, self.amp22,
                c=colorsDict["default"], label=labelsDict["amp22"])
        ax.plot(self.t[self.pericenters_location],
                self.amp22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label=labelsDict["pericenters"])
        ax.plot(self.t[self.apocenters_location],
                self.amp22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label=labelsDict["apocenters"])
        # set reasonable ylims
        ymin = min(self.amp22)
        ymax = max(self.amp22)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["amp22"])
        ax.legend(handlelength=1, labelspacing=0.2, columnspacing=1)
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_phase_diff_ratio_between_extrema(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=True,
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot phase diff ratio between consecutive extrema as function of time.

        Plots deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
        change in orbital phase from the previous extrema to the ith extrema.
        This helps to look for missing extrema, as there will be a drastic
        (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
        extrema, and the ratio will go from ~1 to ~2.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
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
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        tpericenters = self.t[self.pericenters_location[1:]]
        if self.orb_phase_diff_ratio_at_pericenters is None:
            self.orb_phase_diff_at_pericenters, \
                self.orb_phase_diff_ratio_at_pericenters \
                = self.check_extrema_separation(self.pericenters_location,
                                                "pericenters",
                                                always_return=True)
            self.orb_phase_diff_at_apocenters, \
                self.orb_phase_diff_ratio_at_apocenters \
                = self.check_extrema_separation(self.apocenters_location,
                                                "apocenters",
                                                always_return=True)
        ax.plot(tpericenters[1:], self.orb_phase_diff_ratio_at_pericenters[1:],
                c=colorsDict["pericenter"],
                marker=".", label="Pericenter phase diff ratio")
        tapocenters = self.t[self.apocenters_location[1:]]
        ax.plot(tapocenters[1:], self.orb_phase_diff_ratio_at_apocenters[1:],
                c=colorsDict["apocenter"],
                marker=".", label="Apocenter phase diff ratio")
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(r"$\Delta \Phi_{orb}[i] / \Delta \Phi_{orb}[i-1]$")
        if add_help_text:
            ax.text(
                0.5,
                0.98,
                ("If phase difference ratio exceeds 1.5,\n"
                 "there may be missing extrema.\n"
                 "If the ratio falls below 0.5,\n"
                 "there may be spurious extrema."),
                ha="center",
                va="top",
                transform=ax.transAxes)
        ax.set_title("Ratio of phase difference between consecutive extrema")
        ax.legend(frameon=True, loc="center left",
                  handlelength=1, labelspacing=0.2, columnspacing=1)
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
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual omega22, the locations of the apocenters and pericenters.

        Useful to look for bad omega22 data near merger.
        We also throw away post merger before since it makes the plot
        unreadble.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.  Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t, self.res_omega22, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_omega22[self.pericenters_location],
                marker=".", ls="", label=labelsDict["pericenters"],
                c=colorsDict["pericenter"])
        ax.plot(self.t[self.apocenters_location],
                self.res_omega22[self.apocenters_location],
                marker=".", ls="", label=labelsDict["apocenters"],
                c=colorsDict["apocenter"])
        # set reasonable ylims
        ymin = min(self.res_omega22)
        ymax = max(self.res_omega22)
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim  # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["res_omega22"])
        ax.legend(frameon=True, loc="center left",
                  handlelength=1, labelspacing=0.2, columnspacing=1)
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
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual amp22, the locations of the apocenters and pericenters.

        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
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
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t, self.res_amp22, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_amp22[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label=labelsDict["pericenters"])
        ax.plot(self.t[self.apocenters_location],
                self.res_amp22[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label=labelsDict["apocenters"])
        # set reasonable ylims
        ymin = min(self.res_amp22)
        ymax = max(self.res_amp22)
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim  # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(labelsDict["res_amp22"])
        ax.legend(frameon=True, loc="center left", handlelength=1,
                  labelspacing=0.2,
                  columnspacing=1)
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
            style="Notebook",
            use_fancy_settings=True,
            add_vline_at_tref=True,
            **kwargs):
        """Plot the data that is being used.

        Also the locations of the apocenters and pericenters.
        Parameters:
        -----------
        fig:
            Figure object to add the plot to. If None, initiates a new figure
            object.  Default is None.
        ax:
            Axis object to add the plot to. If None, initiates a new axis
            object.  Default is None.
        add_help_text:
            If True, add text to describe features in the plot.
            Default is True.
        usetex:
            If True, use TeX to render texts.
            Default is True.
        style:
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
            Default is Notebook.
        use_fancy_settings:
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
            Default is True.
        add_vline_at_tref:
            If tref_out is scalar and add_vline_at_tref is True then add a
            vertical line to indicate the location of tref_out on the time
            axis.

        Returns:
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t, self.data_for_finding_extrema,
                c=colorsDict["default"])
        ax.plot(
            self.t[self.pericenters_location],
            self.data_for_finding_extrema[self.pericenters_location],
            c=colorsDict["pericenter"],
            marker=".", ls="",
            label=labelsDict["pericenters"])
        apocenters, = ax.plot(
            self.t[self.apocenters_location],
            self.data_for_finding_extrema[self.apocenters_location],
            c=colorsDict["apocenter"],
            marker=".", ls="",
            label=labelsDict["apocenters"])
        # set reasonable ylims
        ymin = min(self.data_for_finding_extrema)
        ymax = max(self.data_for_finding_extrema)
        # we want to make the ylims symmetric about y=0 when Residual data is
        # used
        if "Residual" in self.method:
            ylim = max(ymax, -ymin)
            pad = 0.05 * ylim  # 5 % buffer for better visibility
            ax.set_ylim(-ylim - pad, ylim + pad)
        else:
            pad = 0.05 * ymax
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(self.label_for_data_for_finding_extrema)
        # Add vertical line to indicate the latest time used for extrema
        # finding
        ax.axvline(
            self.t[-1],
            c=colorsDict["vline"], ls="--",
            label="Latest time used for finding extrema.")
        # if tref_out/fref_out is scalar then add vertical line to indicate
        # corresponding reference time.
        if self.tref_out.size == 1 and add_vline_at_tref:
            ax.axvline(self.tref_out, c=colorsDict["pericentersvline"], ls=":",
                       label=labelsDict["t_ref"])
        # add legends
        ax.legend(frameon=True, handlelength=1, labelspacing=0.2,
                  columnspacing=1,
                  loc="upper left")
        # set title
        ax.set_title(
            "Data being used for finding the extrema.",
            ha="center")
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def save_debug_fig(self, fig, fname, fig_name=None, format="pdf"):
        """Save debug plots in fig using fname.

        parameters:
        -----------
        fig:
            fig object to use for saving the plots.
        fname:
            A path, or a Python file-like object, or possibly some
            backend-dependent object such as
            `matplotlib.backends.backend_pdf.PdfPages`. See `fname` in
            plt.savefig for more documentation.
        fig_name:
            fig_name to print before saving the plot. If None, fname is
            used as message assuming that fname is a string. If
            message is None and fname is not a string, Exception is
            raised.
            Default is None.
        format:
            Format for saving the plot. Default is pdf.
        """
        if fig_name is None:
            if type(fname) != str:
                raise Exception("Message cannot be None when fname is not a"
                                " string.")
            fig_name = fname
        print(f"Saving debug plot to {fig_name}")
        fig.savefig(fname, format=format)

    def get_apocenters_from_pericenters(self, pericenters):
        """Get apocenters locations from pericenetrs locations.

        This function treats the mid points between two successive pericenters
        as the location of the apocenter in between the same two pericenters.
        Thus it does not find the locations of the apocenters using pericenter
        finder at all. It is useful in situation where finding pericenters is
        easy but finding the apocenters in between is difficult. This is the
        case for highly eccentric systems where eccentricity approaches 1. For
        such systems the amp22/omega22 data between the pericenters is almost
        flat and hard to find the local minima.

        returns:
        ------
        locations of apocenters
        """
        # NOTE: Assuming uniform time steps.
        # TODO: Make it work for non
        # uniform time steps In the following we get the location of mid point
        # between ith pericenter and (i+1)th pericenter as (loc[i] +
        # loc[i+1])/2 where loc is the array that contains the pericenter
        # locations. This works because time steps are assumed to be uniform
        # and hence proportional to the time itself.
        apocenters = (pericenters[:-1]
                      + pericenters[1:]) / 2
        apocenters = apocenters.astype(int)  # convert to ints
        return apocenters

    def get_width_for_peak_finder_for_dimless_units(
            self,
            width_for_unit_timestep=50):
        """Get the minimal value of `width` parameter for extrema finding.

        See the documentation under
        eccDefinition.get_width_for_peak_finder_from_phase22
        for why this is useful to set when calling scipy.signal.find_peaks.

        This function gets an appropriate width by scaling it with the time
        steps in the time array of the waveform data.  NOTE: As the function
        name mentions, this should be used only for dimensionless units. This
        is because the `width_for_unit_timestep` parameter refers to unit
        timestep in units of M. It is the fiducial width to use if the time
        step is 1M. If using time in seconds, this would depend on the total
        mass.

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
