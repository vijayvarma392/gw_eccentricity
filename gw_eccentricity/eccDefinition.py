"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Classes for eccentricity definitions should be derived from `eccDefinition`
(or one of it's derived classes. See the following wiki
https://github.com/vijayvarma392/gw_eccentricity/wiki/Adding-new-eccentricity-definitions)
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import peak_time_via_quadratic_fit, check_kwargs_and_set_defaults
from .utils import amplitude_using_all_modes
from .utils import time_deriv_4thOrder
from .utils import interpolate
from .utils import get_interpolant
from .utils import get_default_spline_kwargs
from .utils import get_rational_fit
from .utils import get_default_rational_fit_kwargs
from .utils import debug_message
from .plot_settings import use_fancy_plotsettings, colorsDict, labelsDict
from .plot_settings import figWidthsTwoColDict, figHeightsDict


class eccDefinition:
    """Base class to define eccentricity for given waveform data dictionary."""

    def __init__(self, dataDict, num_orbits_to_exclude_before_merger=2,
                 precessing=False,
                 extra_kwargs=None):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict: dict
            Dictionary containing waveform modes dict, time etc. Should follow
            the format::

            dataDict = {"t": time,
                        "hlm": modeDict,
                        "amplm": ampDict,
                        "phaselm": phaseDict,
                        "omegalm": omegaDict,
                        "t_zeroecc": time,
                        "hlm_zeroecc": modeDict,
                        "amplm_zeroecc": ampDict,
                        "phaselm_zeroecc": phaseDict,
                        "omegalm_zeroecc": omegaDict,
                       }

            "t" and one of the following is mandatory:

            1. "hlm" OR
            2. "amplm" and "phaselm"
                but not both 1 and 2 at the same time.

            Apart from specifying "hlm" or "amplm" and "phaselm", the user can
            also provide "omegalm". If the "omegalm" key is not explicitly
            provided, it is computed from the given "hlm" or "phaselm" using
            finite difference method.

            The keys with suffix "zeroecc" are only required for
            `ResidualAmplitude` and `ResidualFrequency` methods, where
            "t_zeroecc" and one of the following is to be provided:

            1. "hlm_zeroecc" OR
            2. "amplm_zeroecc" and "phaselm_zeroecc"
               but not both 1 and 2 at the same time.

            Similar to "omegalm", the user can also provide "omegalm_zeroecc".
            If it is not provided in `dataDict`, it is computed from the given
            "hlm_zeroecc" or "phaselm_zeroecc" using finite difference method.

            If zeroecc data are provided for methods other than
            `ResidualAmplitude` and `ResidualFrequency`, they are used for
            additional diagnostic plots, which can be helpful for all
            methods.

            Any keys in `dataDict` other than the recognized ones will be
            ignored, with a warning.

            The recognized keys are:

            - "t": 1d array of times.

                - Should be uniformly sampled, with a small enough time step so
                  that frequency of gravitational wave from it's phase can be
                  accurately computed, if necessary. We use a 4th-order finite
                  difference scheme. In dimensionless units, we recommend a
                  time step of dtM = 0.1M to be conservative, but one may be
                  able to get away with larger time steps like dtM = 1M. The
                  corresponding time step in seconds would be dtM * M *
                  lal.MTSUN_SI, where M is the total mass in Solar masses.  -
                  We do not require the waveform peak amplitude to occur at any
                  specific time, but tref_in should follow the same convention
                  for peak time as "t".

            - "hlm": Dictionary of waveform modes associated with "t".
                Should have the format::

                    modeDict = {(l1, m1): h_{l1, m1},
                                (l2, m2): h_{l2, m2},
                                ...
                               },
                    where h_{l, m} is a 1d complex array representing the (l,
                    m) waveform mode. Should contain at least the (2, 2) mode,
                    but more modes can be included, as indicated by the
                    ellipsis '...'  above.

            - "amplm": Dictionary of amplitudes of waveform modes associated
              with "t". Should have the same format as "hlm", except that the
              amplitude is real.

            - "phaselm": Dictionary of phases of waveform modes associated
              with "t". Should have the same format as "hlm", except that the
              phase is real. The phaselm is related to hlm as hlm = amplm *
              exp(- i phaselm) ensuring that the phaselm is monotonically
              increasing for m > 0 modes.

            - "omegalm": Dictionary of the frequencies of the waveform modes
              associated with "t". Should have the same format as "hlm", except
              that the omegalm is real. omegalm is related to the phaselm
              (see above) as omegalm = d/dt phaselm, which means that the
              omegalm is positive for m > 0 modes.

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

            - "amplm_zeroecc", "phaselm_zeroecc" and "omegalm_zeroecc":
                Same as "amplm", "phaselm" and "omegalm", respectively, but
                for the quasicircular counterpart to the eccentric waveform.

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

        precessing: bool, default=False
            Whether the system is precessing or not. For precessing systems,
            the `dataDict` should contain modes in the coprecessing frame. For
            nonprecessing systems, there is no distiction between the inertial
            and coprecessing frame since they are the same.

            Default is False which implies the system to be nonprecessing.

        extra_kwargs: dict
            A dictionary of any extra kwargs to be passed. Allowed kwargs
            are:
            spline_kwargs: dict
                Dictionary of arguments to be passed to the spline
                interpolation routine
                (scipy.interpolate.InterpolatedUnivariateSpline) used to
                compute quantities like omega_gw_pericenters(t) and
                omega_gw_apocenters(t).
                Defaults are set using utils.get_default_spline_kwargs

            rational_fit_kwargs: dict
                Dictionary of arguments to be passed to the rational
                fit function. Defaults are set using
                `utils.get_default_rational_fit_kwargs`

            extrema_finding_kwargs: dict
                Dictionary of arguments to be passed to the extrema finder,
                scipy.signal.find_peaks.
                The Defaults are the same as those of scipy.signal.find_peaks,
                except for the "width", which sets the minimum allowed "full
                width at half maximum" for the extrema. Setting this can help
                avoid false extrema in noisy data (for example, due to junk
                radiation in NR). The default for "width" is set using
                phase_gw(t) near the merger. For nonprecessing systems,
                phase_gw = phase of the (2, 2) mode.  Starting from 4 cycles of
                the (2, 2) mode before the merger, we find the number of time
                steps taken to cover 2 cycles, let's call this "the gap". Note
                that 2 cycles of the (2, 2) mode are approximately one orbit,
                so this allows us to approximate the smallest gap between two
                pericenters/apocenters. However, to be conservative, we divide
                this gap by 4 and set it as the width parameter for
                find_peaks. See
                eccDefinition.get_width_for_peak_finder_from_phase_gw for more
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

            omega_gw_averaging_method:
                Options for obtaining omega_gw_average(t) from the
                instantaneous omega_gw(t). For nonprecessing systems,
                omega_gw(t) is the same as the omega(t) of the (2, 2) mode. See
                `get_amp_phase_omega_gw` for more details.

                - "orbit_averaged_omega_gw": First, orbit averages are obtained
                  at each pericenter by averaging omega_gw(t) over the time from
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
                  omega_gw_pericenters(t) and omega_gw_apocenters(t) is used as a
                  proxy for the average frequency.
                - "omega_gw_zeroecc": omega_gw(t) of the quasicircular
                  counterpart is used as a proxy for the average
                  frequency. This can only be used if "t_zeroecc" and
                  "hlm_zeroecc" are provided in dataDict.
                See Sec. IID of arXiv:2302.11257 for more detail description of
                average omega_gw.
                Default is "orbit_averaged_omega_gw".

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

            return_zero_if_small_ecc_failure : bool, default=False
                The code normally raises an exception if sufficient number of
                extrema are not found. This can happen for various reasons
                including when the eccentricity is too small for some methods
                (like the Amplitude method) to measure. See e.g. Fig.4 of
                arxiv.2302.11257. If no extrema are found, we check whether the
                following two conditions are satisfied.

                1. `return_zero_if_small_ecc_failure` is set to `True`.
                2. The waveform is at least
                  (5 + `num_obrits_to_exclude_before_merger`) orbits long. By
                  default, `num_obrits_to_exclude_before_merger` is set to 2,
                  meaning that 2 orbits are removed from the waveform before it
                  is used by the extrema finding routine. Consequently, in the
                  default configuration, the original waveform in the input
                  `dataDict` must have a minimum length of 7 orbits.

                If both of these conditions are met, we assume that small
                eccentricity is the cause, and set the returned eccentricity
                and mean anomaly to zero.
                USE THIS WITH CAUTION!

            omega_gw_extrema_interpolation_method : str, default="spline"
                Specifies the method used to build the interpolations for 
                `omega_gw_pericenters_interp(t)` or `omega_gw_apocenters_interp(t)`.
                The available options are:

                - `spline`: Uses `scipy.interpolate.InterpolatedUnivariateSpline`.
                - `rational_fit`: Uses `polyrat.StabilizedSKRationalApproximation`.

                ### When to Use:
                
                - **`spline`** (default):
                - Best suited for cleaner data, such as when waveform modes are generated 
                    using models like SEOB or TEOB.
                - Faster to construct and evaluate.
                - Since it fits through every data point, it may exhibit oscillatory 
                    behavior, particularly near the merger.
                
                - **`rational_fit`**:
                - More appropriate for noisy data, e.g., waveform modes from numerical 
                    simulations.
                - Minimizes least squares error, resulting in a smoother overall trend 
                    with less oscillation.
                - Significantly slower compared to the `spline` method.
                - Can suppress pathologies in the waveform that might be visible with 
                    `spline`.

                ### Fallback Behavior:
                
                With `omega_gw_extrema_interpolation_method` set to `spline`, if 
                `use_rational_fit_as_fallback` is set to `True`, the method will 
                initially use `spline`. If the first derivative of the spline interpolant 
                exhibits non-monotonicity, the code will automatically fall back to the 
                `rational_fit` method. This ensures a more reliable fit when the spline 
                method shows undesirable behavior.

                Default value: `"spline"`.
                    
            use_rational_fit_as_fallback : bool, default=True
                Use rational fit for interpolant of omega at extrema when the
                interpolant built using spline shows nonmonotonicity in its
                first derivative.

                This is used only when `omega_gw_extrema_interpolation_method` is `spline`.
                If `omega_gw_extrema_interpolation_method` is `rational_fit` then it has
                no use.
        """
        self.precessing = precessing
        # Get data necessary for eccentricity measurement
        self.dataDict, self.t_merger, self.amp_gw_merger, \
            min_width_for_extrema = self.process_data_dict(
                dataDict, num_orbits_to_exclude_before_merger, extra_kwargs)
        self.t = self.dataDict["t"]
        # check if the time steps are equal, the derivative function
        # requires uniform time steps
        self.t_diff = np.diff(self.t)
        if not np.allclose(self.t_diff, self.t_diff[0]):
            raise Exception("Input time array must have uniform time steps.\n"
                            f"Time steps are {self.t_diff}")
        # get amplitude, phase, omega to be used for eccentricity measurement.
        # For precessing systems, even in the coprecessing frame, the (2, 2)
        # mode quantities show some oscillations due to the precession
        # effect. To reduce these oscillations further, we use (2, 2) and (2, -2)
        # mode in the coprecessing frame to define new quantities amp_gw,
        # phase_gw and omega_gw using Eq.(48) and (49) of arXiv:1701.00550. For
        # nonprecessing systems, these quantities reduce to their respective
        # (2, 2) mode values. See `get_amp_phase_omega_gw` for more details.
        self.amp_gw, self.phase_gw, self.omega_gw \
            = self.get_amp_phase_omega_gw()
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
        self.rational_fit_kwargs = check_kwargs_and_set_defaults(
            self.extra_kwargs["rational_fit_kwargs"],
            get_default_rational_fit_kwargs(),
            "rational_fit_kwargs",
            "utils.get_default_rational_fit_kwargs()")
        self.available_averaging_methods \
            = self.get_available_omega_gw_averaging_methods()
        self.debug_level = self.extra_kwargs["debug_level"]
        self.rational_fit_kwargs["verbose"] = True if self.debug_level >=1 else False
        self.debug_plots = self.extra_kwargs["debug_plots"]
        self.return_zero_if_small_ecc_failure = self.extra_kwargs["return_zero_if_small_ecc_failure"]
        self.use_rational_fit_as_fallback = self.extra_kwargs["use_rational_fit_as_fallback"]
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
        # omega_gw_average and the associated time array. omega_gw_average is
        # used to convert a given fref to a tref. If fref is not specified,
        # these will remain as None. However, if get_omega_gw_average() is
        # called, these get set in that function.
        self.t_for_omega_gw_average = None
        self.omega_gw_average = None
        # compute residual data
        if "amplm_zeroecc" in self.dataDict and "omegalm_zeroecc" in self.dataDict:
            self.compute_res_amp_gw_and_res_omega_gw()

    def get_recognized_dataDict_keys(self):
        """Get the list of recognized keys in dataDict."""
        list_of_keys = [
            "t",                # time array of waveform modes
            "hlm",              # Dict of eccentric waveform modes
            "amplm",            # Dict of amplitude of eccentric waveform modes
            "phaselm",          # Dict of phase of eccentric waveform modes
            "omegalm",          # Dict of omega of eccentric waveform modes
            "t_zeroecc",        # time array of quasicircular waveform
            "hlm_zeroecc",      # Dict of quasicircular waveform modes
            "amplm_zeroecc",    # Dict of amplitude of quasicircular waveform modes
            "phaselm_zeroecc",  # Dict of phase of quasicircular waveform modes
            "omegalm_zeroecc",  # Dict of omega of quasicircular waveform modes
        ]
        return list_of_keys

    def get_amp_phase_omega_data(self, dataDict):
        """
        Extract the dictionary of amplitude, omega, and phase from `dataDict`.

        The `dataDict` provided by the user can contain any of the data
        corresponding to the keys in `get_recognized_dataDict_keys`. To compute
        eccentricity and mean anomaly, we need amplitude, phase, and
        omega. This function extracts these data from `dataDict` and creates a
        new dictionary containing only the amplitude, phase, and omega of the
        available modes.

        For example, if `dataDict` contains only the complex hlm modes, this
        function uses the hlm modes to create a new dictionary containing the
        amplitude, phase, and omega by decomposing the hlm modes.  Afterwards,
        only these waveform data in the new dataDict will be used for all
        further computations.

        Parameters
        ----------
        dataDict : dict
            A dictionary containing waveform data.

        Returns
        -------
        amp_phase_omega_dict : dict
            A new dictionary containing only the amplitude, phase, and omega
            dict of available modes.
        """
        amp_phase_omega_dict = {}
        # add amplm and amplm_zeroecc if zeroecc data provided
        amp_phase_omega_dict.update(self.get_amplm_from_dataDict(dataDict))
        # add phaselm and phaselm_zeroecc if zeroecc data provided
        amp_phase_omega_dict.update(self.get_phaselm_from_dataDict(dataDict))
        # add omegalm
        if "omegalm" in dataDict:
            amp_phase_omega_dict.update({"omegalm": dataDict["omegalm"]})
        else:
            # compute it from phaselm that is already in amp_phase_omega_dict
            amp_phase_omega_dict.update(
                {"omegalm": self.get_omegalm_from_phaselm(
                    dataDict["t"],
                    amp_phase_omega_dict["phaselm"])})

        # add omegalm_zeroecc if zeroecc data is provided
        if "omegalm_zeroecc" in dataDict:
            amp_phase_omega_dict.update(
                {"omegalm_zeroecc": dataDict["omegalm_zeroecc"]})
        # Look for zeroecc phaselm is amp_phase_omega_dict and compute
        # omegalm_zeroecc from it if phaselm_zeroecc is present
        elif "phaselm_zeroecc" in amp_phase_omega_dict \
             and "t_zeroecc" in dataDict:
            amp_phase_omega_dict.update(
                {"omegalm_zeroecc": self.get_omegalm_from_phaselm(
                    dataDict["t_zeroecc"],
                    amp_phase_omega_dict["phaselm_zeroecc"])})
        return amp_phase_omega_dict

    def get_amplm_from_dataDict(self, dataDict):
        """Get amplm dict from dataDict.

        Returns the dictionary of amplitudes of waveform modes.
        """
        amplm = {}
        amplm_zeroecc = {}
        for suffix, ampDict in zip(["", "_zeroecc"], [amplm, amplm_zeroecc]):
            if "amplm" + suffix in dataDict:
                # Add the amplitude dictionary from dataDict to ampDict
                ampDict.update(dataDict["amplm" + suffix])
            elif "hlm" + suffix in dataDict:
                # compute amplitude of each hlm mode and add to ampDict
                for k in dataDict["hlm" + suffix]:
                    ampDict.update({k: np.abs(dataDict["hlm" + suffix][k])})
            elif suffix == "":
                raise Exception("`dataDict` should contain either 'amplm' or "
                                " 'hlm' for computing amplitude of the "
                                "waveform modes.")
        amplmDict = {"amplm": amplm}
        if amplm_zeroecc:
            amplmDict.update({"amplm_zeroecc": amplm_zeroecc})
        return amplmDict

    def get_phaselm_from_dataDict(self, dataDict):
        """Get phaselm dict from dataDict.

        When the dataDict contains only the complex hlm modes, the phaselm is
        obtained using the relation hlm = amplm * exp(-i phaselm).
        """
        phaselm = {}
        phaselm_zeroecc = {}
        for suffix, phaseDict in zip(["", "_zeroecc"],
                                     [phaselm, phaselm_zeroecc]):
            if "phaselm" + suffix in dataDict:
                # Add the phase dictionary from dataDict to phaseDict
                phaseDict.update(dataDict["phaselm" + suffix])
            elif "hlm" + suffix in dataDict:
                # Compute phase of each mode in hlm and add to phaselm
                for k in dataDict["hlm" + suffix]:
                    phaseDict.update(
                        {k: - np.unwrap(
                            np.angle(dataDict["hlm" + suffix][k]))})
            elif suffix == "":
                raise Exception("`dataDict` should contain either 'phaselm' "
                                "or 'hlm' for computing phase of the waveform "
                                "modes.")
        phaselmDict = {"phaselm": phaselm}
        if phaselm_zeroecc:
            phaselmDict.update({"phaselm_zeroecc": phaselm_zeroecc})
        return phaselmDict

    def get_omegalm_from_phaselm(self, t, phaselm):
        """Get omegalm dict from phaselm dict.

        omegalm is computed using the relation omegalm = d phaselm/dt.
        """
        omegalm = phaselm.copy()
        for mode in phaselm:
            omegalm[mode] = time_deriv_4thOrder(phaselm[mode], t[1] - t[0])
        return omegalm

    def process_data_dict(self,
                          dataDict,
                          num_orbits_to_exclude_before_merger,
                          extra_kwargs):
        """Get necessary data for eccentricity measurement from `dataDict`.

        To measure the eccentricity, several waveform data are required,
        including:

        - amplitude: Used to locate the merger time from its global maxima and
          may also be utilized to find the pericenters/apocenters depending on
          the method.
        - phase: Helps estimate the time at a given number of orbits before the
          merger.
        - omega: Required to compute the frequency at the
          pericenters/apocenters, which are essential for building the
          interpolant through them. It may also be used to find the
          pericenters/apocenters depending on the method.

        These waveform data are also used for various checks and diagnostic
        plots.

        The `dataDict` provided by the user may contain any of the data
        mentioned in `get_recognized_dataDict_keys`. From `dataDict`, the
        amplitude, phase, and omega data are obtained. If
        `num_orbits_to_exclude_before_merger` is not None, then these data
        (corresponding to the eccentric waveform) are truncated before
        returning.

        In addition to the above data, this function returns a few more
        variables -- `t_merger`, `amp_gw_merger`, and `min_width_for_extrema`
        for future usage. See the details below.

        Parameters
        ----------
        dataDict : dict
            Dictionary containing modes and times.
        num_orbits_to_exclude_before_merger : int or None
            Number of orbits to exclude before the merger to get the truncated
            dataDict. If None, no truncation is performed.
        extra_kwargs:
            Extra keyword arguments passed to the measure eccentricity.

        Returns
        -------
        newDataDict : dict
            Dictionary containing the amplitude, phase, and omega of the
            eccentric waveform modes. Includes the same of the zeroecc waveform
            modes when present in the input `dataDict`. If
            `num_orbits_to_exclude_before_merger` is not None, then the
            eccentric data, i.e., amplitude, phase, and omega of the available
            modes, are truncated before newDataDict is returned.
        t_merger : float
            Merger time evaluated as the time of the global maximum of
            `amplitude_using_all_modes`. This is computed before the
            truncation.
        amp_gw_merger : float
            Value of amp_gw at the merger. For nonprecessing systems, amp_gw is
            the amplitude of the (2, 2) mode. For precessing systems, amp_gw is
            obtained using a symmetric combination of the amplitude of (2, 2)
            and (2, -2) mode in the coprecessing frame. See `get_amp_phase_omega_gw`
            for more details.
            This needs to be computed before the modes are truncated.
            # TODO: Maybe we should use the ampitude from all modes and use
            # it's max value.
        min_width_for_extrema : float
            Minimum width for the `find_peaks` function. This is computed
            before the truncation.
        """
        # Test that exactly one of either hlm or (amplm and phaselm) is
        # provided.
        # Raise exception if hlm is provided and amplm and/or phaselm is also
        # provided.
        for suffix in ["", "_zeroecc"]:
            if ("hlm"+suffix in dataDict) and any(
                    ["amplm"+suffix in dataDict,
                     "phaselm"+suffix in dataDict]):
                raise Exception(
                    f"`dataDict` {'should' if suffix == '' else 'may'} "
                    "contain one of the following: \n"
                    f"1. 'hlm{suffix}' OR \n"
                    f"2. 'amplm{suffix}' and 'phaselm{suffix}'\n"
                    "But not both 1. and 2. at the same time.")
        # Raise exception if hlm is not provided but either amplm and/or
        # phaselm is also not provided
        if ("hlm" not in dataDict) and any(["amplm" not in dataDict,
                                            "phaselm" not in dataDict]):
            raise Exception((
                "`dataDict` should contain one of the following: \n"
                "1. 'hlm' OR \n"
                "2. 'amplm' and 'phaselm'\n"
                "But not both 1. and 2. at the same time."))
        # Create a new dictionary that will contain the data necessary for
        # eccentricity measurement.
        newDataDict = {}
        if "t" in dataDict:
            newDataDict.update({"t": dataDict["t"]})
        else:
            raise Exception("`dataDict` must contain 't', the times associated"
                            " with the eccentric waveform data.")
        # add t_zeroecc if present
        if "t_zeroecc" in dataDict:
            newDataDict.update({"t_zeroecc": dataDict["t_zeroecc"]})
        # From the user dataDict get the amplitude, phase, omega data and add
        # to newDataDict
        newDataDict.update(self.get_amp_phase_omega_data(dataDict))

        # Now we compute data using newDataDict and then truncate it if
        # required.
        # We need to know the merger time of eccentric waveform.
        # This is useful, for example, to subtract the quasi circular
        # amplitude from eccentric amplitude in residual amplitude method
        # We also compute amp_gw and phase_gw at the merger which are needed
        # to compute location at certain number orbits earlier than merger
        # and to rescale amp_gw by it's value at the merger (in AmplitudeFits)
        # respectively.
        t_merger = peak_time_via_quadratic_fit(
            newDataDict["t"],
            amplitude_using_all_modes(newDataDict["amplm"], "amplm"))[0]
        merger_idx = np.argmin(np.abs(newDataDict["t"] - t_merger))
        if not self.precessing:
            amp_gw_merger = newDataDict["amplm"][(2, 2)][merger_idx]
            phase_gw = newDataDict["phaselm"][(2, 2)]
            phase_gw_merger = phase_gw[merger_idx]
            # TODO: we may need to change this in the future.
            # For example, omega_gw could be the invariant angular velocity even
            # for nonprecessing case.
            omega_gw_merger = newDataDict["omegalm"][(2, 2)][merger_idx]
        else:
            raise NotImplementedError("Precessing system is not supported yet.")
        # check if phase_gw is increasing. phase_gw at merger should be greater
        # than the phase_gw at the start of the waveform
        if phase_gw_merger < phase_gw[0]:
            raise Exception(
                f"phase_gw = {phase_gw_merger} at the merger is < "
                f"phase_gw = {phase_gw[0]} at the start. The "
                "phaselm should be related to hlm as "
                "hlm = amplm * exp(- i phaselm) ensuring that "
                "the phaselm is monotonically increasing for m > 0 modes."
                "This might be fixed by changing the overall sign of "
                "phase in the input `dataDict`")
        # check that omega_gw is positive by checking its value at the merger
        if omega_gw_merger < 0:
            raise Exception(f"omega_gw at merger is {omega_gw_merger} < 0. "
                            "omega_gw must be positive.")
        # Minimum width for peak finding function
        min_width_for_extrema = self.get_width_for_peak_finder_from_phase_gw(
            newDataDict["t"],
            phase_gw,
            phase_gw_merger)
        if num_orbits_to_exclude_before_merger is not None:
            # Truncate the last num_orbits_to_exclude_before_merger number of
            # orbits before merger.
            # This helps in avoiding non-physical features in the omega_gw
            # interpolants through the pericenters and the apocenters due
            # to the data being too close to the merger.
            if num_orbits_to_exclude_before_merger < 0:
                raise ValueError(
                    "num_orbits_to_exclude_before_merger must be non-negative."
                    " Given value was {num_orbits}")
            index_num_orbits_earlier_than_merger \
                = self.get_index_at_num_orbits_earlier_than_merger(
                    phase_gw,
                    phase_gw_merger,
                    num_orbits_to_exclude_before_merger)
            # Truncate amp, phase, omega in eccentric waveform data.
            for k in ["amplm", "phaselm", "omegalm"]:
                for mode in newDataDict[k]:
                    newDataDict[k][mode] = newDataDict[k][mode][
                        :index_num_orbits_earlier_than_merger]
                    newDataDict["t"] = newDataDict["t"][
                        :index_num_orbits_earlier_than_merger]
        return newDataDict, t_merger, amp_gw_merger, min_width_for_extrema

    def get_amp_phase_omega_gw(self):
        """Get the gw quanitities from modes dict in the coprecessing frame.

        For nonprecessing systems, the amp_gw, phase_gw and omega_gw are the same
        as those obtained using the (2, 2) mode, i. e., amp22, phase22 and omega22,
        respectively.
        
        For precessing systems, the amplitude and omega of the (2, 2) are not
        the best quantities to use for eccentricity measurement even in the
        coprecessing frame.  Ideally we want to use quantities that have the
        least imprint of precession in the coprecessing frame. The symmetric
        amplitude and antisymmetric phase defined in Eq.(48) and (49) of
        arXiv:1701.00550 which are the symmetric and antisymmetric combination
        of the same from (2, 2) and (2, -2) modes are used for eccentricity
        measurement.
        
        amp_gw = (1/2) * (amp(2, 2) + amp(2, -2))
        phase_gw = (1/2) * (phase(2, 2) - phase(2, -2))
        omega_gw = d(phase_gw)/dt

        These quantities reduce to the corresponding (2, 2) mode data when the
        system is nonprecessing.
        """
        if not self.precessing:
            amp_gw, phase_gw, omega_gw = (self.dataDict["amplm"][(2, 2)],
                                          self.dataDict["phaselm"][(2, 2)],
                                          self.dataDict["omegalm"][(2, 2)])
        else:
            amp_gw = 0.5 * (self.dataDict[(2, 2)] + self.dataDict[(2, -2)])
            phase_gw = 0.5 * (np.unwrap(np.angle(self.dataDict[(2, 2)]))
                              - np.unwrap(np.angle(self.dataDict[(2, -2)])))
            omega_gw = time_deriv_4thOrder(
                phase_gw,
                self.dataDict["t"][1] - self.dataDict["t"][0])
        return amp_gw, phase_gw, omega_gw

    def get_width_for_peak_finder_from_phase_gw(self,
                                               t,
                                               phase_gw,
                                               phase_gw_merger,
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

        This function uses phase_gw (phase of the (2, 2) mode for nonprecessing
        case or phase from an antisymmetric combination of (2, 2) and (2, -2)
        mode phases in the coprecessing frame for precessing case, see
        `get_amp_phase_omega_gw` for more details) to get a reasonable value of
        `width` by looking at the time scale over which the phase_gw changes by
        about 4pi because the change in phase_gw over one orbit would be
        approximately twice the change in the orbital phase which is about 2pi.
        Finally, we divide this by 4 so that the `width` is always smaller than
        the separation between the two troughs surrounding the current
        peak. Otherwise, we risk missing a few extrema very close to the
        merger.

        Parameters:
        -----------
        t: array-like
            Time array.
        phase_gw: array-like
            phase of the (2, 2) mode for nonprecessing systems. For precessing
            systems, phase_gw is obtained using an antisymmetric combination of
            (2, 2) and (2, -2) mode phases in the coprecessing frame. see
            `get_amp_phase_omega_gw` for more details.
        phase_gw_merger: float
            Value of phase_gw at the merger.
        num_orbits_before_merger: float, default=2
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
        # phase_gw changes about 2 * 2pi for each orbit.
        t_at_num_orbits_before_merger = t[
            self.get_index_at_num_orbits_earlier_than_merger(
                phase_gw, phase_gw_merger, num_orbits_before_merger)]
        t_at_num_minus_one_orbits_before_merger = t[
            self.get_index_at_num_orbits_earlier_than_merger(
                phase_gw, phase_gw_merger, num_orbits_before_merger-1)]
        # change in time over which phase_gw change by 4 pi
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
                                                    phase_gw,
                                                    phase_gw_merger,
                                                    num_orbits):
        """Get the index of time num orbits earlier than merger.

        parameters:
        -----------
        phase_gw: array-like
            phase of the (2, 2) mode for nonprecessing systems. For precessing
            systems, phase_gw is obtained using an antisymmetric combination of
            (2, 2) and (2, -2) mode phases in the coprecessing frame. see
            `get_amp_phase_omega_gw` for more details.
        phase_gw_merger: float
            Value of phase_gw at the merger.
        num_orbits: float
            Number of orbits earlier than merger to use for computing
            the index of time.
        """
        # one orbit changes the phase_gw by 4 pi since
        # omega_gw = 2 * omega_orb
        phase_gw_num_orbits_earlier_than_merger = (phase_gw_merger
                                                  - 4 * np.pi
                                                  * num_orbits)
        # check if the waveform is longer than num_orbits
        if phase_gw_num_orbits_earlier_than_merger < phase_gw[0]:
            raise Exception(f"Trying to find index at {num_orbits}"
                            " orbits earlier than the merger but the waveform"
                            f" has less than {num_orbits} orbits of data.")
        return np.argmin(np.abs(
            phase_gw - phase_gw_num_orbits_earlier_than_merger))

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
            "rational_fit_kwargs": {},
            "extrema_finding_kwargs": {},   # Gets overridden in methods like
                                            # eccDefinitionUsingAmplitude
            "debug_level": 0,
            "debug_plots": False,
            "omega_gw_averaging_method": "orbit_averaged_omega_gw",
            "treat_mid_points_between_pericenters_as_apocenters": False,
            "refine_extrema": False,
            "kwargs_for_fits_methods": {},  # Gets overriden in fits methods
            "return_zero_if_small_ecc_failure": False,
            "use_rational_fit_as_fallback": True,
            "omega_gw_extrema_interpolation_method": "spline"
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
                                      max_r_delta_phase_gw_extrema=1.5,
                                      extrema_type="extrema"):
        """Drop the extrema if jump in extrema is detected.

        It might happen that an extremum between two successive extrema is
        missed by the extrema finder, This would result in the two extrema
        being too far from each other and therefore a jump in extrema will be
        introduced.

        To detect if an extremum has been missed we do the following:

        - Compute the phase_gw difference between i-th and (i+1)-th extrema:
          delta_phase_gw_extrema[i] = phase_gw_extrema[i+1] - phase_gw_extrema[i]
        - Compute the ratio of delta_phase_gw: r_delta_phase_gw_extrema[i] =
          delta_phase_gw_extrema[i+1]/delta_phase_gw_extrema[i]

        For correctly separated extrema, the ratio r_delta_phase_gw_extrema
        should be close to 1.

        Therefore if anywhere r_delta_phase_gw_extrema[i] >
        max_r_delta_phase_gw_extrema, where max_r_delta_phase_gw_extrema = 1.5 by
        default, then delta_phase_gw_extrema[i+1] is too large and implies that
        phase_gw difference between (i+2)-th and (i+1)-th extrema is too large
        and therefore an extrema is missing between (i+1)-th and (i+2)-th
        extrema. We therefore keep extrema only upto (i+1)-th extremum.

        It might also be that an extremum is missed at the start of the
        data. In such case, the phase_gw difference would drop from large value
        due to missing extremum to normal value. Therefore, in this case, if
        anywhere r_delta_phase_gw_extrema[i] < 1 / max_r_delta_phase_gw_extrema
        then delta_phase_gw_extrema[i] is too large compared to
        delta_phase_gw_extrema[i+1] and therefore an extremum is missed between
        i-th and (i+1)-th extrema. Therefore, we keep only extrema starting
        from (i+1)-th extremum.
        """
        # Look for extrema jumps at the end of the data.
        phase_gw_extrema = self.phase_gw[extrema_location]
        delta_phase_gw_extrema = np.diff(phase_gw_extrema)
        r_delta_phase_gw_extrema = (delta_phase_gw_extrema[1:] /
                                   delta_phase_gw_extrema[:-1])
        idx_too_large_ratio = np.where(r_delta_phase_gw_extrema >
                                       max_r_delta_phase_gw_extrema)[0]
        mid_index = int(len(r_delta_phase_gw_extrema)/2)
        # Check if ratio is too large near the end of the data. Check also
        # that this occurs within the second half of the extrema locations
        if len(idx_too_large_ratio) > 0 and (idx_too_large_ratio[0]
                                             > mid_index):
            first_idx = idx_too_large_ratio[0]
            first_pair_indices = [extrema_location[first_idx+1],
                                  extrema_location[first_idx+2]]
            first_pair_times = [self.t[first_pair_indices[0]],
                                self.t[first_pair_indices[1]]]
            phase_diff_current = delta_phase_gw_extrema[first_idx+1]
            phase_diff_previous = delta_phase_gw_extrema[first_idx]
            debug_message(
                f"At least a pair of {extrema_type} are too widely separated"
                " from each other near the end of the data.\n"
                f"This implies that a {extrema_type[:-1]} might be missing.\n"
                f"First pair of such {extrema_type} are {first_pair_indices}"
                f" at t={first_pair_times}.\n"
                f"phase_gw difference between this pair of {extrema_type}="
                f"{phase_diff_current/(4*np.pi):.2f}*4pi\n"
                "phase_gw difference between the previous pair of "
                f"{extrema_type}={phase_diff_previous/(4*np.pi):.2f}*4pi\n"
                f"{extrema_type} after idx={first_pair_indices[0]}, i.e.,"
                f"t > {first_pair_times[0]} are therefore dropped.",
                self.debug_level, important=False)
            extrema_location = extrema_location[extrema_location <=
                                                extrema_location[first_idx+1]]
        # Check if ratio is too small
        idx_too_small_ratio = np.where(r_delta_phase_gw_extrema <
                                       (1 / max_r_delta_phase_gw_extrema))[0]
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
            phase_diff_current = delta_phase_gw_extrema[last_idx+1]
            phase_diff_previous = delta_phase_gw_extrema[last_idx]
            debug_message(
                f"At least a pair of {extrema_type} are too widely separated"
                " from each other near the start of the data.\n"
                f"This implies that a {extrema_type[:-1]} might be missing.\n"
                f"Last pair of such {extrema_type} are {last_pair_indices} at "
                f"t={last_pair_times}.\n"
                f"phase_gw difference between this pair of {extrema_type}="
                f"{phase_diff_previous/(4*np.pi):.2f}*4pi\n"
                f"phase_gw difference between the next pair of {extrema_type}="
                f"{phase_diff_current/(4*np.pi):.2f}*4pi\n"
                f"{extrema_type} before {last_pair_indices[1]}, i.e., t < t="
                f"{last_pair_times[-1]} are therefore dropped.",
                self.debug_level, important=False)
            extrema_location = extrema_location[extrema_location >=
                                                extrema_location[last_idx]]
        return extrema_location

    def drop_extrema_if_too_close(self, extrema_location,
                                  min_phase_gw_difference=4*np.pi,
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
        phase_gw_extrema = self.phase_gw[extrema_location]
        phase_gw_diff_extrema = np.diff(phase_gw_extrema)
        idx_too_close = np.where(phase_gw_diff_extrema
                                 < min_phase_gw_difference)[0]
        mid_index = int(len(phase_gw_diff_extrema)/2)
        if len(idx_too_close) > 0:
            # Look for too close pairs in the second half
            if idx_too_close[0] > mid_index:
                first_index = idx_too_close[0]
                first_pair = [extrema_location[first_index],
                              extrema_location[first_index+1]]
                first_pair_times = self.t[first_pair]
                debug_message(
                    f"At least a pair of {extrema_type} are too close to "
                    "each other with phase_gw difference = "
                    f"{phase_gw_diff_extrema[first_index]/(4*np.pi):.2f}*4pi.\n"
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
                    "each other with phase_gw difference = "
                    f"{phase_gw_diff_extrema[last_index]/(4*np.pi):.2f}*4pi.\n"
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
                         max_r_delta_phase_gw_extrema=1.5):
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
        max_r_delta_phase_gw_extrema:
            Maximum value for ratio of successive phase_gw difference between
            consecutive extrema. If the ratio is greater than
            max_r_delta_phase_gw or less than 1/max_r_delta_phase_gw then
            an extremum is considered to be missing.
        returns:
        --------
        pericenters:
            1d array of pericenters after dropping pericenters as necessary.
        apocenters:
            1d array of apocenters after dropping apocenters as necessary.
        """
        pericenters = self.drop_extrema_if_extrema_jumps(
            pericenters, max_r_delta_phase_gw_extrema, "pericenters")
        apocenters = self.drop_extrema_if_extrema_jumps(
            apocenters, max_r_delta_phase_gw_extrema, "apocenters")
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
                                   self.omega_gw[extrema])
        else:
            raise Exception(
                f"Sufficient number of {extrema_type} are not found."
                " Can not create an interpolant.")
        
    def get_rat_fit(self, x, y):
        """Get rational fit.

        A wrapper of `utils.get_rational_fit` with check_kwargs=False. This is
        to make sure that the checking of kwargs is not performed everytime the
        rational fit function is called. Instead, the kwargs are checked once
        in the init and passed to the rational fit function without repeating
        checks.
        """
        return get_rational_fit(x, y,
                                rational_fit_kwargs=self.rational_fit_kwargs,
                                check_kwargs=False)

    def rational_fit(self, x, y):
        """Get rational fit with adaptive numerator and denominator degree.

        This function ensures that the rational fit we obtain is using the
        optimal degree for the numerator and the denominator. We start with an
        initial estimate of what these degrees should be based on the length of
        the waveform. We check the first derivative of the resultant fit to
        check for any nonmonotonicity. In case of nonmonocity detected, we
        lower the degree by 1 and repeat until the check passes successfully.
        """
        if (self.rational_fit_kwargs["num_degree"] is None) or (
            self.rational_fit_kwargs["denom_degree"] is None):
            self.rational_fit_kwargs["num_degree"],\
                self.rational_fit_kwargs["denom_degree"] \
                    = self.get_optimal_degree_for_rational_fit()
        rat_fit = self.get_rat_fit(x, y)
        t = np.arange(x[0], x[-1], self.t[1] - self.t[0])
        while self.check_domega_dt(t, rat_fit(t), 1.0):
            self.rational_fit_kwargs["num_degree"] -= 1
            self.rational_fit_kwargs["denom_degree"] -= 1
            debug_message(f"Rational fit with degree {self.rational_fit_kwargs['num_degree'] + 1} "
                          " has non-monotonic time derivative. Lowering degree to "
                          f"{self.rational_fit_kwargs['num_degree']} and trying again.",
                          debug_level=self.debug_level,
                          important=True)
            rat_fit = self.get_rat_fit(x, y)
        return rat_fit

    def get_optimal_degree_for_rational_fit(self):
        """Get optimal degree based on the approximate number of orbits.

        Assign degree based on approximate number of orbits if user provided
        degree is None. The number of orbits is approximated by the change in
        phase_gw divided by 4*pi. The degree of the rational fit is then chosen
        based on this approximate number of orbits. The degree is increased as
        the number of orbits increases.
        """
        # TODO: Optimize this.
        # assign degree based on approximate number of orbits if user provided
        # degree is None.
        approximate_num_orbits = ((self.phase_gw[-1] - self.phase_gw[0])
                                  / (4 * np.pi))
        if approximate_num_orbits <= 5:
            return 1, 1
        elif (approximate_num_orbits > 5) and (approximate_num_orbits <= 20):
            return 2, 2
        elif (approximate_num_orbits > 20) and (approximate_num_orbits <= 50):
            return 3, 3
        else:
            return 4, 4

    def rational_fit_extrema(self, extrema_type="pericenters"):
        """Build rational fit through extrema.

        parameters:
        -----------
        extrema_type:
            Either "pericenters" or "apocenters".

        returns:
        ------
        Rational fit through extrema
        """
        if extrema_type == "pericenters":
            extrema = self.pericenters_location
        elif extrema_type == "apocenters":
            extrema = self.apocenters_location
        else:
            raise Exception("extrema_type must be either "
                            "'pericenrers' or 'apocenters'.")
        if len(extrema) >= 2:
            return self.rational_fit(self.t[extrema],
                                     self.omega_gw[extrema])
        else:
            raise Exception(
                f"Sufficient number of {extrema_type} are not found."
                " Can not create an interpolant.")

    def check_domega_dt(self, t, omega, tol=1.0):
        """Check first derivative of omega.

        The first derivative of interpolant/fit of omega at the extrema
        should be monotonically increasing.
        """
        domega_dt = np.gradient(omega, t[1] - t[0])
        return any(domega_dt[1:]/domega_dt[:-1] < tol)

    def check_num_extrema(self, extrema, extrema_type="extrema"):
        """Check number of extrema.

        Check the number of extrema to determine if there are enough for
        building the interpolants through the pericenters and apocenters. In
        cases where the number of extrema is insufficient, i.e., less than 2,
        we further verify if the waveform is long enough to have a sufficient
        number of extrema.

        We verify that the waveform sent to the peak finding routine is a
        minimum of 5 orbits long. By default,
        `num_orbits_to_exclude_before_merger` is set to 2, which means that 2
        orbits are subtracted from the original waveform within the input
        dataDict. Consequently, in the default configuration, the original
        waveform must be at least 7 orbits in length to be considered as
        sufficiently long.

        If it is sufficiently long, but the chosen method fails to detect any
        extrema, it is possible that the eccentricity is too small. If
        `return_zero_if_small_ecc_failure` is set to True, then we set
        `insufficient_extrema_but_long_waveform` to True and return it.

        Parameters
        ----------
        extrema : array-like
            1d array of extrema to determine if the length is sufficient for
            building interpolants of omega_gw values at these extrema. We
            require the length to be greater than or equal to two.
        extrema_type: str, default="extrema"
            String to indicate whether the extrema corresponds to pericenters
            or the apocenters.

        Returns
        -------
        insufficient_extrema_but_long_waveform : bool
            True if the waveform has more than approximately 5 orbits but the
            number of extrema is less than two. False otherwise.
        """
        num_extrema = len(extrema)
        if num_extrema < 2:
            # Check if the waveform is sufficiently long by estimating the
            # approximate number of orbits contained in the waveform data using
            # the phase of the (2, 2) mode, assuming that a phase change of
            # 4*pi occurs over one orbit.
            # NOTE: Since we truncate the waveform data by removing
            # `num_orbits_to_remove_before_merger` orbits before the merger,
            # phase_gw[-1] corresponds to the phase of the (2, 2) mode
            # `num_orbits_to_remove_before_merger` orbits before the merger.
            approximate_num_orbits = ((self.phase_gw[-1] - self.phase_gw[0])
                                      / (4 * np.pi))
            if approximate_num_orbits > 5:
                # The waveform is sufficiently long but the extrema finding
                # method fails to find enough number of extrema. This may
                # happen if the eccentricity is too small and, therefore, the
                # modulations in the amplitude/frequency is too small for the
                # method to detect them.
                insufficient_extrema_but_long_waveform = True
            else:
                insufficient_extrema_but_long_waveform = False
            if insufficient_extrema_but_long_waveform \
               and self.return_zero_if_small_ecc_failure:
                debug_message(
                    "The waveform has approximately "
                    f"{approximate_num_orbits:.2f}"
                    f" orbits but number of {extrema_type} found is "
                    f"{num_extrema}. Since `return_zero_if_small_ecc_failure` is set to "
                    f"{self.return_zero_if_small_ecc_failure}, no exception is raised. "
                    "Instead the eccentricity and mean anomaly will be set to "
                    "zero.",
                    important=True,
                    debug_level=0)
            else:
                recommended_methods = ["ResidualAmplitude", "AmplitudeFits"]
                if self.method not in recommended_methods:
                    method_message = (
                        "It's possible that the eccentricity is too small for "
                        f"the {self.method} method to detect the "
                        f"{extrema_type}. Try one of {recommended_methods} "
                        "which should work even for a very small eccentricity."
                    )
                else:
                    method_message = ""
                raise Exception(
                    f"Number of {extrema_type} found = {num_extrema}.\n"
                    "Can not build frequency interpolant through the "
                    f"{extrema_type}.\n"
                    f"{method_message}")
            return insufficient_extrema_but_long_waveform

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

        #TODO: We may decide to use invariant angular velocity as omega in the future
        
        Eccentricity is measured using the GW frequency omega_gw(t) =
        d(phase_gw)/dt. Throughout this documentation, we will refer to
        phase_gw, omega_gw and amp_gw. For nonprecessing systems, these
        quantities are simply the corresponding values of the (2, 2) mode,

        amp_gw = amp22, phase_gw = phase22 and omega_gw = omega22.

        On the other hand, for precessing systems, we use Eq.(48) and (49) of
        arXiv:1701.00550 to define amp_gw and phase_gw. amp_gw [phase_gw] is
        defined using a symmetric [antisymmetric] combination of amplitude
        [phase] of (2, 2) and (2, -2) mode in the coprecessing frame,

        amp_gw = (1/2) * (amp(2, 2) + amp(2, -2))
        phase_gw = (1/2) * (phase(2, 2) - phase(2, -2))
        omega_gw = d(phase_gw)/dt.

        These quantities reduce to the corresponding (2, 2) mode data when the
        system is nonprecessing, but we treat nonprecessing cases differently
        by allowing the user to include only the (2, 2) mode. See
        `eccDefinition.get_amp_phase_omega_gw` for more details.
        
        We currently only allow time-domain waveforms. We evaluate omega_gw(t)
        at pericenter times, t_pericenters, and build a spline interpolant
        omega_gw_pericenters(t) using those data points. Similarly, we build
        omega_gw_apocenters(t) using omega_gw(t) at the apocenter times,
        t_apocenters.

        Using omega_gw_pericenters(t) and omega_gw_apocenters(t), we first
        compute e_omega_gw(t), as described in Eq.(4) of `arXiv:2302.11257`_
        (e_omega_gw is called e_omega_22 in the paper). We then use
        e_omega_gw(t) to compute the eccentricity egw(t) using Eq.(8) of
        `arXiv:2302.11257`_. Mean anomaly is defined using t_pericenters, as
        described in Eq.(10) of `arXiv:2302.11257`_.

        To find t_pericenters/t_apocenters, one can look for extrema in
        different waveform data, like omega_gw(t) or amp_gw(t). Pericenters
        correspond to the local maxima, while apocenters correspond to the
        local minima in the data. The method option (described below) lets the
        user pick which waveform data to use to find
        t_pericenters/t_apocenters.

        .. _arXiv:2302.11257: https://arxiv.org/abs/2302.11257

        Parameters
        ----------
        tref_in : array or float
            Input reference time at which to measure eccentricity and mean
            anomaly.

        fref_in : array or float
            Input reference GW frequency at which to measure the eccentricity
            and mean anomaly. Only one of *tref_in*/*fref_in* should be
            provided.

            Given an *fref_in*, we find the corresponding tref_in such that::

                omega_gw_average(tref_in) = 2 * pi * fref_in

            Here, omega_gw_average(t) is a monotonically increasing average
            frequency obtained from the instantaneous
            omega_gw(t). omega_gw_average(t) defaults to the orbit averaged
            omega_gw, but other options are available (see
            omega_gw_averaging_method below).

            Eccentricity and mean anomaly measurements are returned on a subset
            of *tref_in*/*fref_in*, called *tref_out*/*fref_out*, which are
            described below.  If *dataDict* is provided in dimensionless units,
            *tref_in* should be in units of M and *fref_in* should be in units
            of cycles/M. If dataDict is provided in MKS units, *t_ref* should
            be in seconds and *fref_in* should be in Hz.

        Returns
        -------
        A dictionary with the following keys
            tref_out
                *tref_out* is the output reference time at which eccentricity
                and mean anomaly are measured.  *tref_out* is included in the
                returned dictionary only when *tref_in* is provided.  Units of
                *tref_out* are the same as that of *tref_in*.

                tref_out is set as::

                    tref_out = tref_in[tref_in >= tmin & tref_in <= tmax],

                where, ::

                    tmax = min(t_pericenters[-1], t_apocenters[-1])
                    tmin = max(t_pericenters[0], t_apocenters[0])

                As eccentricity measurement relies on the interpolants
                omega_gw_pericenters(t) and omega_gw_apocenters(t), the above
                cutoffs ensure that we only compute the eccentricity where both
                omega_gw_pericenters(t) and omega_gw_apocenters(t) are within
                their bounds.

            fref_out
                *fref_out* is the output reference frequency at which
                eccentricity and mean anomaly are measured.  *fref_out* is
                included in the returned dictionary only when *fref_in* is
                provided.  Units of *fref_out* are the same as that of
                *fref_in*.

                *fref_out* is set as::

                    fref_out = fref_in[fref_in >= fref_min && fref_in <= fref_max]

                where, fref_min/fref_max are minimum/maximum allowed reference
                frequency, with::

                    fref_min = omega_gw_average(tmin_for_fref)/2/pi
                    fref_max = omega_gw_average(tmax_for_fref)/2/pi

                tmin_for_fref/tmax_for_fref are close to tmin/tmax, see
                :meth:`eccDefinition.get_fref_bounds()` for details.

            eccentricity
                Measured eccentricity at *tref_out*/*fref_out*. Same type as
                *tref_out*/*fref_out*.

            mean_anomaly
                Measured mean anomaly at *tref_out*/*fref_out*. Same type as
                *tref_out*/*fref_out*.
        """
        # check that only one of tref_in or fref_in is provided
        if (tref_in is not None) + (fref_in is not None) != 1:
            raise KeyError("Exactly one of tref_in and fref_in"
                           " should be specified.")
        elif tref_in is not None:
            # Identify whether the reference point is in time or frequency
            self.domain = "time"
            # Identify whether the reference point is scalar or array-like
            self.ref_ndim = np.ndim(tref_in)
            self.tref_in = np.atleast_1d(tref_in)
        else:
            self.domain = "frequency"
            self.ref_ndim = np.ndim(fref_in)
            self.fref_in = np.atleast_1d(fref_in)
        # Get the pericenters and apocenters
        pericenters = self.find_extrema("pericenters")
        original_pericenters = pericenters.copy()
        # Check if there are a sufficient number of extrema. In cases where the
        # waveform is long enough (at least 5 +
        # `num_orbits_to_exclude_before_merger`, i.e., 7 orbits long with
        # default settings) but the method fails to detect any extrema, it
        # might be that the eccentricity is too small for the current method to
        # detect it. See Fig.4 in arxiv.2302.11257. In such cases, the
        # following variable will be true.
        insufficient_pericenters_but_long_waveform \
            = self.check_num_extrema(pericenters, "pericenters")
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
        insufficient_apocenters_but_long_waveform \
            = self.check_num_extrema(apocenters, "apocenters")

        # If the eccentricity is too small for a method to find the extrema,
        # and `return_zero_if_small_ecc_failure` is true, then we set the eccentricity and
        # mean anomaly to zero and return them. In this case, the rest of the
        # code in this function is not executed, and therefore, many variables
        # that are needed for making diagnostic plots are not computed. Thus,
        # in such cases, the diagnostic plots may not work.
        if any([insufficient_pericenters_but_long_waveform,
                insufficient_apocenters_but_long_waveform]) \
                and self.return_zero_if_small_ecc_failure:
            # Store this information that we are setting ecc and mean anomaly
            # to zero to use it in other places
            self.setting_ecc_to_zero = True
            return self.set_eccentricity_and_mean_anomaly_to_zero()
        else:
            self.setting_ecc_to_zero = False

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

        # Build the interpolants of omega_gw at the extrema
        if self.extra_kwargs["omega_gw_extrema_interpolation_method"] == "spline":
            self.omega_gw_pericenters_interp = self.interp_extrema("pericenters")
            self.omega_gw_apocenters_interp = self.interp_extrema("apocenters")
        elif self.extra_kwargs["omega_gw_extrema_interpolation_method"] == "rational_fit":
            self.omega_gw_pericenters_interp = self.rational_fit_extrema("pericenters")
            self.omega_gw_apocenters_interp = self.rational_fit_extrema("apocenters")
        else:
            raise Exception("Unknown method for `omega_gw_extrema_interpolation_method`. "
                            "Must be one of `spline` or `rational_fit`.")

        self.t_pericenters = self.t[self.pericenters_location]
        self.t_apocenters = self.t[self.apocenters_location]
        self.tmax = min(self.t_pericenters[-1], self.t_apocenters[-1])
        self.tmin = max(self.t_pericenters[0], self.t_apocenters[0])
        
        if self.domain == "frequency":
            # get the tref_in and fref_out from fref_in
            self.tref_in, self.fref_out \
                = self.compute_tref_in_and_fref_out_from_fref_in(self.fref_in)
        # We measure eccentricity and mean anomaly from tmin to tmax.
        self.tref_out = self.tref_in[
            np.logical_and(self.tref_in <= self.tmax,
                           self.tref_in >= self.tmin)]
        # set time for checks and diagnostics
        self.t_for_checks = self.dataDict["t"][
            np.logical_and(self.dataDict["t"] >= self.tmin,
                           self.dataDict["t"] <= self.tmax)]

        # Verify the monotonicity of the first derivative of the omega_gw interpolant.
        # If a spline is used for interpolation (as specified by 'omega_gw_extrema_interpolation_method'), 
        # non-monotonicity may occur in the first derivative. 
        # If 'use_rational_fit_as_fallback' is set to True, the spline interpolant 
        # will be replaced with a rational fit to ensure monotonic behavior.
        if self.extra_kwargs["omega_gw_extrema_interpolation_method"] == "spline":
        # Check if the first derivative of omega_gw at pericenters or apocenters is non-monotonic
            has_non_monotonicity = (
                self.check_domega_dt(self.t_for_checks, self.omega_gw_pericenters_interp(self.t_for_checks)) or
                self.check_domega_dt(self.t_for_checks, self.omega_gw_apocenters_interp(self.t_for_checks))
                )      
    
            if has_non_monotonicity:
                if self.use_rational_fit_as_fallback:
                    debug_message(
                        "Non-monotonic time derivative detected in the spline interpolant through extrema. "
                        "Switching to rational fit.",
                        debug_level=self.debug_level, important=True
                        )
                    # Use rational fit for both pericenters and apocenters
                    self.omega_gw_pericenters_interp = self.rational_fit_extrema("pericenters")
                    self.omega_gw_apocenters_interp = self.rational_fit_extrema("apocenters")
                else:
                    debug_message(
                        "Non-monotonic time derivative detected in the spline interpolant through extrema. "
                        "Consider the following options to avoid this: \n"
                        " - Set 'use_rational_fit_as_fallback' to True in 'extra_kwargs' to switch to rational fit.\n"
                        " - Use rational fit directly by setting 'omega_gw_extrema_interpolation_method' to 'rational_fit'.",
                        debug_level=self.debug_level, important=True
                        )

        # Sanity checks
        # check that fref_out and tref_out are of the same length
        if self.domain == "frequency":
            if len(self.fref_out) != len(self.tref_out):
                raise Exception(
                    "length of fref_out and tref_out do not match."
                    f"fref_out has length {len(self.fref_out)} and "
                    f"tref_out has length {len(self.tref_out)}.")

        # Check if tref_out is reasonable
        if len(self.tref_out) == 0:
            self.check_input_limits(self.tref_in, self.tmin, self.tmax)
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

        if self.debug_plots:
            # make a plot for diagnostics
            fig, axes = self.make_diagnostic_plots()
            self.save_debug_fig(fig, f"gwecc_{self.method}_diagnostics.pdf")
            plt.close(fig)
        # return measured eccentricity, mean anomaly and reference time or
        # frequency where these are measured.
        return self.make_return_dict_for_eccentricity_and_mean_anomaly()

    def set_eccentricity_and_mean_anomaly_to_zero(self):
        """Set eccentricity and mean_anomaly to zero."""
        if self.domain == "time":
            # This function sets eccentricity and mean anomaly to zero when a
            # method fails to detect any extrema, and therefore, in such cases,
            # we can set tref_out to be the same as tref_in.
            self.tref_out = self.tref_in
            out_len = len(self.tref_out)
        else:
            # similarly we can set fref_out to be the same as fref_in
            self.fref_out = self.fref_in
            out_len = len(self.fref_out)
        self.eccentricity = np.zeros(out_len)
        self.mean_anomaly = np.zeros(out_len)
        return self.make_return_dict_for_eccentricity_and_mean_anomaly()

    def make_return_dict_for_eccentricity_and_mean_anomaly(self):
        """Prepare a dictionary with reference time/freq, ecc, and mean anomaly.

        In this function, we prepare a dictionary containing the measured
        eccentricity, mean anomaly, and the reference time or frequency where
        these are measured.

        We also make sure that if the input reference time/frequency is scalar,
        then the returned eccentricity and mean anomaly are also scalars. To do
        this, we use the information about tref_in/fref_in that is provided by
        the user. At the top of the measure_ecc function, we set ref_ndim to
        identify whether the original input was scalar or array-like and use
        that here.
        """
        # If the original input was scalar, convert the measured eccentricity,
        # mean anomaly, etc., to scalar.
        if self.ref_ndim == 0:
            # check if ecc, mean ano have more than one elements
            for var, arr in zip(["eccentricity", "mean_anomaly"],
                                [self.eccentricity, self.mean_anomaly]):
                if len(arr) != 1:
                    raise Exception(f"The reference {self.domain} is scalar "
                                    f"but measured {var} does not have "
                                    "exactly one element.")
            self.eccentricity = self.eccentricity[0]
            self.mean_anomaly = self.mean_anomaly[0]
            if self.domain == "time":
                self.tref_out = self.tref_out[0]
            else:
                self.fref_out = self.fref_out[0]

        return_dict = {
            "eccentricity": self.eccentricity,
            "mean_anomaly": self.mean_anomaly
        }
        # Return either tref_out or fref_out, depending on whether the input
        # reference point was in time or frequency, respectively.
        if self.domain == "time":
            return_dict.update({
              "tref_out": self.tref_out})
        else:
            return_dict.update({
              "fref_out": self.fref_out})
        return return_dict

    def et_from_ew22_0pn(self, ew22):
        """Get temporal eccentricity at Newtonian order.

        Parameters
        ----------
        ew22:
            eccentricity measured from the 22-mode frequency.

        Returns
        -------
        et:
            Temporal eccentricity at Newtonian order.
        """
        psi = np.arctan2(1. - ew22*ew22, 2.*ew22)
        et = np.cos(psi/3.) - np.sqrt(3) * np.sin(psi/3.)

        return et

    def compute_eccentricity(self, t):
        """
        Compute eccentricity at time t.

        Compute e_omega_gw from the value of omega_gw_pericenters_interpolant and
        omega_gw_apocenters_interpolant at t using Eq.(4) in arXiv:2302.11257
        and then use Eq.(8) in arXiv:2302.11257 to compute e_gw from e_omega_gw.

        Paramerers
        ----------
        t:
            Time to compute the eccentricity at. Could be scalar or an array.

        Returns
        -------
        Eccentricity at t.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_input_limits(t, self.tmin, self.tmax)

        omega_gw_pericenter_at_t = self.omega_gw_pericenters_interp(t)
        omega_gw_apocenter_at_t = self.omega_gw_apocenters_interp(t)
        self.e_omega_gw = ((np.sqrt(omega_gw_pericenter_at_t)
                           - np.sqrt(omega_gw_apocenter_at_t))
                          / (np.sqrt(omega_gw_pericenter_at_t)
                             + np.sqrt(omega_gw_apocenter_at_t)))
        # get the  temporal eccentricity from e_omega_gw
        return self.et_from_ew22_0pn(self.e_omega_gw)

    def derivative_of_eccentricity(self, t, n=1):
        """Get time derivative of eccentricity.

        Parameters
        ----------
        t
            Times to get the derivative at.
        n : int
            Order of derivative. Should be 1 or 2, since it uses
            cubic spine to get the derivatives.

        Returns
        -------
            nth order time derivative of eccentricity.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_input_limits(t, self.tmin, self.tmax)

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

        Parameters
        ----------
        t:
            Time to compute mean anomaly at. Could be scalar or an array.

        Returns
        -------
        Mean anomaly at t.
        """
        # Check that t is within tmin and tmax to avoid extrapolation
        self.check_input_limits(t, self.tmin, self.tmax)

        # Get the mean anomaly at the pericenters
        mean_ano_pericenters = np.arange(len(self.t_pericenters)) * 2 * np.pi
        # Using linear interpolation since mean anomaly is a linear function of
        # time.
        mean_ano = np.interp(t, self.t_pericenters, mean_ano_pericenters)
        # Modulo 2pi to make the mean anomaly vary between 0 and 2pi
        return mean_ano % (2 * np.pi)

    def check_input_limits(self, input_vals, min_allowed_val, max_allowed_val):
        """Check that the input time/frequency is within the allowed range.

        To avoid any extrapolation, check that the times or frequencies are
        always greater than or equal to the minimum allowed value and always
        less than the maximum allowed value.

        Parameters
        ----------
        input_vals: float or array-like
            Input times or frequencies where eccentricity/mean anomaly is to
            be measured.

        min_allowed_val: float
            Minimum allowed time or frequency where eccentricity/mean anomaly
            can be measured.

        max_allowed_val: float
            Maximum allowed time or frequency where eccentricity/mean anomaly
            can be measured.
        """
        input_vals = np.atleast_1d(input_vals)
        if any(input_vals > max_allowed_val):
            message = (f"Found reference {self.domain} later than maximum "
                       f"allowed {self.domain}={max_allowed_val}")
            if self.domain == "time":
                # Add information about the maximum allowed time
                message += " which corresponds to "
                if self.setting_ecc_to_zero:
                    message += ("time at `num_orbits_to_exclude_before_merger`"
                                " orbits before the merger.")
                else:
                    message += "min(last pericenter time, last apocenter time)."
            raise Exception(
                f"Reference {self.domain} is outside the allowed "
                f"range [{min_allowed_val}, {max_allowed_val}]."
                f"\n{message}")
        if any(input_vals < min_allowed_val):
            message = (f"Found reference {self.domain} earlier than minimum "
                       f"allowed {self.domain}={min_allowed_val}")
            if self.domain == "time":
                # Add information about the minimum allowed time
                message += " which corresponds to "
                if self.setting_ecc_to_zero:
                    message += "the starting time in the time array."
                else:
                    message += "max(first pericenter time, first apocenter time)."
            raise Exception(
                f"Reference {self.domain} is outside the allowed "
                f"range [{min_allowed_val}, {max_allowed_val}]."
                f"\n{message}")

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

        Parameters
        ----------
        always_return : bool, default: False
            The return values of this function are used by some plotting
            functions, so if always_return=True, we execute the body and
            return values regardless of debug_level. However, the warnings
            will still be suppressed for debug_level < 1.
        """
        # This function only has checks with the flag important=False, which
        # means that warnings are suppressed when debug_level < 1.
        # We return without running the rest of the body to avoid unnecessary
        # computations, unless always_return=True.
        if self.debug_level < 1 and always_return is False:
            return None, None

        orb_phase_at_extrema = self.phase_gw[extrema_location] / 2
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

        Parameters
        ----------
        check_convexity : bool, default: False
            In addition to monotonicity, it will check for
            convexity as well.
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

    def compute_res_amp_gw_and_res_omega_gw(self):
        """Compute residual amp_gw and residual omega_gw."""
        self.t_zeroecc = self.dataDict["t_zeroecc"]
        # check that the time steps are equal
        self.t_zeroecc_diff = np.diff(self.t_zeroecc)
        if not np.allclose(self.t_zeroecc_diff, self.t_zeroecc_diff[0]):
            raise Exception(
                "Input time array t_zeroecc must have uniform time steps\n"
                f"Time steps are {self.t_zeroecc_diff}")
        # get amplitude and omega of 22 mode
        self.amp_gw_zeroecc = self.dataDict["amplm_zeroecc"][(2, 2)]
        self.omega_gw_zeroecc = self.dataDict["omegalm_zeroecc"][(2, 2)]
        # to get the residual amplitude and omega, we need to shift the
        # zeroecc time axis such that the merger of the zeroecc is at the
        # same time as that of the eccentric waveform
        amp = amplitude_using_all_modes(
            self.dataDict["amplm_zeroecc"], "amplm")  # total amplitude
        self.t_merger_zeroecc = peak_time_via_quadratic_fit(
            self.t_zeroecc, amp)[0]
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
        self.amp_gw_zeroecc_interp = self.interp(
            self.t, self.t_zeroecc_shifted, self.amp_gw_zeroecc,
            allowExtrapolation=True)
        self.res_amp_gw = self.amp_gw - self.amp_gw_zeroecc_interp
        self.omega_gw_zeroecc_interp = self.interp(
            self.t, self.t_zeroecc_shifted, self.omega_gw_zeroecc,
            allowExtrapolation=True)
        self.res_omega_gw = (self.omega_gw - self.omega_gw_zeroecc_interp)

    def get_t_average_for_orbit_averaged_omega_gw(self):
        """Get the times associated with the fref for orbit averaged omega_gw.

        t_average_pericenters are the times at midpoints between consecutive
        pericenters. We associate time (t[i] + t[i+1]) / 2 with the orbit
        averaged omega_gw calculated between ith and (i+1)th pericenter. That
        is, omega_gw_average((t[i] + t[i+1])/2) = int_t[i]^t[i+1] omega_gw(t) dt
        / (t[i+1] - t[i]), where t[i] is the time at the ith pericenter.  And
        similarly, we calculate the t_average_apocenters. We combine
        t_average_pericenters and t_average_apocenters, and sort them to obtain
        t_average.

        Returns
        -------
        t_for_orbit_averaged_omega_gw
            Times associated with orbit averaged omega_gw
        sorted_idx_for_orbit_averaged_omega_gw
            Indices used to sort the times associated with orbit averaged
            omega_gw
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
        t_for_orbit_averaged_omega_gw = np.append(
            self.t_average_apocenters,
            self.t_average_pericenters)
        sorted_idx_for_orbit_averaged_omega_gw = np.argsort(
            t_for_orbit_averaged_omega_gw)
        t_for_orbit_averaged_omega_gw = t_for_orbit_averaged_omega_gw[
            sorted_idx_for_orbit_averaged_omega_gw]
        return [t_for_orbit_averaged_omega_gw,
                sorted_idx_for_orbit_averaged_omega_gw]

    def get_orbit_averaged_omega_gw_between_pericenters(self):
        """Get orbital average of omega_gw between two consecutive pericenters.

        Given N pericenters at times t[i], i=0...N-1, this function returns a
        np.array of length N-1, where result[i] is the frequency averaged over
        [t[i], t[i+1]]. result[i] is associated with the time at the temporal
        midpoint between t[i] and t[i+1], i.e (t[i] + t[i+1])/2. See Eq.(12)
        and Eq.(13) in arXiv:2302.11257 for details.

        Orbital average of omega_gw between two consecutive pericenters
        i-th and (i+1)-th is given by
        <omega_gw>_i = (int_t[i]^t[i+1] omega_gw(t) dt)/(t[i+1] - t[i])
        t[i] is the time at the i-th extrema.
        Integration of omega_gw(t) from t[i] to t[i+1] is the same
        as taking the difference of phase_gw(t) between t[i] and t[i+1]
        <omega_gw>_i = (phase_gw[t[i+1]] - phase_gw[t[i]])/(t[i+1] - t[i])
        """
        return (np.diff(self.phase_gw[self.pericenters_location]) /
                np.diff(self.t[self.pericenters_location]))

    def get_orbit_averaged_omega_gw_between_apocenters(self):
        """Get orbital average of omega_gw between two consecutive apocenters.

        The procedure to get the orbital average of omega_gw between apocenters
        is the same as that between pericenters. See documentation of
        `get_orbit_averaged_omega_gw_between_pericenters` for details.
        """
        return (np.diff(self.phase_gw[self.apocenters_location]) /
                np.diff(self.t[self.apocenters_location]))

    def compute_orbit_averaged_omega_gw_between_extrema(self, t):
        """Compute reference frequency by orbital averaging omega_gw between extrema.

        We compute the orbital average of omega_gw between two consecutive
        extrema as following:
        <omega_gw>_i = (int_t[i]^t[i+1] omega_gw(t) dt) / (t[i+1] - t[i])
        where t[i] is the time of ith extrema and the suffix `i` stands for the
        i-th orbit between i-th and (i+1)-th extrema
        <omega_gw>_i is associated with the temporal midpoint between the i-th
        and (i+1)-th extrema,
        <t>_i = (t[i] + t[i+1]) / 2
        See Eq.(12) and Eq.(13) in arXiv:2302.11257 for more details.

        We do this averaging between consecutive pericenters and consecutive
        apocenters using the functions
        `get_orbit_averaged_omega_gw_between_pericenters` and
        `get_orbit_averaged_omega_gw_between_apocenters` and combine the
        results. The combined array is then sorted using the sorting indices
        from `get_t_average_for_orbit_averaged_omega_gw`.

        Finally we interpolate the data {<t>_i, <omega_gw>_i} and evaluate the
        interpolant at the input times `t`.
        """
        # get orbit averaged omega_gw between consecutive pericenrers
        # and consecutive apoceneters
        self.orbit_averaged_omega_gw_pericenters \
            = self.get_orbit_averaged_omega_gw_between_pericenters()
        self.orbit_averaged_omega_gw_apocenters \
            = self.get_orbit_averaged_omega_gw_between_apocenters()
        # check monotonicity of the omega_gw average
        self.check_monotonicity_of_omega_gw_average(
            self.orbit_averaged_omega_gw_pericenters,
            "omega_gw averaged [pericenter to pericenter]")
        self.check_monotonicity_of_omega_gw_average(
            self.orbit_averaged_omega_gw_apocenters,
            "omega_gw averaged [apocenter to apocenter]")
        # combine the average omega_gw between consecutive pericenters and
        # consecutive apocenters
        orbit_averaged_omega_gw = np.append(
            self.orbit_averaged_omega_gw_apocenters,
            self.orbit_averaged_omega_gw_pericenters)

        # get the times associated to the orbit averaged omega_gw
        if not hasattr(self, "t_for_orbit_averaged_omega_gw"):
            self.t_for_orbit_averaged_omega_gw,\
                self.sorted_idx_for_orbit_averaged_omega_gw\
                = self.get_t_average_for_orbit_averaged_omega_gw()
        # We now sort omega_gw_average using
        # `sorted_idx_for_orbit_averaged_omega_gw`, the same array of indices
        # that was used to obtain the `t_for_orbit_averaged_omega_gw` in the
        # function `eccDefinition.get_t_average_for_orbit_averaged_omega_gw`.
        orbit_averaged_omega_gw = orbit_averaged_omega_gw[
            self.sorted_idx_for_orbit_averaged_omega_gw]
        # check that omega_gw_average in strictly monotonic
        self.check_monotonicity_of_omega_gw_average(
            orbit_averaged_omega_gw,
            "omega_gw averaged [apocenter to apocenter] and "
            "[pericenter to pericenter]")
        return self.interp(
            t, self.t_for_orbit_averaged_omega_gw, orbit_averaged_omega_gw)

    def check_monotonicity_of_omega_gw_average(self,
                                              omega_gw_average,
                                              description="omega_gw average"):
        """Check that omega average is monotonically increasing.

        Parameters
        ----------
        omega_gw_average : array-like
            1d array of omega_gw averages to check for monotonicity.
        description : str
            String to describe what the the which omega_gw average we are
            looking at.
        """
        idx_non_monotonic = np.where(
            np.diff(omega_gw_average) <= 0)[0]
        if len(idx_non_monotonic) > 0:
            first_idx = idx_non_monotonic[0]
            change_at_first_idx = (
                omega_gw_average[first_idx+1]
                - omega_gw_average[first_idx])
            if self.debug_plots:
                style = "APS"
                use_fancy_plotsettings(style=style)
                nrows = 4
                fig, axes = plt.subplots(
                    nrows=nrows,
                    figsize=(figWidthsTwoColDict[style],
                             nrows * figHeightsDict[style]))
                axes[0].plot(omega_gw_average, marker=".",
                             c=colorsDict["default"])
                axes[1].plot(np.diff(omega_gw_average), marker=".",
                             c=colorsDict["default"])
                axes[2].plot(self.t_average_pericenters,
                             self.orbit_averaged_omega_gw_pericenters,
                             label=labelsDict["pericenters"],
                             c=colorsDict["pericenter"],
                             marker=".")
                axes[2].plot(self.t_average_apocenters,
                             self.orbit_averaged_omega_gw_apocenters,
                             label=labelsDict["apocenters"],
                             c=colorsDict["apocenter"],
                             marker=".")
                axes[3].plot(self.t, self.omega_gw, c=colorsDict["default"])
                axes[3].plot(self.t_pericenters,
                             self.omega_gw[self.pericenters_location],
                             c=colorsDict["pericenter"],
                             label=labelsDict["pericenters"],
                             marker=".")
                axes[3].plot(self.t_apocenters,
                             self.omega_gw[self.apocenters_location],
                             c=colorsDict["apocenter"],
                             label=labelsDict["apocenters"],
                             marker=".")
                axes[2].legend()
                axes[2].set_ylabel(labelsDict["omega_gw_average"])
                axes[3].legend()
                axes[3].set_ylim(0,)
                axes[3].set_ylabel(labelsDict["omega_gw"])
                axes[1].axhline(0, c=colorsDict["vline"])
                axes[0].set_ylabel(labelsDict["omega_gw_average"])
                axes[1].set_ylabel(
                    fr"$\Delta$ {labelsDict['omega_gw_average']}")
                axes[0].set_title(
                    self.extra_kwargs["omega_gw_averaging_method"])
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
                f" where omega_gw drops from {omega_gw_average[first_idx]} to"
                f" {omega_gw_average[first_idx+1]}, a decrease by"
                f" {abs(change_at_first_idx)}.\nTotal number of places of"
                f" non-monotonicity is {len(idx_non_monotonic)}.\n"
                f"Last one occurs at peak number {idx_non_monotonic[-1]}.\n"
                f"{plot_info}\n"
                "Possible fixes: \n"
                "   - Increase sampling rate of data\n"
                "   - Add to extra_kwargs the option 'treat_mid_points_between"
                "_pericenters_as_apocenters': True")

    def compute_mean_of_extrema_interpolants(self, t):
        """Find omega_gw average by taking mean of the extrema interpolants".

        Take mean of omega_gw spline through omega_gw pericenters
        and apocenters to get
        omega_gw_average = 0.5 * (omega_gw_p(t) + omega_gw_a(t))
        """
        return 0.5 * (self.omega_gw_pericenters_interp(t) +
                      self.omega_gw_apocenters_interp(t))

    def compute_omega_gw_zeroecc(self, t):
        """Find omega_gw from zeroecc data."""
        return self.interp(
            t, self.t_zeroecc_shifted, self.omega_gw_zeroecc)

    def get_available_omega_gw_averaging_methods(self):
        """Return available omega_gw averaging methods."""
        available_methods = {
            "mean_of_extrema_interpolants": self.compute_mean_of_extrema_interpolants,
            "orbit_averaged_omega_gw": self.compute_orbit_averaged_omega_gw_between_extrema,
            "omega_gw_zeroecc": self.compute_omega_gw_zeroecc
        }
        return available_methods

    def get_omega_gw_average(self, method=None):
        """Get times and corresponding values of omega_gw average.

        Parameters
        ----------
        method : str
            omega_gw averaging method. Must be one of the following:
            - "mean_of_extrema_interpolants"
            - "orbit_averaged_omega_gw"
            - "omega_gw_zeroecc"
            See get_available_omega_gw_averaging_methods for available averaging
            methods and Sec.IID of arXiv:2302.11257 for more details.
            Default is None which uses the method provided in
            `self.extra_kwargs["omega_gw_averaging_method"]`

        Returns
        -------
        t_for_omega_gw_average : array-like
            Times associated with omega_gw_average.
        omega_gw_average : array-like
            omega_gw average using given "method".
            These are data interpolated on the times t_for_omega_gw_average,
            where t_for_omega_gw_average is a subset of tref_in passed to the
            eccentricity measurement function.

            For the "orbit_averaged_omega_gw" method, the original
            omega_gw_average data points <omega_gw>_i are obtained by averaging
            the omega_gw over the ith orbit between ith to i+1-th extrema. The
            associated <t>_i are obtained by taking the times at the midpoints
            between i-th and i+1-the extrema, i.e., <t>_i = (t_i + t_(i+1))/2.

            These original orbit averaged omega_gw data points can be accessed
            using the gwecc_object with the following variables

            - orbit_averaged_omega_gw_apocenters: orbit averaged omega_gw between
              apocenters. This is available when measuring eccentricity at
              reference frequency. If it is not available, it can be computed
              using `get_orbit_averaged_omega_gw_between_apocenters`
            - t_average_apocenters: temporal midpoints between
              apocenters. These are associated with
              `orbit_averaged_omega_gw_apocenters`
            - orbit_averaged_omega_gw_pericenters: orbit averaged omega_gw
              between pericenters. This is available when measuring
              eccentricity at reference frequency. If it is not available, it
              can be computed using
              `get_orbit_averaged_omega_gw_between_pericenters`
            - t_average_pericenters: temporal midpoints between
              pericenters. These are associated with
              `orbit_averaged_omega_gw_pericenters`
        """
        if method is None:
            method = self.extra_kwargs["omega_gw_averaging_method"]
        if method != "orbit_averaged_omega_gw":
            # the average frequencies are using interpolants that use omega_gw
            # values between tmin and tmax, therefore the min and max time for
            # which omega_gw average are the same as tmin and tmax,
            # respectively.
            self.tmin_for_fref = self.tmin
            self.tmax_for_fref = self.tmax
        else:
            self.t_for_orbit_averaged_omega_gw, self.sorted_idx_for_orbit_averaged_omega_gw = \
                self.get_t_average_for_orbit_averaged_omega_gw()
            # for orbit averaged omega_gw, the associated times are obtained
            # using the temporal midpoints of the extrema, therefore we need to
            # make sure that we use only those times that fall within tmin and
            # tmax.
            self.tmin_for_fref = max(self.tmin,
                                     min(self.t_for_orbit_averaged_omega_gw))
            self.tmax_for_fref = min(self.tmax,
                                     max(self.t_for_orbit_averaged_omega_gw))
        t_for_omega_gw_average = self.t[
            np.logical_and(self.t >= self.tmin_for_fref,
                           self.t <= self.tmax_for_fref)]
        omega_gw_average = self.available_averaging_methods[
            method](t_for_omega_gw_average)
        return t_for_omega_gw_average, omega_gw_average

    def compute_tref_in_and_fref_out_from_fref_in(self, fref_in):
        """Compute tref_in and fref_out from fref_in.

        Using chosen omega_gw average method we get the tref_in and fref_out
        for the given fref_in.

        When the input is frequencies where eccentricity/mean anomaly is to be
        measured, we internally want to map the input frequencies to a tref_in
        and then we proceed to calculate the eccentricity and mean anomaly for
        this tref_in in the same way as we do when the input array was time
        instead of frequencies.

        We first compute omega_gw_average(t) using the instantaneous omega_gw(t),
        which can be done in different ways as described below. Then, we keep
        only the allowed frequencies in fref_in by doing
        fref_out = fref_in[fref_in >= fref_min && fref_in < fref_max],
        Where fref_min/fref_max is the minimum/maximum allowed reference
        frequency for the given omega_gw averaging method. See get_fref_bounds
        for more details.
        Finally, we find the times where omega_gw_average(t) = 2*pi*fref_out,
        and set those to tref_in.

        omega_gw_average(t) could be calculated in the following ways

        - Mean of the omega_gw given by the spline through the pericenters and
          the spline through the apocenters, we call this
          "mean_of_extrema_interpolants"
        - Orbital average at the extrema, we call this
          "orbit_averaged_omega_gw"
        - omega_gw of the zero eccentricity waveform, called "omega_gw_zeroecc"

        Users can provide a method through the "extra_kwargs" option with the
        key "omega_gw_averaging_method".
        Default is "orbit_averaged_omega_gw"

        Once we get the reference frequencies, we create a spline to get time
        as a function of these reference frequencies. This should work if the
        reference frequency is monotonic which it should be.
        Finally, we evaluate this spline on the fref_in to get the tref_in.
        """
        method = self.extra_kwargs["omega_gw_averaging_method"]
        if method in self.available_averaging_methods:
            # The fref_in array could have frequencies that is outside the
            # range of frequencies in omega_gw average. Therefore, we want to
            # create a separate array of frequencies fref_out which is created
            # by taking on those frequencies that falls within the omega_gw
            # average. Then proceed to evaluate the tref_in based on these
            # fref_out
            fref_out = self.get_fref_out(fref_in, method)

            # Now that we have fref_out, we want to know the corresponding
            # tref_in such that omega_gw_average(tref_in) = fref_out * 2 * pi
            # This is done by first creating an interpolant of time as function
            # of omega_gw_average.
            # We get omega_gw_average by evaluating the omega_gw_average(t)
            # on t, from tmin_for_fref to tmax_for_fref
            self.t_for_omega_gw_average, self.omega_gw_average = self.get_omega_gw_average(method)

            # check that omega_gw_average is monotonically increasing
            self.check_monotonicity_of_omega_gw_average(
                self.omega_gw_average, "Interpolated omega_gw_average")

            # Get tref_in using interpolation
            tref_in = self.interp(fref_out,
                                  self.omega_gw_average/(2 * np.pi),
                                  self.t_for_omega_gw_average)
            # check if tref_in is monotonically increasing
            if any(np.diff(tref_in) <= 0):
                debug_message(f"tref_in from fref_in using method {method} is"
                              " not monotonically increasing.",
                              self.debug_level, important=False)
            return tref_in, fref_out
        else:
            raise KeyError(f"Omega_Gw averaging method {method} does not exist."
                           " Must be one of "
                           f"{list(self.available_averaging_methods.keys())}")

    def get_fref_bounds(self, method=None):
        """Get the allowed min and max reference frequency of f_gw = omega_gw/2pi.

        Depending on the omega_gw averaging method, this function returns the
        minimum and maximum allowed reference frequency of f_gw = omega_gw/2pi.

        Note: If omega_gw_average is already computed using a `method` and
        therefore is not None, then it returns the minimum and maximum of that
        omega_gw_average and does not recompute the omega_gw_average using the
        input `method`. In other words, if omega_gw_average is already not None
        then input `method` is ignored and the existing omega_gw_average is
        used.  To force recomputation of omega_gw_average, for example, with a
        new method one need to set it to None first.
        Parameters
        ----------
        method : str
            Omega_Gw averaging method.  See
            get_available_omega_gw_averaging_methods for available methods.
            Default is None which will use the default method for omega_gw
            averaging using `extra_kwargs["omega_gw_averaging_method"]`

        Returns
        -------
            Minimum allowed reference frequency, Maximum allowed reference
            frequency.
        """
        if self.omega_gw_average is None:
            self.t_for_omega_gw_average, self.omega_gw_average = self.get_omega_gw_average(method)
        return [min(self.omega_gw_average)/2/np.pi,
                max(self.omega_gw_average)/2/np.pi]

    def get_fref_out(self, fref_in, method):
        """Get fref_out from fref_in that falls within the valid average f_gw range.

        f_gw = omega_gw / 2pi

        Parameters
        ----------
        fref_in : array-like
            Input reference frequency array, i.e., f_gw = omega_gw / 2pi.

        method : str
            method for getting average omega_gw

        Returns
        -------
        fref_out : array-like
            Slice of fref_in that satisfies:
            fref_in >= fref_min && fref_in < fref_max
        """
        fref_min, fref_max = self.get_fref_bounds(method)
        fref_out = fref_in[
            np.logical_and(fref_in >= fref_min,
                           fref_in < fref_max)]
        if len(fref_out) == 0:
            self.check_input_limits(fref_in, fref_min, fref_max)
            raise Exception("fref_out is empty. This can happen if the "
                            "waveform has insufficient identifiable "
                            "pericenters/apocenters.")
        return fref_out

    def make_diagnostic_plots(
            self,
            add_help_text=True,
            usetex=False,
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
        - omega_gw vs time with the pericenters and apocenters shown. This
          would show if the method is missing any pericenters/apocenters or
          selecting one which is not a pericenter/apocenter
        - deltaPhi_orb(i)/deltaPhi_orb(i-1), where deltaPhi_orb is the
          change in orbital phase from the previous extrema to the ith extrema.
          This helps to look for missing extrema, as there will be a drastic
          (roughly factor of 2) change in deltaPhi_orb(i) if there is a missing
          extrema, and the ratio will go from ~1 to ~2.
        - omega_gw_average, where the omega_gw average computed using the
          `omega_gw_averaging_method` is plotted as a function of time.
          omega_gw_average is used to get the reference time for a given
          reference frequency. Therefore, it should be a strictly monotonic
          function of time.

        Additionally, we plot the following if data for zero eccentricity is
        provided and method is not residual method

        - residual amp_gw vs time with the location of pericenters and
          apocenters shown.
        - residual omega_gw vs time with the location of pericenters and
          apocenters shown.

        If the method itself uses residual data, then add one plot for

        - data that is not being used for finding extrema.
          For example, if method is ResidualAmplitude then plot residual omega
          and vice versa.  These two plots further help in understanding any
          unwanted feature in the measured eccentricity vs time plot. For
          example, non smoothness in the residual omega_gw would indicate that
          the data in omega_gw is not good which might be causing glitches in
          the measured eccentricity plot.

        Finally, plot

        - data that is being used for finding extrema.

        Parameters
        ----------
        add_help_text : bool, default: True
            If True, add text to describe features in the plot.
        usetex : bool, default: False
            If True, use TeX to render texts.
        style : str
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for "APS" journals, use style="APS".  For
            showing plots in a jupyter notebook, use "Notebook" so that plots
            are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.  If None, then uses "Notebook"
            when twocol is False and uses "APS" if twocol is True.
            Default is None.
        use_fancy_settings : bool, default: True
            Use fancy settings for matplotlib to make the plot look prettier.
            See plot_settings.py for more details.
        twocol : bool, default: False
            Use a two column grid layout.
        **kwargs
            kwargs to be passed to plt.subplots()

        Returns
        -------
        fig
            Figure object.
        axarr
            Axes object.
        """
        # Make a list of plots we want to add
        list_of_plots = [self.plot_eccentricity,
                         self.plot_mean_anomaly,
                         self.plot_omega_gw,
                         self.plot_data_used_for_finding_extrema,
                         self.plot_decc_dt,
                         self.plot_phase_diff_ratio_between_extrema,
                         self.plot_omega_gw_average]
        if "hlm_zeroecc" in self.dataDict:
            # add residual amp_gw plot
            if self.method != "ResidualAmplitude":
                list_of_plots.append(self.plot_residual_amp_gw)
            # add residual omega_gw plot
            if self.method != "ResidualFrequency":
                list_of_plots.append(self.plot_residual_omega_gw)

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
            usetex=False,
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
            Default is False.
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
            usetex=False,
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
            Default is False.
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
            usetex=False,
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
            Default is False.
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

    def plot_omega_gw(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot omega_gw, the locations of the apocenters and pericenters.

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
            Default is False.
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
                self.omega_gw_pericenters_interp(self.t_for_checks),
                c=colorsDict["pericenter"],
                label=labelsDict["omega_gw_pericenters"],
                **kwargs)
        ax.plot(self.t_for_checks, self.omega_gw_apocenters_interp(
            self.t_for_checks),
                c=colorsDict["apocenter"],
                label=labelsDict["omega_gw_apocenters"],
                **kwargs)
        ax.plot(self.t, self.omega_gw,
                c=colorsDict["default"], label=labelsDict["omega_gw"])
        ax.plot(self.t[self.pericenters_location],
                self.omega_gw[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="")
        ax.plot(self.t[self.apocenters_location],
                self.omega_gw[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="")
        # set reasonable ylims
        ymin = min(self.omega_gw)
        ymax = max(self.omega_gw)
        pad = 0.05 * ymax  # 5 % buffer for better visibility
        ax.set_ylim(ymin - pad, ymax + pad)
        # add help text
        if add_help_text:
            if usetex:
                help_text = (
                    r"\noindent To avoid extrapolation, first and last\\"
                    r"extrema are excluded when\\"
                    r"evaluating $\omega_{a}$/$\omega_{p}$ interpolants")
            else:
                help_text = (
                    "To avoid extrapolation, first and last\n"
                    "extrema are excluded when\n"
                    r"evaluating $\omega_{a}$/$\omega_{p}$ interpolants")
            ax.text(
                0.22,
                0.98,
                help_text,
                ha="left",
                va="top",
                transform=ax.transAxes)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(self.get_label_for_plots("omega"))
        ax.legend(frameon=True,
                  handlelength=1, labelspacing=0.2, columnspacing=1)
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_omega_gw_average(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            plot_omega_gw=True,
            plot_orbit_averaged_omega_gw_between_extrema=False,
            **kwargs):
        """Plot omega_gw_average.

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
            Default is False.
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
        plot_omega_gw: bool
            If True, plot omega_gw also. Default is True.
        plot_orbit_averaged_omega_gw_between_extrema: bool
            If True and method is orbit_averaged_omega_gw, plot the the orbit
            averaged omega_gw between the extrema as well. Default is False.

        Returns:
        --------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        # check if omega_gw_average is already available. If not
        # available, compute it.
        if self.omega_gw_average is None:
            self.t_for_omega_gw_average, self.omega_gw_average = self.get_omega_gw_average()
        ax.plot(self.t_for_omega_gw_average,
                self.omega_gw_average,
                c=colorsDict["default"],
                label="omega_gw_average",
                **kwargs)
        if plot_omega_gw:
            ax.plot(self.t, self.omega_gw,
                    c='k',
                    alpha=0.4,
                    lw=0.5,
                    label=self.get_label_for_plots("omega"))
        if (self.extra_kwargs["omega_gw_averaging_method"] == "orbit_averaged_omega_gw" and
            plot_orbit_averaged_omega_gw_between_extrema):
            ax.plot(self.t_average_apocenters,
                    self.orbit_averaged_omega_gw_apocenters,
                    c=colorsDict["apocenter"],
                    marker=".", ls="",
                    label=labelsDict["orbit_averaged_omega_gw_apocenters"])
            ax.plot(self.t_average_pericenters,
                    self.orbit_averaged_omega_gw_pericenters,
                    c=colorsDict["pericenter"],
                    marker=".", ls="",
                    label=labelsDict["orbit_averaged_omega_gw_pericenters"])
        # set reasonable ylims
        ymin = min(self.omega_gw)
        ymax = max(self.omega_gw)
        pad = 0.05 * ymax  # 5 % buffer for better visibility
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Averaged frequency")
        # add help text
        if add_help_text:
            ax.text(
                0.35,
                0.98,
                (r"omega_gw_average should be "
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

    def plot_amp_gw(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot amp_gw, the locations of the apocenters and pericenters.

        This would show if the method is missing any pericenters/apocenters or
        selecting one which is not a pericenter/apocenter.

        Parameters
        ----------
        fig :
            Figure object to add the plot to. If None, initiates a new figure
            object. Default is None.
        ax :
            Axis object to add the plot to. If None, initiates a new axis
            object. Default is None.
        add_help_text : bool, default: True
            If True, add text to describe features in the plot.
        usetex : bool, default: False
            If True, use TeX to render texts.
        style : str, default: ``Notebook``
            Set font size, figure size suitable for particular use case. For
            example, to generate plot for ``APS`` journals, use ``style='APS'``.
            For showing plots in a jupyter notebook, use ``Notebook`` so that
            plots are bigger and fonts are appropriately larger and so on.  See
            plot_settings.py for more details.
        use_fancy_settings : bool, default: True
            Use fancy settings for matplotlib to make the plot look prettier.
            See :py:func:`plot_settings.use_fancy_plotsettings` for more
            details.

        Returns
        -------
        fig, ax
        """
        if fig is None or ax is None:
            figNew, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 4))
        if use_fancy_settings:
            use_fancy_plotsettings(usetex=usetex, style=style)
        ax.plot(self.t, self.amp_gw,
                c=colorsDict["default"], label=labelsDict["amp_gw"])
        ax.plot(self.t[self.pericenters_location],
                self.amp_gw[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label=labelsDict["pericenters"])
        ax.plot(self.t[self.apocenters_location],
                self.amp_gw[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label=labelsDict["apocenters"])
        # set reasonable ylims
        ymin = min(self.amp_gw)
        ymax = max(self.amp_gw)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(self.get_label_for_plots("amp"))
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
            usetex=False,
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
            Default is False.
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

    def plot_residual_omega_gw(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual omega_gw, the locations of the apocenters and pericenters.

        Useful to look for bad omega_gw data near merger.
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
            Default is False.
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
        ax.plot(self.t, self.res_omega_gw, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_omega_gw[self.pericenters_location],
                marker=".", ls="", label=labelsDict["pericenters"],
                c=colorsDict["pericenter"])
        ax.plot(self.t[self.apocenters_location],
                self.res_omega_gw[self.apocenters_location],
                marker=".", ls="", label=labelsDict["apocenters"],
                c=colorsDict["apocenter"])
        # set reasonable ylims
        ymin = min(self.res_omega_gw)
        ymax = max(self.res_omega_gw)
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim  # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(self.get_label_for_plots("res_omega"))
        ax.legend(frameon=True, loc="center left",
                  handlelength=1, labelspacing=0.2, columnspacing=1)
        if fig is None or ax is None:
            return figNew, ax
        else:
            return ax

    def plot_residual_amp_gw(
            self,
            fig=None,
            ax=None,
            add_help_text=True,
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            **kwargs):
        """Plot residual amp_gw, the locations of the apocenters and pericenters.

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
            Default is False.
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
        ax.plot(self.t, self.res_amp_gw, c=colorsDict["default"])
        ax.plot(self.t[self.pericenters_location],
                self.res_amp_gw[self.pericenters_location],
                c=colorsDict["pericenter"],
                marker=".", ls="", label=labelsDict["pericenters"])
        ax.plot(self.t[self.apocenters_location],
                self.res_amp_gw[self.apocenters_location],
                c=colorsDict["apocenter"],
                marker=".", ls="", label=labelsDict["apocenters"])
        # set reasonable ylims
        ymin = min(self.res_amp_gw)
        ymax = max(self.res_amp_gw)
        # we want to make the ylims symmetric about y=0
        ylim = max(ymax, -ymin)
        pad = 0.05 * ylim  # 5 % buffer for better visibility
        ax.set_ylim(-ylim - pad, ylim + pad)
        ax.set_xlabel(labelsDict["t"])
        ax.set_ylabel(self.get_label_for_plots("res_amp"))
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
            usetex=False,
            style="Notebook",
            use_fancy_settings=True,
            add_vline_at_tref=True,
            **kwargs):
        """Plot the data that is being used.

        Also the locations of the apocenters and pericenters.

        Parameters
        ----------
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
            Default is False.
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

    def get_label_for_plots(self, data_str):
        """Get appropriate label for plots.

        Depending on whether system is precessing or not, generate appropriate
        labels to use in plots.

        Parameters:
        -----------
        data_str: str
            A string representing the data for which label will be generated.
            It must be one of [`amp`, `omega`, `res_amp`, `res_omega`].

        Returns:
        --------
        Appropriate label for the input `data_str`.
        """
        allowd_data_str_list = ["amp", "omega", "res_amp", "res_omega"]
        if data_str not in allowd_data_str_list:
            raise KeyError(f"`data_str` must be one of {allowd_data_str_list}")
        return (labelsDict[data_str + "_gw"]
                + " = "
                + labelsDict[data_str + "22" + ("_copr_symm" if self.precessing else "")])

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
        such systems the amp_gw/omega_gw data between the pericenters is almost
        flat and hard to find the local minima.

        Returns
        -------
        locations of apocenters : array-like
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
        eccDefinition.get_width_for_peak_finder_from_phase_gw
        for why this is useful to set when calling scipy.signal.find_peaks.

        This function gets an appropriate width by scaling it with the time
        steps in the time array of the waveform data.  NOTE: As the function
        name mentions, this should be used only for dimensionless units. This
        is because the `width_for_unit_timestep` parameter refers to unit
        timestep in units of M. It is the fiducial width to use if the time
        step is 1M. If using time in seconds, this would depend on the total
        mass.

        Parameters
        ----------
        width_for_unit_timestep : int
            Width to use when the time step in the wavefrom data is 1.

        Returns
        -------
        width:
            Minimal width to separate consecutive peaks.
        """
        return int(width_for_unit_timestep / (self.t[1] - self.t[0]))
