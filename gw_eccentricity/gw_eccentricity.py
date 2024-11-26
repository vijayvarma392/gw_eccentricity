"""gw_eccentricity wrapper to call different eccDefinition methods.

Measure eccentricity and mean anomaly from gravitational waves.
See our paper https://arxiv.org/abs/2302.11257 and
https://pypi.org/project/gw_eccentricity for more details.
"""
__copyright__ = "Copyright (C) 2023 Md Arif Shaikh, Vijay Varma, Harald Pfeiffer"
__email__ = "arifshaikh.astro@gmail.com, vijay.varma392@gmail.com"
__status__ = "testing"
__author__ = "Md Arif Shaikh, Vijay Varma, Harald Pfeiffer"
__version__ = "1.0.4"
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from .eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude
from .eccDefinitionUsingFrequency import eccDefinitionUsingFrequency
from .eccDefinitionUsingAmplitudeFits import eccDefinitionUsingAmplitudeFits
from .eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from .eccDefinitionUsingResidualAmplitude import eccDefinitionUsingResidualAmplitude
from .eccDefinitionUsingResidualFrequency import eccDefinitionUsingResidualFrequency


def get_available_methods(return_dict=False):
    """Get all available eccDefinition methods.

    If return_dict is True, returns a dictionary of methods.
    Else, just returns a list of method names.
    """
    methods = {
        "Amplitude": eccDefinitionUsingAmplitude,
        "Frequency": eccDefinitionUsingFrequency,
        "ResidualAmplitude": eccDefinitionUsingResidualAmplitude,
        "ResidualFrequency": eccDefinitionUsingResidualFrequency,
        "AmplitudeFits": eccDefinitionUsingAmplitudeFits,
        "FrequencyFits": eccDefinitionUsingFrequencyFits
    }

    if return_dict:
        return methods
    else:
        return list(methods.keys())


def measure_eccentricity(tref_in=None,
                         fref_in=None,
                         method="Amplitude",
                         dataDict=None,
                         num_orbits_to_exclude_before_merger=2,
                         precessing=False,
                         frame="inertial",
                         extra_kwargs=None):
    """Measure eccentricity and mean anomaly from a gravitational waveform.

    Eccentricity is measured using the GW frequency omega_gw(t) =
    d(phase_gw)/dt. Throughout this documentation, we will refer to phase_gw,
    omega_gw and amp_gw. For nonprecessing systems, these quantities are simply
    the corresponding values of the (2, 2) mode,

    amp_gw = amp22, phase_gw = phase22 and omega_gw = omega22.

    On the other hand, for precessing systems, we use Eq.(48) and (49) of
    arXiv:1701.00550 to define amp_gw and phase_gw. amp_gw [phase_gw] is
    defined using a symmetric [antisymmetric] combination of
    amplitude [phase] of (2, 2) and (2, -2) mode in the coprecessing frame,

    amp_gw = (1/2) * (amp(2, 2) + amp(2, -2))
    phase_gw = (1/2) * (phase(2, 2) - phase(2, -2))
    omega_gw = d(phase_gw)/dt.

    These quantities reduce to the corresponding (2, 2) mode data when the
    system is nonprecessing, but we treat nonprecessing cases differently by
    allowing the user to include only the (2, 2) mode. See
    `eccDefinition.get_amp_phase_omega_gw` for more details.

    We currently only allow time-domain waveforms. We evaluate omega_gw(t) at
    pericenter times, t_pericenters, and build a spline interpolant
    omega_gw_pericenters(t) using those data points. Similarly, we build
    omega_gw_apocenters(t) using omega_gw(t) at the apocenter times,
    t_apocenters.

    Using omega_gw_pericenters(t) and omega_gw_apocenters(t), we first compute
    e_omega_gw(t), as described in Eq.(4) of arXiv:2302.11257 (e_omega_gw is
    called e_omega_22 in the paper). We then use e_omega_gw(t) to compute the
    eccentricity e_gw(t) using Eq.(8) of arXiv:2302.11257. Mean anomaly is
    defined using t_pericenters, as described in Eq.(10) of arXiv:2302.11257.

    To find t_pericenters/t_apocenters, one can look for extrema in different
    waveform data, like omega_gw(t) or amp_gw(t). Pericenters correspond to
    the local maxima, while apocenters correspond to the local minima in the
    data. The method option (described below) lets the user pick which waveform
    data to use to find t_pericenters/t_apocenters.

    Parameters
    ----------
    tref_in
        Input reference time at which to measure eccentricity and mean anomaly.
        Can be a single float or an array.

    fref_in
        Input reference GW frequency at which to measure the eccentricity and
        mean anomaly. Can be a single float or an array. Only one of
        tref_in/fref_in should be provided.

        Given an fref_in, we find the corresponding tref_in such that
        omega_gw_average(tref_in) = 2 * pi * fref_in. Here, omega_gw_average(t)
        is a monotonically increasing average frequency obtained from the
        instantaneous omega_gw(t). omega_gw_average(t) defaults to the orbit
        averaged omega_gw, but other options are available (see
        omega_gw_averaging_method below).

        Eccentricity and mean anomaly measurements are returned on a subset of
        tref_in/fref_in, called tref_out/fref_out, which are described below.
        If dataDict is provided in dimensionless units, tref_in should be in
        units of M and fref_in should be in units of cycles/M. If dataDict is
        provided in MKS units, t_ref should be in seconds and fref_in should be
        in Hz.

    method : str, default: ``Amplitude``
        Which waveform data to use for finding extrema. Options are:

        - "Amplitude": Finds extrema of amp_gw(t).
        - "Frequency": Finds extrema of omega_gw(t).
        - "ResidualAmplitude": Finds extrema of resamp_gw(t), the residual
          amplitude, obtained by subtracting the amp_gw(t) of the quasicircular
          counterpart from the amp_gw(t) of the eccentric waveform. The
          quasicircular counterpart is described in the documentation of
          dataDict below.
        - "ResidualFrequency": Finds extrema of resomega_gw(t), the residual
          frequency, obtained by subtracting the omega_gw(t) of the
          quasicircular counterpart from the omega_gw(t) of the eccentric
          waveform.
        - "AmplitudeFits": Uses amp_gw(t) and iteratively subtracts a
          PN-inspired fit of the extrema of amp_gw(t) from it, and finds extrema
          of the residual.
        - "FrequencyFits": Uses omega_gw(t) and iteratively subtracts a
          PN-inspired fit of the extrema of omega_gw(t) from it, and finds
          extrema of the residual.

        The available list of methods can be also obtained from
        gw_eccentricity.get_available_methods().
        Detailed description of these methods can be found in Sec. III of
        arXiv:2302.11257.

        The Amplitude and Frequency methods can struggle for very small
        eccentricities, especially near the merger, as the
        secular amplitude/frequency growth dominates the modulations due to
        eccentricity, making extrema finding difficult.

        The ResidualAmplitude/ResidualFrequency/AmplitudeFits/FrequencyFits
        methods avoid this limitation by removing the secular growth before
        finding extrema. However, methods that use the frequency for finding
        extrema (Frequency/ResidualFrequency/FrequencyFits) can be more
        sensitive to junk radiation in NR data.

        Therefore, the recommended methods are
        ResidualAmplitude/AmplitudeFits/Amplitude

    dataDict : dict
        Dictionary containing waveform modes dict, time etc. Should follow the
        format::

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

        Apart from specifying "hlm" or "amplm" and "phaselm", the user can also
        provide "omegalm". If the "omegalm" key is not explicitly provided, it
        is computed from the given "hlm" or "phaselm" using finite difference
        method.

        The keys with suffix "zeroecc" are only required for
        `ResidualAmplitude` and `ResidualFrequency` methods, where
        "t_zeroecc" and one of the following is to be provided:

        1. "hlm_zeroecc" OR
        2. "amplm_zeroecc" and "phaselm_zeroecc"
            but not both 1 and 2 at the same time.

        Note that providing "hlm_zeroecc" and "amplm_zeroecc" or
        "phaselm_zeroecc" at the same time will raise error.
        Similar to "omegalm", the user can also provide "omegalm_zeroecc". If
        it is not provided in `dataDict`, it is computed from the given
        "hlm_zeroecc" or "phaselm_zeroecc" using finite difference method.

        If zeroecc data are provided for methods other than `ResidualAmplitude`
        and `ResidualFrequency`, they are used for additional diagnostic plots,
        which can be helpful for all methods.

        Any keys in `dataDict` other than the recognized ones will be ignored,
        with a warning.

        The recognized keys are:

        - "t": 1d array of times.

            - Should be uniformly sampled, with a small enough time step so
              that omega_gw(t) can be accurately computed, if necessary. We use
              a 4th-order finite difference scheme. In dimensionless units, we
              recommend a time step of dtM = 0.1M to be conservative, but one
              may be able to get away with larger time steps like dtM = 1M. The
              corresponding time step in seconds would be dtM * M *
              lal.MTSUN_SI, where M is the total mass in Solar masses.
            - We do not require the waveform peak amplitude to occur at any
              specific time, but tref_in should follow the same convention
              for peak time as "t".

        - "hlm": Dictionary of waveform modes associated with "t".
            Should have the format::

                modeDict = {(l1, m1): h_{l1, m1},
                            (l2, m2): h_{l2, m2},
                            ...
                           }

            where h_{l, m} is a 1d complex array representing the (l, m)
            waveform mode. Should contain at least the (2, 2) mode, but more
            modes can be included, as indicated by the ellipsis '...'  above.

        - "amplm": Dictionary of amplitudes of waveform modes associated with
          "t". Should have the same format as "hlm", except that the amplitude
          is real.

        - "phaselm": Dictionary of phases of waveform modes associated with
          "t". Should have the same format as "hlm", except that the phase is
          real. The phaselm is related to hlm as hlm = amplm * exp(- i phaselm)
          ensuring that the phaselm is monotonically increasing for m > 0
          modes.

        - "omegalm": Dictionary of the frequencies of the waveform modes
          associated with "t". Should have the same format as "hlm", except
          that the omegalm is real. omegalm is related the phaselm (see above)
          as omegalm = d/dt phaselm, which means that the omegalm is positive
          for m > 0 modes.

        - "t_zeroecc" and "hlm_zeroecc":

            - Same as above, but for the quasicircular counterpart to the
              eccentric waveform. The quasicircular counterpart can be obtained
              by evaluating a waveform model by keeping the rest of the binary
              parameters fixed (same as the ones used to generate "hlm") but
              setting the eccentricity to zero. For NR, if such a quasicircular
              counterpart is not available, we recommend using quasicircular
              models like NRHybSur3dq8 or IMRPhenomT, depending on the mass
              ratio and spins.
            - "t_zeroecc" should be uniformly spaced, but does not have to
              follow the same time step as that of "t", as long as the step
              size is small enough to compute the frequency. Similarly, peak
              time does not have to match that of "t".
            - We require that "hlm_zeroecc" be at least as long as "hlm" so
              that residual amplitude/frequency can be computed.

        - "amplm_zeroecc", "phaselm_zeroecc" and "omegalm_zeroecc":
            Same as "amplm", "phaselm" and "omegalm", respectively, but for the
            quasicircular counterpart to the eccentric waveform.


    num_orbits_to_exclude_before_merger:
        Can be None or a non negative number.
        If None, the full waveform data (even post-merger) is used for
        finding extrema, but this might cause interpolation issues.
        For a non negative num_orbits_to_exclude_before_merger, that
        many orbits before the merger are excluded when finding extrema.
        If your waveform does not have a merger (e.g. PN/EMRI), use
        num_orbits_to_exclude_before_merger = None.

        The default value is chosen via an investigation on a set of NR
        waveforms. See the following wiki page for more details,
        https://github.com/vijayvarma392/gw_eccentricity/wiki/NR-investigation-to-set-default-number-of-orbits-to-exclude-before-merger
        Default: 2.

    precessing: bool, default=False
        Indicates whether the system is precessing. For precessing systems, the
        (2, 2) and (2, -2) modes in the coprecessing frame are required to
        compute `amp_gw`, `phase_gw`, and `omega_gw` (see
        `get_amp_phase_omega_gw`), which are used to determine eccentricity.

        For precessing systems, waveform modes in the coprecessing frame must
        be provided. This can be done in two ways:
            - Set `frame="coprecessing"` and supply the coprecessing modes
                directly via `dataDict`.
            - Set `frame="inertial"` and provide the inertial frame modes
                via `dataDict`. In this case, the modes in `dataDict` are
                rotated internally (see `transform_inertial_to_coprecessing`)
                before further computation. To get the coprecessing modes from
                the inertial modes accurately, `dataDict` must include all
                modes for `ell=2`, i.e., (2, -2), (2, -1), (2, 0), (2, 1) and
                (2, 2).

        For nonprecessing systems, the inertial and coprecessing frames are
        equivalent, so there is no distinction. For nonprecessing systems, it
        is sufficient to include only the (2, 2) mode.

        Default is `False`, indicating the system is nonprecessing.

    frame: str, default="inertial"
        Specifies the reference frame for the modes in `dataDict`.
            
        This parameter determines the frame in which the mode data is provided.
        It is especially relevant for measuring eccentricity in precessing
        systems, as the choice of reference frame affects the interpretation of
        the modes. Use this in conjunction with the `precessing` parameter (see
        its documentation for more details) to ensure appropriate handling of
        the data.

        Currently `frame` can be "inertial" or "coprecessing".

        Default value is "inertial".

    extra_kwargs: A dict of any extra kwargs to be passed. Allowed kwargs are:
        spline_kwargs:
            Dictionary of arguments to be passed to the spline interpolation
            routine (scipy.interpolate.InterpolatedUnivariateSpline) used to
            compute quantities like omega_gw_pericenters(t) and
            omega_gw_apocenters(t).
            Defaults are set using utils.get_default_spline_kwargs

        extrema_finding_kwargs:
            Dictionary of arguments to be passed to the extrema finder,
            scipy.signal.find_peaks.

            The Defaults are the same as those of scipy.signal.find_peaks,
            except for the "width", which sets the minimum allowed "full width
            at half maximum" for the extrema. Setting this can help avoid false
            extrema in noisy data (for example, due to junk radiation in
            NR). The default for "width" is set using phase_gw(t) near the
            merger. For nonprecessing systems, phase_gw = phase of the (2, 2)
            mode.  Starting from 4 cycles of the (2, 2) mode before the merger,
            we find the number of time steps taken to cover 2 cycles, let's
            call this "the gap". Note that 2 cycles of the (2, 2) mode are
            approximately one orbit, so this allows us to approximate the
            smallest gap between two pericenters/apocenters. However, to be
            conservative, we divide this gap by 4 and set it as the width
            parameter for find_peaks. See
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
            Default is False.

        omega_gw_averaging_method:
            Options for obtaining omega_gw_average(t) from the instantaneous
            omega_gw(t). For nonprecessing systems, omega_gw(t) is the same as
            the omega(t) of the (2, 2) mode. See `get_amp_phase_omega_gw` for
            more details:

            - "orbit_averaged_omega_gw": First, orbit averages are obtained at
              each pericenter by averaging omega_gw(t) over the time from the
              current pericenter to the next one. This average value is
              associated with the time at mid point between the current and the
              next pericenter. Similarly orbit averages are computed at
              apocenters.  Finally, a spline interpolant is constructed between
              all of these orbit averages at extrema locations. However, the
              final time over which the spline is constructed is constrained to
              be between tmin_for_fref and tmax_for_fref which are close to
              tmin and tmax, respectively. See eccDefinition.get_fref_bounds()
              for details.
            - "mean_of_extrema_interpolants": The mean of
              omega_gw_pericenters(t) and omega_gw_apocenters(t) is used as a
              proxy for the average frequency.
            - "omega_gw_zeroecc": omega_gw(t) of the quasicircular counterpart
              is used as a proxy for the average frequency. This can only be
              used if "t_zeroecc" and "hlm_zeroecc" are provided in dataDict.

            See Sec. IID of arXiv:2302.11257 for more detail description of
            average omega_gw. Default is "orbit_averaged_omega_gw".

        treat_mid_points_between_pericenters_as_apocenters:
            If True, instead of trying to find apocenter locations by looking
            for local minima in the data, we simply find the midpoints between
            pericenter locations and treat them as apocenters. This is helpful
            for eccentricities ~1 where pericenters are easy to find but
            apocenters are not.
            Default: False.

        kwargs_for_fits_methods:
            Extra kwargs to be passed to FrequencyFits and AmplitudeFits
            methods. See
            eccDefinitionUsingFrequencyFits.get_default_kwargs_for_fits_methods
            for allowed keys.

        return_zero_if_small_ecc_failure : bool, default=False
            The code normally raises an exception if sufficient number of
            extrema are not found. This can happen for various reasons
            including when the eccentricity is too small for some methods (like
            the Amplitude method) to measure. See e.g. Fig.4 of
            arxiv.2302.11257. If no extrema are found, we check whether the
            following two conditions are satisfied.

            1. `return_zero_if_small_ecc_failure` is set to `True`.
            2. The waveform is at least (5 +
              `num_obrits_to_exclude_before_merger`) orbits long. By default,
              `num_obrits_to_exclude_before_merger` is set to 2, meaning that 2
              orbits are removed from the waveform before it is used by the
              extrema finding routine. Consequently, in the default
              configuration, the original waveform in the input `dataDict` must
              have a minimum length of 7 orbits.

            If both of these conditions are met, we assume that small
            eccentricity is the cause, and set the returned eccentricity and
            mean anomaly to zero.
            USE THIS WITH CAUTION!

    Returns
    -------
    A dictionary containing the following keys
    tref_out:
        tref_out is the output reference time at which eccentricity and mean
        anomaly are measured.
        tref_out is included in the returned dictionary only when tref_in is
        provided.
        Units of tref_out are the same as that of tref_in.

        tref_out is set as
        tref_out = tref_in[tref_in >= tmin & tref_in <= tmax],
        where::

            tmax = min(t_pericenters[-1], t_apocenters[-1])
            tmin = max(t_pericenters[0], t_apocenters[0])

        As eccentricity measurement relies on the interpolants
        omega_gw_pericenters(t) and omega_gw_apocenters(t), the above cutoffs
        ensure that we only compute the eccentricity where both
        omega_gw_pericenters(t) and omega_gw_apocenters(t) are within their
        bounds.

    fref_out:
        fref_out is the output reference frequency at which eccentricity and
        mean anomaly are measured.
        fref_out is included in the returned dictionary only when fref_in is
        provided.
        Units of fref_out are the same as that of fref_in.

        fref_out is set as
        fref_out = fref_in[fref_in >= fref_min && fref_in <= fref_max],
        where fref_min/fref_max are minimum/maximum allowed reference
        frequency, with fref_min = omega_gw_average(tmin_for_fref)/2/pi
        and fref_max = omega_gw_average(tmax_for_fref)/2/pi.
        tmin_for_fref/tmax_for_fref are close to tmin/tmax, see
        eccDefinition.get_fref_bounds() for details.

    eccentricity:
        Measured eccentricity at tref_out/fref_out. Same type as
        tref_out/fref_out.

    mean_anomaly:
        Measured mean anomaly at tref_out/fref_out. Same type as
        tref_out/fref_out.

    gwecc_object:
        eccDefinition object used to compute eccentricity. This can be used to
        make diagnostic plots, call member functions to do further
        investigation or access variables computed internally for measuring
        eccentricity and mean anomaly.
    """
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](
            dataDict, num_orbits_to_exclude_before_merger,
            precessing, frame, extra_kwargs)
        return_dict = gwecc_object.measure_ecc(
            tref_in=tref_in, fref_in=fref_in)
        return_dict.update({"gwecc_object": gwecc_object})
        return return_dict
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {list(available_methods.keys())}")
