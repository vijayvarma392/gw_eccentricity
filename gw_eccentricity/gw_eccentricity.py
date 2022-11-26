"""gw_eccentricity.

Measure eccentricity and mean anomaly from gravitational waves.
See our paper https://arxiv.org/abs/xxxx.xxxx and
https://pypi.org/project/gw_eccentricity for more details.
FIXME ARIF: Add arxiv link when available.
"""
__copyright__ = "Copyright (C) 2022 Md Arif Shaikh, Vijay Varma"
__email__ = "arifshaikh.astro@gmail.com, vijay.varma392@gmail.com"
__status__ = "testing"
__author__ = "Md Arif Shaikh, Vijay Varma"
__version__ = "0.0.dev"
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
                         extra_kwargs=None):
    """Measure eccentricity and mean anomaly from a gravitational waveform.

    Eccentricity is measured using the GW frequency omega22(t) = dphi22(t)/dt,
    where phi22(t) is the phase of the (2, 2) waveform mode. We currently only
    allow time-domain, nonprecessing waveforms. We evaluate omega22(t) at
    pericenter times, t_pericenters, and build a spline interpolant
    omega22_pericenters(t) using those points. Similarly, we build
    omega22_apocenters(t) using omega22(t) at the apocenter times,
    t_apocenters. Finally, eccentricity is defined using omega22_pericenters(t)
    and omega22_apocenters(t), as described in Eq.(1) of arxiv:xxxx.xxxx. Mean
    anomaly is defined using t_pericenters, as described in Eq.(2) of
    arxiv:xxxx.xxxx.

    FIXME ARIF: In the above text, fill in arxiv number when available. Make
    sure the above Eq numbers are right, once the paper is finalized.

    To find t_pericenters/t_apocenters, one can look for extrema in different
    waveform data, like omega22(t) or Amp22(t), the amplitude of the (2, 2)
    mode. Pericenters correspond to peaks, while apocenters correspond to
    troughs in the data. The method option (described below) lets you pick
    which waveform data to use to find t_pericenters/t_apocenters.

    Parameters:
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
        is a monotonically increasing average frequency obtained from the
        instantaneous omega22(t). omega22_average(t) defaults to the mean
        motion, but other options are available (see omega22_averaging_method
        below).

        Eccentricity and mean anomaly measurements are returned on a subset of
        tref_in/fref_in, called tref_out/fref_out, which are described below.
        If dataDict is provided in dimensionless units, tref_in should be in
        units of M and fref_in should be in units of cycles/M. If dataDict is
        provided in MKS units, t_ref should be in seconds and fref_in should be
        in Hz.

    method: str
        Which waveform data to use for finding extrema. Options are:
        - "Amplitude": Finds extrema of Amp22(t).
        - "Frequency": Finds extrema of omega22(t).
        - "ResidualAmplitude": Finds extrema of resAmp22(t), the residual
          amplitude, obtained by subtracting the Amp22(t) of the quasicircular
          counterpart from the Amp22(t) of the eccentric waveform. The
          quasicircular counterpart is described in the documentation of
          dataDict below.
        - "ResidualFrequency": Finds extrema of resomega22(t), the residual
          frequency, obtained by subtracting the omega22(t) of the
          quasicircular counterpart from the omega22(t) of the eccentric
          waveform.
        - "AmplitudeFits": Uses Amp22(t) and iteratively subtracts a
          PN-inspired fitting function from it, and finds extrema of the
          residual.
        - "FrequencyFits": Uses omega22(t) and iteratively subtracts a
          PN-inspired fitting function from it, and finds extrema of the
          residual.
        Default is "Amplitude".
        Available list of methods can be also obtained from
        gw_eccentricity.get_available_methods().

        The Amplitude and Frequency methods can struggle for very small
        eccentricities (~1e-3), especially near the merger, as the secular
        amplitude/frequency growth dominates the modulations due to
        eccentricity, making extrema finding difficult. This is the main reason
        for using the residual methods,
        ResidualAmplitude/ResidualFrequency/FrequencyFits, which first remove
        the secular growth before finding extrema. However, methods that use
        the frequency for finding extrema
        (Frequency/ResidualFrequency/FrequencyFits) can be more sensitive to
        junk radiation in NR data.

        Therefore, the recommended methods are
        Amplitude/ResidualAmplitude/FrequencyFits.

    dataDict:
        Dictionary containing waveform modes dict, time etc. Should follow the
        format:
        dataDict = {"t": time,
                    "hlm": modeDict,
                    "t_zeroecc": time,
                    "hlm_zeroecc": modeDict,
                   },
        "t" and "hlm" are mandatory. "t_zeroecc" and "hlm_zeroecc" are only
        required for ResidualAmplitude and ResidualFrequency methods, but if
        provided, they are used for additional diagnostic plots, which can be
        helpful for all methods. Any other keys in dataDict will be ignored,
        with a warning.

        The recognized keys are:
        - "t": 1d array of times.
            - Should be uniformly sampled, with a small enough time step
              that omega22(t) can be accurately computed. We use a 4th-order
              finite difference scheme. In dimensionless units, we recommend a
              time step of dtM = 0.1M to be conservative, but you may be able
              to get away with larger time steps like dtM = 1M. The
              corresponding time step in seconds would be
              dtM * M * lal.MTSUN_SI, where M is the total mass in Solar
              masses.
            - We do not require the waveform peak amplitude to occur at any
              specific time, but tref_in should follow the same convention for
              peak time as "t".
        - "hlm": Dictionary of waveform modes associated with "t".
            - Should have the format:
                modeDict = {(l1, m1): h_{l1, m1},
                            (l2, m2): h_{l2, m2},
                            ...
                           },
                where h_{l, m} is a 1d complex array representing the (l, m)
                waveform mode. Should contain at least the (2, 2) mode, but
                more modes can be included, as indicated by the ellipsis '...'
                above.
        - "t_zeroecc" and "hlm_zeroecc":
            - Same as above, but for the quasicircular counterpart to the
              eccentric waveform. The quasicircular counterpart can be obtained
              by evaluating a waveform model by keeping the rest of the binary
              parameters fixed (same as the ones used to generate "hlm") but
              setting the eccentricity to zero. For NR, if such a quasicircular
              counterpart is not available, we recommend using quasicircular
              models like NRHybSur3dq8 or PhenomT, depending on the mass ratio
              and spins.
            - "t_zeroecc" should be uniformly spaced, but does not have to
              follow the same time step as that of "t", as long as the step
              size is small enough to compute the frequency. Similarly, peak
              time does not have to match that of "t".
            - We require that "hlm_zeroecc" be at least as long as "hlm" so
              that residual amplitude/frequency can be computed.

    num_orbits_to_exclude_before_merger:
        Can be None or a non negative number.
        If None, the full waveform data (even post-merger) is used for
        finding extrema, but this might cause interpolation issues.
        For a non negative num_orbits_to_exclude_before_merger, that
        many orbits prior to merger are excluded when finding extrema.
        If your waveform does not have a merger (e.g. PN/EMRI), use
        num_orbits_to_exclude_before_merger = None.

        The default value is chosen via an investigation on a set of NR
        waveforms. See the following wiki page for more details,
        https://github.com/vijayvarma392/gw_eccentricity/wiki/NR-investigation-to-set-default-number-of-orbits-to-exclude-before-merger
        Default: 2.

    extra_kwargs: A dict of any extra kwargs to be passed. Allowed kwargs are:
        spline_kwargs:
            Dictionary of arguments to be passed to the spline
            interpolation routine
            (scipy.interpolate.InterpolatedUnivariateSpline) used to
            compute omega22_pericenters(t) and omega22_apocenters(t).
            Defaults are set using utils.get_default_spline_kwargs

        extrema_finding_kwargs:
            Dictionary of arguments to be passed to the extrema finder,
            scipy.signal.find_peaks.
            The Defaults are the same as those of scipy.signal.find_peaks,
            except for the "width", which sets the minimum allowed "full width
            at half maximum" for the extrema. Setting this can help avoid
            false extrema in noisy data (for example, due to junk radiation in
            NR). The default for "width" is set using phi22(t) near the
            merger. Starting from 4 cycles of the (2, 2) mode before the
            merger, we find the number of time steps taken to cover 2 cycles,
            let's call this "the gap". Note that 2 cycles of the (2, 2) mode
            are approximately one orbit, so this allows us to approximate the
            smallest gap between two pericenters/apocenters. However, to be
            conservative, we divide this gap by 4 and set it as the width
            parameter for find_peaks. See
            eccDefinition.get_width_for_peak_finder_from_phase22 for more
            details.

        debug:
            Run additional sanity checks if debug is True.
            Default: True.

        omega22_averaging_method:
            Options for obtaining omega22_average(t) from the instantaneous
            omega22(t).
            - "orbit_averaged_omega22": First, orbit averages are obtained at each
              pericenter by averaging omega22(t) over the time from the current
              pericenter to the next one. This average value is associated with
              the time at mid point between the current and the next
              pericenter. Similarly orbit averages are computed at apocenters.
              Finally, a spline interpolant is constructed between all of these
              orbit averages at extrema locations. However, the final time over
              which the spline is constructed is constrained to be between
              tmin_for_fref and tmax_for_fref which are close to tmin and tmax,
              respectively. See eccDefinition.get_fref_bounds() for details.
            - "mean_of_extrema_interpolants":
              The mean of omega22_pericenters(t) and omega22_apocenters(t) is
              used as a proxy for the average frequency.
            - "omega22_zeroecc": omega22(t) of the quasicircular counterpart
              is used as a proxy for the average frequency. This can only be
              used if "t_zeroecc" and "hlm_zeroecc" are provided in dataDict.
            Default is "orbit_averaged_omega22".

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

    Returns:
    --------
    A dictionary containing the following keys
    tref_out:
        tref_out is the output reference time at which eccentricity and mean
        anomaly are measured.
        tref_out is included in the returned dictionary only when tref_in is provided.
        Units of tref_out is the same as that of tref_in.

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
        fref_out is the output reference frequency at which eccentricity and
        mean anomaly are measured.
        fref_out is included in the returned dictionary only when fref_in is provided.
        Units of fref_out is the same as that of fref_in.
        
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

    gwecc_object:
        eccDefinition object used to compute eccentricity. This can be used to
        make diagnostic plots and variables computed internally for measuring
        eccentricity and mean anomaly.
    """
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](
            dataDict, num_orbits_to_exclude_before_merger, extra_kwargs)
        return_dict = gwecc_object.measure_ecc(
            tref_in=tref_in, fref_in=fref_in)
        return_dict.update({"gwecc_object": gwecc_object})
        return return_dict
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {list(available_methods.keys())}")


def truncate_at_flow(flow,
                     m_max=None,
                     method="Amplitude",
                     dataDict=None,
                     extra_kwargs=None):
    """Truncate waveform at flow.

    Eccentric waveforms have a non-monotonic instantaneous frequency.
    Therefore, truncating the waveform by demanding that the truncated
    waveform should contain all frequencies that are greater than or equal
    to a given minimum frequency, say flow, must be done carefully since
    the instantaneous frequency can be equal to the given flow at multiple
    points in time.

    We need to find the time tlow, such that all the frequencies at t <
    tlow are < flow and therefore the t >= tlow part of the waveform would
    retain all the frequencies that are >= flow. Note that the t >= tlow
    part could contain some frequencies < flow but that is fine, all we
    need is not to lose any frequencies >= flow.

    This can be done by using the frequency interpolant omega22_p(t)
    through the pericenters because
    1. It is a monotonic function of time.
    2. If at a time tlow, omega22_p(tlow) * (m_max/2) = 2*pi*flow, then all
    frequencies >= flow will be included in the waveform truncated at
    t=tlow. The m_max/2 factor ensures that this statement is true for all
    modes, as the frequency of the h_{l, m} mode scales approximately as
    m/2 * omega_22/(2*pi).

    Thus, we find tlow such that omega22_p(tlow) * (m_max/2) = 2*pi*flow
    and truncate the waveform by keeping only the part where t >= tlow.

    Parameters:
    -----------
    flow: float
        Lower cutoff frequency to truncate the given waveform modes.
        The truncated waveform modes will contain all the frequencies >= flow.

    m_max: int
        Maximum m (index of h_{l, m}) to account for while setting the tlow
        for truncation.  If None, then it is set using the highest available
        m from the modes in the dataDict.
        Default is None.
    
    method: str
        Method to use for finding extrema locations.
        See under `measure_eccentricity` for more details.

    dataDict: dict
        Dictionary containing waveform modes.
        See under `measure_eccentricity` for more details.

    extra_kwargs: dict
        Dictionary of kwargs used to find the extrema or build the interpolants.
        See under `measure_eccentricity` for more details.

    Returns:
    truncatedDataDict: dict
        Dictionary containing waveform data truncated at flow.
    
    gwecc_object: obj
        Object used for truncating data.
    """
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](
            dataDict, num_orbits_to_exclude_before_merger=None,
            extra_kwargs=extra_kwargs)
        truncatedDataDict = gwecc_object.truncate_at_flow(flow, m_max)
        return truncatedDataDict, gwecc_object
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {list(available_methods.keys())}")

    
