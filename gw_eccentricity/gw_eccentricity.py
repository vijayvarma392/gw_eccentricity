"""gw_eccentricity.
========

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
                         return_gwecc_object=False,
                         spline_kwargs=None,
                         extra_kwargs=None):
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

    method: str
        Which waveform data to use for finding extrema. Options are:
        - "Amplitude": Finds extrema of Amp22(t).
        - "Frequency": Finds extrema of omega22(t).
        - "ResidualAmplitude": Finds extrema of resAmp22(t), the residual
          amplitude, obtained by subtracting the Amp22(t) of the quasi-circular
          counterpart from the Amp22(t) of the eccentric waveform. The
          quasi-circular counterpart is described in the documentation of
          dataDict below.
        - "ResidualFrequency": Finds extrema of resomega22(t), the residual
          frequency, obtained by subtracting the omega22(t) of the
          quasi-circular counterpart from the omega22(t) of the eccentric
          waveform.
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

    return_gwecc_object: bool
        If True, returns the eccDefinition object used to compute the
        eccentricity and mean anomaly. This can be used to make diagnostic
        plots.
        Default is False.

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
              the final time over which the spline is constructed is always starts
              after the first extrema and end before the last extrema.
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

    Returns:
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

    gwecc_object:
        eccDefinition object used to compute eccentricity. This can be used to
        make diagnostic plots. Only returned if return_gwecc_object is True.
    """
    available_methods = get_available_methods(return_dict=True)

    if method in available_methods:
        gwecc_object = available_methods[method](dataDict,
                                               spline_kwargs=spline_kwargs,
                                               extra_kwargs=extra_kwargs)

        tref_or_fref_out, ecc_ref, mean_ano_ref = gwecc_object.measure_ecc(
            tref_in=tref_in, fref_in=fref_in)
        if not return_gwecc_object:
            return tref_or_fref_out, ecc_ref, mean_ano_ref
        else:
            return tref_or_fref_out, ecc_ref, mean_ano_ref, gwecc_object
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {list(available_methods.keys())}")
