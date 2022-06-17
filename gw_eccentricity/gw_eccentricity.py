"""gw_eccentricity.
========

Measure eccentricity and mean anomaly from gravitational waves.
See https://pypi.org/project/gw_eccentricity for more details.
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


def get_available_methods():
    """Get dictionary of available methods."""
    models = {
        "Amplitude": eccDefinitionUsingAmplitude,
        "Frequency": eccDefinitionUsingFrequency,
        "ResidualAmplitude": eccDefinitionUsingResidualAmplitude,
        "ResidualFrequency": eccDefinitionUsingResidualFrequency,
        "FrequencyFits": eccDefinitionUsingFrequencyFits
    }
    return models


def measure_eccentricity(tref_in=None,
                         fref_in=None,
                         dataDict=None,
                         method="Amplitude",
                         return_ecc_method=False,
                         spline_kwargs=None,
                         extra_kwargs=None):
    """Measure eccentricity and mean anomaly from a gravitational waveform.

    FIXME ARIF: Explain how the method works briefly, pointing to the equations
    in the paper (and make an issue to update the equations once the paper is
    finalized). Use this to set the context for the different methods,
    omega22_averaging_method, etc.

    parameters:
    ----------
    tref_in:
        Input reference time at which to measure eccentricity and mean anomaly.
        Can be a single float or an array.

    fref_in:
        Input reference GW frequency at which to measure the eccentricity and
        mean anomaly. Can be a single float or an array.

        Given an fref_in, we find the corresponding tref_in such that
        omega22_average(tref_in) = 2 * pi * fref_in. Here, omega22_average(t)
        is a monotonically increasing average frequency that is computed from
        the instantaneous omega22(t) = dphi22(t)/dt, where phi22(t) is the
        phase of the (2,2) mode.

        NOTE:
        omega22_average is not a moving average; depending on which
        averaging method is used (see the omega22_averaging_method option
        below), it means slightly different things.

        Eccentricity and mean anomaly measurements are returned on a subset of
        tref_in/fref_in, called tref_out/fref_out, which are described below.

        If dataDict is provided in dimensionless units, tref_in should in units
        of M and fref_in should be in units of cycles/M. If dataDict is
        provided in MKS units, t_ref should be in seconds and fref_in should be
        in Hz.

    dataDict:
        Dictionary containing waveform modes dict, time etc.
        Should follow the format:
            dataDict = {"t": time,
                        "hlm": modeDict, ...
                        }
            with modeDict = {(l1, m1): h_{l1, m1},
                             (l2, m2): h_{l2, m2}, ...
                            }.

        We currently only allow time-domain, nonprecessing waveforms with a
        uniform time array. Please make sure that the time step is small enough
        that omega22(t) can be accurately computed; we use a 4th-order finite
        difference scheme. In dimensionless units, we recommend a time step of
        dtM = 0.1M to be conservative, but you may be able to get away with
        larger time steps like dtM = 1M. The corresponding time step in seconds
        would be dtM * M * lal.MTSUN_SI, where M is the total mass in Solar
        masses.

        #FIXME ARIF: Maybe the documentation of the method option should be
        above the dataDict option?

        Some methods require additional data in dataDict. In particular, the
        ResidualAmplitude and ResidualFrequency methods require additional
        keys, "t_zeroecc" and "hlm_zeroecc", which should include the time and
        modeDict for the quasicircular limit of the eccentric waveform in
        "hlm". For a waveform model, "hlm_zeroecc" can be obtained by
        evaluating the model by keeping the rest of the binary parameters fixed
        but setting the eccentricity to zero. For NR, if such a quasicircular
        counterpart is not available, we recommend using quasicircular
        waveforms like NRHybSur3dq8 or PhenomT, depending on the mass ratio and
        spins. We require that "hlm_zeroecc" be at least as long as "hlm" so
        that residual amplitude/frequency can be computed.

        The (2,2) mode is always required in "hlm"/"hlm_zeroecc". If additional
        modes are included, they will be used in determining the peak time
        following Eq.(5) from arxiv:1905.09300. The peak time is used to
        time-align the two waveforms before computing the residual
        amplitude/frequency.

    method:
        Method to define eccentricity. See get_available_methods() for
        available methods.
        #FIXME ARIF: Actually, now that things are more mature, please
        spell out the different methods here.

    return_ecc_method:
        #FIXME ARIF: This should be called: return_gwecc_object
        If true, returns the method object used to compute the eccentricity.
        Default is False.

    spline_kwargs:
        Dictionary of arguments to be passed to
        scipy.interpolate.InterpolatedUnivariateSpline.

    extra_kwargs: A dict of any extra kwargs to be passed. Allowed kwargs are:
        num_orbits_to_exclude_before_merger:
            Can be None or a non negative real number.
            If None, the full waveform data (even post-merger) is used for
            finding extrema, but this might cause issues when interpolating
            trough the extrema.
            For a non negative real num_orbits_to_exclude_before_merger, that
            many orbits prior to merger are excluded when finding extrema.
            Default: 1.
        extrema_finding_kwargs:
            Dictionary of arguments to be passed to the peak finding function,
            where it will be (typically) passed to scipy.signal.find_peaks.
            #FIXME ARIF: Explain how the defaults are set.
        debug:
            Run additional sanity checks if debug is True.
            Default: True.
        omega22_averaging_method:
            Options for obtaining omega22_average(t) from the instantaneous
            omega22(t).
            #FIXME ARIF: the first two options should be called:
            "mean_of_extrema_interpolants" and
            "interpolate_orbit_averages_at_extrema". And please check if my
            descriptions are correct.
            - "average_between_extrema": Mean of omega22_peaks(t) and
              omega22_troughs(t), where omega22_peaks(t) is a spline
              interpolant between omega22(t) evaluated at periastron locations,
              and omega22_troughs(t) is a spline interpolant between omega22(t)
              evaluated at apastron locations.
            - "orbital_average_at_extrema": A spline through the orbit averaged
              omega22(t) evaluated at all available extrema. First, orbit
              averages are obtained at each periastron by averaging over the
              time from the previous periastron to the current one. Similar
              orbit averages are done at apastrons. Finally, a spline
              interpolant is constructed between all of these orbit averages at
              extrema locations. Due to the nature of the averaging, the first
              and last extrema need to be excluded, which is why this is not
              the default method.
            - "omega22_zeroecc": omega22(t) of the zero eccentricity waveform
              is used as a proxy for the average frequency of the eccentric
              waveform.
            Default is "average_between_extrema".
        treat_mid_points_between_peaks_as_troughs:
            If True, instead of trying to find local minima in the data, we
            simply find the midpoints between local maxima and treat them as
            apastron locations. This is helpful for eccentricities ~1 where
            periastrons are easy to find but apastrons are not.
            Default: False

    returns:
    --------
    tref_out/fref_out:
        If tref_in is provided, tref_out is returned, and if fref_in provided,
        fref_out is returned. tref_out/fref_out is the output reference
        time/frequency at which eccentricity and mean anomaly are measured.
        Units of tref_out/fref_out are the same as those of tref_in/fref_in.

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

        #FIXME ARIF: This should be explained using omega22_average(t).
        fref_out is set as fref_out = fref_in[fref_in >= fmin && fref_in < fmax].
        where fmin is the frequency at tmin, and fmax is the frequency at tmax.
        tmin/tmax are defined above.

    ecc_ref:
        #FIXME ARIF: Don't you mean tref_in/fref_in?
        Measured eccentricity at t_ref. Same type as t_ref.

    mean_ano_ref:
        Measured mean anomaly at t_ref. Same type as t_ref.

    #FIXME ARIF: This should be called gwecc_object
    ecc_method:
        Method object used to compute eccentricity. Only returned if
        return_ecc_method is True.
    """
    available_methods = get_available_methods()

    if method in available_methods:
        ecc_method = available_methods[method](dataDict,
                                               spline_kwargs=spline_kwargs,
                                               extra_kwargs=extra_kwargs)

        tref_or_fref_out, ecc_ref, mean_ano_ref = ecc_method.measure_ecc(
            tref_in=tref_in, fref_in=fref_in)
        if not return_ecc_method:
            return tref_or_fref_out, ecc_ref, mean_ano_ref
        else:
            return tref_or_fref_out, ecc_ref, mean_ano_ref, ecc_method
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {list(available_methods.keys())}")
