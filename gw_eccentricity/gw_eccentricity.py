"""gw_eccentricity.
========

Measure eccentricity and mean anomaly from gravitational waves.
See https://pypi.org/project/gw_eccentricity for more details.
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
    """Measure eccentricity and mean anomaly at reference time.

    parameters:
    ----------
    tref_in:
        Input reference time at which to measure eccentricity and mean anomaly.
        Can be a single float or an array. NOTE: eccentricity/mean_ano are
        returned on a different time array tref_out, described below.

        If dataDict is provided in dimensionless units, then tref_in should be
        in units of M. If dataDict is provided in MKS units, tref_in should be
        in seconds.

    fref_in:
        Input reference frequency at which to measure the eccentricity and
        mean anomaly. It can be a single float or an array.
        NOTE: eccentricity/mean anomaly are returned on a different freq
        array fref_out, described below.

        If dataDict is provided in dimensionless units, then fref_in should be
        in units of cycles/M. If dataDict is provided in MKS units, then
        fref_in should be in Hz.

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

    dataDict:
        Dictionary containing waveform modes dict, time etc.
        Should follow the format:
            {"t": time, "hlm": modeDict, ...}
            with modeDict = {(l1, m1): h_{l1, m1},
                             (l2, m2): h_{l2, m2}, ...
                            }.
        Some methods may need extra data. For example, the ResidualAmplitude
        method, requires "t_zeroecc" and "hlm_zeroecc" as well in dataDict.

    method:
        Method to define eccentricity. See get_available_methods() for
        available methods.

    return_ecc_method:
        If true, returns the method object used to compute the eccentricity.
        Default is False.

    spline_kwargs:
        Dictionary of arguments to be passed to
        scipy.interpolate.InterpolatedUnivariateSpline.

    extra_kwargs: Any extra kwargs to be passed. Allowed kwargs are
        num_orbits_to_exclude_before_merger:
            Can be None or a non negative real number.
            If None, the full waveform data (even post-merger) is used to
            measure eccentricity, but this might cause issues when
            interpolating trough extrema.
            For a non negative real num_orbits_to_exclude_before_merger, that
            many orbits prior to merger are excluded when finding extrema.
            Default: 1.
        extrema_finding_kwargs:
            Dictionary of arguments to be passed to the peak finding function,
            where it will be (typically) passed to scipy.signal.find_peaks.
        debug:
            Run additional sanity checks if debug is True.
            Default: True.
        omega22_averaging_method:
            Methods for getting average omega22. For more see fref_in.
            Default is "average_between_extrema".
        treat_mid_points_between_peaks_as_troughs:
            If True, instead of trying to find local minima in the
            data, we simply find the midpoints between local maxima
            and treat them as apastron locations. This is helpful for
            eccentricities ~1 where periastrons are easy to find but
            apastrons are not.
            Default: False

    returns:
    --------
    tref_out/fref_out:
        tref_out is the output reference time where eccentricity and mean
        anomaly are measured and fref_out is the output reference frequency
        where eccentricity and mean anomaly are measured.
        Units of tref_out/fref_out is the same as that of tref_in/fref_in. For
        more see tref_in/fref_in above.

        NOTE: Only of these is returned depending on whether tref_in or
        fref_in is provided. If tref_in is provided then tref_out is returned
        and if fref_in provided then fref_out is returned.

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
        where fmin is the frequency at tmin, and fmax is the frequency at tmax.
        tmin/tmax are defined above.

    ecc_ref:
        Measured eccentricity at t_ref. Same type as t_ref.

    mean_ano_ref:
        Measured mean anomaly at t_ref. Same type as t_ref.

    ecc_method:
        Method object used to compute eccentricity. Only returne if
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
