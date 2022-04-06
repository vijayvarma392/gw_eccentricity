"""measureEccentricity.

========
Measure eccentricity and mean anomaly from gravitational waves.
"""
__copyright__ = "Copyright (C) 2021 Md Arif Shaikh, Vijay Varma"
__email__ = "arif.shaikh@icts.res.in, vijay.varma392@gmail.com"
__status__ = "testing"
__author__ = "Md Arif Shaikh, Vijay Varma"
__version__ = "0.1"
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
# from .eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from .eccDefinitionUsingResidualAmplitude import eccDefinitionUsingResidualAmplitude


def get_available_methods():
    """Get dictionary of available methods."""
    models = {
            "Amplitude": eccDefinitionUsingAmplitude,
            "Frequency": eccDefinitionUsingFrequency,
            "ResidualAmplitude": eccDefinitionUsingResidualAmplitude,
            # "FrequencyFits": eccDefinitionUsingFrequencyFits
            }
    return models


def measure_eccentricity(tref_in, dataDict, method="Amplitude",
                         return_ecc_method=False,
                         extrema_finding_keywords=None,
                         spline_keywords=None,
                         extra_keywprds=None):
    """Measure eccentricity and mean anomaly at reference time.

    parameters:
    ----------
    tref_in:
        Input Reference time at which to measure eccentricity and mean anomaly.
        Can be a single float or an array.

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
        method to define eccentricity. See get_available_methods for available
        methods.

    return_ecc_method:
        If true, returns the method object used to compute the eccentricity.

    extrema_finding_keywords:
        Dictionary of arguments to be passed to the peak finding function,
        where it will be passed to scipy.signal.find_peaks.

    spline_keywords:
        Dictionary of arguments to be passed to
        scipy.interpolate.InterpolatedUnivariateSpline.

    extra_keywords: any extra keywords to be passed. Allowed keywords are
        exclude_num_orbits_before_merger:
        could be either None or non negative real number. If None, then
        the full data even after merger is used but this might cause
        issues with he interpolaion trough exrema. For non negative real
        number, that many orbits prior to merger is exculded.
        Default is 1.

    returns:
    --------
    tref_out:
         Output reference time where eccenricity and mean anomaly is
         measured. This would be different than tref_in if
         exclude_num_obrits_before_merger in the extra_keyword is not None

    ecc_ref:
        Measured eccentricity at t_ref. Same type as t_ref.

    mean_ano_ref:
        Measured mean anomaly at t_ref. Same type as t_ref.

    ecc_method:
       method object used to compute eccentricity only if
       return_ecc_method is True
    """
    available_methods = get_available_methods()

    if method in available_methods:
        ecc_method = available_methods[method](dataDict)
        tref_out, ecc_ref, mean_ano_ref = ecc_method.measure_ecc(
            tref_in,
            extrema_finding_keywords,
            spline_keywords,
            extra_keywprds)
        if not return_ecc_method:
            return tref_out, ecc_ref, mean_ano_ref
        else:
            return tref_out, ecc_ref, mean_ano_ref, ecc_method
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {available_methods.keys()}")
