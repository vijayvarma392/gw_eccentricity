"""measureEccentricity
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
from .eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from .eccDefinitionUsingResidualAmplitude import eccDefinitionUsingResidualAmplitude


def get_available_methods():
    """Get dictionary of available methods."""
    models = {
            "Amplitude": eccDefinitionUsingAmplitude,
            "Frequency": eccDefinitionUsingFrequency,
            "ResidualAmplitude": eccDefinitionUsingResidualAmplitude,
            #"FrequencyFits": eccDefinitionUsingFrequencyFits
            }
    return models


def measure_eccentricity(t_ref, dataDict, method="Amplitude",
                         extrema_finding_keywords=None,
                         spline_keywords=None):
    """Measure eccentricity and mean anomaly at reference time.

    parameters:
    ----------
    t_ref: reference time to measure eccentricity and mean anomaly.
    dataDict: dictionary containing waveform modes dict, time etc
    should follow the format {"t": time, "hlm": modeDict, ..}
    and modeDict = {(l, m): hlm_mode_data}
    for ResidualAmplitude method, provide "t0" and "hlm0" as well
    in the dataDict.

    extrema_finding_keywords: Dictionary of arguments to be passed to the
    peak finding function.
    spline_keywords: arguments to be passed to InterpolatedUnivariateSpline

    returns:
    --------
    ecc_ref: measured eccentricity at t_ref
    mean_ano_ref: measured mean anomaly at t_ref
    """
    available_methods = get_available_methods()

    if method in available_methods:
        ecc_method = available_methods[method](dataDict)
        return ecc_method.measure_ecc(t_ref, extrema_finding_keywords,
                                      spline_keywords)
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {available_methods.keys()}")
