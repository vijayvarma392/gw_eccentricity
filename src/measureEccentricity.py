"""Simple script to use different methods and measure eccentricity."""

from eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude
from eccDefinitionUsingFrequency import eccDefinitionUsingFrequency
from eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from eccDefinitionUsingResidualAmplitude import eccDefinitionUsingResidualAmplitude


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
