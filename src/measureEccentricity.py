"""Simple script to use different methods and measure eccentricity."""

from eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude
from eccDefinitionUsingFrequency import eccDefinitionUsingFrequency
from eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from eccDefinitionUsingResidualAmplitude import eccDefinitionUsingResidualAmplitude


def get_available_methods():
    """Get dictionary of available methods."""
    return {"Amplitude": eccDefinitionUsingAmplitude,
            "Frequency": eccDefinitionUsingFrequency,
            "ResidualAmplitude": eccDefinitionUsingResidualAmplitude,
            "FrequencyFits": eccDefinitionUsingFrequencyFits}


def measure_eccentricity(t_ref, dataDict, method="Amplitude", height=None,
                         threshold=None, distance=None, prominence=None,
                         width=10, wlen=None, rel_height=0.5,
                         plateau_size=None, **kwargs):
    """Measure eccentricity and mean anomaly at reference time.

    parameters:
    ----------
    t_ref: reference time to measure eccentricity and mean anomaly.
    dataDict: dictionary containing waveform modes dict, time etc
    should follow the format {"t": time, "hlm": modeDict, ..}
    and modeDict = {(l, m): hlm_mode_data}
    for ResidualAmplitude method, provide "t0" and "hlm0" as well
    in the dataDict.

    see scipy.signal.find_peaks for rest or the arguments.
    kwargs: to be passed to the InterpolatedUnivariateSpline

    returns:
    --------
    ecc_ref: measured eccentricity at t_ref
    mean_ano_ref: measured mean anomaly at t_ref
    """
    available_methods = get_available_methods()

    if method in available_methods:
        ecc_method = available_methods[method](dataDict)
        return ecc_method.measure_ecc(t_ref, height, threshold, distance,
                                      prominence, width, wlen, rel_height,
                                      plateau_size, **kwargs)
    else:
        raise Exception(f"Invalid method {method}, has to be one of"
                        f" {available_methods.keys()}")
