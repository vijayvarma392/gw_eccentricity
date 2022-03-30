"""Simple script to use different methods and measure eccentricity."""

from eccDefinitionUsingAmplitude import measureEccentricityUsingAmplitude
from eccDefinitionUsingFrequency import measureEccentricityUsingFrequency
from eccDefinitionUsingFrequencyFits import measureEccentricityUsingFrequencyFits
from eccDefinitionUsingResidualAmplitude import measureEccentricityUsingResidualAmplitude


def measure_eccentricity(method, dataDict, t_ref, height=None, threshold=None,
                         distance=None, prominence=None, width=10, wlen=None,
                         rel_height=0.5, plateau_size=None, **kwargs):
    """Measure eccentricity and mean anomaly at reference time.

    parameters:
    ----------
    t_ref: reference time to measure eccentricity and mean anomaly.
    see scipy.signal.find_peaks for rest or the arguments.
    kwargs: to be passed to the InterpolatedUnivariateSpline

    returns:
    --------
    ecc_ref: measured eccentricity at t_ref
    mean_ano_ref: measured mean anomaly at t_ref
    """
    if method == "Amplitude":
        ecc_method = measureEccentricityUsingAmplitude(dataDict)
    elif method == "Frequency":
        ecc_method = measureEccentricityUsingFrequency(dataDict)
    elif method == "ResidualAmplitude":
        ecc_method = measureEccentricityUsingResidualAmplitude(dataDict)
    elif method == "FrequencyFits":
        ecc_method = measureEccentricityUsingFrequencyFits(dataDict)
    else:
        raise NotImplementedError(f"{method} method is not implemented.")

    return ecc_method.measure_ecc(t_ref, height, threshold, distance,
                                  prominence, width, wlen, rel_height,
                                  plateau_size, **kwargs)
