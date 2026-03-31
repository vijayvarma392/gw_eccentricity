"""Module to reconstruct eccentricity posterior using gw_eccentricity."""
import matplotlib.pyplot as plt
import logging
from ..gw_eccentricity import measure_eccentricity

logger = logging.getLogger(__name__)

def get_data_dict(params, data_dict_generator,
                  extra_kwargs=None):
    """Get data_dict for given params in the posterior.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the sample.

    data_dict_generator : function
        data_dict is generated using function call as below::

            data_dict = data_dict_generator(params, extra_kwargs)

    extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.

    Returns
    -------
    data_dict : dict
        Dictionary of waveform modes data compatible with
        ``gw_eccentricity.measure_eccentricity``.
    """
    if extra_kwargs is None:
        extra_kwargs = {}
    data_dict = data_dict_generator(
        params, extra_kwargs)
    if not isinstance(data_dict, dict):
        raise TypeError(
            f"The data_dict generator `{data_dict_generator}` should "
            f"return a dict and not a {type(data_dict)}")
    return data_dict


def get_fref_bounds_for_sample(
        params,
        data_dict_generator,
        data_dict_generator_extra_kwargs=None,
        method="Amplitude",
        gw_eccentricity_kwargs=None):
    """Get the min and max allowed fref for a given sample.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the sample.
    data_dict_generator : function
        Function to generate the data dictionary for the sample.
    data_dict_generator_extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.
    method : str, default="Amplitude"
        Method to use in ``gw_eccentricity.measure_eccentricity``.
    gw_eccentricity_kwargs : dict, optional
        Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

    Returns
    -------
    res_dict : dict
        Dictionary with keys: ``params``, ``method``, ``status``,
        ``fref_min``, ``fref_max``, and on failure ``error_message``.
    """
    if gw_eccentricity_kwargs is None:
        gw_eccentricity_kwargs = {}
    res_dict = {"method": method}
    try:
        data_dict = get_data_dict(params, data_dict_generator, data_dict_generator_extra_kwargs)
        res = measure_eccentricity(
            dataDict=data_dict,
            tref_in=data_dict["t"], # pass the full time array to get the fref bounds for the entire waveform
            method=method,
            **gw_eccentricity_kwargs)
        gw_obj = res["gwecc_object"]
        fref_bounds = gw_obj.get_fref_bounds()
        res_dict.update({
            "status": "success",
            "fref_min": fref_bounds[0],
            "fref_max": fref_bounds[1]})
    except Exception as e:
        logger.warning(f"Sample {params} failed to get fref bounds: {e}")
        res_dict.update({
            "status": "fail",
            "fref_min": None,
            "fref_max": None,
            "error_message": str(e)})
    res_dict["params"] = params
    return res_dict


def postprocess_sample(
        params,
        fref,
        data_dict_generator,
        data_dict_generator_extra_kwargs=None,
        method="Amplitude",
        gw_eccentricity_kwargs=None):
    """Measure eccentricity and mean anomaly from waveform modes for a sample.

    A wrapper around ``gw_eccentricity.measure_eccentricity`` to measure
    eccentricity from the waveform modes for a sample with given ``params``.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters for the sample.
    fref : float
        Reference frequency where eccentricity is to be measured.
    data_dict_generator : function
        data_dict is generated using function call as below::
            data_dict = data_dict_generator(params, data_dict_generator_extra_kwargs)
    data_dict_generator_extra_kwargs : dict, optional
        Extra kwargs passed to ``data_dict_generator``.
    method : str, default="Amplitude"
        Method to use in ``gw_eccentricity.measure_eccentricity``.
    gw_eccentricity_kwargs : dict, optional
        Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

    Returns
    -------
    res_dict : dict
        Dictionary with keys: ``params``, ``method``, ``fref``,
        ``status``, ``eccentricity``, ``mean_anomaly``, and on failure
        ``error_message``.
    """
    try:
        data_dict = get_data_dict(
            params,
            data_dict_generator,
            data_dict_generator_extra_kwargs)
        res = measure_eccentricity(
            dataDict=data_dict,
            fref_in=fref,
            method=method,
            **(gw_eccentricity_kwargs or {}))
        return {
            "status": "success",
            "eccentricity": res["eccentricity"],
            "mean_anomaly": res["mean_anomaly"]}
    except Exception as e:
        logger.warning(f"Sample {params} failed: {e}")
        return {
            "status": "fail",
            "eccentricity": None,
            "mean_anomaly": None,
            "error_message": str(e)}
