"""Module to reconstruct eccentricity posterior using gw_eccentricity."""
import logging
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from ..plot_settings import (
    use_fancy_plotsettings, figWidthsOneColDict, figWidthsTwoColDict,
    figHeightsDict, labelsDict
)
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
        Dictionary with keys: ``status``, ``egw``, ``lgw``, and on failure
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
            "egw": res["eccentricity"],
            "lgw": res["mean_anomaly"]}
    except Exception as e:
        logger.warning(f"Sample {params} failed: {e}")
        return {
            "status": "fail",
            "egw": None,
            "lgw": None,
            "error_message": str(e)}


class PostProcess:
    """Reconstruct eccentricity posterior from posterior samples.

    Use `gw_eccentricity` to measure eccentricity directly from the waveform
    modes generated at the posterior samples.
    """

    def __init__(self, posterior_file, data_dict_generator,
                 data_dict_generator_extra_kwargs=None, injection_file=None):
        """Init for PostProcess class.

        Parameters
        ----------
        posterior_file : str
            Path to file containing the posterior from a MCMC parameter
            estimation run.
        data_dict_generator : function
            data_dict is generated using function call as below::

            data_dict = data_dict_generator(
                sample_params,
                data_dict_generator_extra_kwargs
            )

            where

            - `sample_params` is dict containing the parameters for a sample in
              the posterior.
            - `data_dict_generator_extra_kwargs` is an optional dict of parameters to
              be passed to generate the waveform modes.
        data_dict_generator_extra_kwargs : dict, optional
            Extra kwargs to be provided to data_dict_generator.
        injection_file : str, optional
            Path to file containing the injection parameters.
        """
        self.posterior_file = posterior_file
        self.posterior = self.get_posterior()
        if not isinstance(self.posterior, pd.core.frame.DataFrame):
            raise TypeError(
                f"{self.posterior} should be a pandas DataFrame "
                f"and not a {type(self.posterior)}.")
        self.postprocess_result = None
        self.fref_bounds = None
        if not callable(data_dict_generator):
            raise TypeError(
                "`data_dict_generator` must be a `function` and "
                f"not a {type(data_dict_generator)}")
        self.data_dict_generator = data_dict_generator
        self.data_dict_generator_extra_kwargs = (data_dict_generator_extra_kwargs
                                           if data_dict_generator_extra_kwargs is not None
                                           else {})
        self.injection_file = injection_file

    def get_posterior(self):
        """Get the posterior from posterior file.

        The returned posterior should be a Pandas DataFrame.
        """
        raise NotImplementedError(
            "Please override this function to fetch the posterior samples "
            f"from {self.posterior_file}.")
    
    def get_injection(self):
        """Get the injection parameters from injection file.
        
        The returned object should be a dict containing the parameters for the
        injection, which the ``data_dict_generator`` can use to generate the 
        waveform modes for the injection.
        """
        raise NotImplementedError(
            "Please override this function to fetch the injection parameters "
            f"from {self.injection_file}.")
    
    def get_injection_data_dict(self):
        """Get the data dict for the injection.

        Returns
        -------
        data_dict : dict
            Data dict for the injection, generated using the injection parameters
            and the data_dict_generator.
        """
        if self.injection_file is None:
            raise ValueError("injection_file is not provided.")
        injection = self.get_injection()
        data_dict = get_data_dict(
            params=injection,
            data_dict_generator=self.data_dict_generator,
            extra_kwargs=self.data_dict_generator_extra_kwargs)
        return data_dict
    
    def get_injection_eccentricity(self, fref, method="Amplitude", 
                                   gw_eccentricity_kwargs=None, debug=False):
        """Get the eccentricity of the injection.

        Returns
        -------
        eccentricity : float
            Eccentricity of the injection, measured using the data dict for the
            injection.
        mean_anomaly : float
            Mean anomaly of the injection, measured using the data dict for the
            injection.
        """
        if gw_eccentricity_kwargs is None:
            gw_eccentricity_kwargs = {}
        data_dict = self.get_injection_data_dict()
        result = measure_eccentricity(
            fref_in=fref,
            dataDict=data_dict,
            method=method,
            **gw_eccentricity_kwargs)
        if debug:
            gwecc_object = result["gwecc_object"]
            gwecc_object.make_diagnostic_plots()
        return result["eccentricity"], result["mean_anomaly"]

    def plot_eccentricity_posterior(self, fig=None, ax=None, figsize=(6, 4),
                                    **kwargs):
        """Plot the eccentricity posterior as a histogram.

        Parameters
        ----------
        fig : object, default=None
            Figure object to add the plot to. If None, a new figure is created.
        ax : object, default=None
            Axis object to add the plot to. If None, a new axis is created.
        figsize : tuple, default=(6, 4)
            Figure size, used only when creating a new figure.
        **kwargs : dict, optional
            Extra arguments passed to ``matplotlib.pyplot.Axes.hist``.

        Returns
        -------
        fig, ax : tuple
            Returned when ``fig`` or ``ax`` is None (new objects were created).
        ax : object
            Returned when both ``fig`` and ``ax`` were provided by the caller.
        """
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.hist(self.posterior["eccentricity"], **kwargs)
        return fig, ax

    def postprocess(self, fref, samples=None, method="Amplitude",
                    gw_eccentricity_kwargs=None, n_jobs=-1):
        """Post-process to build the egw posterior.

        Parameters
        ----------
        fref : float
            Reference frequency where eccentricity and mean anomaly are to be measured.
        samples : array-like, default=None
            Indices of samples to process. Default is all samples.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.
        n_jobs : int, default=-1
            Number of joblib workers. ``-1`` uses all available cores.

        Returns
        -------
        post_process_result : list of dict
            List of per-sample result dicts, in the same order as ``samples``.
        """
        samples = list(self.posterior.index if samples is None else samples)
        param_list = self.posterior.loc[samples].to_dict(orient="records")
        data_dict_generator = self.data_dict_generator
        data_dict_generator_extra_kwargs = self.data_dict_generator_extra_kwargs

        self.postprocess_result = Parallel(
            n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
            delayed(postprocess_sample)(
                params, fref, data_dict_generator, data_dict_generator_extra_kwargs,
                method, gw_eccentricity_kwargs
            )
            for params in tqdm(param_list, desc="Postprocessing samples")
        )
        return self.postprocess_result

    def postprocess_summary(self):
        """Summarize postprocess result.

        Returns
        -------
        summary_dict : dict
            Dictionary with keys: ``total_samples``, ``success_percentage``,
            ``egw``, ``lgw``.
        """
        if self.postprocess_result is None:
            raise ValueError(
                "postprocess_result is empty. Run postprocess first.")
        total_samples = len(self.postprocess_result)
        success = [s for s in self.postprocess_result
                   if s["status"] == "success"]
        eccentricity = [s["egw"] for s in success]
        mean_anomaly = [s["lgw"] for s in success]
        return {"total_samples": total_samples,
                "success_percentage": (len(success) / total_samples) * 100,
                "egw": eccentricity,
                "lgw": mean_anomaly}

    def get_egw_posterior(self):
        """Return eccentricity and mean anomaly from the post-processed result.

        Returns
        -------
        dict
            Dictionary with keys ``egw`` and ``lgw``.
        """
        if self.postprocess_result is None:
            raise ValueError(
                "Run postprocess first to obtain the post-processed "
                "result from gw_eccentricity.")
        summary = self.postprocess_summary()
        return {"egw": summary["egw"],
                "lgw": summary["lgw"]}
    
    def plot_egw_posterior(self, fig=None, ax=None,
                           usetex=False,
                           style="Notebook",
                           two_col=False,
                           **kwargs):
        """Plot the eccentricity posterior from gw_eccentricity as a histogram.

        Parameters
        ----------
        fig : object, default=None
            Figure object to add the plot to. If None, a new figure is created.
        ax : object, default=None
            Axis object to add the plot to. If None, a new axis is created.
        usetex : bool, default=False
            Whether to use LaTeX for text rendering.
        style : str, default="Notebook"
            Plot style to use. See plot_settings.use_fancy_plotsettings for
            available styles.
        two_col : bool, default=False
            Whether to use a two-column layout for the figure.
        **kwargs : dict, optional
            Extra arguments passed to ``matplotlib.pyplot.Axes.hist``.

        Returns
        -------
        fig, ax : tuple
            Returned when ``fig`` or ``ax`` is None (new objects were created).
        ax : object
            Returned when both ``fig`` and ``ax`` were provided by the caller.
        """
        use_fancy_plotsettings(style=style, usetex=usetex)
        egw_posterior = self.get_egw_posterior()
        if fig is None or ax is None:
            figsize = (figWidthsTwoColDict[style] if two_col else figWidthsOneColDict[style], 
                       figHeightsDict[style])
            fig, ax = plt.subplots(figsize=figsize)
        ax.hist(egw_posterior["egw"], **kwargs)
        ax.set_xlabel(labelsDict["eccentricity"])
        ax.set_ylabel("Number of samples")
        return fig, ax

    def get_fref_bounds(self, samples=None, method="Amplitude",
                        gw_eccentricity_kwargs=None, n_jobs=-1):
        """Get the range of frequencies where eccentricity can be measured.

        Parameters
        ----------
        samples : array-like, default=None
            Indices of samples to process. Default is all samples.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.
        n_jobs : int, default=-1
            Number of joblib workers. ``-1`` uses all available cores.

        Returns
        -------
        result : dict
            Dictionary with keys ``fref_bounds``, ``success_percentage``,
            and ``failed_cases``.
        """
        samples = list(self.posterior.index if samples is None else samples)
        param_list = self.posterior.loc[samples].to_dict(orient="records")

        data_dict_generator = self.data_dict_generator
        data_dict_generator_extra_kwargs = self.data_dict_generator_extra_kwargs
        results = Parallel(
            n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
            delayed(get_fref_bounds_for_sample)(
                param, data_dict_generator, data_dict_generator_extra_kwargs,
                method, gw_eccentricity_kwargs)
            for param in tqdm(param_list, desc="Getting fref bounds")
        )

        fref_mins, fref_maxs, failed_cases = [], [], []
        for result in results:
            if result["status"] == "success":
                fref_mins.append(result["fref_min"])
                fref_maxs.append(result["fref_max"])
            else:
                failed_cases.append(result["params"])

        success_percentage = ((len(samples) - len(failed_cases)) / len(samples)) * 100
        if success_percentage == 0:
            raise ValueError(
                "Failed to get fref bounds for all samples.\n"
                "This could be due to insufficient number of orbits in the "
                "waveform modes for all samples.\n"
                "Consider increasing the length of the waveforms by using "
                "backward evolution or by excluding fewer orbits before merger.")

        self.fref_bounds = max(fref_mins), min(fref_maxs)
        return {"fref_bounds": self.fref_bounds,
                "success_percentage": success_percentage,
                "failed_cases": failed_cases}