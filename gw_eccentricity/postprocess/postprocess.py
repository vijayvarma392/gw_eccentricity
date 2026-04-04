"""Module to reconstruct eccentricity posterior using gw_eccentricity."""
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from ..plot_settings import (
    use_fancy_plotsettings, figWidthsOneColDict, figWidthsTwoColDict,
    figHeightsDict, labelsDict
)
from ..gw_eccentricity import measure_eccentricity
from .core import (
    get_data_dict, get_fref_bounds_for_sample, postprocess_sample,
    PostProcessResults, FrefBoundsResults
)


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
        postprocess_result : PostProcessResults
            Container with per-sample result objects, in the same order as ``samples``.
        """
        samples = list(self.posterior.index if samples is None else samples)
        param_list = self.posterior.loc[samples].to_dict(orient="records")
        data_dict_generator = self.data_dict_generator
        data_dict_generator_extra_kwargs = self.data_dict_generator_extra_kwargs

        results_list = Parallel(
            n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
            delayed(postprocess_sample)(
                sample_index, params, fref, data_dict_generator, data_dict_generator_extra_kwargs,
                method, gw_eccentricity_kwargs
            )
            for sample_index, params in tqdm(zip(samples, param_list), desc="Postprocessing samples")
        )
        self.postprocess_result = PostProcessResults(results_list)
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
        return self.postprocess_result.get_summary()

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
        result : FrefBoundsResults
        """
        samples = list(self.posterior.index if samples is None else samples)
        param_list = self.posterior.loc[samples].to_dict(orient="records")

        data_dict_generator = self.data_dict_generator
        data_dict_generator_extra_kwargs = self.data_dict_generator_extra_kwargs
        results = Parallel(
            n_jobs=n_jobs, pre_dispatch="2*n_jobs")(
            delayed(get_fref_bounds_for_sample)(
                sample_index, param, data_dict_generator, data_dict_generator_extra_kwargs,
                method, gw_eccentricity_kwargs)
            for sample_index, param in tqdm(zip(samples, param_list), desc="Getting fref bounds")
        )

        return FrefBoundsResults(results=results)