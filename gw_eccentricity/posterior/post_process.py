"""Module to reconstruct eccentricity posterior using gw_eccentricity."""
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from ..gw_eccentricity import measure_eccentricity


class PostProcess:
    """Reconstruct eccentricity posterior from posterior samples.

    Use `gw_eccentricity` to measure eccentricity directly from the waveform
    modes generated at the posterior samples.
    """

    def __init__(self, posterior_file, data_dict_generator,
                 data_dict_generator_kwargs=None):
        """Init for PostProcess class.

        Parameters
        ----------
        posterior_file : str
            Path to file containing the posterior from a MCMC parameter
            estimation run.
        data_dict_generator : function
            data_dict is generated using function call as below::

            data_dict = data_dict_generator(sample_index, posterior,
                                          **kwargs)

            where

            - `sample_index` is the row index referring to the sample
              parameters in the `posterior` DataFrame.
            - `posterior` is a Pandas DataFrame containing the
              posterior samples where each row refers to a sample
              in the posterior.
            - `kwargs` is an optional dict of parameters to be passed
              to generate the waveform modes.
        data_dict_generator_kwargs : dict, optional
            Extra kwargs to be provided to data_dict_generator.
        """
        self.posterior_file = posterior_file
        self.posterior = self.get_posterior()
        if not isinstance(self.posterior, pd.core.frame.DataFrame):
            raise TypeError(
                f"{self.posterior} should be a pandas DataFrame "
                f"and not a {type(self.posterior)}.")
        self.posterior_meta_data = None
        self.posterior_fref = None
        self.post_process_result = None
        self.fref_bounds = None
        if not callable(data_dict_generator):
            raise TypeError(
                "`data_dict_generator` must be a `function` and "
                f"not a {type(data_dict_generator)}")
        self.data_dict_generator = data_dict_generator
        self.data_dict_generator_kwargs = (data_dict_generator_kwargs
                                           if data_dict_generator_kwargs is not None
                                           else {})

    def get_posterior(self):
        """Get the posterior from posterior file.

        The returned posterior should be a Pandas DataFrame.
        """
        raise NotImplementedError(
            "Please override this function to fetch the posterior samples "
            f"from {self.posterior_file}.")

    def plot_eccentricity_posterior(self, fig=None, axis=None, figsize=(6, 4),
                                    **kwargs):
        """Plot the eccentricity posterior as a histogram.

        Parameters
        ----------
        fig : object, default=None
            Figure object to add the plot to. If None, a new figure is created.
        axis : object, default=None
            Axis object to add the plot to. If None, a new axis is created.
        figsize : tuple, default=(6, 4)
            Figure size, used only when creating a new figure.
        **kwargs : dict, optional
            Extra arguments passed to ``matplotlib.pyplot.Axes.hist``.

        Returns
        -------
        fig, axis : tuple
            Returned when ``fig`` or ``axis`` is None (new objects were created).
        axis : object
            Returned when both ``fig`` and ``axis`` were provided by the caller.
        """
        if fig is None or axis is None:
            fig, axis = plt.subplots(figsize=figsize)
            axis.hist(self.posterior["eccentricity"], **kwargs)
            return fig, axis
        axis.hist(self.posterior["eccentricity"], **kwargs)
        return axis

    def get_data_dict(self, sample_index):
        """Get data_dict for given sample_index in the posterior.

        Parameters
        ----------
        sample_index : int
            Index to pick a row in the posterior DataFrame.

        Returns
        -------
        data_dict : dict
            Dictionary of waveform modes data compatible with
            ``gw_eccentricity.measure_eccentricity``.
        """
        data_dict = self.data_dict_generator(
            sample_index, self.posterior, **self.data_dict_generator_kwargs)
        if not isinstance(data_dict, dict):
            raise TypeError(
                f"The data_dict generator `{self.data_dict_generator}` should "
                f"return a dict and not a {type(data_dict)}")
        return data_dict

    def measure_eccentricity_for_sample(self, sample_index, fref_in,
                                        method="Amplitude", gw_eccentricity_kwargs=None):
        """Measure eccentricity from waveform modes for a sample.

        A wrapper around ``gw_eccentricity.measure_eccentricity`` to measure
        eccentricity from the waveform modes for a sample at ``sample_index``
        in the posterior DataFrame.

        Parameters
        ----------
        sample_index : int
            Row index in the posterior DataFrame.
        fref_in : float
            Reference frequency where eccentricity is to be measured.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

        Returns
        -------
        res_dict : dict
            Dictionary with keys: ``sample_index``, ``method``, ``fref``,
            ``status``, ``eccentricity``, ``mean_anomaly``, and on failure
            ``error_message``.
        """
        if gw_eccentricity_kwargs is None:
            gw_eccentricity_kwargs = {"num_orbits_to_exclude_before_merger": 2}
        data_dict = self.get_data_dict(sample_index)
        res_dict = {"sample_index": sample_index,
                    "method": method,
                    "fref": fref_in}
        try:
            res = measure_eccentricity(dataDict=data_dict,
                                       fref_in=fref_in,
                                       method=method,
                                       **gw_eccentricity_kwargs)
            res_dict.update({
                "status": "success",
                "eccentricity": res["eccentricity"],
                "mean_anomaly": res["mean_anomaly"]})
        except Exception as exception_message:
            res_dict.update({
                "status": "fail",
                "eccentricity": None,
                "mean_anomaly": None,
                "error_message": exception_message})
        return res_dict

    def post_process(self, fref, samples=None, method="Amplitude",
                     gw_eccentricity_kwargs=None):
        """Post-process to build the eccentricity posterior.

        Parameters
        ----------
        fref : float
            Reference frequency where eccentricity is to be measured.
        samples : array-like, default=None
            Indices of samples to process. Default is all samples.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

        Returns
        -------
        post_process_result : list of dict
            List of per-sample result dicts.
        """
        if samples is None:
            samples = self.posterior.index
        self.post_process_result = []
        for sample in tqdm(samples):
            result = self.measure_eccentricity_for_sample(
                sample, fref, method, gw_eccentricity_kwargs)
            self.post_process_result.append(result)
        return self.post_process_result

    def post_process_summary(self):
        """Summarize post-process result.

        Returns
        -------
        summary_dict : dict
            Dictionary with keys: ``total_samples``, ``success_percentage``,
            ``fref``, ``method``, ``eccentricity``, ``mean_anomaly``.
        """
        if not self.post_process_result:
            raise ValueError(
                "post_process_result is empty. Run post_process first.")
        total_samples = len(self.post_process_result)
        success = [s for s in self.post_process_result
                   if s["status"] == "success"]
        eccentricity = [s["eccentricity"] for s in success]
        mean_anomaly = [s["mean_anomaly"] for s in success]
        method = self.post_process_result[0]["method"]
        fref = self.post_process_result[0]["fref"]
        return {"total_samples": total_samples,
                "success_percentage": (len(success) / total_samples) * 100,
                "fref": fref,
                "method": method,
                "eccentricity": eccentricity,
                "mean_anomaly": mean_anomaly}

    def get_egw_posterior(self):
        """Return eccentricity and mean anomaly from the post-processed result.

        Returns
        -------
        dict
            Dictionary with keys ``eccentricity`` and ``mean_anomaly``.
        """
        if self.post_process_result is None:
            raise ValueError(
                "Run post_process first to obtain the post-processed "
                "result from gw_eccentricity.")
        summary = self.post_process_summary()
        return {"eccentricity": summary["eccentricity"],
                "mean_anomaly": summary["mean_anomaly"]}

    def get_fref_bounds_for_sample(self, sample_index, method="Amplitude",
                                   gw_eccentricity_kwargs=None):
        """Get the min and max allowed fref for a given sample.

        Parameters
        ----------
        sample_index : int
            Row index in the posterior DataFrame.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

        Returns
        -------
        res_dict : dict
            Dictionary with keys: ``sample_index``, ``method``, ``status``,
            ``fref_min``, ``fref_max``, and on failure ``error_message``.
        """
        if gw_eccentricity_kwargs is None:
            gw_eccentricity_kwargs = {"num_orbits_to_exclude_before_merger": 2}
        data_dict = self.get_data_dict(sample_index)
        res_dict = {"sample_index": sample_index,
                    "method": method}
        try:
            res = measure_eccentricity(dataDict=data_dict,
                                       tref_in=data_dict["t"],
                                       method=method,
                                       **gw_eccentricity_kwargs)
            gw_obj = res["gwecc_object"]
            fref_bounds = gw_obj.get_fref_bounds()
            res_dict.update({
                "status": "success",
                "fref_min": fref_bounds[0],
                "fref_max": fref_bounds[1]})
        except Exception as exception_message:
            res_dict.update({
                "status": "fail",
                "fref_min": None,
                "fref_max": None,
                "error_message": exception_message})
        return res_dict

    def get_fref_bounds(self, samples=None, method="Amplitude",
                        gw_eccentricity_kwargs=None):
        """Get the range of frequencies where eccentricity can be measured.

        Parameters
        ----------
        samples : array-like, default=None
            Indices of samples to process. Default is all samples.
        method : str, default="Amplitude"
            Method to use in ``gw_eccentricity.measure_eccentricity``.
        gw_eccentricity_kwargs : dict, optional
            Extra kwargs passed to ``gw_eccentricity.measure_eccentricity``.

        Returns
        -------
        result : dict
            Dictionary with keys ``fref_bounds``, ``success_percentage``,
            and ``failed_cases``.
        """
        samples = list(self.posterior.index if samples is None else samples)
        failed_cases = []
        fref_mins = []
        fref_maxs = []

        for sample in tqdm(samples):
            result = self.get_fref_bounds_for_sample(sample, method, gw_eccentricity_kwargs)
            if result["status"] == "success":
                fref_mins.append(result["fref_min"])
                fref_maxs.append(result["fref_max"])
            else:
                failed_cases.append(result["sample_index"])

        self.fref_bounds = max(fref_mins), min(fref_maxs)
        return {"fref_bounds": self.fref_bounds,
                "success_percentage": ((len(samples) - len(failed_cases))
                                       / len(samples)) * 100,
                "failed_cases": failed_cases}
