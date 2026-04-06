"""Reconstruct eccentricity posterior from Bilby generated posterior."""
import bilby
import pandas as pd
from .postprocess import PostProcess
from .core import filter_posterior_columns


class PostProcessForBilby(PostProcess):
    """Derived class for reconstructing eccentricity posterior for Bilby.

    Uses built-in methods in Bilby to read posterior results from a
    Bilby-generated posterior file (.json, .hdf5, or .pkl).
    """

    def __init__(self, *args, **kwargs):
        """Init for PostProcessForBilby class."""
        self.posterior_type = "Bilby"
        self.posterior_result = None
        super().__init__(*args, **kwargs)
    
    def get_posterior(self, parameter_columns: list[str] | None = None
                      ) -> pd.DataFrame:
        """Get the posterior.
        
        Returns
        -------
        posterior : pandas.DataFrame
            Posterior samples as a pandas DataFrame.
            If `parameter_columns` is provided, only the specified columns will
            be returned. Otherwise, the full posterior will be returned.

        Raises
        ------
        FileNotFoundError
            If the posterior file does not exist.
        """
        if self.posterior_result is None:
            try:
                self.posterior_result = bilby.result.read_in_result(
                    filename=self.posterior_file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Cannot find Bilby posterior file: {self.posterior_file}")
        posterior = self.posterior_result.posterior
        if parameter_columns is not None:
            return filter_posterior_columns(posterior, parameter_columns)
        else:
            return posterior

    def get_injection(self):
        """Get the injection parameters from injection file."""
        if self.injection_file is None:
            raise ValueError("Injection file is not provided.")
        try:
            injection = pd.read_csv(self.injection_file, sep=" ").to_dict(orient="records")[0]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot find injection file: {self.injection_file}")
        # add minimum frequency to the params from result
        injection["minimum_frequency"] = self.posterior.iloc[0]["minimum_frequency"]
        return injection