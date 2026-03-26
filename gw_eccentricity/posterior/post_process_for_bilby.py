"""Reconstruct eccentricity posterior from Bilby generated posterior."""
import bilby
import pandas as pd
from .post_process import PostProcess


class PostProcessForBilby(PostProcess):
    """Derived class for reconstructing eccentricity posterior for Bilby.

    Uses built-in methods in Bilby to read posterior results from a
    Bilby-generated posterior file (.json, .hdf5, or .pkl).
    """

    def __init__(self, *args, **kwargs):
        """Init for PostProcessForBilby class."""
        super().__init__(*args, **kwargs)
        self.posterior_type = "Bilby"

    def get_bilby_result(self):
        """Get Bilby result object for the given posterior file.

        Returns
        -------
        result : bilby.core.result.Result
            Bilby result object.

        Raises
        ------
        FileNotFoundError
            If the posterior file does not exist.
        """
        if not hasattr(self, "posteriror_result"):
            try:
                self.posterior_result = bilby.result.read_in_result(
                    filename=self.posterior_file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Cannot find Bilby posterior file: {self.posterior_file}")
        return self.posterior_result
    
    def get_posterior(self):
        """Get the posterior.
        
        Returns
        -------
        posterior : pandas.DataFrame
            Posterior samples as a pandas DataFrame.

        Sets the following attributes:
        - posterior_result: Bilby result object.
        - posterior_meta_data: Meta data from the Bilby result object.
        - posterior: Posterior samples as a pandas DataFrame.

        Raises
        ------
        FileNotFoundError
            If the posterior file does not exist.
        """
        self.posterior_result = self.get_bilby_result()
        self.posterior_meta_data = self.posterior_result.meta_data
        self.posterior = self.posterior_result.posterior
        return self.posterior

    def get_injection(self):
        """Get the injection parameters from injection file."""
        if self.injection_file is None:
            raise ValueError("Injection file is not provided.")
        try:
            injection = pd.read_csv(self.injection_file, sep=" ")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot find injection file: {self.injection_file}")
        # add other params from result
        injection["minimum_frequency"] = self.posterior.iloc[0]["minimum_frequency"]
        injection["mean_per_ano"] = 0 #self.posterior.iloc[0]["mean_per_ano"]
        return injection