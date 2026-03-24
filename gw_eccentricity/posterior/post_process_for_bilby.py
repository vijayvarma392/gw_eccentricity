"""Reconstruct eccentricity posterior from Bilby generated posterior."""
import os
import bilby
import pandas as pd
from .post_process import PostProcess


class PostProcessForBilby(PostProcess):
    """Derived class for reconstructing eccentricity posterior for Bilby.

    Uses built-in methods in Bilby to read posterior results from a
    Bilby-generated posterior file (.json, .hdf5, or .pkl).
    """

    def __init__(self, *args, **kwargs):
        """Init for PostProcessForBilby class.

        Notes
        -----
        ``super().__init__()`` calls ``self.get_posterior()``, which sets
        ``self.bilby_result`` as a side effect. Therefore
        ``self.bilby_result`` is safe to access after the super().__init__
        call returns.
        """
        super().__init__(*args, **kwargs)
        self.posterior_type = "Bilby"
        # self.bilby_result is set inside get_posterior(),
        # which is called by super().__init__() above.
        self.posterior_meta_data = self.bilby_result.meta_data

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
        if not os.path.exists(self.posterior_file):
            raise FileNotFoundError(
                f"Cannot find Bilby posterior file: {self.posterior_file}")
        return bilby.result.read_in_result(filename=self.posterior_file)

    def get_posterior(self):
        """Get posterior DataFrame from Bilby result file.

        Sets ``self.bilby_result`` as a side effect so that
        ``__init__`` can access Bilby metadata after this call returns.

        Returns
        -------
        posterior : pandas.DataFrame
            DataFrame of posterior samples.
        """
        self.bilby_result = self.get_bilby_result()
        return self.bilby_result.posterior  # already a DataFrame
