"""
Find peaks and troughs using amplitude fits.

Part of Eccentricity Definition project.
"""
from .eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from .plot_settings import labelsDict
import numpy as np


class eccDefinitionUsingAmplitudeFits(eccDefinitionUsingFrequencyFits):
    """Measure eccentricity by finding extrema location using amp fits."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingWithAmplitudeFits class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)
        self.label_for_data_for_finding_extrema = labelsDict["amp22"]
        self.data_for_finding_extrema = self.amp22
        self.method = "AmplitudeFits"
        self.data_str = "amp22"

        # create the shortened data-set for analysis
        merger_idx = np.argmin(np.abs(self.t - self.t_merger))
        idx_end = merger_idx
        if False and (self.extra_kwargs["num_orbits_to_exclude_before_merger"]
                      is not None):
            phase22_at_merger = self.phase22[merger_idx]
            # one orbit changes the 22 mode phase by 4 pi since
            # omega22 = 2 omega_orb
            phase22_num_orbits_earlier_than_merger = (
                phase22_at_merger
                - 4 * np.pi
                * self.extra_kwargs["num_orbits_to_exclude_before_merger"])
            idx_end = np.argmin(
                np.abs(self.phase22
                       - phase22_num_orbits_earlier_than_merger))

        self.t_analyse = self.t[:idx_end] - self.t_merger
        self.data_analyse = self.data_for_finding_extrema[:idx_end]
        self.phase22_analyse = self.phase22[:idx_end]
