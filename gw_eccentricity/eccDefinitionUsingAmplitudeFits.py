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
        self.data_str = "amp22"
        self.method = "AmplitudeFits"
        self.data_for_finding_extrema = self.amp22
        self.data_analyse = self.data_for_finding_extrema[:self.idx_end]
        # FIXME: Find a better solution
        # It turns out that since in MKS units amplitude is very small
        # The envelope fitting does not work properly. Maybe there is a better
        # way to do this but scaling the amp22 data by its value at the global
        # peak (the merger time) solves this issue.
        # However, we don't want to do this for dimless units.
        self.amp22_merger = self.data_for_finding_extrema[
            np.argmin(self.t - self.t_merger)]
        if self.amp22_merger <= 1e-1:
            self.data_analyse /= self.amp22_merger
