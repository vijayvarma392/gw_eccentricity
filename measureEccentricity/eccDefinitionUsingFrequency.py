"""
Find peaks and troughs using frequency.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from .eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude


class eccDefinitionUsingFrequency(eccDefinitionUsingAmplitude):
    """Measure eccentricity by finding extrema location from frequency."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingFrequency class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)

    def get_data_for_finding_extrema(self):
        """Get the data for extrema finding."""
        return self.omega22
