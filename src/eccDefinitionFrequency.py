"""
Find peaks and troughs using frequency.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from eccDefinitionAmplitude import measureEccentricityAmplitude


class measureEccentricityFrequency(measureEccentricityAmplitude):
    """Measure eccentricity by finding extrema location from frequency."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithFrequency class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(dataDict)
        self.data_to_find_extrema = self.omega22
