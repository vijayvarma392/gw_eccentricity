"""
Find peaks and troughs using frequency fits.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from eccDefinition import eccDefinition


class measureEccentricityUsingFrequencyFits(eccDefinition):
    """Measure eccentricity by finding extrema location using freq fits."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithFrequency class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        eccDefinition.__init__(self, dataDict)

    def find_peaks(self, order):
        """Find the peaks in the frequency.

        parameters:
        -----------
        order: window/width of peaks

        return:
        ------
        indices for peaks.
        """
        print("...To be implemented by Harald.")

    def find_troughs(self, order):
        """Find the troughs in the frequency.

        parameters:
        -----------
        order: window/width of peaks

        return:
        ------
        indices for troughs.
        """
        print("...To be implemented by Harald.")
