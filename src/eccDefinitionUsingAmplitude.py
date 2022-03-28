"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from eccDefinition import eccDefinition
from scipy import signal
import numpy as np


class measureEccentricityUsingAmplitude(eccDefinition):
    """Measure eccentricity by finding extrema location from amplitude."""

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
        """
        return signal.argrelextrema(self.amp22, np.greater,
                                    order=order)[0]

    def find_troughs(self, order):
        """Find the troughs in the frequency.

        parameters:
        -----------
        order: window/width of peaks
        """
        return signal.argrelextrema(self.amp22, np.less,
                                    order=order)[0]
