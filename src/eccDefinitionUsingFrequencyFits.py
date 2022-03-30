"""
Find peaks and troughs using frequency fits.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from eccDefinition import eccDefinition


class measureEccentricityUsingFrequencyFits(eccDefinition):
    """Measure eccentricity by finding extrema location using freq fits."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithFrequencyFits class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        eccDefinition.__init__(self, dataDict)

    def find_extrema(self, which="maxima", height=None, threshold=None,
                     distance=None, prominence=None, width=10, wlen=None,
                     rel_height=0.5, plateau_size=None):
        """Find the extrema in the frequency using power law fits.

        parameters:
        -----------
        which: either maxima or minima
        see scipy.signal.find_peaks for rest or the arguments.

        returns:
        ------
        array of positions of extrema in the frequency.
        """
        raise NotImplementedError("...To be implemented by Harald.")
