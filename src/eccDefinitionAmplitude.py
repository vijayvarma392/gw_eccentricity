"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from eccDefinition import eccDefinition
from scipy.signal import find_peaks


class measureEccentricityAmplitude(eccDefinition):
    """Measure eccentricity by finding extrema location from amplitude."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(dataDict)
        # data_to_find_extrema is the data that will be used to find the
        # location of the peaks and troughs. In some of the derived classes
        # using measureEccentricityAmplitude as base class, all that is needed
        # to do is to assign this variable to something else like omega22
        # or residul amp22. This way, these classes do not have to define the
        # find_extrema function again, the inherited one will work fine
        self.data_to_find_extrema = self.amp22

    def find_extrema(self, which="maxima", height=None, threshold=None,
                     distance=None, prominence=None, width=10, wlen=None,
                     rel_height=0.5, plateau_size=None):
        """Find the extrema in the amp22.

        parameters:
        -----------
        which: either maxima or minima
        see scipy.signal.find_peaks for rest or the arguments.

        returns:
        ------
        array of positions of extrema.
        """
        if which == "maxima" or which == "peaks":
            sign = 1
        elif which == "minima" or which == "troughs":
            sign = - 1
        else:
            raise Exception("`which` must be one of ['maxima', 'minima',"
                            " 'peaks', 'troughs']")
        return find_peaks(sign * self.data_to_find_extrema, height, threshold,
                          distance, prominence, width, wlen, rel_height,
                          plateau_size)[0]
