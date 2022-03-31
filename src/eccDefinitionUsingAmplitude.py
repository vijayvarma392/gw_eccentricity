"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from eccDefinition import eccDefinition
from scipy.signal import find_peaks


class eccDefinitionUsingAmplitude(eccDefinition):
    """Define eccentricity by finding extrema location from amplitude."""

    def __init__(self, dataDict):
        """Init for eccDefinitionUsingAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(dataDict)

    def set_data_for_finding_extrema(self):
        """Set data to be used for finding extrema location.

        In the derived classes, one need to override this function
        to return the appropriate data that is to be used. For example,
        in residual amplitude method, this function would return
        residual amp22, whereas for frequency method, it would
        return omega22 and so on.
        """
        return self.amp22

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
        data_to_find_extrema = self.set_data_for_finding_extrema()
        if which == "maxima" or which == "peaks":
            sign = 1
        elif which == "minima" or which == "troughs":
            sign = - 1
        else:
            raise Exception("`which` must be one of ['maxima', 'minima',"
                            " 'peaks', 'troughs']")
        return find_peaks(sign * data_to_find_extrema, height, threshold,
                          distance, prominence, width, wlen, rel_height,
                          plateau_size)[0]
