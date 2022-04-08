"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from .eccDefinition import eccDefinition
from scipy.signal import find_peaks
from .utils import check_kwargs_and_set_defaults


class eccDefinitionUsingAmplitude(eccDefinition):
    """Define eccentricity by finding extrema location from amplitude."""

    def __init__(self, dataDict):
        """Init for eccDefinitionUsingAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(dataDict)
        self.data_for_finding_extrema = self.get_data_for_finding_extrema()

    def get_data_for_finding_extrema(self):
        """Get data to be used for finding extrema location.

        In the derived classes, one need to override this function
        to return the appropriate data that is to be used. For example,
        in residual amplitude method, this function would return
        residual amp22, whereas for frequency method, it would
        return omega22 and so on.
        """
        return self.amp22

    def find_extrema(self, which="maxima", extrema_finding_kwargs=None):
        """Find the extrema in the amp22.

        parameters:
        -----------
        which: either maxima, peaks, minima or troughs
        extrema_finding_kwargs: Dictionary of arguments to be passed to the
        peak finding function. Here we use scipy.signal.find_peaks for finding
        peaks. Hence the arguments are those that are allowed in that function

        returns:
        ------
        array of positions of extrema.
        """
        default_extrema_finding_kwargs = {"height": None,
                                          "threshold": None,
                                          "distance": None,
                                          "prominence": None,
                                          "width": None,
                                          "wlen": None,
                                          "rel_height": 0.5,
                                          "plateau_size": None}
        # make it iterable
        if extrema_finding_kwargs is None:
            extrema_finding_kwargs = {}

        # Sanity check for arguments passed to the find_peak function
        # and set default values of not given by user
        extrema_finding_kwargs = check_kwargs_and_set_defaults(
            extrema_finding_kwargs,
            default_extrema_finding_kwargs,
            "extrema_finding_kwargs")

        # If width is None, make a reasonable guess
        if extrema_finding_kwargs["width"] is None:
            extrema_finding_kwargs["width"] = 10

        if which == "maxima" or which == "peaks":
            sign = 1
        elif which == "minima" or which == "troughs":
            sign = - 1
        else:
            raise Exception("`which` must be one of ['maxima', 'minima',"
                            " 'peaks', 'troughs']")

        return find_peaks(
            sign * self.data_for_finding_extrema,
            **extrema_finding_kwargs)[0]
