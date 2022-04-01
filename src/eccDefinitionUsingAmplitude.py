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

    def get_data_for_finding_extrema(self):
        """Get data to be used for finding extrema location.

        In the derived classes, one need to override this function
        to return the appropriate data that is to be used. For example,
        in residual amplitude method, this function would return
        residual amp22, whereas for frequency method, it would
        return omega22 and so on.
        """
        return self.amp22

    def find_extrema(self, which="maxima", extrema_finding_keywords=None):
        """Find the extrema in the amp22.

        parameters:
        -----------
        which: either maxima, peaks, minima or troughs
        extrema_finding_keywords: Dictionary of arguments to be passed to the
        peak finding function. Here we use scipy.signal.find_peaks for finding
        peaks. Hence the arguments are those that are allowed in that function

        returns:
        ------
        array of positions of extrema.
        """
        data_to_find_extrema = self.get_data_for_finding_extrema()

        default_extrema_finding_keywords = {"height": None,
                                            "threshold": None,
                                            "distance": None,
                                            "prominence": None,
                                            "width": None,
                                            "wlen": None,
                                            "rel_height": 0.5,
                                            "plateau_size": None}

        # make it iterable
        if extrema_finding_keywords is None:
            extrema_finding_keywords = {}

        # Sanity check for arguments passed to the find_peak function
        for keyword in extrema_finding_keywords:
            if keyword not in default_extrema_finding_keywords:
                raise ValueError(f"Invalid key {keyword} is "
                                 "extrema_finding_keywords. Should be one of "
                                 f"{default_extrema_finding_keywords.keys()}")

        # Update keyword if passed by user
        for keyword in default_extrema_finding_keywords:
            if keyword in extrema_finding_keywords:
                default_extrema_finding_keywords[keyword] \
                    = extrema_finding_keywords[keyword]

        # If width is None, make a reasonable guess
        if default_extrema_finding_keywords["width"] is None:
            default_extrema_finding_keywords["width"] = 10

        if which == "maxima" or which == "peaks":
            sign = 1
        elif which == "minima" or which == "troughs":
            sign = - 1
        else:
            raise Exception("`which` must be one of ['maxima', 'minima',"
                            " 'peaks', 'troughs']")

        return find_peaks(
            sign * data_to_find_extrema, **default_extrema_finding_keywords)[0]
