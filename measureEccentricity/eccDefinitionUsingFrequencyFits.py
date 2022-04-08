"""
Find peaks and troughs using frequency fits.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from .eccDefinition import eccDefinition


class eccDefinitionUsingFrequencyFits(eccDefinition):
    """Measure eccentricity by finding extrema location using freq fits."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingWithFrequencyFits class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)


    def find_extrema(self, which="maxima"):
        """Find the extrema in the data.

        parameters:
        -----------
        which:
            One of 'maxima', 'peaks', 'minima' or 'troughs'.

        returns:
        ------
        array of positions of extrema.
        """
        raise NotImplementedError("To be implemented by Harald.")
