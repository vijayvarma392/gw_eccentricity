"""
Find pericenters and apocenters using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from scipy.signal import find_peaks
from .eccDefinition import eccDefinition


class eccDefinitionUsingAmplitude(eccDefinition):
    """Define eccentricity by finding extrema location from amplitude."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)
        self.data_for_finding_extrema = self.get_data_for_finding_extrema()
        self.label_for_data_for_finding_extrema = self.get_label_for_plots("amp")
        self.method = "Amplitude"
        # use a flag so the the function
        # `get_segment_of_data_for_finding_extrema` to get relevant
        # segment of is called only once.
        self.get_segment_of_data = self.extra_kwargs["use_segment"]

    def get_data_for_finding_extrema(self):
        """Get data to be used for finding extrema location.

        In the derived classes, one need to override this function
        to return the appropriate data that is to be used. For example,
        in residual amplitude method, this function would return
        residual amp_gw, whereas for frequency method, it would
        return omega_gw and so on.
        """
        return self.amp_gw

    def find_extrema(self, extrema_type="pericenters"):
        """Find the extrema in the amp_gw.

        parameters:
        -----------
        extrema_type:
            Either "pericenters" or "apocenters".

        returns:
        ------
        array of positions of extrema.
        """
        if extrema_type == "pericenters":
            sign = 1
        elif extrema_type == "apocenters":
            sign = - 1
        else:
            raise Exception("`extrema_type` must be either 'pericenters'"
                            " or 'apocenters'")
        # We need to get the relevant segment from the full data only
        # once, whereas `find_extrema` is called twice, once for the
        # pericenters and then for the apocenters. We handle this
        # by setting the flag `get_segment_of_data` to False after
        # `self.get_segment_of_data_for_finding_extrema` is called for
        # the first time.
        if self.get_segment_of_data:
            self.get_segment_of_data_for_finding_extrema()
            self.get_segment_of_data = False

        return find_peaks(
            sign * self.data_for_finding_extrema,
            **self.extrema_finding_kwargs)[0]
