"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from .eccDefinition import eccDefinition
from scipy.signal import find_peaks
from .utils import check_kwargs_and_set_defaults
import numpy as np


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

        # Sanity check extrema_finding_kwargs and set default values
        self.extrema_finding_kwargs = check_kwargs_and_set_defaults(
            self.extra_kwargs['extrema_finding_kwargs'],
            self.get_default_extrema_finding_kwargs(),
            "extrema_finding_kwargs")

    def get_default_extrema_finding_kwargs(self):
        """Defaults for extrema_finding_kwargs."""
        # TODO: Set width more smartly
        default_extrema_finding_kwargs = {
            "height": None,
            "threshold": None,
            "distance": None,
            "prominence": None,
            "width": self.get_default_width_for_peak_finder(),
            "wlen": None,
            "rel_height": 0.5,
            "plateau_size": None}
        return default_extrema_finding_kwargs

    def get_data_for_finding_extrema(self):
        """Get data to be used for finding extrema location.

        In the derived classes, one need to override this function
        to return the appropriate data that is to be used. For example,
        in residual amplitude method, this function would return
        residual amp22, whereas for frequency method, it would
        return omega22 and so on.
        """
        return self.amp22

    def find_extrema(self, extrema_type="maxima"):
        """Find the extrema in the amp22.

        parameters:
        -----------
        extrema_type: either maxima, peaks, minima or troughs
        extrema_finding_kwargs: Dictionary of arguments to be passed to the
        peak finding function. Here we use scipy.signal.find_peaks for finding
        peaks. Hence the arguments are those that are allowed in that function

        returns:
        ------
        array of positions of extrema.
        """
        if extrema_type in ["maxima", "peaks"]:
            sign = 1
        elif extrema_type in ["minima", "troughs"]:
            sign = - 1
        else:
            raise Exception("`extrema_type` must be one of ['maxima', "
                            "'minima', 'peaks', 'troughs']")

        return find_peaks(
            sign * self.data_for_finding_extrema,
            **self.extrema_finding_kwargs)[0]

    def get_default_width_for_peak_finder(
            self,
            width_for_unit_timestep=50):
        """Get the minimal value of width parameter for extrema finding.

        The extrema finding method, i.e., find_peaks from scipy.signal
        needs a "width" parameter that is used to determine the minimal
        separation between consecutive extrema. If the "width" is too small
        then some noisy features in the signal might be mistaken for extrema
        and on the other hand if the "width" is too large then we might miss
        an extrema.

        This function gets an appropriate width by scaling it with the
        time steps in the time array of the waveform data.

        Parameters:
        ----------
        width_for_unit_timestep:
            Width to use when the time step in the wavefrom data is 1.

        Returns:
        -------
        width:
            Minimal width to separate consecutive peaks.
        """
        return int(width_for_unit_timestep / (self.t[1] - self.t[0]))

    def get_width_for_peak_finder_from_phase22(self, t_for_width=-100):
        """Get the minimal value of width parameter for extrema finding.

        The extrema finding method, i.e., find_peaks from scipy.signal
        needs a "width" parameter that is used to determine the minimal
        separation between consecutive extrema. If the "width" is too small
        then some noisy features in the signal might be mistaken for extrema
        and on the other hand if the "width" is too large then we might miss
        an extrema.

        This function tries to use the phase22 to get a reasonable value of
        "width" by looking at the time scale over which the phase22 changes by
        about 4pi because the change in phase22 over one orbit would be
        approximately twice the change in the orbital phase which is about 2pi.

        Parameters:
        ----------
        t_for_width:
            The time at which the width parameter is determined. We want to do
            this near the merger as this is where the time between extrema is
            the smallest, and the width parameter sets the minimal separation
            between extrema. Default is -100.
            NOTE: Here we assume dimensionless units for time. This should be
            changed in the future to make it work with physical units as well.

        Returns:
        -------
        width:
            Minimal width to separate consecutive peaks.
        """
        # get phase22 at t_for_width
        phase22_t_for_width = self.phase22[
            np.argmin(np.abs(self.t - t_for_width))]
        # get the time where phase22 = phase22_t_for_width + 4 pi
        t_at_one_orbit_after_t_for_width = self.t[np.argmin(
            np.abs(self.phase22 - (phase22_t_for_width + 4 * np.pi)))]
        # change in time over which phase22 change by 4 pi from t_for_width
        dt = t_at_one_orbit_after_t_for_width - t_for_width
        # get the width using dt and the time step
        width = dt / (self.t[1] - self.t[0])
        # we want to use a width that is always smaller than the separation
        # between extrema, otherwise we might miss a few peaks near merger
        return int(width / 4)
