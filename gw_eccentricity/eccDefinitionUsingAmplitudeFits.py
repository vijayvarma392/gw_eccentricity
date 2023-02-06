"""
Find peaks and troughs using amplitude fits.

Part of Eccentricity Definition project.
"""
from .eccDefinitionUsingFrequencyFits import eccDefinitionUsingFrequencyFits
from .plot_settings import labelsDict
from .utils import check_kwargs_and_set_defaults


class eccDefinitionUsingAmplitudeFits(eccDefinitionUsingFrequencyFits):
    """Measure eccentricity by finding extrema location using amp fits."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingWithAmplitudeFits class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)
        self.data_str = "amp22"
        self.label_for_data_for_finding_extrema = labelsDict[self.data_str]
        self.label_for_fit_to_data_for_finding_extrema \
            = labelsDict[f"{self.data_str}_fit"]
        self.method = "AmplitudeFits"
        self.kwargs_for_fits_methods = check_kwargs_and_set_defaults(
            self.extra_kwargs['kwargs_for_fits_methods'],
            self.get_default_kwargs_for_fits_methods(),
            "kwargs_for_fits_methods",
            "eccDefinitionUsingAmplitudeFits.get_default_kwargs_for_fits_methods()")
        self.set_fit_variables()
        # Make a copy of amp22 and use it to set data_for_finding_extrema.
        # This would ensure that any modification of data_for_finding_extrema
        # does not modify amp22.
        self.data_for_finding_extrema = self.amp22.copy()
        # FIXME: Find a better solution
        # It turns out that since in MKS units amplitude is very small
        # The envelope fitting does not work properly. Maybe there is a better
        # way to do this but scaling the amp22 data by its value at the global
        # peak (the merger time) solves this issue.
        self.data_for_finding_extrema /= self.amp22_merger

    def get_default_kwargs_for_fits_methods(self):
        """Get default fits kwargs.

        See eccDefinitionUsingFrequencyFits.get_default_kwargs_for_fits_methods
        for documentation.
        """
        return {
            # The PN exponent as approximation
            # Jolien and Creighton Chapter 3, Equation 3.190a, 3.190b
            "nPN": -1./4,
            "fit_bounds_max_amp_factor": 10,
            "fit_bounds_max_nPN_factor": 10,
            "prominence_factor": 0.025,
            "distance_factor": 0.5,
            "num_orbits": 3,
            "num_orbits_for_global_fit": 10,
            "return_diagnostic_data": False
        }
