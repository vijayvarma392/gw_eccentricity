"""
Find peaks and troughs using Residual Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from .eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude


class eccDefinitionUsingResidualAmplitude(eccDefinitionUsingAmplitude):
    """Measure eccentricity by finding extrema from residual amplitude."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingResidualAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the eccentric and quasi
        circular waveform data.
        For residual amplitude method we need quasi-circular modes
        in additionn to the eccentric modes. Provide it as a dictionary
        for the key `hlm_zeroecc` and `t_zeroecc` in the dataDict dictionary.
        Keys for the modes in the mode dictionary should be
        of the form `(l, m)`.
        """
        super().__init__(*args, **kwargs)

    def get_data_for_finding_extrema(self):
        """Get the data for extrema finding."""
        return self.res_amp22
