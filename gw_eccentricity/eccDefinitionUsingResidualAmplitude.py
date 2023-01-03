"""
Find pericenters and apocenters using Residual Amplitude.

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
        self.method = "ResidualAmplitude"
        self.label_for_data_for_finding_extrema = r"$\Delta A_{22}$"

    def check_and_raise_zeroecc_data_not_found(self, method):
        """Raise exception if zeroecc data not found for Residual method."""
        if "hlm_zeroecc" not in self.dataDict:
            raise Exception(f"Method {method} must have zeroecc data in "
                            "dataDict. 'hlm_zeroecc' data not found.")
        if "t_zeroecc" not in self.dataDict:
            raise Exception(f"Method {method} must have zeroecc data in "
                            "dataDict. 't_zeroecc' data not found.")

    def get_data_for_finding_extrema(self):
        """Get the data for extrema finding."""
        self.check_and_raise_zeroecc_data_not_found("ResidualAmplitude")
        return self.res_amp22
