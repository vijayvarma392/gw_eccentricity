"""
Find pericenters and apocenters using Residual Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from .eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude
from .plot_settings import labelsDict


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
        self.label_for_data_for_finding_extrema = labelsDict["res_amp_gw"]

    def check_and_raise_zeroecc_data_not_found(self, method):
        """Raise exception if zeroecc data not found for Residual method.

        For Residual methods, waveform data of the quasicircular counterpart
        is required to compute residual amplitude/frequency which is then
        used to find the locations of the pericenters/apocenters.

        In the dataDict, these are provided using the keys
        - t_zeroecc: 1d array of times of quasicircular waveform.
        - hlm_zeroecc: Dictionary of modes of the quasicircular waveform.
        For more details on the format of the dataDict, see documentation
        of gw_eccentricity.measure_eccentricity.
        """
        # suggest what data to provide when zeroecc amplitude/omega is missing
        # for Residual methods
        suggested_keys = {
            "t_zeroecc": ["t_zeroecc"],
            "amplm_zeroecc": ["amplm_zeroecc", "hlm_zeroecc"],
            "omegalm_zeroecc": ["omegalm_zeroecc", "hlm_zeroecc", "phaselm_zeroecc"]}
        for k in suggested_keys:
            if k not in self.dataDict:
                raise Exception(f"Method {method} must have zeroecc data in "
                                f"dataDict. {k} data not found. "
                                "At least one of the following data should be "
                                f"provided: {suggested_keys[k]}")

    def get_data_for_finding_extrema(self):
        """Get the data for extrema finding."""
        self.check_and_raise_zeroecc_data_not_found("ResidualAmplitude")
        return self.res_amp_gw
