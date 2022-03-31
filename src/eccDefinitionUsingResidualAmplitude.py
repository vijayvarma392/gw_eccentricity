"""
Find peaks and troughs using Residual Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from eccDefinitionUsingAmplitude import eccDefinitionUsingAmplitude
from utils import get_peak_via_quadratic_fit
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


class eccDefinitionUsingResidualAmplitude(eccDefinitionUsingAmplitude):
    """Measure eccentricity by finding extrema from residual amplitude."""

    def __init__(self, dataDict):
        """Init for eccDefinitionUsingResidualAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the eccentric and quasi
        circular waveform data.
        For residual amplitude method we need quasi-circular modes
        in additionn to the eccentric modes. Provide it as a dictionary
        for the key `hlm0` in the dataDict dictionary. Keys for the
        modes in the mode dictionary should be of the form `(l,m)`.
        """
        super().__init__(dataDict)

    def get_data_for_finding_extrema(self):
        """Get the data for extrema finding."""
        self.hlm_zeroecc = self.dataDict["hlm_zeroecc"]
        self.t_zeroecc = self.dataDict["t_zeroecc"]
        self.h22_zeroecc = self.hlm_zeroecc[(2, 2)]
        self.t_zeroecc = self.t_zeroecc - get_peak_via_quadratic_fit(
            self.t_zeroecc,
            self.h22_zeroecc)[0]
        self.quasi_circ_amp_interp = InterpolatedUnivariateSpline(
            self.t_zeroecc, np.abs(self.h22_zeroecc))
        self.res_amp22 = self.amp22 - self.quasi_circ_amp_interp(self.t)
        return self.res_amp22
