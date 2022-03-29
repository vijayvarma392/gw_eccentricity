"""
Find peaks and troughs using Residual Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from eccDefinition import eccDefinition, get_peak_via_quadratic_fit
from scipy import signal
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class measureEccentricityUsingResidualAmplitude(eccDefinition):
    """Measure eccentricity by finding extrema from residual amplitude."""

    def __init__(self, dataDict):
        """Init for measureEccentricityWithResidualAmplitude class.

        parameters:
        ----------
        dataDict: Dictionary containing the eccentric and quasi
        circular waveform data.
        For residual amplitude method we need quasi-circular modes
        in additionn to the eccentric modes. Provide it as a dictionary
        for the key `hlm0` in the dataDict dictionary. Keys for the
        modes in the mode dictionary should be of the form `(l,m)`.
        """
        eccDefinition.__init__(self, dataDict)
        self.hlm0 = self.dataDict["hlm0"]
        self.time0 = self.dataDict["t0"]
        self.h220 = self.hlm0[(2, 2)]
        self.time0 = self.time0 - get_peak_via_quadratic_fit(
            self.time0,
            self.h220)[0]
        self.quasi_circ_amp_interp = InterpolatedUnivariateSpline(
            self.time0, np.abs(self.h220))
        self.res_amp22 = self.amp22 - self.quasi_circ_amp_interp(self.time)

    def find_peaks(self, order):
        """Find the peaks in the residual amplitude.

        parameters:
        -----------
        order: window/width of peaks

        return:
        ------
        indices: 1d array conndtainging the indices for location of peaks
        """
        return signal.argrelextrema(self.res_amp22, np.greater,
                                    order=order)[0]

    def find_troughs(self, order):
        """Find the troughs in the residual amplitude.

        parameters:
        -----------
        order: window/width of peaks

        return:
        ------
        indices: 1d array conndtainging the indices for location of troughs
        """
        return signal.argrelextrema(self.res_amp22, np.less,
                                    order=order)[0]
