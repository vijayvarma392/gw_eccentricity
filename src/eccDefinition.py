"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 28, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict):
        """Init eccDefinition class.

        parameters:
        ----------
        dataDict: Dictionary conntaining time, modes, etc
        """
        self.dataDict = dataDict
        self.time = self.dataDict["t"]
        self.h22 = self.dataDict["(2,2)"]
        self.amp22 = np.abs(self.h22)
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = np.gradient(self.phase22, self.time)

    def find_peaks(self, order=10):
        """Find the peaks in the data.

        parameters:
        -----------
        order: window/width of peaks
        """
        "Not Implemented. Override it."

    def find_troughs(self, order=10):
        """Find the troughs in the data.

        parameters:
        -----------
        order: window/width of troughs
        """
        "Not Implemented. Override it."

    def peaks_interp(self, order=10, **kwargs):
        """Interpolator through peaks."""
        peak_idx = self.find_peaks(order)
        if len(peak_idx) >= 2:
            return InterpolatedUnivariateSpline(self.time[peak_idx],
                                                self.omega22[peak_idx],
                                                **kwargs)
        else:
            print("...Number of peaks is less than 2. Not able"
                  " to create an interpolator.")
            return None

    def troughs_interp(self, order=10, **kwargs):
        """Interpolator through troughs."""
        trough_idx = self.find_troughs(order)
        if len(trough_idx) >= 2:
            return InterpolatedUnivariateSpline(self.time[trough_idx],
                                                self.omega22[trough_idx],
                                                **kwargs)
        else:
            print("...Number of troughs is less than 2. Not able"
                  " to create an interpolator.")
            return None

    def measure_ecc(self, t_ref, order=10, **kwargs):
        """Measure eccentricity and meann anomaly at reference time.

        parameters:
        ----------
        t_ref: reference time to measure eccentricity and mean anomaly.
        order: width of peaks/troughs
        kwargs: any extra kwargs to the peak/trough findining functions.

        returns:
        --------
        ecc_ref: measured eccentricity at t_ref
        mean_ano_ref: measured mean anomaly at t_ref
        """
        default_kwargs = {"w": None,
                          "bbox": [None, None],
                          "k": 3,
                          "ext": 0,
                          "check_finite": False}
        for kw in default_kwargs.keys():
            if kw in kwargs:
                default_kwargs[kw] = kwargs[kw]

        peaks_interpolator = self.peaks_interp(order, **default_kwargs)
        troughs_interpolator = self.troughs_interp(order, **default_kwargs)

        if peaks_interpolator is None or troughs_interpolator is None:
            print("...Sufficiemt number of peaks/troughs are not found."
                  " Can not creator interpolator. Most probably the "
                  "excentricity is too small. Returning eccentricity to be"
                  " zero")
            ecc_ref = 0
        else:
            eccVals = ((np.sqrt(np.abs(peaks_interpolator(self.time)))
                        - np.sqrt(np.abs(troughs_interpolator(self.time))))
                       / (np.sqrt(np.abs(peaks_interpolator(self.time)))
                          + np.sqrt(np.abs(troughs_interpolator(self.times)))))
            ecc_interpolator = InterpolatedUnivariateSpline(self.time, eccVals)
            ecc_ref = ecc_interpolator(t_ref)

        return ecc_ref
