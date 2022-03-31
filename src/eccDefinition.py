"""
Base module to measure eccentricity and mean anomaly for given waveform data.

Part of Defining eccentricity project
Md Arif Shaikh, Mar 29, 2022
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from utils import get_peak_via_quadratic_fit


class eccDefinition:
    """Measure eccentricity from given waveform data dictionary."""

    def __init__(self, dataDict):
        """Init eccDefinition class.

        parameters:
        ---------
        dataDict: dictionary containing waveform modes dict, time etc
        should follow the format {"t": time, "hlm": modeDict, ..}
        and modeDict = {(l, m): hlm_mode_data}
        for ResidualAmplitude method, provide "t_zeroecc" and "hlm_zeroecc"
        as well in the dataDict.
        """
        self.dataDict = dataDict
        self.t = self.dataDict["t"]
        self.hlm = self.dataDict["hlm"]
        self.h22 = self.hlm[(2, 2)]
        self.amp22 = np.abs(self.h22)
        # shift the time axis to make t = 0 at merger
        # t_ref would be then negative. This helps
        # when subtracting quasi circular amplitude from
        # eccentric amplitude in residual amplitude method
        self.t = self.t - get_peak_via_quadratic_fit(
            self.t, self.amp22)[0]
        self.phase22 = - np.unwrap(np.angle(self.h22))
        self.omega22 = np.gradient(self.phase22, self.t)

    def find_extrema(self, which="maxima", height=None, threshold=None,
                     distance=None, prominence=None, width=10, wlen=None,
                     rel_height=0.5, plateau_size=None):
        """Find the extrema in the data.

        parameters:
        -----------
        which: either maxima or minima
        see scipy.signal.find_peaks for rest or the arguments.

        returns:
        ------
        array of positions of extrema.
        """
        raise NotImplementedError("Please override me.")

    def interp_extrema(self, which="maxima", height=None, threshold=None,
                       distance=None, prominence=None, width=10, wlen=None,
                       rel_height=0.5, plateau_size=None, **kwargs):
        """Interpolator through extrema.

        parameters:
        -----------
        which: either maxima or minima
        see scipy.signal.find_peaks for rest or the arguments.
        **kwargs for Interpolatedunivariatespline

        returns:
        ------
        spline through extrema, positions of extrema
        """
        extrema_idx = self.find_extrema(which, height, threshold, distance,
                                        prominence, width, wlen, rel_height,
                                        plateau_size)
        if len(extrema_idx) >= 2:
            return InterpolatedUnivariateSpline(self.t[extrema_idx],
                                                self.omega22[extrema_idx],
                                                **kwargs), extrema_idx
        else:
            print("...Number of extrema is less than 2. Not able"
                  " to create an interpolator.")
            return None

    def measure_ecc(self, t_ref, height=None, threshold=None,
                    distance=None, prominence=None, width=10, wlen=None,
                    rel_height=0.5, plateau_size=None, **kwargs):
        """Measure eccentricity and mean anomaly at reference time.

        parameters:
        ----------
        t_ref: reference time to measure eccentricity and mean anomaly.
        see scipy.signal.find_peaks for rest or the arguments.
        kwargs: to be passed to the InterpolatedUnivariateSpline

        returns:
        --------
        ecc_ref: measured eccentricity at t_ref
        mean_ano_ref: measured mean anomaly at t_ref
        """
        t_ref = np.atleast_1d(t_ref)
        default_spline_kwargs = {"w": None,
                                 "bbox": [None, None],
                                 "k": 3,
                                 "ext": 0,
                                 "check_finite": False}
        for kw in default_spline_kwargs.keys():
            if kw in kwargs:
                default_spline_kwargs[kw] = kwargs[kw]

        omega_peaks_interp, omega_peaks_idx = self.interp_extrema(
            "maxima", height, threshold,
            distance, prominence, width,
            wlen, rel_height, plateau_size,
            **default_spline_kwargs)
        omega_troughs_interp = self.interp_extrema(
            "minima", height, threshold,
            distance, prominence, width,
            wlen, rel_height, plateau_size,
            **default_spline_kwargs)[0]

        if omega_peaks_interp is None or omega_troughs_interp is None:
            print("...Sufficient number of peaks/troughs are not found."
                  " Can not create an interpolator. Most probably the "
                  "excentricity is too small. Returning eccentricity to be"
                  " zero")
            ecc_ref = 0
            mean_ano_ref = 0
        else:
            # compute eccentricty from the value of omega_peaks_interp
            # and omega_troughs_interp at t_ref using the fromula in
            # ref. arXiv:2101.11798 eq. 4
            omega_peak_at_t_ref = omega_peaks_interp(t_ref)
            omega_trough_at_t_ref = omega_troughs_interp(t_ref)
            ecc_ref = ((np.sqrt(omega_peak_at_t_ref)
                        - np.sqrt(omega_trough_at_t_ref))
                       / (np.sqrt(omega_peak_at_t_ref)
                          + np.sqrt(omega_trough_at_t_ref)))
            t_peaks = self.t[omega_peaks_idx]
            # check if the t_ref has a peak before and after
            # and compute the mean anomaly using ref. arXiv:2101.11798 eq. 7
            # mean anomaly grows linearly from 0 to 2 pi over
            # the range [t_at_last_peak, t_at_next_peak]
            if t_ref[0] >= t_peaks[0] and t_ref[-1] < t_peaks[-1]:
                mean_ano_ref = np.zeros(len(t_ref))
                for idx, time in enumerate(t_ref):
                    idx_at_last_peak = np.where(t_peaks <= time)[0][-1]
                    t_at_last_peak = t_peaks[idx_at_last_peak]
                    t_at_next_peak = t_peaks[idx_at_last_peak + 1]
                    mean_ano = time - t_at_last_peak
                    mean_ano_ref[idx] = (2 * np.pi * mean_ano
                                         / (t_at_next_peak - t_at_last_peak))
            else:
                raise Exception("Reference time must be within two peaks.")
            if len(t_ref) == 1:
                mean_ano_ref = mean_ano_ref[0]
                ecc_ref = ecc_ref[0]

        return ecc_ref, mean_ano_ref
