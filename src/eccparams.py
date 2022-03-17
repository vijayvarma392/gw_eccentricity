from scipy import signal
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt


class estimate_parameters:
    """estimate eccentricity of waveform form waveform data.

    parameters:
    ----------
    data: list
        In the form of [times, strain], strain is the complex strain.
    order: int
        Window for the argrelextrema function from scipy.signal. Default is 1.
    method: str
        Method to obtain the position of minia and maxima of the
        frequency of the signal. Must be one of "amp", "freq" or "res_amp".
        Default is "amp".
    use_MR: bool
        Whether to use Merger Ringdown part or not. Default is False.
    inspiral_cutoff: int
        Where to cut the signal at. Merger is assumed to be
        at t = 0. Default value is -100 in geometrical units.
    circ_data: list
        Quasi circular waveform data with all the parameters the same
        as for data with ecc set to zero. Default is None.
        This should be not None when method is "res_amp".
    **kwargs: for the InterpolatedUnivariateSpline.
    """

    def __init__(self,
                 data,
                 order=1,
                 method="amp",
                 use_MR=False,
                 inspiral_cutoff=-100,
                 circ_data=None,
                 **kwargs):
        self.data = data
        self.order = order
        self.times, self.strain = self.data
        self.times = self.times - self.times[np.argmax(abs(self.strain))]
        self.use_MR = use_MR
        if not self.use_MR:
            if self.times[0] < inspiral_cutoff:
                inspiral_idxs = self.times < inspiral_cutoff
                self.times = self.times[inspiral_idxs]
                self.strain = self.strain[inspiral_idxs]
            else:
                raise Exception("inspiral length of the waveform is "
                                "shorter than the inspiral_cutoff.")
        self.amp = np.abs(self.strain)
        self.phase = np.unwrap(np.angle(self.strain))
        self.method = method
        self.phase_interp = InterpolatedUnivariateSpline(self.times,
                                                         self.phase,
                                                         **kwargs)
        self.amp_interp = InterpolatedUnivariateSpline(self.times, self.amp,
                                                       **kwargs)
        self.omega_interp = self.phase_interp.derivative()
        self.omega = np.abs(self.omega_interp(self.times))
        self.freq = np.abs(self.omega_interp(self.times)) / (2 * np.pi)
        self.circ_data = circ_data
        if self.method == "res_amp":
            if self.circ_data is None:
                raise Exception("res_amp method needs circ_data.")
            else:
                self.circ_times, self.circ_strain = self.circ_data
                self.circ_times = (
                    self.circ_times
                    - self.circ_times[
                        np.argmax(abs(self.circ_strain))])
                # check if the circular waveform is longer than the ecc one
                if abs(self.times[0]) > abs(self.circ_times[0]):
                    raise Exception("The eccentric waveform is longer than the"
                                    "circular one. This would cause inaccurate"
                                    " result due to extrapolation. Rerun with "
                                    "longer circular waveform. For example "
                                    "start with smaller initial frequency.")
                if not self.use_MR:
                    if self.circ_times[0] < inspiral_cutoff:
                        inspiral_idxs = self.circ_times < inspiral_cutoff
                        self.circ_times = self.circ_times[inspiral_idxs]
                        self.circ_strain = self.circ_strain[inspiral_idxs]
                    else:
                        raise Exception("inspiral length of the waveform is "
                                        "shorter than the inspiral_cutoff.")
                self.circ_amp = np.abs(self.circ_strain)
                self.circ_phase = np.unwrap(
                    np.angle(self.circ_strain))
                self.circ_phase_interp = InterpolatedUnivariateSpline(
                    self.circ_times,
                    self.circ_phase,
                    **kwargs)
                self.circ_amp_interp = InterpolatedUnivariateSpline(
                    self.circ_times,
                    self.circ_amp,
                    **kwargs)
                self.circ_omega_interp = self.circ_phase_interp.derivative()
                self.circ_omega = np.abs(
                    self.circ_omega_interp(self.circ_times))
                self.circ_freq = np.abs(
                    self.circ_omega_interp(self.circ_times)) / (2 * np.pi)

    def get_max_idx(self):
        """get the indices where frequency has maxima
        """
        if self.method == "amp":
            return signal.argrelextrema(self.amp, np.greater, order=self.order)
        elif self.method == "freq":
            return signal.argrelextrema(self.freq, np.greater,
                                        order=self.order)
        elif self.method == "res_amp":
            res_amp = self.amp - self.circ_amp_interp(self.times)
            return signal.argrelextrema(res_amp, np.greater, order=self.order)
        else:
            raise Exception(
                "Unknown method."
                " `method` should be one of ['amp', 'freq', 'res_amp']")

    def get_min_idx(self):
        """get the indices where frequency has minima
        """
        if self.method == "amp":
            return signal.argrelextrema(self.amp, np.less, order=self.order)
        elif self.method == "freq":
            return signal.argrelextrema(self.freq, np.less, order=self.order)
        elif self.method == "res_amp":
            res_amp = self.amp - self.circ_amp_interp(self.times)
            return signal.argrelextrema(res_amp, np.less, order=self.order)
        else:
            raise Exception(
                "Unknown method."
                " `method` should be one of ['amp', 'freq', 'res_amp']")

    def get_f_max_interp(self, **kwargs):
        """get the interpolating function for finding maxima of frequency
        """
        idx_f_maxs = self.get_max_idx()
        if len(idx_f_maxs[0]) >= 2:
            return InterpolatedUnivariateSpline(
                self.times[idx_f_maxs],
                self.freq[idx_f_maxs],
                **kwargs)
        else:
            print("...Number of maxima is less than 2."
                  " Not able to interpolate.")
            return None

    def get_f_min_interp(self, **kwargs):
        """get the interpolating function for finding minima of frequency
        """
        idx_f_mins = self.get_min_idx()
        if len(idx_f_mins[0]) > 2:
            return InterpolatedUnivariateSpline(
                self.times[idx_f_mins],
                self.freq[idx_f_mins],
                **kwargs)
        else:
            print("...Number of minima is less than 2."
                  " Not able to interpolate.")
            return None

    def get_amp_max_interp(self, **kwargs):
        """get the interpolating function for finding maxima of amplitude
        """
        idx_amp_maxs = self.get_max_idx()
        if len(idx_amp_maxs[0]) >= 2:
            return InterpolatedUnivariateSpline(
                self.times[idx_amp_maxs],
                self.amp[idx_amp_maxs],
                **kwargs)
        else:
            print("...Number of maxima is less than 2."
                  " Not able to interpolate.")
            return None

    def get_amp_min_interp(self, **kwargs):
        """get the interpolating function for finding minima of amplitude
        """
        idx_amp_mins = self.get_min_idx()
        if len(idx_amp_mins[0]) > 2:
            return InterpolatedUnivariateSpline(
                self.times[idx_amp_mins],
                self.amp[idx_amp_mins],
                **kwargs)
        else:
            print("...Number of minima is less than 2."
                  " Not able to interpolate.")
            return None

    def get_ecc_interp(self, **kwargs):
        """get the interpolating function for eccentricity
        """
        f_max_interp = self.get_f_max_interp(**kwargs)
        f_min_interp = self.get_f_min_interp(**kwargs)

        if f_max_interp is not None and f_min_interp is not None:
            eccvalues = ((np.sqrt(np.abs(f_max_interp(self.times)))
                          - np.sqrt(np.abs(f_min_interp(self.times))))
                         / (np.sqrt(np.abs(f_max_interp(self.times)))
                            + np.sqrt(np.abs(f_min_interp(self.times)))))
            return InterpolatedUnivariateSpline(
                self.times, eccvalues, **kwargs)
        else:
            print("...Sufficient number of minima or maxima in orbital "
                  "frequency is not found. Cannot get eccentricity "
                  "interpolator. Most probably the eccentricity is too small."
                  " Returning eccentricity to be zero.")
            return None

    def get_ecc_at_times(self, selected_times, **kwargs):
        """get the value of eccentricity at given time
        """

        ecc_interp = self.get_ecc_interp(**kwargs)
        if ecc_interp is None:
            if type(selected_times) == np.ndarray:
                return np.zeros(len(selected_times))
            else:
                return np.array(0)
        else:
            return ecc_interp(selected_times)

    def get_times_at_peri(self):
        """get time of pericenter passage
        """
        idx_peri = self.get_max_idx()
        return self.times[idx_peri]

    def get_meanPerAno_interp(self, **kwargs):
        """get the interpolating function for mean periastron anomaly
        """
        idx_peris = self.get_max_idx()[0]

        meanperAnoVals = np.array([])
        timeVals = np.array([])

        idx = 0
        while idx < len(idx_peris) - 1:
            orbital_period = (self.times[idx_peris[idx + 1]]
                              - self.times[idx_peris[idx]])
            time_since_last_peri = (
                self.times[idx_peris[idx]: idx_peris[idx + 1]]
                - self.times[idx_peris[idx]])
            meanperAno = 2 * np.pi * time_since_last_peri / orbital_period
            meanperAnoVals = np.append(meanperAnoVals, meanperAno)
            timeVals = np.append(
                timeVals,
                self.times[idx_peris[idx]: idx_peris[idx + 1]])
            idx += 1

        return InterpolatedUnivariateSpline(
            timeVals, meanperAnoVals, **kwargs)

    def get_meanPerAno_at_times(self, selected_times, **kwargs):
        """get value mean periastron anomaly at given times.
        """
        meanperAno_interp = self.get_meanPerAno_interp(**kwargs)
        return meanperAno_interp(selected_times)

    def get_valid_times(self):
        idx_f_mins = self.get_min_idx()[0]
        idx_f_maxs = self.get_max_idx()[0]
        return (
            self.times[max(idx_f_mins[0], idx_f_maxs[0]):
                       min(idx_f_mins[-1], idx_f_maxs[-1])])

    def plot_freq(self, fig=None, ax=None,
                  marker="o", figsize=(6, 4), **kwargs):
        """plot the freq marking the max and min points. This
        could be used to visualize how the peaks and troughs are being
        picked by the np.argrelextrema algorithm"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.times, np.pi * self.get_f_max_interp()(self.times),
                label=r"$\omega_p$", **kwargs)
        ax.scatter(self.times[self.get_max_idx()],
                   np.pi * self.freq[self.get_max_idx()],
                   marker=marker,
                   **kwargs)
        ax.plot(self.times, np.pi * self.get_f_min_interp()(self.times),
                label=r"$\omega_a$", **kwargs)
        ax.scatter(self.times[self.get_min_idx()],
                   np.pi * self.freq[self.get_min_idx()],
                   marker=marker,
                   **kwargs)
        ax.plot(self.times, np.pi * self.freq, label=r"$\omega_{orb}$",
                **kwargs)
        ax.set_ylabel(r"$\omega$")
        ax.set_xlabel("times")
        ax.legend()
        return fig, ax

    def plot_ecc_and_freq(self,
                          fig=None,
                          ax=None,
                          marker="o",
                          figsize=(6, 8),
                          sharex=True,
                          **kwargs):
        """plot the ecc marking the max and min points. This
        could be used to visualize how the peaks and troughs are being
        picked by the np.argrelextrema algorithm"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=2, figsize=figsize, sharex=sharex)
        ax[1].plot(self.times, np.pi * self.get_f_max_interp()(self.times),
                   **kwargs)
        ax[1].scatter(self.times[self.get_max_idx()],
                      np.pi * self.freq[self.get_max_idx()],
                      marker=marker, **kwargs, label=r"$\omega_p$")
        ax[1].plot(self.times, np.pi * self.get_f_min_interp()(self.times),
                   **kwargs)
        ax[1].scatter(self.times[self.get_min_idx()],
                      np.pi * self.freq[self.get_min_idx()],
                      marker=marker, **kwargs, label=r"$\omega_a$")
        ax[1].plot(self.times, np.pi * self.freq, label=r"$\omega_{orb}$",
                   **kwargs)
        ax[0].plot(self.times, self.get_ecc_at_times(self.times),
                   label="eccentricity",
                   **kwargs)
        ax[1].set_xlabel("times")
        ax[1].legend()
        ax[0].legend()
        plt.subplots_adjust(hspace=0)
        return fig, ax

    def plot_amp(self, marker="o", **kwargs):
        """plot the amplitude marking the max and min points. This
        could be used to visualize how the peaks and troughs are being
        picked by the np.argrelextrema algorithm"""
        fig, ax = plt.subplots()
        ax.plot(self.times, self.get_amp_max_interp()(self.times))
        ax.plot(self.times[self.get_max_idx()], self.amp[self.get_max_idx()],
                marker=marker, **kwargs)
        ax.plot(self.times, self.get_amp_min_interp()(self.times))
        ax.plot(self.times[self.get_min_idx()], self.amp[self.get_min_idx()],
                marker=marker, **kwargs)
        ax.plot(self.times, self.amp)
        ax.set_ylabel("amp")
        ax.set_xlabel("times")
        return fig, ax
