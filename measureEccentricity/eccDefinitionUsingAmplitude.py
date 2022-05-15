"""
Find peaks and troughs using Amplitude.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 28, 2022
"""
from .eccDefinition import eccDefinition
from scipy.signal import find_peaks
from .utils import check_kwargs_and_set_defaults
import matplotlib.pyplot as plt


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
            "width": 50,
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

    def make_diagnostic_plots(self, **kwargs):
        """Make dignostic plots for the eccDefinition method."""
        fig, ax = plt.subplots(nrows=5, figsize=(12, 20), **kwargs)
        self.plot_measured_ecc(fig, ax[0])
        self.plot_decc_dt(fig, ax[1])
        self.plot_mean_ano(fig, ax[2])
        self.plot_extrema_in_omega(fig, ax[3])
        self.plot_phase_diff_ratio_between_peaks(fig, ax[4])
        return fig, ax

    def plot_measured_ecc(self, fig=None, ax=None, **kwargs):
        """Plot measured ecc as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.tref_out, self.ecc_ref, **kwargs)
        axNew.set_xlabel("time")
        axNew.set_ylabel("eccentricity")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_decc_dt(self, fig=None, ax=None, **kwargs):
        """Plot decc_dt as function of time to check monotonicity."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.t_for_ecc_test, self.decc_dt, **kwargs)
        axNew.set_xlabel("time")
        axNew.set_ylabel("decc/dt")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_mean_ano(self, fig=None, ax=None, **kwargs):
        """Plot measured mean anomaly as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.tref_out, self.mean_ano_ref, **kwargs)
        axNew.set_xlabel("time")
        axNew.set_ylabel("mean anomaly")
        axNew.grid()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_extrema_in_omega(self, fig=None, ax=None, **kwargs):
        """Plot omega and the Apastron Periastron as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        axNew.plot(self.tref_out, self.omega_peak_at_tref_out,
                   label="Periastron", **kwargs)
        axNew.plot(self.tref_out, self.omega_trough_at_tref_out,
                   label="Apastron", **kwargs)
        axNew.plot(self.t, self.omega22)
        axNew.plot(self.t[self.peaks_location],
                   self.omega22[self.peaks_location],
                   marker="o", ls="")
        axNew.plot(self.t[self.troughs_location],
                   self.omega22[self.troughs_location],
                   marker="o", ls="")
        axNew.set_xlabel("time")
        axNew.grid()
        axNew.set_ylabel(r"$\omega_{22}$")
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew

    def plot_phase_diff_ratio_between_peaks(self, fig=None, ax=None, **kwargs):
        """Plot phase diff ratio between consecutive as function of time."""
        if fig is None or ax is None:
            figNew, axNew = plt.subplots()
        else:
            axNew = ax
        tpeaks = self.t[self.peaks_location[1:]]
        axNew.plot(tpeaks[1:], self.orb_phase_diff_ratio_at_peaks[1:],
                   marker="o", label="peaks phase diff ratio")
        ttroughs = self.t[self.troughs_location[1:]]
        axNew.plot(ttroughs[1:], self.orb_phase_diff_ratio_at_troughs[1:],
                   marker="o", label="troughs phase diff ratio")
        axNew.set_xlabel("time")
        axNew.set_ylabel(r"$\Delta \Phi_{orb}[i] / \Delta \Phi_{orb}[i-1]$")
        axNew.grid()
        axNew.legend()
        if fig is None or ax is None:
            return figNew, axNew
        else:
            return axNew
