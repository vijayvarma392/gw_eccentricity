"""Module to filter spin-induced oscillations."""
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from warnings import warn
from .plot_settings import (
    use_fancy_plotsettings, labelsDict, figWidthsTwoColDict, figHeightsDict
    )


def _find_intersection_points(t, x, y):
    """Find where x and y intersect.
    
    Parameters
    ----------
    t : array-like
        Time array.
    x : array-like
        First signal array.
    y : array-like
        Second signal array.

    Returns
    -------
    array
        Indices of intersection points.
    """
    diff = x - y
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) < 2:
        raise ValueError(
            "Not enough intersection points found to compute delta T. "
            f"Found {len(sign_changes)} intersection points, but at least 2 "
            "are required.")

    t_cross = t[sign_changes]
    delta_t = np.diff(t_cross)
    t_mid = 0.5 * (t_cross[:-1] + t_cross[1:])

    return t_cross, t_mid, delta_t

def _build_delta_T_interpolant(t_mid, delta_t):
    """Build an interpolant for delta T.
    
    Parameters
    ----------
    t_mid : array-like
        Midpoints of crossing times.
    delta_t : array-like
        Time differences between crossings.

    Returns
    -------
    function
        Interpolant function for delta T.
    """
    return interp1d(t_mid, delta_t, kind='cubic', fill_value='extrapolate')


def _pn_model(t, t_merger, A, power):
    """Post-Newtonian model for secular trend.
    
    Parameters
    ----------
    t : array-like
        Time array.
    t_merger : float
        Merger time.
    A : float
        Amplitude parameter.
    power : float
        Power-law exponent.
    
    Returns
    -------
    array
        Evaluated model at time t.
    """
    return A * (t_merger - t)**power


def _pn_fit_for_secular_trend(t, x, kind, t_merger):
    """Fit a post-Newtonian model to the secular trend of the signal.
    
    Parameters
    ----------
    t : array-like
        Time array.
    x : array-like
        Signal array.
    kind : str
        Type of data to fit, one of ["omega", "amp"].
    t_merger : float
        Time of the merger of the binary, used as a reference for fitting.

    Returns
    -------
    function
        Fitted function for the secular trend.
    """
    allowed_kinds = ["omega", "amp"]
    if kind not in allowed_kinds:
        raise ValueError(
            f"Unknown fit kind: {kind}. Allowed kinds are: {allowed_kinds}")
    
    power = -3.0/8.0 if kind == "omega" else -1.0/4.0
    A_guess = np.mean(x) # typical amplitude of the data
    
    def residuals(params):
        t_merger_p, A, power = params
        model = _pn_model(t, t_merger_p, A, power)
        return (x - model)
    
    initial_guess = [t_merger, A_guess, power]
    result = least_squares(residuals, initial_guess)
    t_merger_fit, A_fit, power_fit = result.x

    return lambda t: _pn_model(t, t_merger_fit, A_fit, power_fit)

def check_filter_requirement(t, data_copr_22, data_copr_2m2, 
                             t_merger, data_type, threshold=0.2, 
                             debug_plots=False, style="Notebook"):
    """Check if the filter requirements are met.
    
    We set the requirement for filtering to True when the mode asymmetry
    becomes greater than a threshold fraction of (2, 2) mode in the
    coprecessing frame.

    Parameters
    ----------
    t : array-like
        Time array.
    data_copr_22 : array-like
        Coprecessing frame data for the (2, 2) mode.
    data_copr_2m2 : array-like
        Coprecessing frame data for the (2, -2) mode.
    data_type : str
        Type of data to check, one of ["omega", "amp"]. Used for debugging.
    t_merger : float
        Time of the merger of the binary, used in fitting the secular trend.
    threshold : float, optional, default=0.2
        Threshold fraction for the mode asymmetry to be considered
        significant compared to the (2, 2) mode in the coprecessing frame.
    debug_plots : bool, optional, default=False
        Whether to generate debug plots to visualize the filtering process.
    style : str, optional, default="Notebook"
        Style for debug plots, if generated.
    
    Returns
    -------
    bool
        True if the filter requirements are met, False otherwise.
    """
    if data_type not in ["omega", "amp"]:
        raise ValueError(
            f"Unknown data type: {data_type}. Allowed types are: ['omega', 'amp']")
    combination_sign = 1 if data_type == "amp" else -1
    asymmetry = 0.5 * (data_copr_22 - combination_sign * data_copr_2m2)
    secular_trend = _pn_fit_for_secular_trend(t, data_copr_22, data_type, t_merger)
    residual = data_copr_22 - secular_trend(t)

    max_res_amp = np.max(np.abs(residual[:-10000]))
    max_asym_amp = np.max(np.abs(asymmetry[:-10000]))
    ratio = max_asym_amp / max_res_amp

    if debug_plots:
        use_fancy_plotsettings(style=style)
        fig, ax = plt.subplots(
            figsize=(figWidthsTwoColDict[style], figHeightsDict[style]))
        ax.plot(t, asymmetry, label="Asymmetry")
        ax.plot(t, residual, label="Residual")
        ax.set_xlabel(labelsDict["t"])
        ax.legend()
        ax.set_title(f"Max asymmetry amplitude: {max_asym_amp:.3e}, "
                     f"Max residual amplitude: {max_res_amp:.3e}, "
                     f"Ratio: {ratio:.3f}")
        fig.tight_layout()
        fig.savefig("debug_filter_requirement.png", dpi=300)
        plt.close(fig)

    return ratio > threshold


@dataclass
class FilterSegmentResult:
    """Container storing results and diagnostics from segment filtering."""
    t: np.ndarray
    data: np.ndarray
    secular: np.ndarray
    residual: np.ndarray
    filtered_residual: np.ndarray
    filtered_data: np.ndarray
    frequency: np.ndarray
    spectrum: np.ndarray
    f_spin: np.float64
    f_ecc: np.float64
    f_spin_guess: np.float64
    f_cutoff: np.float64


def _get_fcut_fecc_fspin(t, data, f_spin_guess,
                         fspin_lo_frac, fspin_hi_frac,
                         fecc_lo_frac, fecc_hi_frac, verbose):
    """Get f_cutoff, f_ecc, and f_spin from the amplitude spectrum.

    The amplitude spectrum of the residual signal has two peaks: one at
    the eccentricity-induced oscillation frequency f_ecc and one at the
    spin-induced oscillation frequency f_spin. We identify both peaks
    using the f_spin_guess as a prior, then set f_cutoff to the trough
    between them so the low-pass filter removes spin-induced oscillations
    while preserving the eccentricity signal.

    Parameters
    ----------
    t : np.ndarray
        Time array of the segment.
    data : np.ndarray
        Residual signal with secular trend removed.
    f_spin_guess : float
        Rough estimate of f_spin from the intersection timescale.
    fspin_lo_frac : float
        Lower bound for f_spin search as a fraction of f_spin_guess.
        Default value is set in ``get_default_kwargs_for_filtering``.
    fspin_hi_frac : float
        Upper bound for f_spin search as a fraction of f_spin_guess.
        Default value is set in ``get_default_kwargs_for_filtering``.
    fecc_lo_frac : float
        Lower bound for f_ecc search as a fraction of f_spin.
        Default value is set in ``get_default_kwargs_for_filtering``.
    fecc_hi_frac : float
        Upper bound for f_ecc search as a fraction of f_spin.
        Default value is set in ``get_default_kwargs_for_filtering``.

    Returns
    -------
    f_cutoff : float
        Cutoff frequency for the low-pass filter, set to the trough
        between f_ecc and f_spin in the amplitude spectrum.
    f_ecc : float
        Eccentricity-induced oscillation frequency.
    f_spin : float
        Spin-induced oscillation frequency.
    freqs : np.ndarray
        Frequency array corresponding to the FFT of the input data.
    spectrum : np.ndarray
        Normalised amplitude spectrum ``|FFT(data)| / len(data)``.
    """
    fd = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=t[1] - t[0])
    spectrum = np.abs(fd) / len(data)

    # f_spin: argmax in [fspin_lo_frac, fspin_hi_frac] * f_spin_guess
    mask_spin = ((freqs >= fspin_lo_frac * f_spin_guess)
                & (freqs <= fspin_hi_frac * f_spin_guess))
    if np.any(mask_spin):
        f_spin = freqs[mask_spin][np.argmax(spectrum[mask_spin])]
        if verbose:
            print(f"✓ f_spin={f_spin:.5f}")
    else:
        f_spin = f_spin_guess
        if verbose:
            print(f"? No spin peak found in "
                f"[{fspin_lo_frac}*f_spin_guess, {fspin_hi_frac}*f_spin_guess], "
                f"using f_spin_guess={f_spin_guess:.5f} directly")

    # f_ecc: argmax in [fecc_lo_frac, fecc_hi_frac] * f_spin
    mask_ecc = ((freqs >= fecc_lo_frac * f_spin)
                & (freqs <= fecc_hi_frac * f_spin))
    if np.any(mask_ecc):
        f_ecc = freqs[mask_ecc][np.argmax(spectrum[mask_ecc])]
        if verbose:
            print(f"✓ f_ecc={f_ecc:.5f}, ratio f_spin/f_ecc={f_spin/f_ecc:.2f}")
    else:
        f_ecc = f_spin / 2.5
        if verbose:
            print(f"? No eccentricity peak found in "
                f"[{fecc_lo_frac}*f_spin, {fecc_hi_frac}*f_spin], "
                f"setting f_ecc=f_spin/2.5={f_ecc:.5f}")

    # f_cutoff: argmin in [f_ecc, f_spin]
    mask_trough = (freqs >= f_ecc) & (freqs <= f_spin)
    if np.any(mask_trough):
        f_cutoff = freqs[mask_trough][np.argmin(spectrum[mask_trough])]
        if verbose:
            print(f"✓ f_cutoff={f_cutoff:.5f} (trough between f_ecc and f_spin)")
    else:
        f_cutoff = 0.5 * (f_spin + f_ecc)
        if verbose:
            print(f"? No trough found between f_ecc and f_spin, "
                f"setting f_cutoff={f_cutoff:.5f}")

    return f_cutoff, f_ecc, f_spin, freqs, spectrum


def _make_lowpass_mask(frequencies, f_cutoff, taper_width):
    """Build a smooth low-pass mask using a cosine taper around f_cutoff.

    Instead of a hard cutoff (Gibbs ringing), the mask transitions
    smoothly from 1 to 0 over a frequency range of taper_width * f_cutoff
    centred on f_cutoff.

    Parameters
    ----------
    frequencies : np.ndarray
        FFT frequency axis.
    f_cutoff : float
        Cutoff frequency — mask is 1 below and 0 above.
    taper_width : float, optional, default=0.2
        Width of the cosine taper as a fraction of f_cutoff.

    Returns
    -------
    np.ndarray
        Smooth mask of the same length as frequencies.
    """
    f_low  = f_cutoff * (1 - taper_width)
    f_high = f_cutoff * (1 + taper_width)

    mask = np.ones(len(frequencies))
    # cosine taper in the transition band [f_low, f_high]
    taper_region = np.logical_and(frequencies >= f_low, frequencies <= f_high)
    mask[taper_region] = 0.5 * (
        1 + np.cos(np.pi * (frequencies[taper_region] - f_low) / (f_high - f_low))
    )
    mask[frequencies > f_high] = 0.0

    return mask


def _condition_data_for_fft(data, padding_length):
    """Condition the data for FFT."""
    # apply windowing
    windowed_data = (np.hamming(len(data)) * data)
    # apply zero-padding
    padded_data_with_window = np.pad(windowed_data, (padding_length, padding_length),
                            mode='constant', constant_values=0)
    padded_data_no_window = np.pad(
            data, (padding_length, padding_length), mode='constant', constant_values=0)
    
    return padded_data_with_window, padded_data_no_window


class FilterSpinInducedOscillations:
    """Class to filter spin-induced oscillations from a signal."""
    def __init__(self, data_dict, data_type, t_merger=None,
                 segment_size=10,
                 data_type_for_fspin_estimate=None,
                 data_type_for_filter_requirement_check=None,
                 data_tag="",
                 debug_plots=False,
                 filter_threshold=0.2,
                 verbose=0):
        """Initialize the filter.

        Parameters
        ----------
        t : array-like
            Time array.
        data_copr_22 : array-like
            Coprecessing frame data for the (2, 2) mode.
        data_copr_2m2 : array-like
            Coprecessing frame data for the (2, -2) mode.
        t_merger : float
            Time of the merger of the binary, used as a reference for fitting
            the secular trend.
        segment_size : float, optional, default=10
            Size of the segments to filter, defined in terms of the local delta T
            between crossings. This allows the filter to adapt to the changing
            frequency of the signal, which is crucial for effectively filtering
            the spin-induced oscillations without distorting the underlying
            secular trend.
        use_this_data_for_fspin_estimate : str, optional, default="amp"
            Data to use for estimating frequency of the spin induced oscillations.
        debug_plots : bool, optional, default=False
            Whether to generate debug plots to visualize the filtering process.
        """
        _allowed_data_type = ["amp", "omega"]
        _allowed_data_tags = ["", "_zeroecc"]
        if data_type not in _allowed_data_type:
            raise ValueError(f"data_type should be one of {_allowed_data_type}.")
        else:
            self.data_type = data_type

        if data_tag not in _allowed_data_tags:
            raise ValueError(f"data_tag should be one of {_allowed_data_tags}.")
        else:
            self.data_tag = data_tag

        if data_type_for_fspin_estimate is None:
            self.data_type_for_fspin_estimate = self.data_type
        elif data_type_for_fspin_estimate not in _allowed_data_type:
            raise ValueError(f"data_type_for_f_estimate should be one of {_allowed_data_type}")
        else:
            self.data_type_for_fspin_estimate = data_type_for_fspin_estimate

        if data_type_for_filter_requirement_check is None:
            self.data_type_for_filter_requirement_check = self.data_type
        elif data_type_for_filter_requirement_check not in _allowed_data_type:
            raise ValueError(f"data_type_for_filter_requirement_check should be one of {_allowed_data_type}")
        else:
            self.data_type_for_filter_requirement_check = data_type_for_filter_requirement_check

        self.t = data_dict["t"]
        self.data_copr_22 = data_dict[self.data_type + "lm" + self.data_tag][(2, 2)]
        self.data_copr_2m2 = data_dict[self.data_type + "lm" + self.data_tag][(2, -2)]
        self.segment_size = segment_size
        self.debug_plots = debug_plots
        self.filter_threshold = filter_threshold
        self.verbose = verbose
        if t_merger is None:
            t_merger = self.t[np.argmax(data_dict["amplm" + data_tag][(2, 2)] + data_dict["amplm" + data_tag][(2, -2)])]
        self.t_merger = t_merger
        _sign = 1 if self.data_type == "amp" else -1
        self.data = 0.5 * (self.data_copr_22 + _sign * self.data_copr_2m2)

        # check filter requirement
        self.filtering_required = check_filter_requirement(
            t=self.t,
            data_copr_22=data_dict[self.data_type_for_filter_requirement_check + "lm" + self.data_tag][(2, 2)],
            data_copr_2m2=data_dict[self.data_type_for_filter_requirement_check + "lm" + self.data_tag][(2, -2)],
            t_merger=self.t_merger,
            debug_plots=self.debug_plots,
            threshold=self.filter_threshold,
            data_type=self.data_type_for_filter_requirement_check
        )

        if not self.filtering_required:
            return None

        # build interpolant for the delta T between crossings
        self.t_cross, self.t_mid, self.delta_T = _find_intersection_points(
            t=self.t,
            x=data_dict[self.data_type_for_fspin_estimate + "lm" + self.data_tag][(2, 2)],
            y=(data_dict[self.data_type_for_fspin_estimate + "lm" + self.data_tag][(2, -2)]
               * (1 if self.data_type_for_fspin_estimate == "amp" else -1))
            )
        if self.t_cross.size > 100:
                if self.verbose:
                    warn(
                        f"{self.t_cross.size} crossing "
                        f"points found, which may be due to noise or numerical"
                        "artifacts. Consider removing noise or applying a "
                        "preliminary smoothing to the data.")
        self.delta_T_interp = _build_delta_T_interpolant(self.t_mid, self.delta_T)
   
    def _make_segment(self, t_mid):
        """Make a segment of the data centered at t_mid.

        The segment size is determined by the local delta T between crossings
        such the total size is segment_size * delta_T.
        """
        delta_T = self.delta_T_interp(t_mid)
        half = 0.5 * self.segment_size * delta_T

        segment_start = max(self.t[0],  t_mid - half)
        segment_end   = min(self.t[-1], t_mid + half)

        start_idx = np.searchsorted(self.t, segment_start, side='left')
        end_idx   = np.searchsorted(self.t, segment_end,   side='right')
        end_idx   = min(end_idx, len(self.t))

        return self.t[start_idx:end_idx], self.data[start_idx:end_idx]
    
    def _f_spin_from_delta_T(self, t):
        """Compute the spin-induced oscillation frequency from delta T."""
        return 1.0 / self.delta_T_interp(t)
    
    def _get_secular_and_residual(self, t, data):
        """Compute the residual of the data after removing the secular trend."""
        secular_trend = _pn_fit_for_secular_trend(
            t, data, self.data_type, self.t_merger)
        secular = secular_trend(t)
        return secular, data - secular

    def _filter_segment(self, t_mid, padding_length, taper_width, f_spin_lo_frac,
                        f_spin_hi_frac, fecc_lo_frac, fecc_hi_frac):
        """Filter a single segment of the data centered at t_mid."""
        # get data segment
        t_seg, data_seg = self._make_segment(t_mid)
        # get the estimated spin oscillation frequency
        f_spin_guess = self._f_spin_from_delta_T(t_mid)
        # get the secular and residual from the data
        secular, residual = self._get_secular_and_residual(t_seg, data_seg)
        # condition data for proper fft
        residual_windowed, residual_no_window = _condition_data_for_fft(
            residual, padding_length=padding_length)
        # get the cutoff frequency from amplitude spectrum
        f_cutoff, f_ecc, f_spin, freq_spectrum, amp_spectrum = _get_fcut_fecc_fspin(
            t_seg, residual_windowed, f_spin_guess, f_spin_lo_frac,
            f_spin_hi_frac, fecc_lo_frac, fecc_hi_frac, self.verbose)
        
        # lowpass filter to remove the faster spin-induced oscillation
        fd = np.fft.rfft(residual_no_window)
        frequencies = np.fft.rfftfreq(
            len(residual_no_window), d=t_seg[1] - t_seg[0])
        # fd[frequencies > f_cutoff] = 0
        mask = _make_lowpass_mask(frequencies, f_cutoff, taper_width)
        residual_smooth = np.fft.irfft(
                fd * mask, n=len(residual_no_window))[padding_length:-padding_length]

        return FilterSegmentResult(
            t=t_seg,
            data=data_seg,
            secular=secular,
            residual=residual,
            filtered_residual=residual_smooth,
            filtered_data=secular+residual_smooth,
            frequency=freq_spectrum,
            spectrum=amp_spectrum,
            f_spin=f_spin,
            f_ecc=f_ecc,
            f_spin_guess=f_spin_guess,
            f_cutoff=f_cutoff)

    def _combine_segments(self, results):
        """Combine the filtered segments back into a single signal."""
        y_full = np.zeros_like(self.t)
        w_full = np.zeros_like(self.t)

        for res in results:
            start_idx = np.searchsorted(self.t, res.t[0],  side='left')
            end_idx   = start_idx + len(res.filtered_data)
            end_idx   = min(end_idx, len(self.t))

            n = end_idx - start_idx
            y = res.filtered_data[:n]
            w = np.hanning(n)

            y_full[start_idx:end_idx] += y * w
            w_full[start_idx:end_idx] += w

        # Avoid division by zero
        mask = w_full > 0
        result = np.where(mask, y_full / np.where(mask, w_full, 1.0), 0.0)

        # Fill edge zeros by interpolating from the covered interior
        result[~mask] = np.interp(self.t[~mask], self.t[mask], result[mask])

        return result

    def apply_filter(self, padding_length, taper_width, alpha, f_spin_lo_frac,
                     f_spin_hi_frac, fecc_lo_frac, fecc_hi_frac):
        """Apply the filter to the data."""
        # We start at the start of the data and then advance by 
        # dt = alpha * delta_T_spin, where delta_T_spin is the timescale of
        # the local spin-induced oscillations.
        t_s = self.t[0]
        results = []

        iter = 0
        while t_s < self.t[-1]:
            if self.verbose:
                print(f"========== iter = {iter}: segment center at = {t_s} =================")
            result = self._filter_segment(
                t_s, padding_length, taper_width, f_spin_lo_frac, f_spin_hi_frac,
                fecc_lo_frac, fecc_hi_frac)
            t_s += alpha * self.delta_T_interp(t_s)
            iter += 1
            results.append(result)

        # Combine the filtered segments
        filtered_data = self._combine_segments(results)

        # debug plot
        if self.debug_plots:
            self._make_diagnostic_plots(
                results=results,
                filtered_data=filtered_data,
                padding_length=padding_length)

        self.filter_segment_results = results

        return filtered_data
    
    def _plot_final_signal(self, filtered_data):
        """Plot the final filtered signal."""
        nrows = 2
        style = "Notebook"
        use_fancy_plotsettings(usetex=False, style=style)
        fig, axes = plt.subplots(
            nrows=nrows, figsize=(figWidthsTwoColDict[style], 3 * nrows))
        axes[0].plot(self.t, self.data, label="Original data")
        axes[0].plot(self.t, filtered_data, label="Filtered data")
        axes[0].set_xlabel(labelsDict["t"])
        axes[0].set_ylabel(labelsDict[self.data_type + "_gw"])
        axes[0].legend()

        # add residual plot
        # get the secular trend for the original data
        secular_trend = _pn_fit_for_secular_trend(
            self.t, self.data, self.data_type, self.t_merger)
        residual_original = self.data - secular_trend(self.t)
        residual_filtered = filtered_data - secular_trend(self.t)
        axes[1].plot(self.t, residual_original, label="Original residual")
        axes[1].plot(self.t, residual_filtered, label="Filtered residual")
        axes[1].set_xlabel(labelsDict["t"])
        axes[1].set_ylabel(labelsDict[self.data_type + "_gw"] + " residual")
        axes[1].legend()

        fig.tight_layout()
        return fig, axes

    def _make_diagnostic_plots(self, results, filtered_data, padding_length):
        """Make diagnostic plots for debuging and visualizing filtering."""
        debug_plot_file_name = f"gwecc_{self.data_type}_lowpass_filter_diagnostics.pdf"
        debug_pdf = PdfPages(debug_plot_file_name)
        print(f"Saving debug plot to {debug_plot_file_name}")
        if self.data_type == "omega":
            label_str = r"\omega"
        elif self.data_type == "amp":
            label_str = "A"
        else:
            raise ValueError("data_type must be one of ['amp', 'omega']")

        for res in results:
            nrows = 3
            style = "Notebook"
            use_fancy_plotsettings(usetex=False, style=style)
            fig, axes = plt.subplots(
                nrows=nrows, figsize=(figWidthsTwoColDict[style], 3 * nrows))
            # plot time domain original
            axes[0].plot(res.t, res.residual,
                            label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
            axes[0].plot(res.t, res.filtered_residual, ls="-", c="tab:brown",
                            label=rf"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
            axes[0].set_xlabel(labelsDict["t"])
            axes[0].set_ylabel(fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
            axes[0].legend()
            # plot the amplitude spectrum
            axes[1].plot(res.frequency, res.spectrum,
                            label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
            # get amplitude spectrum of the filtered residual
            padded_smooth_residual = np.pad(
                res.filtered_residual * np.hamming(len(res.filtered_residual)),
                (padding_length, padding_length),
                mode='constant', constant_values=0)
            fd_smooth = np.fft.rfft(padded_smooth_residual)
            frequencies_smooth = np.fft.rfftfreq(
                len(padded_smooth_residual), d=res.t[1] - res.t[0])
            axes[1].plot(frequencies_smooth,
                            np.abs(fd_smooth)/len(padded_smooth_residual),
                            c="tab:brown", ls="-",
                            label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
            axes[1].axvline(res.f_spin, label=r"$f_{\mathrm{spin}}$",
                            c="tab:blue", ls="--")
            axes[1].axvline(res.f_ecc, label=r"$f_{\mathrm{ecc}}$", c="tab:green")
            axes[1].axvline(res.f_cutoff, label=r"$f_{\mathrm{cutoff}}$", c="tab:pink")
            axes[1].set_xlim(0, right=2*res.f_spin)
            axes[1].text(
                0.5, 0.95,
                fr"$f_{{\mathrm{{spin}}}}/f_{{\mathrm{{ecc}}}} = {res.f_spin/res.f_ecc:.2f}$",
                ha="left", va="top",
                transform=axes[1].transAxes)
            axes[1].set_xlabel("$f$")
            axes[1].set_ylabel(fr"$\mathrm{{FFT}}(\Delta {label_str}_{{\mathrm{{gw}}}})$")
            axes[1].legend(loc="upper right")

            # plot only the smoothed residual
            axes[2].plot(res.t, res.filtered_residual, ls="-", c="tab:brown")
            axes[2].set_ylabel(rf"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
            axes[2].set_xlabel(labelsDict["t"])
            
            fig.subplots_adjust(hspace=0.4, left=0.15, right=0.95)
            debug_pdf.savefig(fig)
            plt.close(fig)
        # add final filtered signal plot
        fig_final, _  = self._plot_final_signal(filtered_data)
        debug_pdf.savefig(fig_final)
        plt.close(fig_final)
        debug_pdf.close()


def get_default_kwargs_for_filtering():
    """Default kwargs for filtering spin-induced oscillations.

    Returns
    -------
    dict
        Dictionary containing allowed filter parameters and their default
        values:

        - "filter_threshold": Threshold for the mode asymmetry to be considered
          significant compared to the (2, 2) mode in the coprecessing frame.
          Default is 0.2, meaning that if the maximum amplitude of the
          asymmetry is greater than 20% of the maximum amplitude of the
          residual, the filtering will be applied. The residual is the obtained
          by subtracting a fitted secular trend from the (2, 2) mode in the
          coprecessing frame. Setting this threshold too low may lead to the
          filter being applied to data where the spin-induced oscillations are
          not significant, and in high eccentricity cases, it may even lead to
          removal of eccentric higher harmonics.
        - "alpha": Factor determining how much to advance the segment center
          for each iteration of filtering, in units of the local delta T between
          crossings. A smaller alpha means more overlap between segments and
          potentially smoother results, but also more computational cost.
        - "segment_size": Size of each segment to filter, defined in terms of
          the local delta T between crossings. This allows the filter to adapt
          to the changing frequency of the signal.
        - "data_type_for_filter_requirement_check": Data type ("amp" or "omega")
          to use for checking if filter requirements are met. If None, defaults
          to the same data type being filtered.
        - "data_type_for_fspin_estimate": Data type ("amp" or "omega") to use
          for estimating the frequency of spin-induced oscillations. If None,
          defaults to the same data type being filtered.
        - "verbose": Whether to print detailed information about the filtering
          process.
        - "debug_plots": Whether to generate debug plots to visualize the
          filtering process.
        - "do_not_filter": If True, skip filtering even if requirements are met,
          and return original data. Useful for testing and debugging.
        - "padding_length": Length of zero-padding to apply on either side of
          the data segment before performing FFT-based filtering. This helps
          mitigate edge effects in the FFT.
        - "taper_width": Width of the cosine taper used in the low-pass filter,
          as a fraction of the cutoff frequency. This controls how smoothly the
          filter transitions from passing frequencies below cutoff to attenuating
          frequencies above cutoff.
        - "f_spin_lo_frac": Lower bound for the spin-frequency search window,
          as a fraction of ``f_spin_guess``.
        - "f_spin_hi_frac": Upper bound for the spin-frequency search window,
          as a fraction of ``f_spin_guess``.
        - "fecc_lo_frac": Lower bound for the eccentricity-frequency search
          window, as a fraction of ``f_spin``.
        - "fecc_hi_frac": Upper bound for the eccentricity-frequency search
          window, as a fraction of ``f_spin``.
    """
    return {
        "filter_threshold": 0.2,
        "alpha": 2,
        "segment_size": 15,
        "data_type_for_filter_requirement_check": None,
        "data_type_for_fspin_estimate": None,
        "verbose": False,
        "debug_plots": False,
        "do_not_filter": False,
        "padding_length": 2*10**4,
        "taper_width": 0.2,
        "f_spin_lo_frac": 0.75,
        "f_spin_hi_frac": 1.25,
        "fecc_lo_frac": 0.25,
        "fecc_hi_frac": 0.5
    }


def check_and_filter_spin_induced_oscillations(data_dict, data_tag, t_merger,
                                               filter_kwargs):
    """Get filtered amp_gw, omega_gw after removing spin-induced oscillations.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the data to be filtered. Should contain the
        coprecessing frame data for the (2, 2) and (2, -2) modes under the keys
        "{data_type}lm{data_tag}[(2, 2)]" and "{data_type}lm{data_tag}[(2,
        -2)]" respectively, where data_type is either "amp", "omega" or "phase".
    data_tag : str
        Tag to identify the data. Either "" for eccentric data or "_zeroecc"
        for zero-eccentricity data.
    t_merger : float
        Time of merger. Used to fit the secular trend of the data, which is
        necessary for effectively identifying and filtering the spin-induced
        oscillations.
    filter_kwargs : dict
        Dictionary containing the filter parameters. See the allowed keys and
        their default values in ``get_default_kwargs_for_filtering``.

    Returns
    -------
    filter_data_dict : dict
        Dictionary containing the filtered data and related diagnostics.
        For each data_type in ["amp", "omega", "phase"], the dictionary contains:

        - ``{data_type}_gw{data_tag}`` : np.ndarray
            Best available data — filtered if filtering was applied,
            original otherwise. This is the key to use downstream.
        - ``{data_type}_gw{data_tag}_original`` : np.ndarray
            Original unmodified data, always present and non-None.
        - ``{data_type}_gw{data_tag}_filtered`` : np.ndarray or None
            Filtered data if filtering was applied, None otherwise.
        - ``{data_type}_gw{data_tag}_filter_segment_results`` : list or None
            Results of the filter segments if filtering was applied,
            None otherwise.
    """
    filter_data_dict = {}
    # We loop over both amplitude and frequency data types, check if filtering
    # is required, and apply the filter if so. The results are stored in
    # filter_data_dict with keys indicating the data type and whether it's
    # original or filtered.
    for data_type in ["amp", "omega"]:
        filter_obj = FilterSpinInducedOscillations(
            data_dict=deepcopy(data_dict),
            data_type=data_type,
            data_tag=data_tag,
            t_merger=t_merger,
            segment_size=filter_kwargs["segment_size"],
            data_type_for_filter_requirement_check=filter_kwargs["data_type_for_filter_requirement_check"],
            data_type_for_fspin_estimate=filter_kwargs["data_type_for_fspin_estimate"],
            debug_plots=filter_kwargs["debug_plots"],
            filter_threshold=filter_kwargs["filter_threshold"],
            verbose=filter_kwargs["verbose"],
        )

        original = filter_obj.data.copy()
        base_key = f"{data_type}_gw{data_tag}"

        if filter_obj.filtering_required and filter_kwargs["do_not_filter"]:
            if filter_kwargs["verbose"]:
                warn(f"Filter requirements are met for {data_type}, but "
                      "filtering is skipped due to do_not_filter=True. Returning "
                      "original data.")

        if filter_obj.filtering_required and not filter_kwargs["do_not_filter"]:
            if filter_kwargs["verbose"]:
                print(f"✅ Filter requirements are met for {data_type}, applying filter.")

            filtered = filter_obj.apply_filter(
                padding_length=filter_kwargs["padding_length"],
                taper_width=filter_kwargs["taper_width"],
                alpha=filter_kwargs["alpha"],
                f_spin_lo_frac=filter_kwargs["f_spin_lo_frac"],
                f_spin_hi_frac=filter_kwargs["f_spin_hi_frac"],
                fecc_lo_frac=filter_kwargs["fecc_lo_frac"],
                fecc_hi_frac=filter_kwargs["fecc_hi_frac"],
            )
            filter_data_dict[base_key]                             = filtered
            filter_data_dict[f"{base_key}_original"]               = original
            filter_data_dict[f"{base_key}_filtered"]               = filtered
            filter_data_dict[f"{base_key}_filter_segment_results"] = filter_obj.filter_segment_results

            if filter_kwargs["verbose"]:
                print(f"✅ Finished filtering {data_type}.")
        else:
            filter_data_dict[base_key]                             = original
            filter_data_dict[f"{base_key}_original"]               = original
            filter_data_dict[f"{base_key}_filtered"]               = None
            filter_data_dict[f"{base_key}_filter_segment_results"] = None

    # Get the phase from the best available omega_gw
    # set the original phase as the anti-symmetric combination of the (2, 2) 
    # and (2, -2) modes in the coprecessing frame.
    filter_data_dict[f"phase_gw{data_tag}_original"] = (
        0.5 * (data_dict[f"phaselm{data_tag}"][(2, 2)]
               - data_dict[f"phaselm{data_tag}"][(2, -2)]))
    # if filtered omega is available, get the phase by integrating the 
    # filtered omega.
    if filter_data_dict[f"omega_gw{data_tag}_filtered"] is not None:
        filter_data_dict[f"phase_gw{data_tag}_filtered"] = (
            np.cumsum(filter_data_dict[f"omega_gw{data_tag}_filtered"])
            * (filter_obj.t[1] - filter_obj.t[0]))
        # set the phase to the filtered phase.
        filter_data_dict[f"phase_gw{data_tag}"] \
            = filter_data_dict[f"phase_gw{data_tag}_filtered"].copy()
    else:
        filter_data_dict[f"phase_gw{data_tag}"] \
            = filter_data_dict[f"phase_gw{data_tag}_original"].copy()
        filter_data_dict[f"phase_gw{data_tag}_filtered"] = None

    return filter_data_dict
