"""Module for filtering gw variables."""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
from .plot_settings import use_fancy_plotsettings, figWidthsTwoColDict

class FilterSpinInducedOscillations:
    def __init__(self, dataDict):
        self.dataDict = dataDict

    def find_intersections(self, data_key="omega"):
        window = 501
        d1 = savgol_filter(self.dataDict[f"{data_key}lm"][(2, 2)], window, 3)
        d2 = savgol_filter(self.dataDict[f"{data_key}lm"][(2, -2)], window, 3)
        if data_key in ["omega", "phase"]:
            d = d1 + d2
        elif data_key == "amp":
            d = d1 - d2
        else:
            raise Exception(
                "Unknown data_key. Should be one of `omega`, `phase` or `amp`")
        return np.argwhere(np.diff(np.sign(d))).flatten()
    
    def get_interpolant_for_intersection_distances(self, data_key="omega"):
        idx = self.find_intersections(data_key)
        distances = np.diff(idx)
        ts = 0.5 * (self.dataDict["t"][idx][:-1] + self.dataDict["t"][idx][1:])
        return InterpolatedUnivariateSpline(ts, distances)
    
    def lowpass_filter_with_overlapping_segments(
            self, x, y, data_key="omega"):
        N = len(y)
        y_smooth = np.zeros_like(y)
        weights = np.zeros_like(y)

        # Compute intersection distances
        intersection_distance_interp \
            = self.get_interpolant_for_intersection_distances(data_key)
        intersection_distances = intersection_distance_interp(x).astype(int)
        # Compute data segment sizes. Twice the local intersection distance.
        data_segments_sizes = (10 * intersection_distances).astype(int)

        debug_plots = False
        
        # create debug plot
        if debug_plots:
            debug_plot_file_name = f"gwecc_{data_key}_lowpass_filter_diagnostics.pdf"
            debug_pdf = PdfPages(debug_plot_file_name)
            print(f"Saving debug plot to {debug_plot_file_name}")
            if data_key == "omega":
                label_str = "\omega"
            elif data_key == "amp":
                label_str = "A"
            elif data_key == "phase":
                label_str = "\Phi"

        # Process data in overlapping segments
        # Use half of the last segment size as the step to process
        # the data
        min_segment_size = int(data_segments_sizes[-1] // 2)
        
        for i in range(0, N, min_segment_size): 
            segment_size = data_segments_sizes[i]
            half_segment = segment_size // 2

            start, end = max(0, i - half_segment), min(N, i + half_segment + 1)
            if start == 0:
                end == min(N, segment_size)
            if end == N:
                start = max(0, N - segment_size)
            # get the residual
            fit_center_time = 0.5 * (x[start] + x[end-1])
            fit = FitSecularTrend(t0=fit_center_time)
            # typial scale of data
            y0 = 0.5 * (y[start] + y[end-1])
            nPN = -3.0/8.0 # for omega
            p0 = [y0,  # function value
                  -nPN * y0 / (- fit_center_time),  # func = f0/t0^n*(t)^n -> dfunc/dt (t0) = n*f0/t0
                  0  # singularity in fit is near t=0, since waveform aligned at max(amp_gw)
                  ]
            bounds0 = [[0., 0., 0.8 * x[end-1]],
                       [10 * y0,
                        10 * y0 / (-fit_center_time), - fit_center_time]
                       ]
            
            p_secular = curve_fit(fit, x[start:end], y[start:end],
                                  p0=p0, bounds=bounds0)[0]

            secular = fit(x[start:end], *p_secular)
            residual = y[start:end] - secular

            # apply windowing
            windowed_residual = (np.hamming(len(residual)) * residual)

            # apply zero-padding
            pad_width = 2 * 10**4
            padded_residual = np.pad(windowed_residual, (pad_width, pad_width),
                                     mode='constant', constant_values=0)
            padded_residual_no_window = np.pad(
                residual, (pad_width, pad_width), mode='constant', constant_values=0)

            # apply lowpass
            fs = 1 / (x[1] - x[0])
            fd = np.fft.rfft(padded_residual)
            frequencies = np.fft.rfftfreq(len(padded_residual), d=1/fs)
            df = (frequencies[1] - frequencies[0])
            fspin_estimated = (
                fs / intersection_distance_interp(x[int(0.5 * (start + end))]))
            distance = int((fspin_estimated * (3 - 1) / 10) / df)
            peaks = find_peaks(np.abs(fd), distance=distance)[0]
            troughs = find_peaks(-np.abs(fd), distance=distance)[0]
            fpeaks = frequencies[peaks]
            fspin = fpeaks[fpeaks > 0.75 * fspin_estimated][0]
            fecc_freq = fpeaks[np.logical_and(fpeaks > fspin_estimated / 4,
                                              fpeaks < fspin_estimated / 2)]
            if len(fecc_freq) > 0:
                fecc = fecc_freq[0]
            else:
                fecc = fspin / 2.5
            ftroughs = frequencies[troughs]
            fcutoff = ftroughs[ftroughs > fecc][0]
            fcutoff_low = ftroughs[ftroughs < fecc]

            fd_filt = np.fft.rfft(padded_residual_no_window)
            fd_filt[frequencies > fcutoff] = 0
            residual_smooth = np.fft.irfft(
                fd_filt, n=len(padded_residual))[pad_width:-pad_width]
            
            if debug_plots:
                nrows = 3
                style = "Notebook"
                use_fancy_plotsettings(usetex=False, style=style)
                fig, axes = plt.subplots(
                    nrows=nrows, figsize=(figWidthsTwoColDict[style], 3 * nrows))
                # plot time domain original
                axes[0].plot(x[start: end], residual,
                             label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
                axes[0].plot(x[start:end], residual_smooth, ls="-", c="tab:brown",
                             label=rf"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
                axes[0].set_xlabel("$t [M]$")
                axes[0].set_ylabel(fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
                axes[0].legend()
                # plot the amplitude spectrum
                axes[1].plot(frequencies, np.abs(fd)/len(padded_residual),
                             label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}$")
                # get amplitude spectrum of the filtered residual
                padded_smooth_residual = np.pad(
                    residual_smooth * np.hamming(len(residual_smooth)),
                    (pad_width, pad_width),
                    mode='constant', constant_values=0)
                fd_smooth = np.fft.rfft(padded_smooth_residual)
                frequencies_smooth = np.fft.rfftfreq(
                    len(padded_smooth_residual), d=1/fs)
                axes[1].plot(frequencies_smooth,
                             np.abs(fd_smooth)/len(padded_smooth_residual),
                             c="tab:brown", ls="-",
                             label=fr"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
                axes[1].axvline(fspin, label="$f_{\mathrm{spin}}$",
                                c="tab:blue", ls="--")
                axes[1].axvline(fecc, label="$f_{\mathrm{ecc}}$", c="tab:green")
                axes[1].axvline(fcutoff, label="$f_{\mathrm{cutoff}}$", c="tab:pink")
                axes[1].set_xlim(0, right=2*fspin)
                axes[1].text(
                    0.5, 0.95,
                    fr"$f_{{\mathrm{{spin}}}}/f_{{\mathrm{{ecc}}}} = {fspin/fecc:.2f}$",
                    ha="left", va="top",
                    transform=axes[1].transAxes)
                axes[1].set_xlabel("$f [1/M]$")
                axes[1].set_ylabel(fr"$\mathrm{{FFT}}(\Delta {label_str}_{{\mathrm{{gw}}}})$")
                axes[1].legend(loc="upper right")

                # plot only the smoothed residual
                axes[2].plot(x[start:end], residual_smooth, ls="-", c="tab:brown")
                # axes[2].plot(x[start:end], sin_fit_for_residual, ls="--", c="tab:blue", label="$\sin$ fit")
                axes[2].set_ylabel(rf"$\Delta {label_str}_{{\mathrm{{gw}}}}^{{\mathrm{{filtered}}}}$")
                axes[2].set_xlabel("$t [M]$")
                
                fig.subplots_adjust(hspace=0.4, left=0.15, right=0.95)
                debug_pdf.savefig(fig)
                plt.close(fig)
            
            # get smooth signal
            segment_smooth = residual_smooth + secular
            
            # Apply blending using Hann weights for smooth transitions
            blend_weights = np.hanning(len(segment_smooth))  # Smooth window blending
            y_smooth[start:end] += (segment_smooth * blend_weights)
            weights[start:end] += blend_weights

        if debug_plots:
            debug_pdf.close()
            
        # Normalize weights to avoid intensity variations
        y_smooth /= np.maximum(weights, 1e-10)
        return y_smooth

    def check_if_filtering_is_required(self):
        omega_asymmetry = 0.5 * (self.dataDict["omegalm"][(2, 2)]
                                 + self.dataDict["omegalm"][(2, -2)])
        omega_copr22 = self.dataDict["omegalm"][(2, 2)]
        # find the residual omega_gw
        fit_center_time = 0.5 * (self.dataDict["t"][0] + self.dataDict["t"][-1])
        fit = FitSecularTrend(t0=fit_center_time)
        # typial scale of data
        y0 = 0.5 * (omega_copr22[0] + omega_copr22[-1])
        nPN = -3.0/8.0 # for omega
        p0 = [y0,  # function value
              -nPN * y0 / (- fit_center_time),  # func = f0/t0^n*(t)^n -> dfunc/dt (t0) = n*f0/t0
              0  # singularity in fit is near t=0, since waveform aligned at max(amp_gw)
              ]
        bounds0 = [[0., 0., 0.8 * self.dataDict["t"][-1]],
                   [10 * y0,
                    10 * y0 / (-fit_center_time), - fit_center_time]
                   ]

        p_secular = curve_fit(fit, self.dataDict["t"], omega_copr22,
                              p0=p0, bounds=bounds0)[0]

        secular = fit(self.dataDict["t"], *p_secular)
        omega_copr22_residual = omega_copr22 - secular

        debug_plots = False
        
        # We find the maximum values of omega_asymmetry and omega_gw_residual.
        # Compare their values to find if omega_asymmetry is comparable to
        # omega_gw_residual
        omega_asymmetry_peaks = find_peaks(omega_asymmetry, width=300, rel_height=0.5)[0]
        omega_copr22_residual_peaks = find_peaks(omega_copr22_residual, width=300, rel_height=0.5)[0]
        avg_omega_asymmetry_max = np.mean(omega_asymmetry[omega_asymmetry_peaks])
        avg_omega_copr22_residual_max = np.mean(omega_copr22_residual[omega_copr22_residual_peaks])

        threshold = 0.2

        if debug_plots:
            debug_plot_file_name = f"gwecc_lowpass_filter_requirement_check.pdf"
            print(f"Saving debug plot to {debug_plot_file_name}")
            style = "Notebook"
            use_fancy_plotsettings(usetex=False, style=style)
            fig, ax = plt.subplots(figsize=(figWidthsTwoColDict[style], 6))
            ax.plot(self.dataDict["t"], omega_asymmetry,
                    label=r"$0.5\times(|\omega^{\mathrm{copr}}_{2,2}| - |\omega^{\mathrm{copr}}_{2,-2}|)$",
                    c="tab:blue")
            ax.plot(self.dataDict["t"], omega_copr22_residual,
                    label="$\Delta\omega_{\mathrm{2,2}}^{\mathrm{copr}} = \omega_{\mathrm{2,2}}^{\mathrm{copr}} - \omega_{\mathrm{2,2}}^{\mathrm{copr,secular}}$",
                    c="tab:brown")
            ax.plot(self.dataDict["t"][omega_asymmetry_peaks],
                    omega_asymmetry[omega_asymmetry_peaks],
                    ls="", marker="s", c="tab:blue")
            ax.plot(self.dataDict["t"][omega_copr22_residual_peaks],
                    omega_copr22_residual[omega_copr22_residual_peaks],
                    ls="", marker="o", c="tab:brown")
            ax.axhline(avg_omega_asymmetry_max, ls="--", c="tab:blue",
                       label="Average max of $|\omega^{\mathrm{copr}}_{2,2}| - |\omega^{\mathrm{copr}}_{2,-2}|$")
            ax.axhline(avg_omega_copr22_residual_max, ls="--", c="tab:brown",
                       label="Average max of $\Delta\omega_{\mathrm{gw}}$")
            ax.axhline(threshold * avg_omega_copr22_residual_max, label="Threshold for filtering.")
            ax.text(0.05, 0.05, "Filtering is done if Average max of "
                    r"$0.5\times(|\omega^{\mathrm{copr}}_{2,2}| - |\omega^{\mathrm{copr}}_{2,-2}|)>$ "
                    "the threshold line\n which is at "
                    f"${threshold}\\times$"
                    " average max of $\Delta\omega_{\mathrm{gw}}$",
                    va="bottom", ha="left", transform=ax.transAxes, alpha=1,
                    bbox=dict(
                        boxstyle='round,pad=0.5',    # Box shape and padding
                        facecolor='white',            # Box background color
                        edgecolor='black',           # Box border color
                        linewidth=1,                 # Border line width
                        alpha=0.7                  # Transparency
                    ))
            ax.legend(loc="upper right", frameon=False, ncols=3, bbox_to_anchor=(1,1.2))
            ax.set_xlabel("$t$ [$M$]")
            fig.tight_layout()
            fig.savefig(debug_plot_file_name)
            plt.close(fig)

        return (avg_omega_asymmetry_max >= threshold * avg_omega_copr22_residual_max)
    
    def filter_spin_induced_oscillations(self, t, data, data_key="omega"):
        valid_data_keys = ["amp", "phase", "omega"]
        if data_key not in valid_data_keys:
            raise Exception("Invalid data_key. Should be one of"
                            f"{valid_data_keys}")

        if not self.check_if_filtering_is_required():
            return data
        else:
            print("Filtering is required.")
        
        x = np.copy(t)
        y = np.copy(data)

        y = self.lowpass_filter_with_overlapping_segments(
            x, y, data_key
        )

        # Filter may introuce sharp features at the edges which affects extrema finding.
        # Set the first and last 10 values to the 11th values value
        # from the start and the the end, respectively.
        y[:10] = y[10]
        y[-10:] = y[-10]
        
        return y

    
class FitSecularTrend:
    def __init__(self, t0):
        self.t0 = t0

    def __call__(self, t, f0, f1, T):
        # f0, f1 are function values and first time-derivatives
        # at t0. Re-expfress as T, n, A, then evalate A*(T-t)^n
        n = - (T - self.t0) * (f1 / f0)
        A = f0 * (T - self.t0)**(-n)
        if t.max() > T:
            raise Exception(
                "Fit reached parameters where merger "
                "time T is within time-series to be fitted\n"
                f"f0={f0}, f1={f1}, T={T}; n={n}, A={A}, max(t)={max(t)}")
        return A * (T - t)**n
