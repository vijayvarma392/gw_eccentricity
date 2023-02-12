"""
Find peaks and troughs using frequency fits.

Part of Eccentricity Definition project.
"""
from .eccDefinition import eccDefinition
from .plot_settings import labelsDict
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .utils import check_kwargs_and_set_defaults


class envelope_fitting_function:
    """Fit envelope.

    Re-parameterize A*(T-t)^n in terms of function value and 
    first derivative at the time t0, and T.
    """

    def __init__(self, t0, verbose=False):
        """Init."""
        self.t0 = t0
        self.verbose = verbose

    def format(self, f0, f1, T):
        """Return a string representation for use in legends and output."""
        n = -(T-self.t0)*f1/f0
        A = f0*(T-self.t0)**(-n)
        return f"{A:.3g}({T:+.2f}-t)^{n:.3f}"

    def __call__(self, t, f0, f1, T):
        """Call."""
        # f0, f1 are function values and first time-derivatives
        # at t0.  Re-expfress as T, n, A, then evalate A*(T-t)^n
        n = -(T-self.t0)*f1/f0
        A = f0*(T-self.t0)**(-n)
        if self.verbose:
            print(f"f0={f0}, f1={f1}, T={T}; n={n}, A={A}, max(t)={t.max()}")
        if t.max() > T:
            print(end="", flush=True)
            raise Exception(
                "envelope_fitting_function reached parameters where merger "
                "time T is within time-series to be fitted\n"
                f"f0={f0}, f1={f1}, T={T}; n={n}, A={A}, max(t)={max(t)}")
        return A*(T-t)**n


class eccDefinitionUsingFrequencyFits(eccDefinition):
    """Measure eccentricity by finding extrema location using freq fits."""

    def __init__(self, *args, **kwargs):
        """Init for eccDefinitionUsingWithFrequencyFits class.

        parameters:
        ----------
        dataDict: Dictionary containing the waveform data.
        """
        super().__init__(*args, **kwargs)
        self.data_str = "omega22"
        self.label_for_data_for_finding_extrema = labelsDict[self.data_str]
        self.label_for_fit_to_data_for_finding_extrema \
            = labelsDict[f"{self.data_str}_fit"]
        # Make a copy of omega22 and use it to set data_for_finding_extrema.
        # This would ensure that any modification of data_for_finding_extrema
        # does not modify omega22.
        self.data_for_finding_extrema = self.omega22.copy()
        self.method = "FrequencyFits"
        # Get dictionary of kwargs to be used for Fits methods.
        self.kwargs_for_fits_methods = check_kwargs_and_set_defaults(
            self.extra_kwargs['kwargs_for_fits_methods'],
            self.get_default_kwargs_for_fits_methods(),
            "kwargs_for_fits_methods",
            "eccDefinitionUsingFrequencyFits.get_default_kwargs_for_fits_methods()")
        # Set variables needed for envelope fits and find_peaks
        self.set_fit_variables()
        # show more verbose output if debug_level is >= 1
        self.verbose = self.debug_level >= 1
        # If return_diagnostic_data is true then return a dictionary of data for diagnostics.
        if self.return_diagnostic_data:
            self.diagnostic_data_dict = {
                # Initialize with empty lists to fill in during iteration over the extrema
                "params": {"pericenters": [], "apocenters": []},
                "t_extrema": {"pericenters": [], "apocenters": []},
                "data_extrema": {"pericenters": [], "apocenters": []},
                "t_ref": {"pericenters": [], "apocenters": []},
                "f_fit": {"pericenters": [], "apocenters": []},
                "data": self.data_for_finding_extrema,
                "t": self.t
            }

    def get_default_kwargs_for_fits_methods(self):
        """Get default kwargs to be used for Fits methods.

        The kwargs are:
        - "nPN": The PN exponent to use in the fit function. It is
          inspired by the functional form of frequency/amplitude in
          the leading Post-Newtonian order ~(t - t_merger)^nPN
        - "fit_bounds_max_amp_factor": To set the upper bound on the
          Amplitude A of the fitting function of the form A(t-T)^n.
          The upper bound of A is set as f0*fit_bounds_max_amp_factor,
          where f0 is the mean of the first and the last values of data
          to be fitted.  f0 = 0.5*(data[0]+data[-1])
        - "fit_bounds_max_nPN_factor": To set the upper bound on the
          exponent n of the fitting function. The upper bound on n is
          set as f0*fit_bounds_max_nPN_factor/(-fit_center_time),
          where fit_center_time is the time at the midpoint of the
          data.
          fit_center_time = 0.5*(t[0]+t[-1]).
          The merger is assumed to be at t=0 here.
        - "prominence_factor": To set the prominence for find_peaks
          function. The prominence is set as
          prominence = prominence_factor * residual_data_amp,
          where,
          residual_data_amp = max(residual_data) - min(residual_data)
        - "distance_factor": To set the distance for find_peaks
          function.
          distance = distance_factor * average_orbital_period
        - "num_orbits": Number of extrema to look for during
          fitting. It looks for num_orbits on the left and num_orbits+1
          on the right.
        - "num_orbits_for_global_fit": Number of orbits to use for
          global fit in the Fits methods.
        - "return_diagnostic_data": If True, Retuns a dictionary
          of data useful for diagnostics. Default is False.
        """
        return {
            "nPN": -3./8,
            "fit_bounds_max_amp_factor": 10,
            "fit_bounds_max_nPN_factor": 10,
            "prominence_factor": 0.03,  # prominence = 3% of residual_amp_max
            "distance_factor": 0.75,  # 75% of the average orbital period,
            "num_orbits": 3,
            "num_orbits_for_global_fit": 10,
            "return_diagnostic_data": False
        }

    def set_fit_variables(self):
        """Set variables to be used for Fits Methods.

        See under get_default_kwargs_for_fits_methods for documentation
        on these variables.
        """
        self.fit_bounds_max_amp_factor = self.kwargs_for_fits_methods[
            "fit_bounds_max_amp_factor"]
        self.fit_bounds_max_nPN_factor = self.kwargs_for_fits_methods[
            "fit_bounds_max_nPN_factor"]
        self.nPN = self.kwargs_for_fits_methods["nPN"]
        self.prominence_factor = self.kwargs_for_fits_methods[
            "prominence_factor"]
        self.distance_factor = self.kwargs_for_fits_methods["distance_factor"]
        self.num_orbits = self.kwargs_for_fits_methods["num_orbits"]
        self.num_orbits_for_global_fit = self.kwargs_for_fits_methods[
            "num_orbits_for_global_fit"]
        self.return_diagnostic_data = self.kwargs_for_fits_methods["return_diagnostic_data"]

    def find_extrema(self, extrema_type="pericenters"):
        """Find the extrema in the data.

        parameters:
        -----------
        extrema_type:
            Either "pericenters" or "apocenters".

        returns:
        ------
        array of positions of extrema.
        """
        # STEP 0 - setup

        if extrema_type == "pericenters":
            sign = +1
        elif extrema_type == "apocenters":
            sign = -1
        else:
            raise Exception(f"extrema_type='{extrema_type}' unknown.")
        # The fit function assume the merger to be at t=0. So we align the time
        # axis such that merger occurs at t=0.
        # After the peaks are found, the time is reshifted to to its original
        # values at the end of this function so the merger is again at t_merger.
        self.t -= self.t_merger

        # DESIRED NUMBER OF EXTREMA left/right DURING FITTING
        # Code will look for N extrema to the left of idx_ref, and N+1 extrema
        # to the right
        N = self.num_orbits

        # if True, perform an additional fitting step to find the position of
        # extrema to sub-gridspacing accuracy.
        #
        # The results of this more accurate extrema-determination are collected
        # in self.periastron_info = [t_extrema_refined,
        # data_extrema_refined, phase22_extrema_refined] and/or
        # self.apastron_info = [t_extrema_refined, data_extrema_refined,
        # phase22_extrema_refined]
        # if False, then
        # self.periastron_info/apastron_info contains the data at the
        # grid-points
        refine_extrema = self.extra_kwargs["refine_extrema"]

        # diagnostic output?
        # setting diag_file to a valid pdf-filename will trigger diagnostic
        # plots
        # Print more verbose output if debug_level >= 1
        # Create debug plots if debug_plots is True
        diag_file = (f"gwecc_{self.method}_diagnostics_{extrema_type}.pdf") if self.debug_plots else ""
        pp = PdfPages(diag_file) if diag_file != "" else False
        # STEP 1:
        # global fit as initialization of envelope-subtraced extrema

        # use this many orbits from the start of the waveform for the initial
        # global fit.
        # Keeping this initial fit-interval away from merger helps
        # to obtain a good fit that also allows to discern small eccentricities
        N_orbits_for_global_fit = self.num_orbits_for_global_fit
        idx_end = np.argmax(self.phase22 > self.phase22[0]
                            + N_orbits_for_global_fit*4*np.pi)

        if idx_end == 0:  # don't have that much data, so use all
            idx_end = -1

        if self.verbose:
            print(f"t[0]={self.t[0]}, t[-1]="
                  f"{self.t[-1]}, "
                  f"global fit to t<={self.t[idx_end]}")

        # create fitting function object, set initial guess and bounds
        fit_center_time = 0.5*(self.t[0] + self.t[-1])
        f_fit = envelope_fitting_function(t0=fit_center_time,
                                          verbose=False)
        # typial scale of data
        f0 = 0.5 * (self.data_for_finding_extrema[0]+self.data_for_finding_extrema[idx_end])
        p0 = [f0,  # function value
              -self.nPN*f0/(-fit_center_time),  # func = f0/t0^n*(t)^n -> dfunc/dt (t0) = n*f0/t0
              0  # singularity in fit is near t=0, since waveform aligned at max(amp22)
              ]
        bounds0 = [[0., 0., 0.8*self.t[-1]],
                   [self.fit_bounds_max_amp_factor*f0,
                    self.fit_bounds_max_nPN_factor*f0/(-fit_center_time),
                    -fit_center_time]]
        if self.verbose:
            print(f"global fit: guess p0={p0},  t_center={fit_center_time}")
            print(f"            bounds={bounds0}")

        if pp:
            # first diagnostic plots.  Will be available even if
            # scipy.optimize fails
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].set_title(self.label_for_data_for_finding_extrema)
            axs[1].set_title(
                f"residual:  {self.label_for_data_for_finding_extrema}"
                f"-{self.label_for_fit_to_data_for_finding_extrema} "
                "(fitted region only)")
            axs[2].set_title(
                f"residual:  {self.label_for_data_for_finding_extrema}-"
                f"{self.label_for_fit_to_data_for_finding_extrema}")
            axs[0].plot(self.t, self.data_for_finding_extrema,
                        label=self.label_for_data_for_finding_extrema)
            axs[0].plot(
                self.t, f_fit(self.t, *p0),
                linewidth=0.5, color='grey',
                label=(f"{self.label_for_fit_to_data_for_finding_extrema}"
                       "(*p_guess)"))

        p_global, pconv = scipy.optimize.curve_fit(
            f_fit, self.t[:idx_end],
            self.data_for_finding_extrema[:idx_end], p0=p0,
            bounds=bounds0)
        if self.verbose:
            print(f"            result p_global={p_global}")

        if pp:
            line, = axs[0].plot(self.t[:idx_end],
                                f_fit(self.t[:idx_end], *p_global),
                                label='fit [first 10 orbits]')
            axs[0].plot(self.t, f_fit(self.t, *p_global),
                        color=line.get_color(), linewidth=0.5)

            line, = axs[1].plot(
                self.t[:idx_end],
                self.data_for_finding_extrema[:idx_end]
                - f_fit(self.t[:idx_end], *p_global),
                label='residual (fitted region only)')
            line, = axs[2].plot(
                self.t[:idx_end],
                self.data_for_finding_extrema[:idx_end]
                - f_fit(self.t[:idx_end], *p_global),
                label='residual (fitted region)')
            axs[2].plot(self.t,
                        self.data_for_finding_extrema-f_fit(self.t, *p_global),
                        linewidth=0.5, color=line.get_color(),
                        label='residual (all data)')
            axs[0].legend()
            axs[2].legend()
            fig.tight_layout()
            # fig.savefig(pp, format='pdf')
            self.save_debug_fig(fig, pp, diag_file)
            plt.close(fig)

        # STEP 2 From start of data, move through data and do local fits across
        # (2N+1) extrema.  For each fit, take the middle one as part of the
        # output

        # collects indices of extrema
        extrema = []

        # collects floating point values associated with extrema
        # if refine_extrema==true, these will in general be betweem
        # grid-points
        t_extrema_refined = []
        data_extrema_refined = []
        phase22_extrema_refined = []

        # estimates for initial start-up values. Because idx_ref
        # is only allowed to increase, be rather conservative with
        # its initial guess
        K = 1.1   # periastron-advance rate
        idx_ref = np.argmax(self.phase22
                            > self.phase22[0] + K*(N-1)*4*np.pi)
        if idx_ref == 0:
            raise Exception("data set too short.")
        p = p_global
        count = 0
        extra_extrema = []
        extra_extrema_t = []
        extra_extrema_data = []
        extra_extrema_phase22 = []
        while True:
            count = count+1
            if self.verbose:
                print(f"=== count={count} "+"="*60)
            idx_extrema, p, K, idx_ref, extrema_refined \
                = self.FindExtremaNearIdxRef(
                    idx_ref,
                    sign, N, N+1, K,
                    f_fit, p, bounds0,
                    1e-8,
                    increase_idx_ref_if_needed=True,
                    refine_extrema=refine_extrema,
                    verbose=self.verbose,
                    pp=pp,
                    plot_info=f"count={count}")
            if self.verbose:
                print(f"IDX_EXTREMA={idx_extrema}, "
                      f"{self.data_str}_fit"
                      f"={f_fit.format(*p)}, "
                      f"K={K:5.3f}, idx_ref={idx_ref}")

            # decide whether we are going to stop iterating:
            terminate = False
            if len(idx_extrema) < 2*N+1:  # didn't find the desired number of extrema
                terminate = True
            elif idx_extrema[N] < idx_ref:
                # The N+1-st extremum should be *later* than idx_ref.  If not, sth wrong
                terminate = True

            if len(idx_extrema) >= 2*N-1:
                # at most two extrema short of target.  Assume the fit is
                # good enough to report extrema obtained through it.
                if count == 1:
                    # at first call (at start of waveform, report the extrema
                    # identified in the left part of the fitting interval
                    for k in range(0, N):
                        extrema.append(idx_extrema[k])
                        t_extrema_refined.append(extrema_refined[0][k])
                        data_extrema_refined.append(extrema_refined[1][k])
                        phase22_extrema_refined.append(extrema_refined[2][k])
                if idx_extrema[N] > idx_ref:
                    # the N+1-st extremum should be *after* idx_ref.  If it is
                    # not, something went wrong, so do not take this extremum

                    # take the extremum in the middle of the fitting interval.
                    # (if we are short extrema, then there will be fewer to the
                    # right. To not report those, due to potentially inaccurate
                    # fits that close to merger)
                    extrema.append(idx_extrema[N])
                    t_extrema_refined.append(extrema_refined[0][N])
                    data_extrema_refined.append(extrema_refined[1][N])
                    phase22_extrema_refined.append(extrema_refined[2][N])

                    # also store additional extrema, just in case we want to
                    # return them
                    extra_extrema = idx_extrema[N+1:]
                    extra_extrema_t = extrema_refined[0][N+1:]
                    extra_extrema_data = extrema_refined[1][N+1:]
                    extra_extrema_phase22 = extrema_refined[2][N+1:]

                if terminate:
                    # be greedy and take any further extrema we know
                    for k in range(N+1, len(idx_extrema)):
                        if idx_extrema[k] > idx_ref:
                            extrema.append(idx_extrema[k])
                            t_extrema_refined.append(extrema_refined[0][k])
                            data_extrema_refined.append(
                                extrema_refined[1][k])
                            phase22_extrema_refined.append(
                                extrema_refined[2][k])
            else:
                # more than two extrema short.  In this case, don't trust the
                # fit anymore, but rather report the extema found in the last
                # iteration

                # sanity check: if we have too few extrema, ten terminate must
                # be set, too.
                if not terminate:
                    raise Exception("Logical error -- should never get here")
                if self.verbose:
                    print("terminating with very few extrema in this "
                          "iteration. Take left-over extrema from previous "
                          "iteration")
                    print(f"extra_extrema_t={extra_extrema_t}")
                extrema.extend(extra_extrema)
                t_extrema_refined.extend(extra_extrema_t)
                data_extrema_refined.extend(extra_extrema_data)
                phase22_extrema_refined.extend(extra_extrema_phase22)

            if terminate:
                # print("WARNING - TOO FEW EXTREMA FOUND. THIS IS LIKELY SIGNAL
                # THAT WE ARE AT MERGER")
                break

            # shift idx_ref one extremum to the right, in preparation for the
            # next extrema-search
            idx_ref = int(0.5*(idx_extrema[N]+idx_extrema[N+1]))

            if count > 10000:
                if pp:
                    pp.close()
                raise Exception("Detected more than 10000 extrema. "
                                "This has triggered a saftey exception."
                                "If your waveform is really this long, "
                                "please remove this exception and run again.")
        if self.verbose:
            print(f"Reached end of data.  Identified extrema = {extrema}")
        if pp:
            pp.close()

        if sign > 0:
            self.periastron_info = [np.array(t_extrema_refined),
                                    np.array(data_extrema_refined),
                                    np.array(phase22_extrema_refined)]
        else:
            self.apastron_info = [np.array(t_extrema_refined),
                                  np.array(data_extrema_refined),
                                  np.array(phase22_extrema_refined)]

        # Now that we are done with finding peaks, we shift the time axis to
        # it's original values
        self.t += self.t_merger
        return np.array(extrema)

        # Procedure:
        # - find length of useable dataset
        #     - exlude from start
        #     - find Max(A) and exclude from end
        # - one global fit for initial fitting parameters
        # - check which trefs we can do:
        #     - delineate N_extrema * 0.5 orbits from start
        #     - delineate N_extrema * 0.5 orbits from end
        #     - discard tref outside this interval (this places the trefs at
        #       least mostly into the middle of the fitting intervals. Not
        #       perfectly, since due to periastron advance the radial periods
        #       are longer than the orbital ones)
        # - set K=1
        # - set fitting_func=global_fit
        # - Loop over tref:
        #     - set old_extrema = [0.]*N_extrema
        #     - Loop over fitting-iterations:
        #         - (A) find interval that covers phase from
        #           K*(0.5*N_extrema+0.2) orbits before to after t_ref
        #         - find extrema of data - fit
        #         - update K based on the identified extrema
        #         - if  number of extrema != N_extrema:
        #             goto (A)  [i.e. compute a larger/smaller data-interval
        #             with new K]
        #         - if |extrema - old_extrema| < tol:  break
        #         - old_extrema=extrema
        #         - update fitting_func by fit to extrema


    def FindExtremaNearIdxRef(self,
                              idx_ref,
                              sign, Nbefore, Nafter, K,
                              f_fit, p_initial, bounds,
                              TOL,
                              increase_idx_ref_if_needed=True,
                              refine_extrema=False,
                              verbose=False,
                              pp=None,
                              plot_info=""):
        """given a 22-GW mode (t, phase22, data), identify a stretch of data
        [idx_lo, idx_hi] centered roughly around the index idx_ref which
        satisfies the following properties:
          - The interval [idx_lo, idx_hi] contains Nbefore+Nafter maxima
            (if sign==+1) or minimia (if sign==-1)
            of trend-subtracted data, where Nbefore exrema are before idx_ref
            and Nafter extrema are after idx_ref
          - The trend-subtraction is specified by the fitting function
            data_trend = f_fit(t, *p).
            Its fitting parameters *p are self-consistently fitted to the
            N_extrema extrema.
          - if increase_idx_ref_if_needed, idx_ref is allowed to increase in
            order to reach the desired Nbefore.

        INPUT
          - idx_ref   - the reference index, i.e. the approximate middle of the
            interval of data to be sought
          - sign      - if +1, look for maxima, if -1, look for minima
          - Nbefore   - number of extrema to identify before idx_ref
          - Nafter    - number of extrema to identify after idx_ref
                          if Nafter=Nbefore-1, then the Nbefore'th extremum
                          will be centered
          - K         - an estimate for the periastron advance of the binary,
                        i.e. the increase of phase22/4pi between two extrema
          - f_fit     - fitting function f_fit(t, *p) to use for
                        trend-subtraction
          - p_initial - initial guesses for the best-fit parametes
          - p_bounds  - bounds for the fit-parameters
          - TOL       - iterate until the maximum change in any one data at an
                        extremum is less tha this TOL
          - increase_idx_ref_if_needed -- if true, allows to increase idx_ref
                                          in order to achieve Nbefore extrema
                                          between start of dataset and idx_ref
                                          (idx_ref will never be decreased, in
                                          order to preserve monotonicity to
                                          help tracing out an inspiral)

          - pp a PdfPages object for a diagnostic output plot
          - plot_info -- string placed into the title of the diagnostic plot

        RETURNS:
              idx_extrema, p, K, idx_ref, extrema_refined
        where
          - idx_extrema -- the indices of the identified extrema
                           USUALLY len(idx_extrema) == Nbefore+Nafter HOWEVER,
                           if not enough extrema can be identified (e.g.  end
                           of data), then a shorter or even empty list can be
                           returned

          - p -- the fitting parameters of the best fit through the extrema
          - K -- an updated estimate of the periastron advance K (i.e. the
                 average increase of phase22 between extrema divided by 4pi)
          - idx_ref -- a (potentially increased) value of idx_ref, so that
                       Nbefore extrema were found between the start of the data
                       and idx_ref
          - extrema_refined=(t_extrema_refined, data_extrema_refined,
                 phase22_extrema_refined) information about the
                 parabolic-fit-refined extrema.  If RefineExtrema==True, these
                 arrays have same length as idx_extrema.  Otherwise, empty.


        ASSUMPTIONS & POSSIBLE FAILURE MODES
          - if increase_idx_ref_if_needed == False, and idx_lo cannot be
            reduced enough to reach Nbefore -> raise Exception
          - if fewer extrema are identified than requested, then the function
            will return normally, but with len(idx_extrema) **SMALLER** than
            Nbefore+Nafter. This signals that the end of the data is reached,
            and that the user should not press to even larger idx_ref.
        """
        extrema_type = {+1: "pericenters", -1: "apocenters"}[sign]
        if verbose:
            print(f"FindExtremaNearIdxRef  idx_ref={idx_ref}, "
                  f"K_initial={K:5.3f}, "
                  f"p_initial={f_fit.format(*p_initial)}"
                  f", refine_extrema={refine_extrema}")

        # look for somewhat more data than we (probably) need
        DeltaPhase = 4.2*np.pi*K
        idx_lo = np.argmax(
            self.phase22 > self.phase22[idx_ref]
            - DeltaPhase*Nbefore)
        idx_hi = np.argmax(
            self.phase22 > self.phase22[idx_ref]
            + DeltaPhase*Nafter)
        if idx_hi == 0:
            idx_hi = len(self.phase22)
            if verbose:
                print("WARNING: reaching end of data, so close to merger")
        p = p_initial
        it = 0

        old_extrema = np.zeros(Nbefore+Nafter)
        old_idx_lo, old_idx_hi = -1, -1

        # this variable counts the number of iterations in which Nright was one
        # too few.  This is used to detect limiting cycles, where the interval
        # adjustment oscillates between
        #   short interval with Nleft, Nright-1    extrema
        #   long interval with  Nleft, Nright      extrema
        # the oscillations can occur, because the fit with one more/left
        # extremum is so different as to make the extremum appear/vanish
        Count_Nright_short = 0

        if pp:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(plot_info, x=0.02, ha='left')
            axs[0].set_title(
                'trend-subtracted:  '
                f"{'-' if sign == -1 else ''}"
                f"({self.label_for_data_for_finding_extrema}-"
                f'{self.label_for_fit_to_data_for_finding_extrema})',
                fontsize='small')
            axs[1].set_title(f'{self.label_for_data_for_finding_extrema}(t)')
            axs[2].set_title('residual of fit')
            plot_offset = None

        # compute prominence
        distance, prominence = self.compute_distance_and_prominence(
            idx_lo, idx_hi, f_fit, p)
        if verbose:
            print(f"       find_peaks: distance={distance}, "
                  f"prominence={prominence}")
        interval_changed_on_it = -1
        if self.return_diagnostic_data:
            # Append an empty dict and for each iteration add data to this dict with iteration
            # number as the key.
            self.diagnostic_data_dict['params'][extrema_type].append({})
            self.diagnostic_data_dict['t_extrema'][extrema_type].append({})
            self.diagnostic_data_dict['data_extrema'][extrema_type].append({})
            self.diagnostic_data_dict['t_ref'][ extrema_type].append({})
            self.diagnostic_data_dict['f_fit'][ extrema_type].append({})
        while True:
            it = it+1
            if verbose:
                print(f"it={it}:  [{idx_lo} / {idx_ref} / {idx_hi}],  "
                      f"K={K:5.3f}")
            data_residual = (self.data_for_finding_extrema[idx_lo:idx_hi]
                             - f_fit(self.t[idx_lo:idx_hi], *p))
            data_residual_amp = max(data_residual)-min(data_residual)
            # TODO -- pass user-specified arguments into find_peaks
            # POSSIBLE UPGRADE
            # find_peaks on discrete data will not identify a peak to a
            # location better than the time-spacing.  To improve, one can add a
            # parabolic fit to the data around the time of extremum, and then
            # take the (fractional) index where the fit reaches its maximum.
            # Harald has some code to do this, but he hasn't moved it over yet
            # to keep the base implementation simple.

            # width used as exclusion in find_peaks
            #    1/2 phi-orbit  (at highest data)
            #    translated into samples using the maximum time-spacing

            idx_extrema, properties = scipy.signal.find_peaks(
                sign*data_residual,
                distance=distance,
                prominence=prominence)

            # add offset due to to calling find_peaks with sliced data
            idx_extrema = idx_extrema+idx_lo
            Nleft = sum(idx_extrema < idx_ref)
            Nright = sum(idx_extrema >= idx_ref)

            # remember info about extrema to be used in rest of this function
            N_extrema = len(idx_extrema)
            t_extrema = self.t[idx_extrema]
            data_extrema = self.data_for_finding_extrema[idx_extrema]
            phase22_extrema = self.phase22[idx_extrema]
            # data_residual is shorter array
            data_residual_extrema = data_residual[idx_extrema-idx_lo]
            # update K based on identified peaks
            if N_extrema >= 2:
                K = ((phase22_extrema[-1] - phase22_extrema[0])
                     / (4*np.pi * (N_extrema - 1)))
            if verbose:
                with np.printoptions(precision=2):
                    print(f"       idx_extrema=   {idx_extrema}, "
                          f"Nleft={Nleft},"
                          f" Nright={Nright}")
                    print(f"       t[idx_extrema]={t_extrema}")

            if refine_extrema:
                t_extrema, data_extrema, phase22_extrema \
                    = self.get_refined_extrema(t_extrema,
                                               data_extrema,
                                               phase22_extrema,
                                               verbose, f_fit, p)
                if verbose:
                    with np.printoptions(precision=4):
                        print("")
                        print("       Delta t_extrema = "
                              f"{t_extrema - self.t[idx_extrema]}")
            if N_extrema > 0 and self.return_diagnostic_data:
                # Update the dictionary for current extrema count with data for this iteration
                self.diagnostic_data_dict['params'][extrema_type][-1].update({it: p})
                self.diagnostic_data_dict['t_extrema'][extrema_type][-1].update({it: t_extrema})
                self.diagnostic_data_dict['data_extrema'][extrema_type][-1].update({it: data_extrema})
                self.diagnostic_data_dict['t_ref'][ extrema_type][-1].update({it: self.t[idx_ref]})
                self.diagnostic_data_dict['f_fit'][ extrema_type][-1].update({it: f_fit})
            if pp:
                # offset data vertically by 10^k*it
                if plot_offset is None:
                    plot_offset = 10**np.ceil(np.log10(data_residual_amp/2.))
                    axs[0].axvline(self.t[idx_ref], linestyle='--',
                                   color='grey', linewidth=1)

                line, = axs[0].plot(self.t[idx_lo:idx_hi],
                                    it*plot_offset
                                    + sign*data_residual, label=f"it={it}")
                if N_extrema > 0:
                    axs[0].plot(t_extrema,
                                it*plot_offset+sign*data_residual_extrema,
                                'o',
                                color=line.get_color())
                line, = axs[1].plot(
                    self.t[idx_lo:idx_hi],
                    self.data_for_finding_extrema[idx_lo:idx_hi]
                    + plot_offset*it)
                # note: 'line.get_color()` is also used below in axs[2].plot
                axs[1].plot(t_extrema, data_extrema+plot_offset*it, 'o',
                            color=line.get_color(), label=f"it={it}")

            if Nright < Nafter:  # and Nleft==Nbefore:
                Count_Nright_short = Count_Nright_short+1
                if verbose:
                    print(f"       Count_Nright_short={Count_Nright_short}")

            if N_extrema == 0 or it > 20:
                if verbose:
                    if N_extrema == 0:
                        print("could not identify a single extremum. "
                              "This can happen, for instance\n"
                              "for low eccentricity late in the inspiral where"
                              " the range of data\n"
                              "is so large that the prominence = "
                              f"{0.03*data_residual_amp} cannot be\n"
                              "reached by the small eccentricity oscillations")
                    else:
                        print(f"interval finding failed after it={it} "
                              "iterations. Exit")
                if pp:
                    # plt.legend()
                    fig.tight_layout()
                    fig.savefig(pp, format='pdf')
                    plt.close(fig)
                # don't really know what to do if we didn't identify any
                # extrema.  so return with empty idx_extrema and let upstream
                # code handle this
                return idx_extrema, p, K, idx_ref, [t_extrema, data_extrema,
                                                    phase22_extrema]

            if Nleft != Nbefore or Nright != Nafter:
                # number of extrema not as we wished, so update [idx_lo, idx_hi]
                # too many peaks left, discard by placing idx_lo between N and
                # N+1's peak to left
                if Nleft > Nbefore:
                    idx_lo = int(
                        (idx_extrema[Nleft-Nbefore-1]
                         + idx_extrema[Nleft-Nbefore])/2)
                    if verbose:
                        print(f"       idx_lo increased to {idx_lo}")
                elif Nleft < Nbefore:  # reduce idx_lo to capture one more peak
                    if idx_lo == 0:
                        # no more data to the left, so consider shifting
                        # idx_ref
                        if increase_idx_ref_if_needed:
                            if Nright >= 2:
                                # we need at least two maxima to the right to
                                # average for the new idx_ref
                                tmp = np.argmax(idx_extrema >= idx_ref)
                                # shift idx_ref one extremum to the right
                                idx_ref = int((idx_extrema[tmp]
                                               + idx_extrema[tmp + 1])/2)
                                # reflect the change in idx_ref to aid in
                                # updating idx_hi
                                Nright = Nright - 1
                                if verbose:
                                    print("       idx_ref increased to"
                                          f" {idx_ref}")
                            else:
                                pass
                                # First, wait for the idx_hi-updating below to
                                # widen the interval.  The next iteration will
                                # come back here and update idx_ref

                        else:
                            raise Exception(
                                f"could not identify {Nbefore}"
                                f" extrema to the left of idx_ref={idx_ref}")
                    else:
                        # decrease idx_lo by 0.6 radial periods.  This should
                        # get idx_lo toward seeing one earlier extremum.
                        # Rationale for 0.6: The next extremum should be 1
                        # radial period earlier.  We rather prefer to err on
                        # the low side, than overshooting and adding two
                        # extrema at once.
                        phase_lo = self.phase22[idx_lo] - K*4*np.pi*0.6
                        idx_lo = np.argmax(self.phase22 > phase_lo)
                        if verbose:
                            print(f"       idx_lo reduced to {idx_lo}")
                # too many peaks to the right right, discard by placing idx_hi
                # between N and N+1's peak to right
                if Nright > Nafter:
                    idx_hi = int((idx_extrema[Nafter-Nright]
                                  + idx_extrema[Nafter-Nright-1])/2)
                    if verbose:
                        print(f"        idx_hi reduced to {idx_hi}")
                elif Nright < Nafter:
                    # increase idx_hi to capture one more peak
                    # do we have extra data?
                    if idx_hi < len(self.phase22):
                        # target phase on right 0.6 radial periods beyond
                        # current end of interval rationale for 0.6: The next
                        # extremum should be 1 radial period away, we are
                        # worried that near the end of the run, this prediction
                        # may not be accurate.  Therefore, go more slowly.
                        phase_hi = self.phase22[idx_hi] + K*4*np.pi*0.6
                        idx_hi = np.argmax(self.phase22 > phase_hi)
                        if idx_hi == 0:
                            # coulnd't get as much data as we wished, take all
                            # we have
                            idx_hi = len(self.phase22)
                        if verbose and idx_hi != old_idx_hi:
                            print(f"       idx_hi increased to {idx_hi}")
                    else:
                        # we had already fully extended idx_hi in earlier
                        # iteration
                        if verbose:
                            print("        idx_hi at its maximum, "
                                  "but still insufficient "
                                  f"Nright={Nright}")

                if (idx_lo, idx_hi) != (old_idx_lo, old_idx_hi):
                    interval_changed_on_it = it
                    # remember when we last changed the search interval
                    (old_idx_lo, old_idx_hi) = (idx_lo, idx_hi)
                    # data-interval was updated; go back to start of loop to
                    # re-identify extrema
                    continue

            # if the code gets here, we have an interval [idx_lo,idx_high] with
            # either
            #  - Nleft + Nright envelope-subtracted extrema,
            # *or*
            #  - fewer envelope subtracted extrema and idx_hi at the end of the
            #  - data
            #
            # The following arrays are filled with information at the extrema
            #   t_extrema, phase22_extrema, data_extrema, where:
            #      * If refine_extrema==False: the arrays correspond to index
            #      * positions idx_extrema If refine_extrema==True: the arrays
            #      * are refined via fits.
            #
            # Now check whether data-envelope fitting has already converged.
            # If yes: return
            # If no:  re-fit envelope

            if N_extrema != len(old_extrema):
                # number of extrema has changed since last iteration, so avoid
                # to compute differences to last-iteration's extrema
                max_delta_data = 1e99
            else:
                max_delta_data = max(np.abs(data_extrema-old_extrema))
            if it >= 16 \
               and Nleft == Nbefore and Nright == Nafter \
               and interval_changed_on_it == it-1:
                # we have been interating for a while, and the recent change in
                # N_extrema points toward a limiting cycle.  Since we just
                # happened to have hit the right number of extrema, let's take
                # them and exit.
                if verbose:
                    print("looks like we hit a limiting cycle; presently the "
                          "number of extrema is correct, so exit")
                if pp:
                    # plt.legend()
                    fig.tight_layout()
                    fig.savefig(pp, format='pdf')
                    plt.close(fig)
                return idx_extrema, p, K, idx_ref, [t_extrema, data_extrema,
                                                    phase22_extrema]

            if Count_Nright_short >= 5 or N_extrema < 5 or it > 20:
                # safety exit to catch periodic loops note that
                # Count_Nright_short is only increased if Nright<Nafter,
                # therefore, this will coincide with Nright<Nafter, also
                # signaling that the overall extrema searching is ending.  we
                # require **5** extrema, in order to have safety for the **3**
                # parameter fit below
                if verbose:
                    print("exiting because Count_right_short="
                          f"{Count_Nright_short}"
                          f" is large, or N_extrema={N_extrema} is "
                          "insufficient")
                if pp:
                    # plt.legend()
                    fig.tight_layout()
                    fig.savefig(pp, format='pdf')
                    plt.close(fig)
                return idx_extrema, p, K, idx_ref, [t_extrema, data_extrema,
                                                    phase22_extrema]

            if max_delta_data < TOL:
                # (this cannot trigger on first iteration, due to
                # initialization of old_extrema)
                if verbose:
                    print(f"max_delta_{self.data_str}={max_delta_data:5.4g}<"
                          f"TOL={TOL}. Done.")
                if pp:
                    # plt.legend()
                    fig.tight_layout()
                    fig.savefig(pp, format='pdf')
                    plt.close(fig)
                return idx_extrema, p, K, idx_ref, [t_extrema, data_extrema,
                                                    phase22_extrema]
            p, pconv = scipy.optimize.curve_fit(f_fit, t_extrema,
                                                data_extrema, p0=p,
                                                bounds=bounds, maxfev=10000)
            if verbose and False:
                print("    PRODUCTION FIT: residual="
                      f"{np.linalg.norm(f_fit(t_extrema,*p)-data_extrema)},"
                      f" f={f_fit.format(*p)}")
            if pp:
                axs[2].plot(t_extrema, f_fit(t_extrema, *p)-data_extrema, "o",
                            color=line.get_color())
            old_extrema = data_extrema
            if verbose:
                print(f"       max_delta_{self.data_str}={max_delta_data:5.4g}"
                      f" => fit updated to f_fit={f_fit.format(*p)}")
        raise Exception("Should never get here")

    def compute_distance_and_prominence(self, idx_lo, idx_hi, f_fit, p):
        """Compute distance and prominence for current data section."""
        maxdt = np.max(np.diff(self.t[idx_lo:idx_hi]))

        # average orbital period during [idx_lo, idx_hi] idx_hi-1 also
        # works in the case when idx_hi = one-past-last-element
        T_orbit = ((self.t[idx_hi-1] - self.t[idx_lo])
                   / (self.phase22[idx_hi-1]
                      - self.phase22[idx_lo])
                   * 4*np.pi)

        # set distance = distance_factor * period.  This should exclude
        # spurious extrema due to noise
        distance = int(self.distance_factor*T_orbit/maxdt)
        data_residual = (self.data_for_finding_extrema[idx_lo:idx_hi]
                         - f_fit(self.t[idx_lo:idx_hi], *p))
        data_residual_amp = max(data_residual)-min(data_residual)
        prominence = data_residual_amp*self.prominence_factor

        return distance, prominence

    def get_refined_extrema(self, t_extrema, data_extrema, phase22_extrema,
                            verbose, f_fit, p):
        """Get refined extrema.

        Perform parabolic fits to data_residual around each extremum
        in order to find the time of extrema to sub-index
        accuracy.
        """
        if verbose:
            print(f"       Refine {len(t_extrema)} extremas, "
                  f"local fit with "
                  "Npoints=", end='')
        for k in range(len(t_extrema)):
            # length of fitting interval = 0.05radians left/right
            deltaT = 0.05 / data_extrema[k]
            idx_refine = np.abs(self.t - t_extrema[k]) < deltaT
            # number of points to be used in fit
            N_refine = sum(idx_refine)
            if verbose:
                print(f"{N_refine}  ", end='')
            if N_refine >= 7:  # enough data for fit
                t_parafit = self.t[idx_refine]
                # re-compute fit-subtracted data_residual, to avoid
                # indexing problems, should idx_lo/idx_high be so close
                # that the parabolic fitting interval extends beyond it
                data_resi_parafit = (
                    self.data_for_finding_extrema[idx_refine]
                    - f_fit(t_parafit, *p))

                parabola = np.polynomial.polynomial.Polynomial.fit(
                    t_parafit, data_resi_parafit, 2)
                t_max = parabola.deriv().roots()[0]

                # update extrema information
                t_extrema[k] = t_max

                # interpolate data from fits *assumption* the
                # fitting-interval is short enough that this is
                # accurate
                data_extrema[k] = parabola(t_max) + f_fit(t_max, *p)

                # 3rd order fit to phase to interpolate *assumption*
                # the fitting-interval is short enough that this is
                # accurate
                phase_fit = np.polynomial.polynomial.Polynomial.fit(
                    t_parafit, self.phase22[idx_refine], 3)
                phase22_extrema[k] = phase_fit(t_max)
            else:
                pass
                # if verbose:
                #    print(f"refinement of k={k} has too few points - skip")
        return t_extrema, data_extrema, phase22_extrema
