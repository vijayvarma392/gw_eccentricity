"""
Find peaks and troughs using frequency fits.

Part of Eccentricity Definition project.
Md Arif Shaikh, Mar 29, 2022
"""
from .eccDefinition import eccDefinition
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class envelope_fitting_function:
    """Re-parameterize A*(t-T)^n in terms of function value, first derivative at the time t0, and T"""    

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
        # at t0.  Re-expfress as T, n, A, then evalate A*(t-T)^n
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


class envelope_fitting_function2:
    """Re-parameterize A*(t-T)^n in terms of function value, first derivative at the time t0, and 
    second time-derivative at time t0.  The second time-derivative is re-parameterized
    into alpha in [-1,1], such that the value alpha=-1 corresponds to Tmin<0, the value
    alpha=0 corresponds to T=0, and alpha=1 to T->infty.  This reparameterization is hoped
    to improve convergence of the fitting procedure while simultaneously allowing for square
    bounds -1<alpha<1.
    """
    def __init__(self, t0, Tmin,verbose=False):
        """Init."""
        self.t0 = t0
        self.Tmin=Tmin
        self.alpha1 = (Tmin - t0) / (-Tmin)
        self.alpha1 = -2*(Tmin - t0) / (2*Tmin-t0)
        self.Gamma = -(-t0 + 2*Tmin)/(2*(Tmin-t0)*(-t0))
        self.verbose = verbose
        if verbose:
            print(f"fitting_function2:  to={t0}, Tmin={Tmin}, alpha1={self.alpha1}, Gamma={self.Gamma}")
        if -t0<= -2*Tmin:
            raise Exception("t0 must be at least twice as large as Tmin, "\
                "otherwise the alpha-reparameterization is multi-valued"
            )

    def format(self, f0, f1, alpha):
        """Return a string representation for use in legends and output."""
        T_m_t0 =  (self.Tmin-self.t0)/(1.-alpha)
        n = -f1/f0*T_m_t0
        A = f0*(T_m_t0)**(-n)
        T = T_m_t0 + self.t0
        return f"{A:.3g}({T:+.2f}-t)^{n:.3f}"

    def __call__(self, t, f0, f1, alpha):
        """Call."""
        # f0, f1 are function values and first time-derivatives
        # at t0.  Re-expfress as T, n, A, then evalate A*(t-T)^n

        # T-t0
        T_m_t0 =  (self.Tmin-self.t0)/(1.-alpha)
        n = -f1/f0*T_m_t0
        A = f0*(T_m_t0)**(-n)
        T = T_m_t0 + self.t0
        if self.verbose:
            print(f"f0={f0}, f1={f1}, alpha={alpha}; T={T}, n={n}, A={A}, max(t)={t.max()}")
        if t.max() > T:
            print(end="", flush=True)
            raise Exception(
                "envelope_fitting_function reached parameters where merger "
                "time T is within time-series to be fitted\n"
                f"f0={f0}, f1={f1}, alpha={alpha}; T={T}, n={n}, A={A}, max(t)={max(t)}\n"
                f"self.to={self.t0}, self.Tmin={self.Tmin}")
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

        self.debug=self.extra_kwargs["debug"];

        # create the shortened data-set for analysis 
        if self.extra_kwargs["num_orbits_to_exclude_before_merger"] is not None:
            merger_idx = np.argmin(np.abs(self.t - self.t_merger))
            phase22_at_merger = self.phase22[merger_idx]
            # one orbit changes the 22 mode phase by 4 pi since
            # omega22 = 2 omega_orb
            phase22_num_orbits_earlier_than_merger = (
                phase22_at_merger
                - 4 * np.pi
                * self.extra_kwargs["num_orbits_to_exclude_before_merger"])
            idx_num_orbit_earlier_than_merger = np.argmin(np.abs(
                self.phase22 - phase22_num_orbits_earlier_than_merger))

            self.tC_analyse = self.t[:idx_num_orbit_earlier_than_merger]
            self.omega22_analyse = self.omega22[
                :idx_num_orbit_earlier_than_merger]
            self.phase22_analyse = self.phase22[
                :idx_num_orbit_earlier_than_merger]
            self.t_analyse = self.t[:idx_num_orbit_earlier_than_merger]
        else:
            self.t_analyse = self.t
            self.omega22_analyse = self.omega22
            self.phase22_analyse = self.phase22
            self.t_analyse = self.t

        # TODO - consider whether to also cut from the start
        # (e.g. NR junk radiation)

        return

    def find_extrema(self, extrema_type="maxima"):
        """Find the extrema in the data.

        parameters:
        -----------
        extrema_type:
            One of 'maxima', 'peaks', 'minima' or 'troughs'.

        returns:
        ------
        array of positions of extrema.
        """
        # STEP 0 - setup

        if extrema_type in ['maxima', 'peaks']:
            sign = +1
        elif extrema_type in ['minima', 'troughs']:
            sign = -1
        else:
            raise Exception(f"extrema_type='{extrema_type}' unknown.")

        # data-sets to operate on (stored as member data)
        # self.t_analyse
        # self.phase22_analyse
        # self.omega22_analyse

        # DESIRED NUMBER OF EXTREMA left/right DURING FITTING
        # Code will look for N+1 extrema to the left of idx_ref, and N extrema to the right
        # TODO - make user-specifiable via option
        N = 3

        verbose=self.debug
        if verbose:
            diag_file="eccDefinitionUsingFrequencyFitsDiagnosticOutput.pdf"
        else:
            diag_file=""

        if diag_file!="":
            pp = PdfPages(diag_file)
        else:
            pp=False

        # STEP 1:
        # global fit as initialization of envelope-subtraced extrema

        UseAlphaFit=False
        fit_center_time = 0.5*(self.t_analyse[0] + self.t_analyse[-1])
        if UseAlphaFit:
            Tmin = 0.8*self.t_analyse[-1]
            f_fit = envelope_fitting_function2(t0=fit_center_time,
                                               Tmin=Tmin,
                                               verbose=False
                                              #verbose=verbose
                                              )
            # some reasonable initial guess for c urve_fit
            nPN = -3./8  # the PN exponent as approximation
            f0 = 0.5 * (self.omega22_analyse[0]+self.omega22_analyse[-1])
            p0 = [f0,  # function value ~ omega
                  -nPN*f0/(-fit_center_time),  # func = f0/t0^n*(t)^n -> dfunc/dt (t0) = n*f0/t0
                  0.  # singularity in fit is near t=0, since waveform aligned at max(amp22)
                  ]
            # some hopefully reasonable bounds for global curve_fit
            bounds0 = [[0., 0., 0.],
                       [1., 10./(-fit_center_time), 1.]]
        else:
            f_fit = envelope_fitting_function(t0=fit_center_time,
                                              verbose=False
                                              #verbose=verbose
                                              )
            # some reasonable initial guess for curve_fit
            nPN = -3./8  # the PN exponent as approximation
            f0 = 0.5 * (self.omega22_analyse[0]+self.omega22_analyse[-1])
            p0 = [f0,  # function value ~ omega
                  -nPN*f0/(-fit_center_time),  # func = f0/t0^n*(t)^n -> dfunc/dt (t0) = n*f0/t0
                  0.  # singularity in fit is near t=0, since waveform aligned at max(amp22)
                  ]
            # some hopefully reasonable bounds for global curve_fit
            bounds0 = [[0., 0., 0.8*self.t_analyse[-1]],
                       [1., 10./(-fit_center_time), -fit_center_time]]
        if verbose:
            print(f"global fit: guess p0={p0}, bounds={bounds0}")
        p_global, pconv = scipy.optimize.curve_fit(
            f_fit, self.t_analyse,
            self.omega22_analyse, p0=p0,
            bounds=bounds0)

        # STEP 2
        # starting at start of data, move through data and do local fits across (2N+1) extrema
        # for each fit, take the middle one as part of the outptu 

        # to collect extrema
        extrema = []

        # estimates for initial start-up values (will be updated as needed)
        K = 1.2   # periastron-advance rate
        idx_ref = np.argmax(self.phase22_analyse
                            > self.phase22_analyse[0] + K*N*4*np.pi)
        if idx_ref == 0:
            raise Exception("data set too short.")
        p = p_global
        count = 0
        while True:
            count = count+1
            if verbose:
                print(f"=== count={count} "+"="*60)
            idx_extrema, p, K, idx_ref = FindExtremaNearIdxRef(
                self.t_analyse, self.phase22_analyse,
                self.omega22_analyse,
                idx_ref,
                sign, N+1, N, K,
                f_fit, p, bounds0,
                1e-8,
                increase_idx_ref_if_needed=True,
                verbose=verbose,
                pp=pp)
            if verbose:
                print(f"IDX_EXTREMA={idx_extrema}, f_fit={f_fit.format(*p)}, "
                      f"K={K:5.3f}, idx_ref={idx_ref}")
            # use the one just to the left of idx_ref
            extrema.append(idx_extrema[N])
            idx_ref = int(0.5*(idx_extrema[N+1]+idx_extrema[N+2]))
            if len(idx_extrema) <= 2*N:
                # print("WARNING - TOO FEW EXTREMA FOUND.
                # THIS IS LIKELY SIGNAL THAT WE ARE AT MERGER")
                break
            if count > 1000:
                if pp:
                    pp.close()
                raise Exception("count large??")
        if verbose:
            print(f"Reached end of data.  Identified extrema = {extrema}")
        if pp:  
            pp.close()
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
        #         - find extrema of omega - fit
        #         - update K based on the identified extrema
        #         - if  number of extrema != N_extrema:
        #             goto (A)  [i.e. compute a larger/smaller data-interval
        #             with new K]
        #         - if |extrema - old_extrema| < tol:  break
        #         - old_extrema=extrema
        #         - update fitting_func by fit to extrema


def FindExtremaNearIdxRef(t, phase22, omega22,
                          idx_ref,
                          sign, Nbefore, Nafter, K,
                          f_fit, p_initial, bounds,
                          TOL,
                          increase_idx_ref_if_needed=True,
                          verbose=False,
                          pp=None):
    """given a 22-GW mode (t, phase22, omega22), identify a stretch of data
    [idx_lo, idx_hi] centered roughly around the index idx_ref which satisfies
    the following properties:
      - The interval [idx_lo, idx_hi] contains Nbefore+Nafter maxima
        (if sign==+1) or minimia (if sign==-1)
        of trend-subtracted omega22, where Nbefore exrema are before idx_ref
        and Nafter extrema are after idx_ref
      - The trend-subtraction is specified by the fitting function
        omega22_trend = f_fit(t, *p).
        Its fitting parameters *p are self-consistently fitted to the
        N_extrema extrema.
      - if increase_idx_ref_if_needed, idx_ref is allowed to increase in order
        to reach the desired Nbefore.

    INPUT
      - t, phase22, omega22 -- data to analyse
      - idx_ref   - the reference index, i.e. the approximate middle of the
        interval of data to be sought
      - sign      - if +1, look for maxima, if -1, look for minima
      - Nbefore   - number of extrema to identify before idx_ref
      - Nafter    - number of extrema to identify after idx_ref
                      if Nafter=Nbefore-1, then the Nbefore'th extremum will be
                      centered
      - K         - an estimate for the periastron advance of the binary,
                    i.e. the increase of phase22/4pi between two extrema
      - f_fit     - fitting function f_fit(t, *p) to use for trend-subtraction
      - p_initial - initial guesses for the best-fit parametes
      - p_bounds  - bounds for the best-fit parameters
      - TOL       - iterate until the maximum change in any one omega at an
                    extremum is less tha this TOL
      - increase_idx_ref_if_needed -- if true, allows to increase idx_ref in
                                      order to achieve Nbefore extrema between
                                      start of dataset and idx_ref (idx_ref
                                      will never be decreased, in order to
                                      preserve monotonicity to help tracing
                                      out an inspiral)

      - pp a PdfPages object for a diagnostic output plot 

    RETURNS:
          idx_extrema, p, K, idx_ref
    where
      - idx_extrema -- the indices of the identified extrema
                       USUALLY len(idx_extrema) == Nbefore+Nafter
                       HOWEVER, if Nafer cannot be reached because of
                       end-of-data, then
                         a) if one extremum too few is identified, returns
                            idx_extrema of length Nbefore+Nafter-1
                         b) if two or more extrema too few are identified,
                            then raise Exception

      - p -- the fitting parameters of the best fit through the extrema
      - K -- an updated estimate of the periastron advance K (i.e. the average
             increase of phase22 between extrema divided by 4pi)
      - idx_ref -- a (potentially increased) value of idx_ref, so that Nbefore
                   extrema were found between the
                   start of the data and idx_ref


    ASSUMPTIONS & POSSIBLE FAILURE MODES
      - if increase_idx_ref_if_needed == False, and idx_lo cannot be reduced
          enough to reach Nbefore -> raise Exception
      - if identified number of maxima after is exactly **ONE** below the
          target Nafter
          -> return normally, but with len(idx_extrema) **ONE SHORTER** than
             Nbefore+Nafter. This signals that the end of the data is reached,
             and that the user should not press to even larger idx_ref.
      - if identified number of maxima after is **two** or more below the
          target Nafter
          -> raise Exception
    """
    if verbose:
        print(f"FindExtremaNearIdxRef  idx_ref={idx_ref}, K_initial={K:5.3f}, "
              f"p_initial={f_fit.format(*p_initial)}")

    # look for somewhat more data than we (probably) need
    DeltaPhase = 4*np.pi*K
    idx_lo = np.argmax(phase22 > phase22[idx_ref] - DeltaPhase*Nbefore)
    idx_hi = np.argmax(phase22 > phase22[idx_ref] + DeltaPhase*Nafter)
    if idx_hi == 0:
        idx_hi = len(phase22)
        if verbose:
            print("WARNING: reaching end of data, so close to merger")
    p = p_initial
    it = 0

    old_extrema = np.zeros(Nbefore+Nafter)
    old_idx_lo, old_idx_hi = -1, -1

    # this variable counts the number of iterations in which Nright was one too few.
    # This is used to detect limiting cycles, where the interval adjustment oscillates
    # betweem
    #   short interval with Nleft, Nright-1    extrema
    #   long interval with  Nleft, Nright      extrema
    # the oscillations can occur, because the fit with one more/left extremum
    # is so different as to make the extremum appear/vanish
    Count_Nright_one_short=0

    if pp:
        fig,axs=plt.subplots(1,3,figsize=(11,4))

    while True:
        it = it+1

        if it > 30:
            if pp:
                axs[0].legend()
                fig.savefig(pp,format='pdf')
                plt.close(fig)
                pp.close()
            raise Exception("FindExtremaNearIdxRef seems to not converge "
                            "(use 'verbose=True' to diagnose)")
        if verbose:
            print(f"it={it}:  [{idx_lo} / {idx_ref} / {idx_hi}]")
        omega_residual = omega22[idx_lo:idx_hi] - f_fit(t[idx_lo:idx_hi], *p)

        # TODO -- pass user-specified arguments into find_peaks
        # POSSIBLE UPGRADE
        # find_peaks on discrete data will not identify a peak to a location
        # better than the time-spacing.  To improve, one can add a parabolic
        # fit to the data around the time of extremum, and then take the
        # (fractional) index where the fit reaches its maximum.
        # Harald has some code to do this, but he hasn't moved it over yet to
        # keep the base implementation simple.

        # width used as exclusion in find_peaks
        #    1/2 phi-orbit  (at highest omega)
        #    translated into samples using the maximum time-spacing
        maxdt = np.max(np.diff(t[idx_lo:idx_hi]))
        width=int( 0.5* 2 * np.pi / np.max(omega22[idx_lo:idx_hi]) / maxdt )
        omega_residual_amp = max(omega_residual)-min(omega_residual)
        if verbose:
            print(f"width for find_peaks = {width}")


        idx_extrema, properties = scipy.signal.find_peaks(
            sign*omega_residual,
                width=width,
                prominence=omega_residual_amp*0.03
        )
        # add offset due to to calling find_peaks with sliced data
        idx_extrema = idx_extrema+idx_lo
        Nleft = sum(idx_extrema < idx_ref)
        Nright = sum(idx_extrema >= idx_ref)

        # update K based on identified peaks
        K = ((phase22[idx_extrema[-1]] - phase22[idx_extrema[0]])
             / (4*np.pi * (len(idx_extrema) - 1)))
        if verbose:
            print(f"idx_extrema={idx_extrema}, Nleft={Nleft}, "
                  f"Nright={Nright}, K={K:5.3f}")
        if pp:
            axs[0].plot(t[idx_lo:idx_hi], sign*omega_residual, label=f"it={it}")
            axs[1].plot(t[idx_extrema], omega22[idx_extrema], 'o', label=f"it={it}")

        if Nright==Nafter-1  and Nleft==Nbefore:
            Count_Nright_one_short=Count_Nright_one_short+1
            if verbose:
                print(f"Count_Nright_one_short={Count_Nright_one_short}")

        if Nleft != Nbefore or Nright != Nafter:
            # number of extrema not as we wished, so update [idx_lo, idx_hi]
            if Nleft > Nbefore:  # too many peaks left, discard by placing idx_lo between N and N+1's peak to left
                idx_lo = int(
                    (idx_extrema[Nleft-Nbefore-1]
                     + idx_extrema[Nleft-Nbefore])/2)
                if verbose:
                    print(f"idx_lo increased to {idx_lo}")
            elif Nleft < Nbefore:  # reduce idx_lo to capture one more peak
                if idx_lo == 0:
                    # no more data to the left, so consider shifting idx_ref
                    if increase_idx_ref_if_needed:
                        if Nright >= 2:
                            # we need at least two maxima to the right to average for the new idx_ref
                            tmp = np.argmax(idx_extrema >= idx_ref)
                            # shift idx_ref one extremum to the right
                            idx_ref = int((idx_extrema[tmp]
                                           + idx_extrema[tmp + 1])/2)
                            Nright = Nright - 1  # reflect the change in idx_ref to aid in updating idx_hi
                            if verbose:
                                print(f"idx_ref increased to {idx_ref}")
                        else:
                            pass
                            # First, wait for the idx_hi-updating below to widen the interval.
                            # The next iteration will come back here and update idx_ref

                    else:
                        raise Exception(f"could not identify {Nbefore} extrema"
                                        f" to the left of idx_ref={idx_ref}")
                else:
                    # # target phase on left 1.5 radial periods before first
                    # # identified peak that should conveniently cover the next
                    # # earlier peak.  If there's not enough
                    # # data then this search returns the first data-point
                    # phase_lo = phase22[idx_extrema[0]] - K*4*np.pi*1.5
                    # idx_lo = np.argmax(phase22 > phase_lo)

                    # decrease idx_lo by 1.1 radial periods.  This should get the next 
                    # lower extremum into the view
                    phase_lo = phase22[idx_lo] - K*4*np.pi*1.1
                    idx_lo = np.argmax(phase22 > phase_lo)
                    if verbose:
                        print(f"idx_lo reduced to {idx_lo}")
            # too many peaks to the right right, discard by placing idx_hi
            # between N and N+1's peak to right
            if Nright > Nafter:
                idx_hi = int((idx_extrema[Nafter-Nright]
                              + idx_extrema[Nafter-Nright-1])/2)
                if verbose:
                    print(f"idx_hi reduced to {idx_hi}")
            elif Nright < Nafter:  # increase idx_hi to capture one more peak

                # do we have extra data?
                if idx_hi < len(phase22):
                    # target phase on right 1.5 radial periods after last
                    # identified peak
                    phase_hi = phase22[idx_extrema[-1]] + K*4*np.pi*1.5
                    idx_hi = np.argmax(phase22 > phase_hi)
                    if idx_hi == 0:
                        # coulnd't get as much data as we wished, take all
                        # we have
                        idx_hi = len(phase22)
                    if verbose and idx_hi != old_idx_hi:
                        print(f"idx_hi increased to {idx_hi}")
                else:
                    # we had already fully extended idx_hi in earlier iteration
                    if Nright < Nafter-1:
                        # data-set at least *two* extrema too short
                        if (idx_lo, idx_hi) != (old_idx_lo, old_idx_hi) \
                           or it <= interval_changed_on_it + 1:
                            # we just changed the interval, or didn't yet have
                            # the extra iteration that does perform a new
                            # curve_fit therefore, continue with another
                            # curve_fit, in the hopes that it might identify
                            # one more extremum
                            pass
                        else:
                            # sorry - don't know what else to do
                            raise Exception(f"data set *two* or more extrema"
                                            f"too short after idx_ref={idx_ref}.")
                    else:
                        # data set only one extrema to the right too short,
                        # perform fit, and use the fact that the returned
                        # number of extrema is too few to signal end-of-data
                        pass
                    if verbose:
                        print("idx_hi at its maximum, but still insufficient"
                              f"Nright={Nright}")

            if (idx_lo, idx_hi) != (old_idx_lo, old_idx_hi):
                interval_changed_on_it = it
                # remember when we last changed the search interval
                (old_idx_lo, old_idx_hi) = (idx_lo, idx_hi)
                # data-interval was updated; go back to start of loop to
                # re-identify extrema
                continue

        # if the code gets here, it has identified [idx_lo, idx_high] with the
        # right number of envelope-subtracted extrema.

        # Now check whether omega-envelope fitting has already converged.
        # If yes: return
        # If no:  re-fit envelope

        t_extrema=t[idx_extrema]
        omega22_extrema = omega22[idx_extrema]
        if len(omega22_extrema) != len(old_extrema):  # number of extrema can change near merger
            max_delta_omega = 1e99
        else:
            max_delta_omega = max(np.abs(omega22_extrema-old_extrema))

        if Count_Nright_one_short>=5:  
            # safety exit to catch periodic loops
            # note that Count_Nright_one_short is only increased if Nright=Nafter-1, 
            # therefore, this will coincide with Nright=Nafter-1, also signaling
            # that the overall extrema searching is ending.
            if verbose:
                print(f"exiting because Count_right_one_short={Count_Nright_one_short} is large")
            if pp:
                plt.legend()
                fig.savefig(pp,format='pdf')
                plt.close(fig)
            return idx_extrema, p, K, idx_ref


        if max_delta_omega < TOL:
            # (this cannot trigger on first iteration, due to initialization of old_extrema)
            if verbose:
                print("extrema & omega(extrema) unchanged.  Done")
            if pp:
                plt.legend()
                fig.savefig(pp,format='pdf')
                plt.close(fig)
            return idx_extrema, p, K, idx_ref

        # termination conditions not met, update fit and continue iterating

        if verbose and False:
            # begin plotting residuals
            for deg in [1,2,3]:
                lin=np.polynomial.polynomial.Polynomial.fit(t_extrema, omega22_extrema, deg)
                print(f"    fit with deg={deg}: residual={np.linalg.norm(lin(t_extrema)-omega22_extrema)}")
            #axs[2].plot(t_extrema, quad(t_extrema)-omega22_extrema], "o--")

        #tmp=bounds[0][2]
        p, pconv = scipy.optimize.curve_fit(f_fit, t_extrema,
                                            omega22_extrema, p0=p,
                                            bounds=bounds,maxfev=10000)
        if verbose and False:
                print(f"    PRODUCTION FIT: residual={np.linalg.norm(f_fit(t_extrema,*p)-omega22_extrema)}, f={f_fit.format(*p)}")
        #bounds[0][2]=t_extrema[-1]*0.8
        #p_unbounded, pconv = scipy.optimize.curve_fit(f_fit, t_extrema,
        #                                    omega22_extrema, p0=p,
        #                                    bounds=bounds,maxfev=10000)

        #bounds[0][2]=tmp
        #if verbose:
        #        print(f"    unbounded     : residual={np.linalg.norm(f_fit(t_extrema,*p_unbounded)-omega22_extrema)}, f={f_fit.format(*p_unbounded)}")


        if pp:
            axs[2].plot(t_extrema, f_fit(t_extrema, *p)-omega22_extrema, "o")
            #axs[2].plot(t_extrema, f_fit(t_extrema, *p_unbounded)-omega22_extrema, "x")

        old_extrema = omega22_extrema
        if verbose:
            print(f"max_delta_omega={max_delta_omega:5.4g} => fit updated to"
                  f" f_fit={f_fit.format(*p)}")
    raise Exception("Should never get here")
