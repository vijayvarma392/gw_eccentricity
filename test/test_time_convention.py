import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np

INTERPOLATION_METHODS = ["spline", "rational_fit"]


def test_time_convention():
    """ Tests that the measure_eccentricity interface is working for any
    time conventions.

    We generate a dataDict using EccentricTD waveform and then create
    two different copy of the dataDict with
    - the fist dataDict time having t=0 at the begining and
    - the second dataDict time having t=0 at the end

    The test expects that the measured eccentricity array and mean anomaly
    array from these two dataDicts should be the same.

    Tested for both tref_in and fref_in, and for both spline and rational_fit
    interpolation methods.
    """
    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 3.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.1,
                  "mean_ano": 0,
                  "include_zero_ecc": True}
    dataDict = load_data.load_waveform(**lal_kwargs)

    # create dataDict with t=0 as the begining
    dataDict1 = dataDict.copy()
    dataDict1["t"] = dataDict1["t"] - dataDict1["t"][0]
    dataDict1["t_zeroecc"] = (dataDict1["t_zeroecc"]
                              - dataDict1["t_zeroecc"][0])

    # create dataDict with t=0 as the end
    dataDict2 = dataDict.copy()
    dataDict2["t"] = dataDict2["t"] - dataDict2["t"][-1]
    dataDict2["t_zeroecc"] = (dataDict2["t_zeroecc"]
                              - dataDict2["t_zeroecc"][-1])

    # time shift between dataDict1 and dataDict2
    time_shift = dataDict1["t"][0] - dataDict2["t"][0]

    available_methods = gw_eccentricity.get_available_methods()

    for interp_method in INTERPOLATION_METHODS:
        extra_kwargs = {"omega_gw_extrema_interpolation_method": interp_method}

        # --- tref_in: tref_out shifts by time_shift, ecc and mean_ano match ---
        for method in available_methods:
            eccs = []
            meananos = []
            tref_outs = []
            for data in [dataDict1, dataDict2]:
                gwecc_dict = measure_eccentricity(
                    tref_in=data["t"],
                    method=method,
                    dataDict=data,
                    extra_kwargs=extra_kwargs)
                tref_outs.append(gwecc_dict["tref_out"])
                eccs.append(gwecc_dict["eccentricity"])
                meananos.append(gwecc_dict["mean_anomaly"])
            np.testing.assert_allclose(tref_outs[0], tref_outs[1] + time_shift)
            np.testing.assert_allclose(eccs[0], eccs[1])
            np.testing.assert_allclose(np.unwrap(meananos[0]),
                                       np.unwrap(meananos[1]))

        # --- fref_in: frequency is time-convention-independent, ecc and
        #     mean_ano should match exactly between dataDict1 and dataDict2 ---
        # pick the orbit-averaged omega_gw at the midpoint of dataDict1 as fref
        mid_idx = len(dataDict1["t"]) // 2
        gwecc_mid = measure_eccentricity(
            tref_in=dataDict1["t"][mid_idx],
            method="Amplitude",
            dataDict=dataDict1,
            extra_kwargs=extra_kwargs)
        fref = gwecc_mid["gwecc_object"].compute_orbit_averaged_omega_gw_between_extrema(
            dataDict1["t"][mid_idx]) / (2 * np.pi)

        for method in available_methods:
            eccs = []
            meananos = []
            for data in [dataDict1, dataDict2]:
                gwecc_dict = measure_eccentricity(
                    fref_in=fref,
                    method=method,
                    dataDict=data,
                    extra_kwargs=extra_kwargs)
                eccs.append(gwecc_dict["eccentricity"])
                meananos.append(gwecc_dict["mean_anomaly"])
            np.testing.assert_allclose(eccs[0], eccs[1])
            # fref_in returns scalar outputs so unwrap is not needed
            np.testing.assert_allclose(meananos[0], meananos[1])
