import gw_eccentricity
from gw_eccentricity import load_data
from gw_eccentricity import measure_eccentricity
import numpy as np


def test_time_convention():
    """ Tests that the measure_eccentricity interface is working for any
    time conventions.

    We generate a dataDict using EccentricTD waveform and then create
    two different copy of the dataDict with
    - the fist dataDict time having t=0 at the begining and
    - the second dataDict time having t=0 at the end

    The test expects that the measured eccentricity array and mean anomaly
    array from these two dataDicts should be the same.
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

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    for method in available_methods:
        eccs = []
        meananos = []
        tref_outs = []
        for data in [dataDict1, dataDict2]:
            gwecc_dict = measure_eccentricity(
                tref_in=data["t"],
                method=method,
                dataDict=data)
            tref_out = gwecc_dict["tref_out"]
            ecc_ref = gwecc_dict["eccentricity"]
            meanano_ref = gwecc_dict["mean_anomaly"]
            tref_outs.append(tref_out)
            eccs.append(ecc_ref)
            meananos.append(meanano_ref)
        np.testing.assert_allclose(tref_outs[0], tref_outs[1] + time_shift)
        np.testing.assert_allclose(eccs[0], eccs[1])
        np.testing.assert_allclose(np.unwrap(meananos[0]),
                                   np.unwrap(meananos[1]))
