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
    extra_kwargs = {"debug": False}
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

    # List of all available methods
    available_methods = gw_eccentricity.get_available_methods()
    for method in available_methods:
        # FIXEME: Does not work for FrequencyFits, so skipping it for now
        if method == "FrequencyFits":
            continue
        eccs = []
        meananos = []
        for data in [dataDict1, dataDict2]:
            tref_out, ecc_ref, meanano_ref, eccMethod = measure_eccentricity(
                tref_in=data["t"],
                method=method,
                dataDict=data,
                return_ecc_method=True,
                extra_kwargs=extra_kwargs)
            eccs.append(ecc_ref)
            meananos.append(meanano_ref)
        np.testing.assert_allclose(eccs[0], eccs[1])
        np.testing.assert_allclose(np.unwrap(meananos[0]),
                                   np.unwrap(meananos[1]))
