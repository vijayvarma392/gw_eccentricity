import measureEccentricity
from measureEccentricity import load_data
from measureEccentricity import measure_eccentricity
import numpy as np
import lal


def test_unit():
    """ Tests that the measure_eccentricity interface is working for both
    MKS and dimensionless units.
    """

    # Load test waveform
    lal_kwargs = {"approximant": "EccentricTD",
                  "q": 1.0,
                  "chi1": [0.0, 0.0, 0.0],
                  "chi2": [0.0, 0.0, 0.0],
                  "Momega0": 0.01,
                  "ecc": 0.1,
                  "mean_ano": 0,
                  "include_zero_ecc": True}
    # load_waveform returns dimensionless dataDict
    dataDict = load_data.load_waveform(**lal_kwargs)
    # get dataDict in MKS units. Use M=10 and D=1Mpc
    M = 10
    D = 1
    tDimlessToMKS = M * lal.MTSUN_SI
    ampDimlessToMKS = lal.C_SI * M * lal.MTSUN_SI / (D * 1e6 * lal.PC_SI)
    dataDictMKS = {
        "t": dataDict["t"] * tDimlessToMKS,
        "hlm": {(2, 2): dataDict["hlm"][(2, 2)] * ampDimlessToMKS},
        "t_zeroecc": dataDict["t_zeroecc"] * tDimlessToMKS,
        "hlm_zeroecc": {
            (2, 2): dataDict["hlm_zeroecc"][(2, 2)] * ampDimlessToMKS}
    }

    # List of all available methods
    available_methods = measureEccentricity.get_available_methods()
    for method in available_methods:
        # Try evaluating at a single dimless time
        tref_out, ecc_ref, meanano_ref = measure_eccentricity(
            tref_in=-12000,
            method=method,
            dataDict=dataDict)
        # Try evaluating at a single MKS time
        tref_out_MKS, ecc_ref_MKS, meanano_ref_MKS = measure_eccentricity(
            tref_in=-12000 * tDimlessToMKS,
            method=method,
            dataDict=dataDictMKS,
            M=10,
            D=1,
            units="mks")
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.isclose([ecc_ref], [ecc_ref_MKS]):
            raise Exception("Eccentricity at a single dimensionless and MKS"
                            "time gives different results. Dimensionless gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose([meanano_ref], [meanano_ref_MKS]):
            raise Exception("Mean anomaly at a single dimensionless and MKS"
                            "time gives different results. Dimensionless gives"
                            f" {meanano_ref} and MKS gives {meanano_ref_MKS}."
                            " Absolute difference is "
                            f"{abs(meanano_ref - meanano_ref_MKS)}")

        # Try evaluating at an array of dimless times
        tref_out, ecc_ref, meanano_ref, eccMethod = measure_eccentricity(
            tref_in=dataDict["t"],
            method=method,
            dataDict=dataDict,
            return_ecc_method=True)
        # Try evaluating at an array of MKS times
        tref_out_MKS, ecc_ref_MKS, meanano_ref_MKS, eccMethod = measure_eccentricity(
            tref_in=dataDict["t"] * tDimlessToMKS,
            method=method,
            dataDict=dataDictMKS,
            M=10,
            D=1,
            units="mks",
            return_ecc_method=True)
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.allclose(ecc_ref, ecc_ref_MKS):
            raise Exception("Eccentricity at an dimensionless and MKS array of"
                            "times are different. Dimensionless gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {np.abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose(meanano_ref, meanano_ref_MKS):
            raise Exception("Mean anomaly at an dimensionless and MKS array of"
                            "times are different. Dimensionless gives"
                            f" {meanano_ref} and MKS gives {meanano_ref_MKS}."
                            " Absolute difference is "
                            f"{np.abs(meanano_ref - meanano_ref_MKS)}")

        # Try evaluating at single dimensionless frequency
        tref_out, ecc_ref, meanano_ref = measure_eccentricity(
            fref_in=0.025 / (2 * np.pi),
            method=method,
            dataDict=dataDict)

        # Try evaluating at an array of frequencies
        tref_out, ecc_ref, meanano_ref, eccMethod = measure_eccentricity(
            fref_in=np.arange(0.025, 0.035, 0.001) / (2 * np.pi),
            method=method,
            dataDict=dataDict,
            return_ecc_method=True)
