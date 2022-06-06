import measureEccentricity
from measureEccentricity import load_data
from measureEccentricity import measure_eccentricity
import numpy as np


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
    # get dataDict in MKS units.
    lal_kwargs.update({"physicalUnits": True})
    dataDictMKS = load_data.load_waveform(**lal_kwargs)

    # List of all available methods
    available_methods = measureEccentricity.get_available_methods()
    for method in available_methods:
        # Try evaluating at a single dimless time
        tref_out, ecc_ref, meanano_ref = measure_eccentricity(
            tref_in=dataDict["t"][15000],
            method=method,
            dataDict=dataDict)
        # Try evaluating at a single MKS time
        tref_out_MKS, ecc_ref_MKS, meanano_ref_MKS = measure_eccentricity(
            tref_in=dataDictMKS["t"][15000],
            method=method,
            dataDict=dataDictMKS)
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.allclose([ecc_ref], [ecc_ref_MKS], atol=1e-5):
            raise Exception("Eccentricity at a single dimensionless and MKS"
                            "time gives different results. Dimensionless gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose([meanano_ref], [meanano_ref_MKS], atol=1e-5):
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
        tref_out_MKS, ecc_ref_MKS, meanano_ref_MKS, eccMethod_MKS = measure_eccentricity(
            tref_in=dataDictMKS["t"],
            method=method,
            dataDict=dataDictMKS,
            return_ecc_method=True)
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.allclose(ecc_ref, ecc_ref_MKS, atol=1e-5):
            raise Exception("Eccentricity at dimensionless and MKS array of"
                            " times are different. Dimensionless gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {np.abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose(meanano_ref, meanano_ref_MKS, atol=1e-2):
            raise Exception("Mean anomaly at dimensionless and MKS array of"
                            " times are different. Dimensionless gives"
                            f" {meanano_ref} and MKS gives {meanano_ref_MKS}."
                            " Absolute difference is "
                            f"{np.abs(meanano_ref - meanano_ref_MKS)}")

        # Try evaluating at single dimensionless frequency
        fref_in = eccMethod.compute_omega22_average_between_extrema(dataDict["t"][15000]) / (2 * np.pi)
        fref_out, ecc_ref, meanano_ref = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict)
        # Try evaluating at single MKS frequency
        fref_in = eccMethod_MKS.compute_omega22_average_between_extrema(dataDictMKS["t"][15000]) / (2 * np.pi)
        fref_out, ecc_ref_MKS, meanano_ref_MKS = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDictMKS)
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.allclose([ecc_ref], [ecc_ref_MKS], atol=1e-5):
            raise Exception("Eccentricity at a single dimensionless and MKS"
                            "frequency gives different results. Dimensionless"
                            "gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose([meanano_ref], [meanano_ref_MKS], atol=1e-5):
            raise Exception("Mean anomaly at a single dimensionless and MKS"
                            "frequency gives different results. Dimensionless"
                            "gives"
                            f" {meanano_ref} and MKS gives {meanano_ref_MKS}."
                            " Absolute difference is "
                            f"{abs(meanano_ref - meanano_ref_MKS)}")

        # Try evaluating at an array of dimensionless frequencies
        fref_in = eccMethod.compute_omega22_average_between_extrema(dataDict["t"][15000: 20000]) / (2 * np.pi)
        tref_out, ecc_ref, meanano_ref, eccMethod = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDict,
            return_ecc_method=True)
        # Try evaluating at an array of MKS frequencies
        fref_in = eccMethod_MKS.compute_omega22_average_between_extrema(dataDictMKS["t"][15000: 20000]) / (2 * np.pi)
        tref_out, ecc_ref_MKS, meanano_ref_MKS, eccMethod = measure_eccentricity(
            fref_in=fref_in,
            method=method,
            dataDict=dataDictMKS,
            return_ecc_method=True)
        # Check if the measured ecc an mean ano are the same from the two units
        if not np.allclose(ecc_ref, ecc_ref_MKS, atol=1e-5):
            raise Exception("Eccentricity at dimensionless and MKS array of"
                            " frequencies are different. Dimensionless gives"
                            f" {ecc_ref} and MKS gives {ecc_ref_MKS}. Absolute"
                            f" difference is {np.abs(ecc_ref - ecc_ref_MKS)}")
        if not np.allclose(meanano_ref, meanano_ref_MKS, atol=1e-2):
            raise Exception("Mean anomaly at dimensionless and MKS array of"
                            " frequencies are different. Dimensionless gives"
                            f" {meanano_ref} and MKS gives {meanano_ref_MKS}."
                            " Absolute difference is "
                            f"{np.abs(meanano_ref - meanano_ref_MKS)}")
