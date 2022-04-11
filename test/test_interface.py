import measureEccentricity
from measureEccentricity import load_data
from measureEccentricity import measure_eccentricity


def test_interface():
    """ Tests that the measure_eccentricity interface is working for all
    implemented methods.
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
    dataDict = load_data.load_waveform(**lal_kwargs)

    # List of all available methods
    available_methods = measureEccentricity.get_available_methods()
    for method in available_methods:
        tref_out, ecc_ref, mean_ano_ref = measure_eccentricity(
            tref_in=-14000,
            method=method,
            dataDict=dataDict)
