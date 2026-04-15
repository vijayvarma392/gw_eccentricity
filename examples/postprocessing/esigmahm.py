"""Example code to generate waveform modes using ESIGMAHM 
for postprocessing PE samples using gw_eccentricity."""

# Importing esigmapy (https://github.com/gwnrtools/esigmapy) for generating ESIGMAHM waveforms.
import esigmapy


def esigma_data_dict_generator(params, extra_kwargs=None):
    params_dict = {
        "mass1": params["mass_1"],
        "mass2": params["mass_2"],
        "spin1z": params["spin_1z"],
        "spin2z": params["spin_2z"],
        "delta_t": 1.0 / 2048.0,
        "f_lower": params["minimum_frequency"],
        "f_ref": params["reference_frequency"],
        "distance": params["luminosity_distance"],
        "modes_to_use": [(2, 2)],  # gw_eccentricity only requires the (2,2)-mode
        "include_conjugate_modes": False,  # Do not return -|m| modes
        "eccentricity": params["eccentricity"],
        # Akash: I had used "mean_anomaly" as the key for mean anomaly in
        # PE runs, instead of "mean_per_ano" which is used in the other models.
        # Change it if your PE run uses a different key.
        "mean_anomaly": params["mean_anomaly"],
    }

    if extra_kwargs is not None:
        params_dict.update(extra_kwargs)

    hlm = esigmapy.get_imr_esigma_modes(
        **params_dict
    )  # We call the generator with the parameters
    times = hlm[(2, 2)].sample_times.data
    hlm = {
        key: value.data for key, value in hlm.items()
    }  # Convert modes from PyCBC to NumPy arrays

    return {"t": times, "hlm": hlm}
