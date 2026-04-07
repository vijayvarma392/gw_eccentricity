#!/usr/bin/env bash
set -euo pipefail

NPROCS=${NPROCS:-1}

CMD=(
	gw-eccentricity-postprocess
	--posterior-path "/Users/arif/Desktop/TEOB_chi0_9_ecc0_3_samples.hdf5"
	--parameter-columns "mass_1,mass_2,spin_1z,spin_2z,luminosity_distance,minimum_frequency,eccentricity,mean_per_ano"
	--output-dir "/Users/arif/Desktop/"
	--output-format csv
	--save-every none
	--samples 0:1000
	--fref 10
	--method Amplitude
	--data-dict-generator "/Users/arif/gw_eccentricity/examples/postprocessing/teobresums_dali.py:teob_data_dict_generator"
	--data-dict-generator-extra-kwargs '{"backwards":"yes","ode_tmax":1}'
	--gw-eccentricity-kwargs '{"extra_kwargs":{"omega_gw_extrema_interpolation_method":"spline"}}'
)

if [ "$NPROCS" -gt 1 ]; then
	mpirun -n "$NPROCS" "${CMD[@]}"
else
	"${CMD[@]}"
fi