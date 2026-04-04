#!/usr/bin/env bash
set -euo pipefail

NPROCS=${NPROCS:-1}

CMD=(
	gw-eccentricity-postprocess
	--posterior-path "/Users/arif/Desktop/TEOB_chi0_9_ecc0_1_samples.hdf5"
	--output-dir "/Users/arif/Desktop/"
	--output-format csv
	--save-every none
	--samples 0:100
	--fref 10
	--method AmplitudeFits
	--data-dict-generator "/Users/arif/gw_eccentricity/gw_eccentricity/postprocess/examples/teob_backward_evolution.py:teob_data_dict_generator"
	--data-dict-generator-extra-kwargs '{"backwards":"yes","ode_tmax":1}'
	--gw-eccentricity-kwargs '{}'
)

if [ "$NPROCS" -gt 1 ]; then
	mpirun -n "$NPROCS" "${CMD[@]}"
else
	"${CMD[@]}"
fi