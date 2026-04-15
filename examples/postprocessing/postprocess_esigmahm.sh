#!/usr/bin/env bash

source /home1/akash.maurya/miniconda3/etc/profile.d/conda.sh

set -euo pipefail

# Temporarily disable unbound variable check for conda activation
set +u

# Activate your env
conda activate esigma-igwn39

# Re-enable
set -u

NPROCS=${NPROCS:-1}

CMD=(
	gw-eccentricity-postprocess
	--posterior-path "/home1/akash.maurya/projects/gitlab@icts/esigma-pe-master/alice/prod_injections/HM-debug/ESIGMAHM-ESIGMAHM/outdir_ESIGMAHM_ESIGMAHM_inj/result/ESIGMAHM-ESIGMAHM-inj_data0_1126259462-4_analysis_H1L1V1_merge_result.hdf5"
	--parameter-columns "mass_1, mass_2, spin_1z, spin_2z, minimum_frequency, reference_frequency, eccentricity, mean_anomaly, luminosity_distance, waveform_approximant"  
        --output-dir "/home1/akash.maurya/projects/gitlab@ligo/gw-eccentricity-review/results/postprocess/esigmahm"
	--output-format csv
	--save-every none
	--samples all
	--fref 15
	--method Amplitude
	--data-dict-generator "/home1/akash.maurya/projects/gitlab@ligo/gw-eccentricity-review/scripts/postprocessing/esigmahm.py:esigma_data_dict_generator"
	--data-dict-generator-extra-kwargs '{"f_lower": 10.0}'
	--gw-eccentricity-kwargs '{"extra_kwargs":{"omega_gw_extrema_interpolation_method":"spline"}}'
)

if [ "$NPROCS" -gt 1 ]; then
	mpirun -n "$NPROCS" "${CMD[@]}"
else
	"${CMD[@]}"
fi
