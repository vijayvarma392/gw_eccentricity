# EOB eccentriciy vs measured eccentriciy test

## Run following command
 - `python test_eob_vs_measured_ecc.py --data_dir path/to/base/directory --method method --set set --fig_dir path/to/save/figure`
 - `method` could be `all` or one or more methods. Example `--method Amplitude ResidualAmplitude`
 - `set` could be `all` or one or more of `1, 2, 3, 4`. Example `--set 1 2`
 - The four sets corresponds to Non precessing EOB waveforms with certain mass ration and spins.
  -  q=1, chi1z=chi2z=0
  -  q=2, chi1z=chi2z=0.5
  -  q=4, chi1z=chi2z=-0.6
  -  q=6, chi1z=0.4, chi2z=-0.4.

