"""Group NR waveforms in precessing and non-precessing directory.

April 13, 2022
Md Arif Shaikh
"""
import h5py
import numpy as np
import argparse
import os
import warnings
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dest_dir",
    type=str,
    help=("Main destination directory where the NR files would be stored after"
          " categorizing. This is where the sub directories would live."))
parser.add_argument(
    "--nr_dir",
    type=str,
    help=("NR directory where the NR files are located before"
          " categorizing. If this is provided then all files"
          "under this directory would be worked on. If you want"
          " to work on a single file, use --nr_file key."))
parser.add_argument(
    "--nr_file",
    type=str,
    help=("NR file to be categorized."))
parser.add_argument(
    "--catalog",
    type=str,
    required=True,
    help=("Name of the waveform catalog. Choose the existing names"
          " from get_standard_catalog_names if possible to avoid duplicating"
          " directories."))

args = parser.parse_args()


def get_standard_catalog_names():
    """Get the list of standard catalog names."""
    dict_of_catalog_names = {
        "SXS": "Numerical Relativity waveforms from SXS collaboration",
        "RIT": "Numerical Relativity waveforms from RIT",
        "MAYA": "Numerical Relativity waveforms from MAYA",
        "ET": "Numerical Relativity waveforms from Einstein Toolkitt",
        "SEOB": "SEOB waveforms from Toni's model."}
    return dict_of_catalog_names


# warnings if catalog name is non standard
if args.catalog not in get_standard_catalog_names():
    warnings.warn(f"Catalog name {args.catalog} is not one of the standard"
                  f" catalog names {get_standard_catalog_names().keys()}")


def check_precessing(waveform_file):
    """Check if the given waveform is precessing."""
    f = h5py.File(waveform_file, "r")
    s1x = f.attrs["spin1x"]
    s1y = f.attrs["spin1y"]
    s1z = f.attrs["spin1z"]
    s2x = f.attrs["spin2x"]
    s2y = f.attrs["spin2y"]
    s2z = f.attrs["spin2z"]
    chi1 = [s1x, s1y, s1z]
    chi2 = [s2x, s2y, s2z]
    f.close()
    return (np.linalg.norm(chi1[:2]) > 1e-3
            or np.linalg.norm(chi2[:2]) > 1e-3)


def create_dir(dir):
    """Create a dir if it does not exist."""
    if not os.path.exists(dir):
        os.system(f"mkdir -p {dir}")


def move_file(waveform_file, dest_dir, catalog):
    """Move a file to correc sub directory."""
    sub_dir = "" if check_precessing(waveform_file) else "Non-"
    dest_dir = f"{dest_dir}/{sub_dir}Precessing/{catalog}"
    create_dir(dest_dir)
    fileName = os.path.basename(waveform_file)  # won't work on windows
    dest_path = f"{dest_dir}/{fileName}"
    os.system(f"cp {waveform_file} {dest_path}")


# move nr files to precessing/Non-precessing directories
if args.nr_file:
    move_file(args.nr_file, args.dest_dir, args.catalog)
else:
    if not args.nr_dir:
        raise Exception("Must provide nr_dir if nr_file is not given.")
    nr_files = glob.glob(f"{args.nr_dir}/*.h5")
    for fileName in tqdm(nr_files):
        move_file(fileName, args.dest_dir, args.catalog)
