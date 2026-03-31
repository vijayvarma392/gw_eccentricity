"""Run eccentricity postprocessing in parallel using MPI."""
from mpi4py import MPI
import pandas as pd
import numpy as np
import bilby

from gw_eccentricity.posterior.postprocess import postprocess_sample
import sys
sys.path.append("/Users/arif/teobresums_reviewed/Python/")
from teob_backward_evolution import teob_data_dict_generator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MASTER = 0
STOP = None


def master(posterior, config):
    samples = config["samples"]
    if samples is None:
        samples = list(posterior.index)
    elif isinstance(samples, (int, np.integer)):
        samples = [samples]
    param_list = posterior.loc[samples].to_dict(orient="records")

    # send initial jobs
    for i, worker in enumerate(range(1, size)):
        if i < len(param_list):
            comm.send((i, param_list[i]), dest=worker)
        else:
            comm.send(STOP, dest=worker)

    next_idx = size - 1
    results = [None] * len(param_list)

    completed = 0
    while completed < len(param_list):
        worker_rank, idx, res = comm.recv(source=MPI.ANY_SOURCE)

        res.update({"params": param_list[idx]})
        results[idx] = res
        completed += 1

        if next_idx < len(param_list):
            comm.send((next_idx, param_list[next_idx]), dest=worker_rank)
            next_idx += 1
        else:
            comm.send(STOP, dest=worker_rank)

        if completed % 100 == 0:
            print(f"Completed {completed}/{len(param_list)}")

    save_results(results, config)


def worker(config):
    while True:
        msg = comm.recv(source=MASTER)

        if msg is STOP:
            break

        idx, params = msg

        res = postprocess_sample(
            params=params,
            fref=config["fref"],
            data_dict_generator=config["data_dict_generator"],
            data_dict_generator_extra_kwargs=config["data_dict_generator_extra_kwargs"],
            method=config["method"],
            gw_eccentricity_kwargs=config["gw_eccentricity_kwargs"]
        )

        comm.send((rank, idx, res), dest=MASTER)


def get_bilby_posterior(config):
    res = bilby.result.read_in_result(config["posterior_path"])
    posterior = res.posterior
    return posterior


def save_results(results, config):
    output_path = f"{config['output_dir']}/eccentricity_results.{config['output_format']}"
    if config["output_format"] == "json":
        pd.DataFrame(results).to_json(output_path)
    elif config["output_format"] == "csv":
        pd.DataFrame(results).to_csv(output_path, index=False)
    elif config["output_format"] == "parquet":
        pd.DataFrame(results).to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {config['output_format']}")


if __name__ == "__main__":
    config = {
        "posterior_path": "/Users/arif/Desktop/TEOB_chi0_9_ecc0_3_samples.hdf5",
        "output_dir": ".",
        "output_format": "csv",
        "samples": range(100), # set to None to process all samples
        "fref": 10,
        "data_dict_generator": teob_data_dict_generator,
        "data_dict_generator_extra_kwargs": {"backwards": "yes", "ode_tmax": 1.0},
        "method": "AmplitudeFits",
        "gw_eccentricity_kwargs": {}
    }

    if rank == MASTER:
        posterior = get_bilby_posterior(config)
        master(posterior, config)
    else:
        worker(config)
