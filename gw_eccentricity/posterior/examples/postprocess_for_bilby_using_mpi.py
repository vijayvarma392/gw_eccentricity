"""Run eccentricity postprocessing in parallel using MPI."""
from mpi4py import MPI
import pandas as pd
import numpy as np
import bilby
from gw_eccentricity.posterior.postprocess import postprocess_sample
from teob_backward_evolution import teob_data_dict_generator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MASTER = 0
STOP = None


def master(posterior, config):
    save_every = config.get("save_every")
    if save_every is not None and save_every <= 0:
        raise ValueError("save_every must be a positive integer or None")

    samples = config["samples"]
    if samples is None:
        samples = list(posterior.index)
    elif isinstance(samples, (int, np.integer)):
        samples = [samples]
    selected_posterior = posterior.loc[samples]
    row_indices = list(selected_posterior.index)
    param_list = selected_posterior.to_dict(orient="records")

    # send initial jobs
    for i, worker in enumerate(range(1, size)):
        if i < len(param_list):
            comm.send((i, param_list[i]), dest=worker)
        else:
            comm.send(STOP, dest=worker)

    next_idx = size - 1
    results = [None] * len(param_list)
    chunk_results = []
    chunk_id = 0

    completed = 0
    while completed < len(param_list):
        worker_rank, idx, res = comm.recv(source=MPI.ANY_SOURCE)

        res.update({"sample_index": row_indices[idx]})
        results[idx] = res
        chunk_results.append(res)
        completed += 1

        if save_every is not None and len(chunk_results) >= save_every:
            chunk_id += 1
            save_chunk_results(chunk_results, selected_posterior, config, chunk_id)
            chunk_results = []

        if next_idx < len(param_list):
            comm.send((next_idx, param_list[next_idx]), dest=worker_rank)
            next_idx += 1
        else:
            comm.send(STOP, dest=worker_rank)

        if completed % 100 == 0:
            print(f"Completed {completed}/{len(param_list)}")

    if save_every is not None and chunk_results:
        chunk_id += 1
        save_chunk_results(chunk_results, selected_posterior, config, chunk_id)

    results_df = pd.DataFrame(results)
    combined_df = selected_posterior.copy()
    combined_df["sample_index"] = row_indices
    combined_df = combined_df.merge(results_df, on="sample_index", how="left")

    save_results(combined_df, config)


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


def save_chunk_results(chunk_results, selected_posterior, config, chunk_id):
    chunk_df = pd.DataFrame(chunk_results)
    chunk_merged = selected_posterior.merge(chunk_df, left_index=True, right_on="sample_index", how="inner")

    output_path = (
        f"{config['output_dir']}/eccentricity_results_chunk_{chunk_id:04d}.{config['output_format']}"
    )
    if config["output_format"] == "json":
        chunk_merged.to_json(output_path)
    elif config["output_format"] == "csv":
        chunk_merged.to_csv(output_path, index=False)
    elif config["output_format"] == "parquet":
        chunk_merged.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {config['output_format']}")


def save_results(results_df, config):
    output_path = f"{config['output_dir']}/eccentricity_results.{config['output_format']}"
    if config["output_format"] == "json":
        results_df.to_json(output_path)
    elif config["output_format"] == "csv":
        results_df.to_csv(output_path, index=False)
    elif config["output_format"] == "parquet":
        results_df.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported output format: {config['output_format']}")


if __name__ == "__main__":
    config = {
        "posterior_path": "/Users/arif/Desktop/TEOB_chi0_9_ecc0_3_samples.hdf5",
        "output_dir": ".",
        "output_format": "csv",
        "save_every": 100, # set to None to save only at the end, or a positive integer to save intermediate results every N samples
        "samples": range(1000), # set to None to process all samples
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
