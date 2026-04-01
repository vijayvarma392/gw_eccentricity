"""Run eccentricity postprocessing in parallel using MPI."""
import argparse
import importlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

import bilby
from gw_eccentricity.posterior.postprocess import postprocess_sample

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MASTER = 0
_STOP = "STOP"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_int_or_none(value):
    if value is None or (isinstance(value, str) and value.lower() == "none"):
        return None
    return int(value)


def parse_samples(value):
    """Parse sample selection from CLI into a list of ints or None (= all).

    Supported formats
    -----------------
    - ``all`` / ``none``       → None  (use all samples)
    - ``100``                  → [100]
    - ``1,5,42``               → [1, 5, 42]
    - ``0:100`` or ``0:100:2`` → list(range(...))
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"all", "none"}:
        return None
    if ":" in text:
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid samples range format: {value!r}")
        start = int(parts[0]) if parts[0] else 0
        stop  = int(parts[1])
        step  = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, stop, step))
    if "," in text:
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(text)]  # always return a list, never a bare int


def parse_json_dict(value, arg_name):
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --{arg_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--{arg_name} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def load_callable(import_path, arg_name):
    """Load a callable from ``module.path:function`` or ``/path/to/file.py:function``."""
    if ":" not in import_path:
        raise ValueError(
            f"--{arg_name} must be 'module.path:function_name' or '/path/to/file.py:function_name'"
        )
    module_path, attr_name = import_path.rsplit(":", 1)
    if module_path.endswith(".py") or "/" in module_path:
        file_path = Path(module_path).expanduser().resolve()
        if not file_path.exists():
            raise ValueError(f"File not found for --{arg_name}: {file_path}")
        spec = importlib.util.spec_from_file_location(f"_dyn_{file_path.stem}", str(file_path))
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module for --{arg_name}: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    target = getattr(module, attr_name)
    if not callable(target):
        raise ValueError(f"Imported object for --{arg_name} is not callable")
    return target


# ---------------------------------------------------------------------------
# Core work unit
# ---------------------------------------------------------------------------

def run_postprocess(params, config):
    return postprocess_sample(
        params=params,
        fref=config["fref"],
        data_dict_generator=config["data_dict_generator"],
        data_dict_generator_extra_kwargs=config["data_dict_generator_extra_kwargs"],
        method=config["method"],
        gw_eccentricity_kwargs=config["gw_eccentricity_kwargs"],
    )


def _safe_run(idx, params, config):
    """Run postprocessing, catching any exception so workers never hang master."""
    try:
        return run_postprocess(params, config)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[ERROR] rank {rank}, sample index {idx}: {exc}\n{tb}", flush=True)
        return {"error": str(exc), "traceback": tb}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_df(df, path, fmt):
    if fmt == "json":
        df.to_json(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")


def save_chunk(chunk_results, selected_posterior, config, chunk_id):
    chunk_df = pd.DataFrame(chunk_results)
    merged = selected_posterior.merge(
        chunk_df, left_index=True, right_on="sample_index", how="left"
    )
    dropped = merged["sample_index"].isna().sum()
    if dropped:
        print(f"[WARNING] chunk {chunk_id}: {dropped} rows missing after merge")
    path = (
        f"{config['output_dir']}/eccentricity_results_chunk_{chunk_id:04d}"
        f".{config['output_format']}"
    )
    write_df(merged, path, config["output_format"])
    print(f"  → chunk {chunk_id} saved: {path}", flush=True)


def save_final(results, row_indices, selected_posterior, config):
    missing_indices = [row_indices[i] for i, r in enumerate(results) if r is None]
    if missing_indices:
        print(
            f"[WARNING] {len(missing_indices)} results missing "
            f"(worker errors?): {missing_indices[:10]}"
            f"{'...' if len(missing_indices) > 10 else ''}"
        )

    results_df = pd.DataFrame([r for r in results if r is not None])
    combined = selected_posterior.copy()
    combined["sample_index"] = row_indices
    combined = combined.merge(results_df, on="sample_index", how="left")

    path = f"{config['output_dir']}/eccentricity_results.{config['output_format']}"
    write_df(combined, path, config["output_format"])
    print(f"\nFinal results saved: {path}", flush=True)


def get_bilby_posterior(posterior_path):
    res = bilby.result.read_in_result(posterior_path)
    return res.posterior


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total, report_every=100):
        self.total = total
        self.report_every = report_every
        self.completed = 0
        self.t0 = time.monotonic()

    def tick(self):
        self.completed += 1
        if self.completed % self.report_every == 0 or self.completed == self.total:
            elapsed = time.monotonic() - self.t0
            rate = self.completed / elapsed if elapsed > 0 else float("inf")
            eta = (self.total - self.completed) / rate if rate > 0 else float("inf")
            print(
                f"  Completed {self.completed}/{self.total}"
                f" | {elapsed:.0f}s elapsed"
                f" | ETA {eta:.0f}s"
                f" | {rate:.1f} samples/s",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Master / worker
# ---------------------------------------------------------------------------

def master(posterior, config):
    save_every = config.get("save_every")
    if save_every is not None and save_every <= 0:
        raise ValueError("save_every must be a positive integer or None")

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    samples = config["samples"]
    if samples is None:
        samples = list(posterior.index)
    selected_posterior = posterior.loc[samples]
    row_indices = list(selected_posterior.index)
    param_list = selected_posterior.to_dict(orient="records")
    n = len(param_list)

    print(f"Processing {n} samples across {size} rank(s).", flush=True)

    progress = _Progress(n)
    results = [None] * n
    chunk_results = []
    chunk_id = 0

    # ---- single-process fast path (no mpirun) --------------------------------
    if size == 1:
        for idx, params in enumerate(param_list):
            res = _safe_run(idx, params, config)
            res["sample_index"] = row_indices[idx]
            results[idx] = res
            chunk_results.append(res)
            progress.tick()

            if save_every and len(chunk_results) >= save_every:
                chunk_id += 1
                save_chunk(chunk_results, selected_posterior, config, chunk_id)
                chunk_results = []

        if save_every and chunk_results:
            chunk_id += 1
            save_chunk(chunk_results, selected_posterior, config, chunk_id)

        save_final(results, row_indices, selected_posterior, config)
        return

    # ---- MPI master/worker loop ----------------------------------------------
    # Prime workers: send one job each (or _STOP if fewer jobs than workers).
    next_idx = 0
    for worker_rank in range(1, size):
        if next_idx < n:
            comm.send((next_idx, param_list[next_idx]), dest=worker_rank)
            next_idx += 1
        else:
            comm.send(_STOP, dest=worker_rank)

    completed = 0
    while completed < n:
        worker_rank, idx, res = comm.recv(source=MPI.ANY_SOURCE)

        res["sample_index"] = row_indices[idx]
        results[idx] = res
        chunk_results.append(res)
        completed += 1
        progress.tick()

        if save_every and len(chunk_results) >= save_every:
            chunk_id += 1
            save_chunk(chunk_results, selected_posterior, config, chunk_id)
            chunk_results = []

        if next_idx < n:
            comm.send((next_idx, param_list[next_idx]), dest=worker_rank)
            next_idx += 1
        else:
            comm.send(_STOP, dest=worker_rank)

    if save_every and chunk_results:
        chunk_id += 1
        save_chunk(chunk_results, selected_posterior, config, chunk_id)

    save_final(results, row_indices, selected_posterior, config)


def worker(config):
    while True:
        msg = comm.recv(source=MASTER)
        if msg == _STOP:
            break
        idx, params = msg
        res = _safe_run(idx, params, config)
        comm.send((rank, idx, res), dest=MASTER)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_config_from_cli():
    parser = argparse.ArgumentParser(
        description="Run eccentricity postprocessing in parallel using MPI"
    )
    parser.add_argument(
        "--posterior-type", default="bilby", choices=["bilby"]
    )
    parser.add_argument(
        "--posterior-path", required=True, help="Path to bilby result file"
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory for output files"
    )
    parser.add_argument(
        "--output-format", default="csv", choices=["json", "csv", "parquet"]
    )
    parser.add_argument(
        "--save-every",
        type=parse_int_or_none,
        default=None,
        metavar="N",
        help="Save intermediate chunk files every N samples (or 'none')",
    )
    parser.add_argument(
        "--samples",
        default="all",
        metavar="SPEC",
        help="Sample selection: all|none, int, comma list (1,5,9), or range (0:100[:step])",
    )
    parser.add_argument("--fref", type=float, default=10.0)
    parser.add_argument("--method", default="AmplitudeFits")
    parser.add_argument(
        "--data-dict-generator",
        required=True,
        metavar="MODULE:FUNC",
        help="Callable as 'module.path:function_name' or '/path/to/file.py:function_name'",
    )
    parser.add_argument(
        "--data-dict-generator-extra-kwargs",
        default="{}",
        metavar="JSON",
        help="JSON dict of extra kwargs for data_dict_generator",
    )
    parser.add_argument(
        "--gw-eccentricity-kwargs",
        default="{}",
        metavar="JSON",
        help="JSON dict passed to gw_eccentricity",
    )

    args = parser.parse_args()

    # Each rank loads its own callable — callables cannot survive comm.bcast.
    return {
        "posterior_type": args.posterior_type,
        "posterior_path": args.posterior_path,
        "output_dir": args.output_dir,
        "output_format": args.output_format,
        "save_every": args.save_every,
        "samples": parse_samples(args.samples),
        "fref": args.fref,
        "method": args.method,
        "data_dict_generator": load_callable(
            args.data_dict_generator, "data-dict-generator"
        ),
        "data_dict_generator_extra_kwargs": parse_json_dict(
            args.data_dict_generator_extra_kwargs,
            "data-dict-generator-extra-kwargs",
        ),
        "gw_eccentricity_kwargs": parse_json_dict(
            args.gw_eccentricity_kwargs, "gw-eccentricity-kwargs"
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Each rank parses CLI independently — callables cannot be bcast'd.
    config = build_config_from_cli()

    if rank == MASTER:
        if config["posterior_type"] == "bilby":
            posterior = get_bilby_posterior(config["posterior_path"])
        else:
            raise ValueError(f"Unsupported posterior type: {config['posterior_type']!r}")
        master(posterior, config)
    else:
        worker(config)


if __name__ == "__main__":
    main()