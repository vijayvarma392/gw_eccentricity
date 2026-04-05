"""Run eccentricity postprocessing in parallel using MPI."""
import argparse
import importlib
import importlib.util
import json
import time
from pathlib import Path
from typing import TypedDict, Callable
import pandas as pd
from mpi4py import MPI
import bilby
from gw_eccentricity.postprocess.core import postprocess_sample, PostProcessResults, PostProcessResult

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MASTER = 0
_STOP = "STOP"


class PostProcessConfig(TypedDict):
    """Configuration dictionary for postprocessing."""
    posterior_type: str
    posterior_path: str
    output_dir: str
    output_format: str
    save_every: int | None
    samples: list[int] | None
    fref: float
    method: str
    batch_size: int
    data_dict_generator: Callable
    data_dict_generator_extra_kwargs: dict
    gw_eccentricity_kwargs: dict


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

def run_postprocess(
        sample_index: int, 
        params: dict, config: 
        PostProcessConfig) -> PostProcessResult:
    return postprocess_sample(
        sample_index=sample_index,
        params=params,
        fref=config["fref"],
        data_dict_generator=config["data_dict_generator"],
        data_dict_generator_extra_kwargs=config["data_dict_generator_extra_kwargs"],
        method=config["method"],
        gw_eccentricity_kwargs=config["gw_eccentricity_kwargs"],
    )


def run_postprocess_batch(
        sample_indices: list[int],
        params_list: list[dict],
        config: PostProcessConfig
        ) -> list[PostProcessResult]:
    """Process a batch of samples."""
    results = []
    for sample_index, params in zip(sample_indices, params_list):
        res = run_postprocess(sample_index, params, config)
        results.append(res)
    return results


def extract_batch_data(
        to_process_batch: list) -> tuple[list[int], list[dict]]:
    """Extract sample indices and params from to_process tuples.

    to_process items are (position, sample_index, params) tuples.
    Returns (sample_indices, params_list).
    """
    sample_indices = [item[1] for item in to_process_batch]
    params_list = [item[2] for item in to_process_batch]
    return sample_indices, params_list


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_df(df: pd.DataFrame, path: str, fmt: str) -> None:
    if fmt == "json":
        df.to_json(path)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")


def load_checkpoint(checkpoint_path: str) -> pd.DataFrame | None:
    """Load checkpoint if it exists."""
    if not Path(checkpoint_path).exists():
        return None
    return pd.read_parquet(checkpoint_path)


def save_checkpoint(results: list, config: PostProcessConfig) -> None:
    """Save results to checkpoint file (parquet format)."""
    results_df = PostProcessResults(results).to_dataframe()
    checkpoint_path = f"{config['output_dir']}/checkpoint.parquet"
    results_df.to_parquet(checkpoint_path)
    print(f"  → checkpoint saved: {checkpoint_path}", flush=True)


def save_final(
        results: list, 
        row_indices: list, 
        selected_posterior: pd.DataFrame, 
        config: PostProcessConfig) -> None:
    missing_indices = [row_indices[i] for i, r in enumerate(results) if r is None]
    if missing_indices:
        print(
            f"[WARNING] {len(missing_indices)} results missing "
            f"(worker errors?): {missing_indices[:10]}"
            f"{'...' if len(missing_indices) > 10 else ''}"
        )

    results_df = PostProcessResults([r for r in results if r is not None]).to_dataframe()
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

def master(posterior: pd.DataFrame, config: PostProcessConfig) -> None:
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_path = f"{config['output_dir']}/checkpoint.parquet"
    checkpoint_df = load_checkpoint(checkpoint_path)

    samples = config["samples"]
    if samples is None:
        samples = list(posterior.index)
    selected_posterior = posterior.loc[samples]
    row_indices = list(selected_posterior.index)
    param_list = selected_posterior.to_dict(orient="records")
    n = len(param_list)

    # Determine which samples to process
    processed_indices = set()
    if checkpoint_df is not None:
        processed_indices = set(checkpoint_df["sample_index"].dropna().unique())

    to_process = [
        (i, idx, params) for i, (idx, params) in enumerate(zip(row_indices, param_list))
        if idx not in processed_indices
    ]

    # Initialize results array and position mapping (only if needed)
    results = [None] * n

    # Build sample_index_to_pos only when needed (checkpoint reconstruction or MPI)
    sample_index_to_pos = None
    if checkpoint_df is not None or size > 1:
        sample_index_to_pos = {idx: pos for pos, idx in enumerate(row_indices)}

    if checkpoint_df is not None and sample_index_to_pos is not None:
        for row_dict in checkpoint_df.to_dict('records'):
            sample_idx = row_dict["sample_index"]
            if sample_idx in sample_index_to_pos:
                pos = sample_index_to_pos[sample_idx]
                res = PostProcessResult(
                    sample_index=int(sample_idx),
                    status=row_dict["status"],
                    egw=row_dict.get("egw"),
                    lgw=row_dict.get("lgw"),
                    error_message=row_dict.get("error_message"),
                )
                results[pos] = res

    n_to_process = len(to_process)
    print(
        f"Total samples: {n}, Already processed: {len(processed_indices)}, "
        f"Remaining: {n_to_process}",
        flush=True,
    )

    if n_to_process == 0:
        print("All samples already processed.", flush=True)
        save_final(results, row_indices, selected_posterior, config)
        # Signal workers to stop if running MPI
        if size > 1:
            for worker_rank in range(1, size):
                comm.send(_STOP, dest=worker_rank)
        return

    print(f"Processing {n_to_process} samples across {size} rank(s).", flush=True)

    progress = _Progress(n_to_process)
    checkpoint_interval = config.get("save_every")  # None = don't checkpoint during processing
    samples_since_checkpoint = 0

    # ---- single-process fast path (no mpirun) --------------------------------
    if size == 1:
        for idx, sample_idx, params in to_process:
            res = run_postprocess(sample_idx, params, config)
            results[idx] = res
            samples_since_checkpoint += 1
            progress.tick()

            if checkpoint_interval and samples_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, config)
                samples_since_checkpoint = 0

        if checkpoint_interval and samples_since_checkpoint > 0:
            save_checkpoint(results, config)

        save_final(results, row_indices, selected_posterior, config)
        return

    # ---- MPI master/worker loop ----------------------------------------------
    batch_size = config["batch_size"]  # Get batch size from config

    # Prime workers: send one batch each (or _STOP if fewer batches than workers).
    next_batch_idx = 0
    for worker_rank in range(1, size):
        if next_batch_idx < len(to_process):
            end_idx = min(next_batch_idx + batch_size, len(to_process))
            batch = to_process[next_batch_idx:end_idx]
            batch_indices, batch_params = extract_batch_data(batch)
            comm.send((batch_indices, batch_params), dest=worker_rank)
            next_batch_idx = end_idx
        else:
            comm.send(_STOP, dest=worker_rank)

    completed = 0
    while completed < n_to_process:
        worker_rank, batch_indices, batch_results = comm.recv(source=MPI.ANY_SOURCE)

        # Process all results in the batch
        for sample_index, res in zip(batch_indices, batch_results):
            pos = sample_index_to_pos[sample_index]
            results[pos] = res
            samples_since_checkpoint += 1
            completed += 1
            progress.tick()

            if checkpoint_interval and samples_since_checkpoint >= checkpoint_interval:
                save_checkpoint(results, config)
                samples_since_checkpoint = 0

        # Send next batch to the freed worker
        if next_batch_idx < len(to_process):
            end_idx = min(next_batch_idx + batch_size, len(to_process))
            batch = to_process[next_batch_idx:end_idx]
            batch_indices, batch_params = extract_batch_data(batch)
            comm.send((batch_indices, batch_params), dest=worker_rank)
            next_batch_idx = end_idx
        else:
            comm.send(_STOP, dest=worker_rank)

    if checkpoint_interval and samples_since_checkpoint > 0:
        save_checkpoint(results, config)

    save_final(results, row_indices, selected_posterior, config)


def worker(config: PostProcessConfig) -> None:
    while True:
        msg = comm.recv(source=MASTER)
        if msg == _STOP:
            break
        sample_indices, params_list = msg
        results = run_postprocess_batch(sample_indices, params_list, config)
        comm.send((rank, sample_indices, results), dest=MASTER)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_config_from_cli() -> PostProcessConfig:
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
        help="Save results to file every N samples (or 'none')",
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
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of samples per batch for MPI communication (default: 100)",
    )
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
        "batch_size": args.batch_size,
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

def main() -> None:
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