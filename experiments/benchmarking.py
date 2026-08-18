"""Conventional solver benchmarking.

Explore how solution time evolves with problem size and structure.
n_zones: number of zones
n_years: number of years
battery_reset: daily, weekly, annual; drives duration of coupling of storage.
modular_build: False (continuous), True (discrete); changes problem from LP to MILP.
presolve: True (use HiGHS presolve), False (disable)

Run locally:
    python experiments/benchmarking.py

Run on a Coiled cluster:
    python experiments/benchmarking.py --coiled
"""

import argparse
import gc
import tempfile

import pandas as pd

RESULTS_PATH = "results.csv" # or "gs://seer-datalake/benchmarking/results.csv"


def run_experiment(n_zones, n_years, battery_reset, modular_build, presolve=True):
    """Generate synthetic data, build+solve a pyomo model, and return one results row.

    Imports are local to this function so it can be sent to a Coiled worker as-is, without the
    caller needing the seer package pre-imported in its own process.
    """
    from seer.inputs.synthetic import generate_all_data
    from seer.models.pyomo_model import build, solve

    years = list(range(2025, 2025 + n_years))
    with tempfile.TemporaryDirectory() as tmp:
        # STEP 1: data generation - write a fresh synthetic seer-schema dataset to disk
        data_path = generate_all_data(n_zones=n_zones, years=years, output_path=tmp)

        # STEP 2: build + solve the pyomo model from that dataset
        print(
            f"Solving: {n_zones} zones, {n_years} years, {battery_reset} battery reset, "
            f"modular_build={modular_build}, presolve={presolve}",
            flush=True,
        )
        instance = build(data_path, discrete_additions=modular_build, battery_reset=battery_reset)
        _, solve_time = solve(instance, verbose=False, solver_options={"presolve": "choose" if presolve else "off"})

    # in coiled, important to delete model once done to free memory
    del instance
    gc.collect()

    print(f"-> solve_time={solve_time:.2f}s", flush=True)
    return {
        "n_zones": n_zones, "n_years": n_years, "battery_reset": battery_reset,
        "modular_build": modular_build, "presolve": presolve, "solve_time": solve_time,
    }


def run_locally(experiments):
    return [run_experiment(**exp) for exp in experiments]


def run_on_coiled(experiments, name="seer-benchmarking", n_workers=2):
    import coiled
    from dask.distributed import Client

    cluster = coiled.Cluster(
        name=name, n_workers=n_workers, worker_cpu=2, worker_memory="8 GiB",
        worker_options={"nthreads": 1}, idle_timeout="20 minutes",
    )
    client = Client(cluster)
    print(f"Coiled dashboard: {cluster.dashboard_link}", flush=True)

    futures = [client.submit(run_experiment, **exp) for exp in experiments]
    results = client.gather(futures)

    # close client connection; cluster will keep running until idle_timeout
    client.close()
    return results


def main(use_coiled, results_path, n_workers=2, presolve=True, repeats=3):
    # ---- default problem size, held fixed while sweeping one dimension at a time ----
    default_n_zones = 2
    default_n_years = 2
    default_battery_reset = "daily"

    # ---- grids: each swept independently, holding the other two dimensions at the defaults ----
    n_zones_grid = [1, 2, 3, 4, 5]
    n_years_grid = [1, 2, 3, 4, 5]
    battery_reset_grid = ["daily", "weekly", "annual"]
    modular_build_grid = [False, True]  # discrete_additions: continuous vs integer unit_size multiples

    # one (n_zones, n_years, battery_reset) triple per sweep point, one sweep per dimension
    sweeps = [
        (n_zones, default_n_years, default_battery_reset) for n_zones in n_zones_grid
    ] + [
        (default_n_zones, n_years, default_battery_reset) for n_years in n_years_grid
    ] + [
        (default_n_zones, default_n_years, battery_reset) for battery_reset in battery_reset_grid
    ]

    # cross with modular_build, then repeat each `repeats` times (fresh synthetic data every
    # time) to capture both average solve time and its run-to-run variance
    experiments = [
        dict(n_zones=n_zones, n_years=n_years, battery_reset=battery_reset, modular_build=modular_build,
             presolve=presolve)
        for n_zones, n_years, battery_reset in sweeps
        for modular_build in modular_build_grid
        for _ in range(repeats)
    ]

    print(
        f"Running {len(experiments)} experiments "
        f"({len(sweeps) * len(modular_build_grid)} configs x {repeats} repeats each, "
        f"{'coiled' if use_coiled else 'local'})...",
        flush=True,
    )
    results = run_on_coiled(experiments, n_workers=n_workers) if use_coiled else run_locally(experiments)

    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False)
    print(f"Wrote {len(df)} rows to {results_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coiled", action="store_true", help="run experiments on a Coiled cluster")
    parser.add_argument("--n-workers", type=int, default=2, help="Coiled cluster size (ignored without --coiled)")
    parser.add_argument("--no-presolve", action="store_true", help="disable HiGHS presolve for every experiment")
    parser.add_argument("--repeats", type=int, default=3, help="times to repeat each config, on fresh synthetic data")
    parser.add_argument(
        "--results-path", default=RESULTS_PATH, help="local path or gs:// URL to write results.csv to"
    )
    args = parser.parse_args()

    main(
        use_coiled=args.coiled, results_path=args.results_path, n_workers=args.n_workers,
        presolve=not args.no_presolve, repeats=args.repeats,
    )
