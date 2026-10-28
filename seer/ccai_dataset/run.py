"""Generate one country's perturbations, one or more per scenario of each of the config's runs (see
design.py), and solve the runs marked solve, locally or on Coiled.

Usage:
    python -m seer.ccai_dataset.run seer/ccai_dataset/configs/Botswana.yaml --plan   # what it would make
    python -m seer.ccai_dataset.run seer/ccai_dataset/configs/Botswana.yaml --workers 8
    python -m seer.ccai_dataset.run <config> --coiled --workers 20 --out gs://seer-datalake/ccai_dataset

Writes <out>/<country>/ (out: a local directory or a gs:// URL; layout in dataset_readme.md):
    config.yaml, README.md, manifest.csv, base.zip
    perturbations/<id>/perturbation.json, inputs.zip      every perturbation
    perturbations/<id>/outputs/<model>.json, <model>.zip  solved ones
ids: <run name>_00000... A perturbation whose inputs equal an earlier one's (e.g. a scenario element with
nothing to act on in this country) isn't solved again: its run.json says which it duplicates.

Workers (local processes or Coiled workers) read base.zip and write each perturbation's files directly to
out. Resumable: perturbations whose inputs.zip, and solves whose run.json, already exist are skipped.
"""
import argparse
import gc
import io
import json
import posixpath
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import fsspec
import pandas as pd
import yaml

from ..scenarios import perturb
from ..scenarios.perturb import round_significant
from ..scenarios.scenario_parameters import SCENARIO_DATA
from . import plan, provenance
from .design import build_entries

DEFAULTS = {"seed": 0, "significant_figures": 6, "cost_dataset": "wb_epm", "noise": SCENARIO_DATA["noise"], "runs": [],
            "solve": {"model": "pyomo", "annualize_capex": False, "years": None}}
README = (Path(__file__).parent / "dataset_readme.md").read_text()


def load_config(path):
    config = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    config["solve"] = {**DEFAULTS["solve"], **config.get("solve", {})}
    return config


# ---------------------------------------------------------------- output files (local or gs://)


def open_out(out_url):
    """(filesystem, country root path, perturbations path) for a country's out_url (local or gs://)."""
    fs, root = fsspec.core.url_to_fs(str(out_url))
    return fs, root, posixpath.join(root, "perturbations")


def write(fs, path, data):
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    fs.pipe(path, data.encode() if isinstance(data, str) else data)


def zip_tables(tables, record_name, record):
    """A zip (bytes) of tables ({name: df}) as csvs, plus record as json."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, df in tables.items():
            z.writestr(f"{name}.csv", df.to_csv(index=False))
        z.writestr(record_name, json.dumps(record, indent=2))
    return buffer.getvalue()


def instance_tables(instance):
    """{name: df} as SeerInstance.write writes them (no empty network)."""
    return {name: df for name, df in instance.as_dict().items() if not (name == "network" and df.empty)}


def ids_with(fs, perturbations, filename):
    """{id: path} of perturbations that have filename (e.g. inputs.zip, outputs/pyomo.json), in one listing."""
    return {posixpath.relpath(p, perturbations).split("/")[0]: p for p in fs.glob(f"{perturbations}/*/{filename}")}


def read_records(fs, perturbations, filename):
    """{id: json} for every perturbations/<id>/<filename>, in one batched read."""
    paths = ids_with(fs, perturbations, filename)
    data = fs.cat(list(paths.values())) if paths else {}
    return {pid: json.loads(data[path]) for pid, path in paths.items()}


# ---------------------------------------------------------------- generate and solve (run on workers)


@cache
def load_base(out_url):
    """The country's base instance, from its base.zip: read once per worker process."""
    fs, root, _ = open_out(out_url)
    return provenance.read_zipped(fs.cat(posixpath.join(root, "base.zip")))


def generate_one(config, entry, base_sha256, code, out_url):
    """entry's perturbation (see design.build_entries), written to perturbations/<id>/."""
    fs, _, perturbations = open_out(out_url)
    instance = perturb(load_base(out_url), entry["scenario"], entry["perturbation"], entry["seed"],
                       config["significant_figures"], config["cost_dataset"], entry["noise"])
    instance.validate()
    tables = instance_tables(instance)
    record = provenance.perturbation_record(config, entry, base_sha256, provenance.tables_sha256(tables), code)
    directory = posixpath.join(perturbations, entry["id"])
    write(fs, posixpath.join(directory, "perturbation.json"), json.dumps(record, indent=2))
    write(fs, posixpath.join(directory, "inputs.zip"), zip_tables(tables, "perturbation.json", record))


def solve_one(config, perturbation_id, code, out_url):
    """Solve perturbation_id's inputs.zip with config's model; write outputs/<model>.zip and .json."""
    fs, _, perturbations = open_out(out_url)
    solve, directory = config["solve"], posixpath.join(perturbations, perturbation_id)
    model = solve["model"]
    run = {"id": perturbation_id, "model": model, "annualize_capex": solve["annualize_capex"],
           "years": solve["years"], "code": code}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(fs.cat(posixpath.join(directory, "inputs.zip")))) as z:
            z.extractall(tmp)
        if solve["years"]:
            years = pd.read_csv(f"{tmp}/years.csv")
            years[years["year"].between(*solve["years"])].to_csv(f"{tmp}/years.csv", index=False)
        try:
            tables, objective, solve_time = SOLVERS[model](tmp, solve["annualize_capex"])
            tables = {name: round_floats(df, config["significant_figures"]) for name, df in tables.items()}
            run.update(status="optimal", objective=objective, solve_seconds=round(solve_time, 1))
        except Exception as e:  # infeasible / non-optimal: record it, it's part of the dataset
            tables = {}
            run.update(status="failed", error=f"{type(e).__name__}: {e}")
    gc.collect()  # solved models are large and workers are long-lived
    run["total_seconds"] = round(time.perf_counter() - start, 1)
    if tables:
        write(fs, posixpath.join(directory, "outputs", f"{model}.zip"), zip_tables(tables, "run.json", run))
    write(fs, posixpath.join(directory, "outputs", f"{model}.json"), json.dumps(run, indent=2))


def solve_pyomo(data_path, annualize_capex):
    import pyomo.environ as pyo
    from ..models import pyomo_model

    instance = pyomo_model.build(f"{data_path}/", annualize_capex=annualize_capex)
    _, solve_time = pyomo_model.solve(instance, verbose=False)
    return pyomo_model.build_output_tables(instance, data_path), pyo.value(instance.objective), solve_time


def solve_pypsa(data_path, annualize_capex):
    from ..models import pypsa_model

    network = pypsa_model.build(f"{data_path}/", annualize_capex=annualize_capex)
    _, _, solve_time = pypsa_model.solve(network, verbose=False)
    return pypsa_model.build_output_tables(network, data_path), network.objective, solve_time


SOLVERS = {"pyomo": solve_pyomo, "pypsa": solve_pypsa}


def round_floats(df, significant_figures):
    df = df.copy()
    for column in df.select_dtypes("float").columns:
        df[column] = round_significant(df[column].to_numpy(), significant_figures)
    return df


# ---------------------------------------------------------------- run


@contextmanager
def make_executor(workers, coiled=False, cluster_name="seer-ccai-dataset", region="us-west2"):
    """A concurrent.futures executor: local processes, or a Coiled cluster (as in ahana/benchmarking's
    experiments/benchmarking.py, forwarding Google credentials so workers can use gs://)."""
    if not coiled:
        with ProcessPoolExecutor(workers) as pool:
            yield pool
        return
    import coiled as coiled_
    from coiled.credentials.google import send_application_default_credentials
    from dask.distributed import Client

    cluster = coiled_.Cluster(name=cluster_name, n_workers=workers, worker_cpu=2, worker_memory="8 GiB",
                              worker_options={"nthreads": 1}, idle_timeout="20 minutes", region=region)
    send_application_default_credentials(cluster)
    client = Client(cluster)
    print(f"Coiled dashboard: {cluster.dashboard_link}", flush=True)
    try:
        yield client.get_executor(pure=False)  # pure=False: every task runs, even with identical arguments
    finally:
        client.close()  # the cluster keeps running until its idle_timeout


def run_parallel(executor, fn, arg_list):
    """fn(*args) for each args on executor, printing progress; raises the first task error."""
    futures = [executor.submit(fn, *args) for args in arg_list]
    for done, future in enumerate(as_completed(futures), 1):
        future.result()
        if done == len(futures) or done % max(1, len(futures) // 20) == 0:
            print(f"  {done}/{len(futures)}", flush=True)


def run(config, out_root, executor):
    """Generate config's perturbations and solve the ones marked solve, into out_root/<country>."""
    out_url = f"{str(out_root).rstrip('/')}/{config['country']}"
    fs, root, perturbations = open_out(out_url)
    model = config["solve"]["model"]
    write(fs, posixpath.join(root, "config.yaml"), yaml.safe_dump(config, sort_keys=False))
    write(fs, posixpath.join(root, "README.md"), README.format(**config, model=model, runs_table=plan.runs_table(config)))
    # the base instance (workers read it from here) and code version, from this machine
    write(fs, posixpath.join(root, "base.zip"), provenance.zip_dir(config["base"]))
    base_sha256, code = provenance.hash_dir(config["base"], "*.csv"), provenance.code_record()

    # Step 1 - generate every perturbation not already written
    entries = build_entries(config)
    done = ids_with(fs, perturbations, "inputs.zip")
    todo = [e for e in entries if e["id"] not in done]
    print(f"generating {len(todo)} of {len(entries)} perturbations", flush=True)
    run_parallel(executor, generate_one, [(config, e, base_sha256, code, out_url) for e in todo])

    # Step 2 - solve entries marked solve, once per distinct inputs
    solved = ids_with(fs, perturbations, f"outputs/{model}.json")
    to_solve = mark_duplicates(fs, perturbations, [e["id"] for e in entries if e["solve"] and e["id"] not in solved], model, code)
    print(f"solving {len(to_solve)} with {model}", flush=True)
    run_parallel(executor, solve_one, [(config, pid, code, out_url) for pid in to_solve])

    # Step 3 - manifest of everything written
    manifest = build_manifest(fs, perturbations, model)
    write(fs, posixpath.join(root, "manifest.csv"), manifest.to_csv(index=False))
    print(f"{len(manifest)} perturbations, solves: {manifest['solve_status'].value_counts().to_dict()} -> {out_url}")


def mark_duplicates(fs, perturbations, ids, model, code):
    """ids to solve: the first with each inputs_sha256; the rest get a run.json naming the one they duplicate."""
    sha = {pid: r["inputs_sha256"] for pid, r in read_records(fs, perturbations, "perturbation.json").items()}
    solved = read_records(fs, perturbations, f"outputs/{model}.json")
    first = {}
    for pid in sorted(p for p, r in solved.items() if r["status"] != "duplicate"):
        first.setdefault(sha[pid], pid)
    to_solve = []
    for pid in ids:
        if sha[pid] in first:
            run = {"id": pid, "model": model, "status": "duplicate", "duplicate_of": first[sha[pid]], "code": code}
            write(fs, posixpath.join(perturbations, pid, "outputs", f"{model}.json"), json.dumps(run, indent=2))
        else:
            first[sha[pid]] = pid
            to_solve.append(pid)
    return to_solve


def build_manifest(fs, perturbations, model):
    """One row per perturbation: its record's design, perturb arguments and code, and its solve."""
    records = pd.json_normalize(list(read_records(fs, perturbations, "perturbation.json").values()), max_level=1)
    runs = pd.DataFrame(list(read_records(fs, perturbations, f"outputs/{model}.json").values()),
                        columns=["id", "status", "duplicate_of", "objective", "solve_seconds"])
    manifest = records[["id", "design.run", "design.scenario_name", "perturb.scenario", "perturb.perturbation",
                        "perturb.noise", "perturb.seed", "perturb.cost_dataset", "inputs_sha256", "code.git_commit",
                        "code.git_dirty"]]
    manifest = manifest.rename(columns=lambda c: c.split(".")[-1]).assign(
        scenario=lambda m: m["scenario"].map(json.dumps), perturbation=lambda m: m["perturbation"].map(json.dumps))
    manifest = manifest.merge(runs.rename(columns={"status": "solve_status"}), on="id", how="left")
    return manifest.sort_values("id")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", default="data/ccai_dataset", help="local directory or gs:// URL")
    parser.add_argument("--plan", action="store_true", help="print what the config would make, and perturbation ranges; run nothing")
    parser.add_argument("--workers", type=int, default=1, help="local processes, or Coiled workers with --coiled")
    parser.add_argument("--coiled", action="store_true", help="run tasks on a Coiled cluster")
    parser.add_argument("--cluster-name", default="seer-ccai-dataset", help="Coiled cluster name (with --coiled)")
    parser.add_argument("--region", default="us-west2", help="Coiled cluster region (with --coiled)")
    parser.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes (recorded as git_dirty)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.plan:
        return plan.print_plan(config)
    if provenance.code_record()["git_dirty"] and not args.allow_dirty:
        parser.error("uncommitted changes: perturbations wouldn't be reproducible from the git commit (--allow-dirty to run anyway)")
    with make_executor(args.workers, args.coiled, args.cluster_name, args.region) as executor:
        run(config, args.out, executor)


if __name__ == "__main__":
    main()
