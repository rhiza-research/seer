"""Generate one country's perturbations, one or more per scenario of each of the config's runs (see
design.py), and solve the runs marked solve, locally or on Coiled.

Usage:
    python -m seer.ccai_dataset.run seer/ccai_dataset/configs/Botswana.yaml --plan   # what it would make
    python -m seer.ccai_dataset.run seer/ccai_dataset/configs/Botswana.yaml --workers 8
    python -m seer.ccai_dataset.run <config> --coiled --workers 20 --out gs://seer-datalake/ccai_dataset

The base instance is the config's base path in a local clone of seer-planning-models (--models-dir). Before a
run, seer and that clone must be committed and pushed (--test skips this, for trial runs), so the commits
recorded resolve on GitHub.

Writes <out>/<country>/ (out: a local directory or a gs:// URL):
    config.yaml                      the config as given, plus a provenance block (commits and links)
    index.csv                        one row per perturbation: id, run, scenario, perturbation, noise, seed, status
    perturbations/<id>/inputs.zip    its seer-schema csvs
    perturbations/<id>/outputs.zip   solved ones: the model's output tables
    perturbations/<id>/failed.json   failed solves: the error (infeasible, time limit, worker lost ...)
A perturbation is regenerated from config.yaml and its id (provenance.regenerate). One whose inputs equal an
earlier one's (only possible with noise 0) is dropped.

Resumable: the bucket is the only state. Rerunning the same command (at the same commits) skips every
inputs.zip / outputs.zip / failed.json already written (--retry-failed retries the failed solves); files appear
only once complete; a failing task is recorded and the run carries on; index.csv is rewritten from the bucket
as tasks finish.
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
from pathlib import Path

import fsspec
import pandas as pd
import yaml

from ..scenarios import perturb
from ..scenarios.perturb import round_significant
from . import plan, provenance
from .design import build_entries

DEFAULTS = {"seed": 0, "significant_figures": 6, "cost_dataset": "wb_epm", "runs": [],
            "solve": {"model": "pyomo", "annualize_capex": False, "years": "all", "time_limit": None}}


def load_config(text):
    config = {**DEFAULTS, **yaml.safe_load(text)}
    config["solve"] = {**DEFAULTS["solve"], **config.get("solve", {})}
    years = config["solve"]["years"]
    if years != "all" and not (isinstance(years, int) and not isinstance(years, bool) and years > 0):
        raise ValueError(f"solve: years must be all or a number of years, got {years!r}")
    return config


# ---------------------------------------------------------------- output files (local or gs://)


def open_out(out_url):
    """(filesystem, country root path, perturbations path) for a country's out_url (local or gs://)."""
    fs, root = fsspec.core.url_to_fs(str(out_url))
    return fs, root, posixpath.join(root, "perturbations")


def write(fs, path, data):
    """Write data to path so it appears only once complete: a gs:// object does by itself; a local file is
    written aside and renamed into place."""
    data = data.encode() if isinstance(data, str) else data
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    if "file" not in fs.protocol:
        fs.pipe(path, data)
        return
    fs.pipe(path + ".partial", data)
    fs.mv(path + ".partial", path)


def zip_csvs(csvs):
    """A zip (bytes) of csvs ({name: csv text})."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, text in csvs.items():
            z.writestr(f"{name}.csv", text)
    return buffer.getvalue()


def read_csvs(data):
    """{name: csv text} in a zip (bytes) from zip_csvs."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return {name.removesuffix(".csv"): z.read(name).decode() for name in z.namelist()}


def ids_with(fs, perturbations, filename):
    """{id: path} of perturbations that have filename (inputs.zip / outputs.zip), in one listing."""
    return {posixpath.relpath(p, perturbations).split("/")[0]: p for p in fs.glob(f"{perturbations}/*/{filename}")}


# ---------------------------------------------------------------- generate and solve (run on workers)


BASES = {}  # base instance by url: read once per worker process


def generate_one(config, entry, base_zip, base_url, out_url):
    """entry's perturbation (see design.build_entries), written to perturbations/<id>/inputs.zip.
    Returns the sha256 of its csvs."""
    if base_url not in BASES:
        BASES[base_url] = provenance.read_zipped(base_zip)
    fs, _, perturbations = open_out(out_url)
    instance = perturb(BASES[base_url], entry["scenario"], entry["perturbation"], entry["seed"],
                       config["significant_figures"], config["cost_dataset"], entry["noise"])
    instance.validate()
    csvs = {name: df.to_csv(index=False) for name, df in instance.as_dict().items() if not (name == "network" and df.empty)}
    write(fs, posixpath.join(perturbations, entry["id"], "inputs.zip"), zip_csvs(csvs))
    return provenance.tables_sha256(csvs)


def solve_one(config, perturbation_id, out_url):
    """Solve perturbation_id's inputs.zip with config's model: write its output tables (summary with the solve
    time) to outputs.zip, or the error to failed.json."""
    fs, _, perturbations = open_out(out_url)
    solve, directory = config["solve"], posixpath.join(perturbations, perturbation_id)
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in read_csvs(fs.cat(posixpath.join(directory, "inputs.zip"))).items():
            Path(tmp, f"{name}.csv").write_text(text)
        if solve["years"] != "all":  # solve only the first that many years
            years = pd.read_csv(f"{tmp}/years.csv").sort_values("year")
            years.head(solve["years"]).to_csv(f"{tmp}/years.csv", index=False)
        try:
            tables, solve_time = SOLVERS[solve["model"]](tmp, solve["annualize_capex"], solve["time_limit"])
        except Exception as e:  # infeasible, non-optimal, time limit
            write_failed(fs, perturbations, perturbation_id, f"{type(e).__name__}: {e}")
            return
    gc.collect()  # solved models are large and workers are long-lived
    tables["summary"] = pd.concat([tables["summary"], pd.DataFrame(
        [{"metric": "solve_seconds", "value": round(solve_time, 1), "unit": "s"}])], ignore_index=True)
    csvs = {name: round_floats(df, config["significant_figures"]).to_csv(index=False) for name, df in tables.items()}
    write(fs, posixpath.join(directory, "outputs.zip"), zip_csvs(csvs))


def write_failed(fs, perturbations, perturbation_id, error):
    write(fs, posixpath.join(perturbations, perturbation_id, "failed.json"), json.dumps({"id": perturbation_id, "error": error}))


def solve_pyomo(data_path, annualize_capex, time_limit):
    from ..models import pyomo_model

    instance = pyomo_model.build(f"{data_path}/", annualize_capex=annualize_capex)
    options = {"time_limit": time_limit} if time_limit else None
    _, solve_time = pyomo_model.solve(instance, verbose=False, solver_options=options)
    return pyomo_model.build_output_tables(instance, data_path), solve_time


def solve_pypsa(data_path, annualize_capex, time_limit):
    from ..models import pypsa_model

    network = pypsa_model.build(f"{data_path}/", annualize_capex=annualize_capex)
    options = {"time_limit": time_limit} if time_limit else {}
    _, _, solve_time = pypsa_model.solve(network, verbose=False, solver_options=options)
    return pypsa_model.build_output_tables(network, data_path), solve_time


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


def run_parallel(executor, fn, arg_list, on_progress=None):
    """fn(*args) for each args on executor. Returns ({args[1]: result}, {args[1]: error}) -- args[1] is the
    perturbation (or its id); a task that raises (or whose worker dies) is recorded, not raised. Every 5% of
    tasks: prints progress and calls on_progress()."""
    futures = {executor.submit(fn, *args): args[1] for args in arg_list}
    results, errors = {}, {}
    for done, future in enumerate(as_completed(futures), 1):
        key = futures[future]
        key = key["id"] if isinstance(key, dict) else key
        try:
            results[key] = future.result()
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"
        if done == len(futures) or done % max(1, len(futures) // 20) == 0:
            print(f"  {done}/{len(futures)}" + (f" ({len(errors)} errors)" if errors else ""), flush=True)
            if on_progress:
                on_progress()
    return results, errors


def start_dataset(fs, root, config, config_text, models_dir):
    """Write config.yaml: the config as given, plus its provenance. Refuse to resume a dataset made at other
    commits, or whose runs changed under the same name (their existing perturbations would no longer match)."""
    record = provenance.provenance_record(config, models_dir)
    path = posixpath.join(root, "config.yaml")
    if fs.exists(path):
        old = yaml.safe_load(fs.cat(path))
        for repo in ["optimizer", "base_instance"]:
            if old["provenance"][repo]["commit"] != record[repo]["commit"]:
                raise SystemExit(f"{path} was made at {repo} commit {old['provenance'][repo]['commit']}, not "
                                 f"{record[repo]['commit']}: use a new --out, or check out that commit")
        old_runs = {r["name"]: r for r in old["runs"]}
        changed = [r["name"] for r in config["runs"] if r["name"] in old_runs and old_runs[r["name"]] != r]
        if changed:
            raise SystemExit(f"runs {changed} differ from {path}'s: rename them, or use a new --out")
    write(fs, path, config_text.rstrip() + "\n\n# added by seer.ccai_dataset.run\n"
          + yaml.safe_dump({"provenance": record}, sort_keys=False))
    return record


def run(config, config_text, out_root, executor, models_dir, retry_failed=False):
    """Generate config's perturbations and solve the ones marked solve, into out_root/<country>, resuming from
    whatever the bucket already holds."""
    out_url = f"{str(out_root).rstrip('/')}/{config['country']}"
    fs, root, perturbations = open_out(out_url)
    record = start_dataset(fs, root, config, config_text, models_dir)
    base_zip = provenance.zip_dir(Path(models_dir) / config["base"]["path"])
    entries = build_entries(config)

    def write_index():
        index = build_index(fs, perturbations, entries)
        write(fs, posixpath.join(root, "index.csv"), index.to_csv(index=False))
        return index

    # Step 1 - generate every perturbation not already written
    have = ids_with(fs, perturbations, "inputs.zip")
    todo = [e for e in entries if e["id"] not in have]
    print(f"generating {len(todo)} of {len(entries)} perturbations", flush=True)
    sha, errors = run_parallel(executor, generate_one, [(config, e, base_zip, record["base_instance"]["url"], out_url) for e in todo],
                               write_index)
    for pid, error in errors.items():
        print(f"  generating {pid} failed: {error}", flush=True)

    # Step 2 - drop duplicates: a perturbation whose inputs equal an earlier one's. Only possible with noise 0
    # (e.g. a scenario element with nothing to act on in this country), so only those are compared
    have = ids_with(fs, perturbations, "inputs.zip")
    first, duplicate_of = {}, {}
    for e in entries:
        if e["noise"] == 0 and e["id"] in have:
            if e["id"] not in sha:  # written by an earlier run
                sha[e["id"]] = provenance.tables_sha256(read_csvs(fs.cat(have[e["id"]])))
            if sha[e["id"]] in first:
                duplicate_of[e["id"]] = first[sha[e["id"]]]
                fs.rm(posixpath.join(perturbations, e["id"]), recursive=True)
            else:
                first[sha[e["id"]]] = e["id"]
    print(f"dropped {len(duplicate_of)} duplicates", flush=True)

    # Step 3 - solve entries marked solve (a dropped duplicate: its original), not already solved or failed
    done = set(ids_with(fs, perturbations, "outputs.zip"))
    failed = ids_with(fs, perturbations, "failed.json")
    if retry_failed:
        for path in failed.values():
            fs.rm(path)
    else:
        done |= set(failed)
    to_solve = list(dict.fromkeys(duplicate_of.get(e["id"], e["id"]) for e in entries if e["solve"]))
    to_solve = [pid for pid in to_solve if pid not in done and pid in have]
    print(f"solving {len(to_solve)} with {config['solve']['model']}", flush=True)
    _, errors = run_parallel(executor, solve_one, [(config, pid, out_url) for pid in to_solve], write_index)
    for pid, error in errors.items():  # the task itself failed (e.g. its worker died): record it as a failed solve
        if error.startswith("BrokenProcessPool"):  # a local worker died, and the pool with it: not this solve's fault
            print(f"  {pid}: local worker pool broke, not recorded; rerun to retry", flush=True)
        else:
            write_failed(fs, perturbations, pid, error)

    # Step 4 - final index
    index = write_index()
    print(f"{len(index)} perturbations, solves: {index['status'].value_counts().to_dict()} -> {out_url}")


def build_index(fs, perturbations, entries):
    """index.csv, from the bucket: one row per perturbation written, status optimal / failed for solves."""
    have, solved, failed = (ids_with(fs, perturbations, name) for name in ["inputs.zip", "outputs.zip", "failed.json"])
    return pd.DataFrame([{
        "id": e["id"], "run": e["run"], "scenario": json.dumps(e["scenario"]), "perturbation": json.dumps(e["perturbation"]),
        "noise": e["noise"], "seed": e["seed"],
        "status": "optimal" if e["id"] in solved else "failed" if e["id"] in failed else None,
    } for e in entries if e["id"] in have], columns=["id", "run", "scenario", "perturbation", "noise", "seed", "status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", default="data/ccai_dataset", help="local directory or gs:// URL")
    parser.add_argument("--plan", action="store_true", help="print what the config would make, and perturbation ranges; run nothing")
    parser.add_argument("--workers", type=int, default=1, help="local processes, or Coiled workers with --coiled")
    parser.add_argument("--coiled", action="store_true", help="run tasks on a Coiled cluster")
    parser.add_argument("--cluster-name", default="seer-ccai-dataset", help="Coiled cluster name (with --coiled)")
    parser.add_argument("--region", default="us-west2", help="Coiled cluster region (with --coiled)")
    parser.add_argument("--models-dir", type=Path, default=provenance.MODELS_DIR, help="local clone of seer-planning-models")
    parser.add_argument("--test", action="store_true", help="a trial run: skip the committed-and-pushed checks")
    parser.add_argument("--retry-failed", action="store_true", help="retry solves that failed in an earlier run")
    args = parser.parse_args()

    config_text = args.config.read_text()
    config = load_config(config_text)
    if args.plan:
        return plan.print_plan(config, args.models_dir)
    if not args.test:
        provenance.check_repo(provenance.SEER_DIR)
        provenance.check_repo(args.models_dir)
    with make_executor(args.workers, args.coiled, args.cluster_name, args.region) as executor:
        run(config, config_text, args.out, executor, args.models_dir, args.retry_failed)


if __name__ == "__main__":
    main()
