"""Provenance for perturbed instances: what a perturbation.json records (base instance, perturb arguments,
code version), and regenerating an instance from one."""
import hashlib
import io
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ..schema import SeerInstance
from ..scenarios import perturb

REPO_DIR = Path(__file__).resolve().parents[2]
REFERENCE_DATA_DIR = REPO_DIR / "seer" / "scenarios" / "reference_data"


def hash_dir(path, pattern="*"):
    """sha256 over the relative paths and contents of path's files matching pattern, in sorted order."""
    digest = hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob(pattern) if p.is_file() and ".cache" not in p.parts):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout


def code_record():
    """The running code: git commit, dirty tree (uncommitted or untracked files), reference data hash."""
    diff = git("diff", "HEAD")
    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain").strip()),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "reference_data_sha256": hash_dir(REFERENCE_DATA_DIR),
        "seer_version": version("seer"),
    }


def perturbation_record(config, entry, base_sha256, inputs_sha256, code):
    """The perturbation.json for one entry of config's run (see design.build_entries)."""
    return {
        "id": entry["id"],
        "country": config["country"],
        "design": {k: entry[k] for k in ["run", "scenario_name"]},
        "base": {"path": config["base"], "sha256": base_sha256},
        "perturb": {
            "scenario": entry["scenario"],
            "perturbation": entry["perturbation"],
            "seed": entry["seed"],
            "significant_figures": config["significant_figures"],
            "cost_dataset": config["cost_dataset"],
            "noise": entry["noise"],
        },
        "inputs_sha256": inputs_sha256,
        "code": code,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def tables_sha256(tables):
    """sha256 of an instance's tables ({name: df}) as written: identical instances share it."""
    digest = hashlib.sha256()
    for name, df in sorted(tables.items()):
        digest.update(name.encode())
        digest.update(df.to_csv(index=False).encode())
    return digest.hexdigest()


def zip_dir(path, pattern="*.csv"):
    """A zip (bytes) of path's files matching pattern, byte for byte."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for file in sorted(Path(path).glob(pattern)):
            z.write(file, file.name)
    return buffer.getvalue()


def read_zipped(data, sha256=None):
    """The SeerInstance in a zip (bytes) of seer-schema csvs; raises if sha256 is given and its csvs differ."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        if sha256 is not None and hash_dir(tmp, "*.csv") != sha256:
            raise ValueError("base instance doesn't match the record's sha256")
        return SeerInstance.read(tmp)


def regenerate(record, base=None):
    """The perturbed SeerInstance a perturbation.json record describes. base: the dataset's base.zip, or a
    directory of csvs; default the record's base path in the seer repo. Raises if the base's csvs differ
    from the record's."""
    base = Path(base or REPO_DIR / record["base"]["path"])
    data = base.read_bytes() if base.suffix == ".zip" else zip_dir(base)
    p = record["perturb"]
    return perturb(read_zipped(data, record["base"]["sha256"]), p["scenario"], p["perturbation"], p["seed"],
                   p["significant_figures"], p["cost_dataset"], p["noise"])
