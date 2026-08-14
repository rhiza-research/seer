"""
gdx: load GAMS EPM .gdx inputs, align them to a canonical per-symbol schema,
enrich generator and VRE profile data, and write one CSV per symbol per region.

    import gdx
    gdx.to_csv(data_dir="data/epm_gams",
               mapping_path="data/symbol_mapping.csv",
               output_dir="data/epm_gams_csv")
"""

from .enrich_data import reconcile_data
from .load_gdx import load_gdx_files
from .to_csv import write_to_csv

__all__ = ["to_csv", "load_gdx_files", "enrich_data", "write_csvs"]


def gams_to_csv(data_dir, mapping_path, output_dir, strict=True, region=None):
    """Run the full pipeline: load, align, enrich, and write CSVs for every region."""

    all_data, all_data_raw = load_gdx_files(data_dir, mapping_path, strict=strict, region=region)
    all_data_enriched = reconcile_data(all_data, all_data_raw)
    write_to_csv(all_data_enriched, output_dir)
    return all_data_enriched
