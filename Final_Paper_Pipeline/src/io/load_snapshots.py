"""Load the read-only data snapshots consolidated in Phase 0.

These are the intermediate CSVs that feed the figure notebooks but currently
have no recoverable generating script anywhere in the project (see
_source_archive/INDEX.md). Loading them from the archive keeps this pipeline
self-contained and independent of the live Google Drive mount.
"""

import os
import yaml
import pandas as pd


def load_config(config_path=None):
    if config_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, "..", "..", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _snapshot_path(config, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "..", "..", config["paths"]["data_snapshots"])
    return os.path.normpath(os.path.join(base, filename))


def load_snapshot(config, filename):
    path = _snapshot_path(config, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected data snapshot not found: {path}\n"
            f"This file should have been copied into _source_archive/data_snapshots/ "
            f"during Phase 0 consolidation -- see _source_archive/INDEX.md."
        )
    return pd.read_csv(path)
