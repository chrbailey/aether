"""Data Pipeline - Loading, normalization, and preparation of SAP event data.

Provides loaders for multiple SAP data sources and a unified pipeline
that combines them into training-ready datasets with shared vocabularies.

Data sources supported:
    - SAP IDES SQLite (sap-extractor output)
    - BPI Challenge 2019 JSON (real P2P event log)
    - CSV event logs (O2C + P2P from process mining datasets)
    - OCEL 2.0 P2P SQLite (simulated SAP procurement)
"""

from .unified_pipeline import (
    AetherDataPipeline,
    DEFAULT_PATHS,
    compute_outcome_heuristic,
    normalize_activity,
    normalize_resource,
)

__all__ = [
    "AetherDataPipeline",
    "DEFAULT_PATHS",
    "compute_outcome_heuristic",
    "normalize_activity",
    "normalize_resource",
]
