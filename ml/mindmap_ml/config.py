"""Central configuration: paths, model-version constants, and the conservative
thresholds ported verbatim from the app's TypeScript engines.

These constants are the single source of truth on the Python side. Where they
mirror a TS engine the reference file is named so the two never silently drift.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all gitignored; see ml/.gitignore)
# --------------------------------------------------------------------------- #
ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "data"
ARTIFACTS_DIR = ML_ROOT / "artifacts"
REPORTS_DIR = ML_ROOT / "reports"

# --------------------------------------------------------------------------- #
# Model versions
# --------------------------------------------------------------------------- #
# Matches frontend/lib/prediction-engine.ts MODEL_VERSION so the ported baseline
# is wire-compatible with rows the app already understands.
BASELINE_MODEL_VERSION = "v1_rule_extended"
# ML models stamp their own version once they pass acceptance (Phase 3+).
ML_MODEL_VERSION = "v2_ml_assistive"

# --------------------------------------------------------------------------- #
# Correlation thresholds — port of frontend/lib/correlation-engine.ts
# --------------------------------------------------------------------------- #
CORR_MIN_SAMPLE_SIZE = 8
CORR_MIN_ABS_R = 0.3
CORR_MAX_RESULTS = 6
CORR_STRENGTH_STRONG = 0.7
CORR_STRENGTH_MODERATE = 0.5

# --------------------------------------------------------------------------- #
# Abstention / data-sufficiency (safety contract — see safety/contract.py)
# --------------------------------------------------------------------------- #
# Minimum logged days before any per-user risk number is allowed; below this we
# abstain with the "not enough data yet" message regardless of model confidence.
MIN_HISTORY_DAYS = 7
# Below this, forecasting is disabled (cold-start) and only rules may run.
MIN_FORECAST_HISTORY_DAYS = 14
# Confidence below this → abstain even with enough history.
MIN_CONFIDENCE_TO_SURFACE = 0.35

INSUFFICIENT_DATA_MESSAGE = (
    "I do not have enough consistent data yet to estimate this reliably. "
    "Keep logging for at least 7–14 days to reveal clearer patterns."
)
