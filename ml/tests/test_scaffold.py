"""Phase 0 smoke test: the package imports and core config constants are sane.

Replaced/expanded by real golden tests as each phase lands.
"""

from mindmap_ml import __version__, config


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_baseline_model_version_matches_app() -> None:
    # Must stay wire-compatible with frontend/lib/prediction-engine.ts MODEL_VERSION.
    assert config.BASELINE_MODEL_VERSION == "v1_rule_extended"


def test_correlation_thresholds_match_ts_engine() -> None:
    # Ported from frontend/lib/correlation-engine.ts defaults.
    assert config.CORR_MIN_SAMPLE_SIZE == 8
    assert config.CORR_MIN_ABS_R == 0.3
    assert config.CORR_MAX_RESULTS == 6
