from mindmap_ml.synthetic.power import (
    days_to_detect,
    lagged_correlation_power,
    recommend_min_days,
)


def test_power_increases_with_days_for_real_effect() -> None:
    pts = lagged_correlation_power(effect=0.4, noise=1.0, n_sims=120, seed=0)
    by_days = {p.n_days: p for p in pts}
    assert by_days[90].power > by_days[14].power  # more data -> more power
    assert by_days[90].power >= 0.8  # a real moderate effect is detectable with enough data
    # the |r|>=0.3 threshold is loose at 14 days but tightens with N
    assert by_days[90].false_positive_rate <= 0.15
    assert by_days[14].false_positive_rate > by_days[90].false_positive_rate


def test_days_to_detect_is_finite_for_strong_effect() -> None:
    pts = lagged_correlation_power(effect=0.7, noise=1.0, n_sims=80, seed=1)
    d = days_to_detect(pts, target_power=0.8, max_fp=0.15)
    assert d is not None and d <= 90


def test_recommend_min_days_returns_sensible_number() -> None:
    d = recommend_min_days(effect=0.6, seed=2)
    assert 14 <= d <= 90


def test_weak_effect_needs_more_or_undetectable() -> None:
    # a very weak effect should have low power at small N (don't over-promise)
    pts = lagged_correlation_power(effect=0.1, noise=1.0, n_sims=120, seed=3)
    by_days = {p.n_days: p for p in pts}
    assert by_days[14].power < 0.6
