"""Tests for the paper-trading sandbox (offline, synthetic data)."""

import numpy as np
import pytest

from src.config import load_config
from src import paper
from tests.conftest import make_ohlcv


def _cfg(tmp_path, symbols):
    cfg = load_config()
    cfg["paper"]["state_file"] = str(tmp_path / "paper_state.json")
    cfg["paper"]["symbols"] = symbols
    cfg["paper"]["initial_capital"] = 9000.0
    return cfg


def test_step_inits_advances_and_persists(tmp_path):
    cfg = _cfg(tmp_path, ["BTC/USD", "SPY"])
    df = make_ohlcv(100.0 * (1.01 ** np.arange(400)))
    prov = lambda _s: df

    s1 = paper.step(cfg, df_provider=prov, today="2026-01-01")
    assert set(s1["accounts"]) == {"BTC/USD", "SPY"}
    assert len(s1["history"]) == 1
    for a in s1["accounts"].values():
        assert "agent_equity" in a and "hold_equity" in a
        assert a["agent"]["strategy"]  # a chosen strategy is recorded

    s2 = paper.step(cfg, df_provider=prov, today="2026-01-02")
    assert len(s2["history"]) == 2
    assert s2["last_run"] == "2026-01-02"

    # round-trips to disk
    loaded = paper.load_state(cfg["paper"]["state_file"])
    assert loaded["created"] == "2026-01-01"
    assert len(loaded["history"]) == 2


def test_hold_marks_to_market_on_price_move(tmp_path):
    cfg = _cfg(tmp_path, ["BTC/USD"])
    flat_100 = make_ohlcv([100.0] * 300)
    flat_110 = make_ohlcv([100.0] * 300 + [110.0])

    seq = [flat_100, flat_110]
    prov = lambda _s: seq.pop(0)

    s1 = paper.step(cfg, df_provider=prov, today="2026-01-01")
    assert s1["accounts"]["BTC/USD"]["hold_equity"] == 9000.0  # bought at inception

    s2 = paper.step(cfg, df_provider=prov, today="2026-01-02")
    # price 100 -> 110 (+10%): the buy-and-hold shadow must rise ~10%.
    assert s2["accounts"]["BTC/USD"]["hold_equity"] == pytest.approx(9900.0, rel=1e-6)


def test_summary_lines_present(tmp_path):
    cfg = _cfg(tmp_path, ["BTC/USD"])
    df = make_ohlcv(100.0 * (1.005 ** np.arange(300)))
    paper.step(cfg, df_provider=lambda _s: df, today="2026-01-01")
    state = paper.load_state(cfg["paper"]["state_file"])
    lines = paper.summary_lines(state)
    assert any("PAPER SANDBOX" in ln for ln in lines)
    assert any("BTC/USD" in ln for ln in lines)


def test_summary_empty_state_is_no_lines():
    assert paper.summary_lines(None) == []
    assert paper.summary_lines({"accounts": {}}) == []
