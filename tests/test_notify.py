"""Tests for the daily digest notifier (pure, offline)."""

from src import notify


def _result(symbol, verdict, pct_vs_ma, drawdown, is_stock, synthetic=False):
    return {
        "symbol": symbol, "verdict": verdict, "synthetic": synthetic,
        "is_stock": is_stock, "price": 100.0, "pct_vs_ma": pct_vs_ma,
        "ma_days": 200, "drawdown_from_high": drawdown,
    }


def test_digest_sends_daily_with_real_data():
    # Even when everything is NEUTRAL, the daily digest still goes out.
    res = [
        _result("AAPL", "NEUTRAL", 0.10, -0.05, is_stock=True),
        _result("BTC/USD", "NEUTRAL", 0.05, -0.10, is_stock=False),
    ]
    should_send, subject, body = notify.build_payload(res)
    assert should_send is True
    assert "TOP STOCKS" in body and "TOP CRYPTO" in body
    assert "AAPL" in body and "BTC/USD" in body


def test_ranks_cheapest_vs_trend_first():
    # NVDA is further below its trend than AAPL, so it should rank first.
    res = [
        _result("AAPL", "NEUTRAL", -0.05, -0.10, is_stock=True),
        _result("NVDA", "FAVORABLE", -0.30, -0.40, is_stock=True),
    ]
    _, _, body = notify.build_payload(res)
    assert body.index("NVDA") < body.index("AAPL")


def test_subject_reflects_favorable_count():
    res = [_result("BTC/USD", "FAVORABLE", -0.25, -0.5, is_stock=False)]
    _, subject, _ = notify.build_payload(res)
    assert "1 market" in subject


def test_no_send_when_only_synthetic():
    res = [_result("BTC/USD", "FAVORABLE", -0.25, -0.5, is_stock=False, synthetic=True)]
    should_send, _, _ = notify.build_payload(res)
    assert should_send is False


def test_body_has_no_prediction_disclaimer():
    res = [_result("SPY", "NEUTRAL", 0.02, -0.03, is_stock=True)]
    _, _, body = notify.build_payload(res)
    low = body.lower()
    assert "not a prediction" in low
    assert "not" in low and "financial advice" in low
