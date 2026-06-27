"""Tests for the daily notifier's send/skip logic (pure, offline)."""

from src import notify


def _result(symbol, verdict, synthetic=False):
    return {
        "symbol": symbol, "verdict": verdict, "synthetic": synthetic,
        "price": 50000.0, "pct_vs_ma": -0.2, "ma_days": 200,
        "drawdown_from_high": -0.5,
    }


def test_sends_when_favorable():
    should_send, subject, body = notify.build_payload(
        [_result("BTC/USD", "FAVORABLE"), _result("ETH/USD", "NEUTRAL")])
    assert should_send is True
    assert "BTC" in subject
    assert "BTC/USD: FAVORABLE" in body


def test_leaning_is_actionable():
    should_send, _, _ = notify.build_payload([_result("BTC/USD", "LEANING FAVORABLE")])
    assert should_send is True


def test_no_send_when_all_neutral():
    should_send, subject, _ = notify.build_payload(
        [_result("BTC/USD", "NEUTRAL"), _result("ETH/USD", "NEUTRAL")])
    assert should_send is False
    assert "no actionable" in subject.lower()


def test_never_sends_on_synthetic_data():
    # Even a FAVORABLE verdict must NOT alert if the data was synthetic/fallback.
    should_send, _, body = notify.build_payload(
        [_result("BTC/USD", "FAVORABLE", synthetic=True)])
    assert should_send is False
    # synthetic rows are excluded from the body entirely
    assert "BTC/USD" not in body


def test_body_has_disclaimer():
    _, _, body = notify.build_payload([_result("BTC/USD", "FAVORABLE")])
    assert "not financial advice" in body.lower()
    assert "not a prediction" in body.lower()
