"""BinanceBroker against an in-memory fake exchange (no network)."""

import pytest

from bot.binance_client import BinanceAPIError
from bot.broker import BinanceBroker
from bot.config import Settings
from bot.risk import TradePlan
from bot.state import BotState, StateStore
from tests.conftest import symbol_info


class FakeExchange:
    def __init__(self):
        self.positions = {}
        self.algo = {}
        self.next_id = 1
        self.orders = []
        self.income_rows = []
        self.fail_stop = False
        self.reject_stops = 0
        self.wallet = 1000.0
        self.dual = False
        self.withdrawals = False

    # --- endpoints used by the broker
    def sync_time(self):
        return 0

    def position_mode(self):
        return {"dualSidePosition": self.dual}

    def api_restrictions(self):
        return {"enableWithdrawals": self.withdrawals, "enableFutures": True,
                "ipRestrict": True}

    def account(self):
        return {"totalWalletBalance": str(self.wallet), "totalMarginBalance": str(self.wallet)}

    def set_margin_type(self, symbol, margin_type):
        return None

    def set_leverage(self, symbol, leverage):
        return {"leverage": leverage}

    def market_order(self, symbol, side, quantity, reduce_only=False, client_order_id=None):
        self.orders.append((symbol, side, quantity, reduce_only))
        amt = float(quantity) * (1 if side == "BUY" else -1)
        self.positions[symbol] = self.positions.get(symbol, 0.0) + amt
        if abs(self.positions[symbol]) < 1e-12:
            del self.positions[symbol]
        return {"executedQty": str(quantity), "avgPrice": "100.0"}

    def conditional_close_order(self, symbol, side, order_type, trigger_price,
                                client_algo_id=None):
        if order_type == "STOP_MARKET" and self.fail_stop:
            raise BinanceAPIError(400, -2021, "Order would immediately trigger.")
        if order_type == "STOP_MARKET" and self.reject_stops > 0:
            self.reject_stops -= 1
            raise BinanceAPIError(400, -1111, "Precision is over the maximum.")
        if any(v[:3] == (symbol, side, order_type) for v in self.algo.values()):
            # Like Binance: one closePosition order per type and direction.
            raise BinanceAPIError(400, -4130, "An open stop or take profit order with GTE "
                                              "and closePosition in the direction is existing.")
        algo_id = self.next_id
        self.next_id += 1
        self.algo[algo_id] = (symbol, side, order_type, float(trigger_price))
        return {"algoId": algo_id}

    def cancel_algo_order(self, algo_id):
        self.algo.pop(algo_id, None)
        return {}

    def cancel_all_algo_orders(self, symbol):
        for k in [k for k, v in self.algo.items() if v[0] == symbol]:
            del self.algo[k]
        return {}

    def position_risk(self, symbol=None):
        return [{"symbol": s, "positionAmt": str(a), "markPrice": "100.0",
                 "unRealizedProfit": "0"} for s, a in self.positions.items()]

    def income(self, start_time, symbol=None, income_type=None, limit=1000):
        return [r for r in self.income_rows
                if r["time"] >= start_time and (symbol is None or r["symbol"] == symbol)]

    def user_trades(self, symbol, start_time, limit=500):
        return [{"side": "SELL", "price": "102.0"}]


def make(tmp_path, **kw):
    s = Settings(mode="testnet", binance_api_key="k", binance_api_secret="s",
                 state_dir=str(tmp_path), **kw)
    state = BotState.fresh("testnet", 50.0)
    state.started_at = state.income_cursor = 0
    ex = FakeExchange()
    broker = BinanceBroker(s, state, StateStore(tmp_path), ex,
                           {"AAAUSDT": symbol_info("AAAUSDT", step="0.001", tick="0.01")})
    return s, state, ex, broker


PLAN = TradePlan(symbol="AAAUSDT", side="LONG", qty=1.0, price=100.0, stop_loss=99.0,
                 take_profit=102.0, leverage=3, atr=0.66, notional=100.0, margin=33.3,
                 risk_amount=1.0)


def test_open_places_market_order_and_exchange_side_protection(tmp_path):
    _, _, ex, broker = make(tmp_path)
    pos = broker.open(PLAN, 100.0, now=1)
    assert ex.orders[0][:2] == ("AAAUSDT", "BUY")
    types = sorted(v[2] for v in ex.algo.values())
    assert types == ["STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert all(v[1] == "SELL" for v in ex.algo.values())
    assert pos.sl_algo_id in ex.algo and pos.tp_algo_id in ex.algo
    assert pos.stop_loss == 99.0 and pos.take_profit == 102.0


def test_position_is_closed_if_stop_cannot_be_placed(tmp_path):
    _, _, ex, broker = make(tmp_path)
    ex.fail_stop = True
    with pytest.raises(RuntimeError):
        broker.open(PLAN, 100.0, now=1)
    assert ex.positions == {}  # never left unprotected
    assert ex.orders[-1][3] is True  # reduce-only close


def test_sync_detects_exchange_side_close_and_uses_income(tmp_path):
    _, state, ex, broker = make(tmp_path)
    pos = broker.open(PLAN, 100.0, now=1000)
    state.positions["AAAUSDT"] = pos
    # The take-profit fills on the exchange while the bot sleeps.
    ex.positions.clear()
    ex.income_rows = [
        {"symbol": "AAAUSDT", "incomeType": "REALIZED_PNL", "income": "2.0", "time": 2000,
         "tranId": 1},
        {"symbol": "AAAUSDT", "incomeType": "COMMISSION", "income": "-0.1", "time": 2000,
         "tranId": 2},
    ]
    closed = broker.sync({"AAAUSDT": 102.0}, now=3000)
    assert len(closed) == 1
    trade = closed[0]
    assert trade.reason == "take_profit"
    assert trade.pnl == pytest.approx(1.9)
    assert trade.fees == pytest.approx(0.1)
    assert ex.algo == {}  # leftover stop cancelled
    assert state.balance == pytest.approx(51.9)
    broker.sync({}, now=4000)  # income is not counted twice
    assert state.balance == pytest.approx(51.9)


def test_trailing_stop_replaces_exchange_order(tmp_path):
    s, state, ex, broker = make(tmp_path)
    pos = broker.open(PLAN, 100.0, now=1)
    state.positions["AAAUSDT"] = pos
    old_sl = pos.sl_algo_id
    pos.best_price = 101.5  # price already went +1.5R
    broker.sync({"AAAUSDT": 101.5}, now=2)
    assert pos.sl_algo_id != old_sl
    assert old_sl not in ex.algo
    assert ex.algo[pos.sl_algo_id][3] > 100.0  # stop now above entry


def test_preflight_refuses_withdrawal_keys_and_hedge_mode(tmp_path):
    s, _, ex, broker = make(tmp_path)
    s.mode = "live"
    ex.withdrawals = True
    with pytest.raises(RuntimeError, match="WITHDRAW"):
        broker.preflight()
    ex.withdrawals = False
    ex.dual = True
    with pytest.raises(RuntimeError, match="Hedge"):
        broker.preflight()
    ex.dual = False
    ex.wallet = 20.0
    ex.positions["ZZZUSDT"] = 1.0
    warnings = broker.preflight()
    assert any("below INITIAL_CAPITAL" in w for w in warnings)
    assert any("ZZZUSDT" in w for w in warnings)


def test_equity_is_capped_by_the_real_wallet(tmp_path):
    _, state, ex, broker = make(tmp_path)
    ex.wallet = 30.0
    assert broker.equity({}) == pytest.approx(30.0)
    ex.wallet = 5000.0
    assert broker.equity({}) == pytest.approx(50.0)  # only the bot's own capital


def test_failed_stop_move_restores_the_old_stop(tmp_path):
    _, state, ex, broker = make(tmp_path)
    pos = broker.open(PLAN, 100.0, now=1)
    state.positions["AAAUSDT"] = pos
    pos.best_price = 101.5
    ex.reject_stops = 1  # the new stop is rejected, the old one is put back
    closed = broker.sync({"AAAUSDT": 101.5}, now=2)
    assert closed == []
    assert pos.stop_loss == 99.0 and not pos.breakeven_done
    stops = [v for v in ex.algo.values() if v[2] == "STOP_MARKET"]
    assert stops == [("AAAUSDT", "SELL", "STOP_MARKET", 99.0)]


def test_position_closed_when_no_stop_can_be_placed(tmp_path):
    _, state, ex, broker = make(tmp_path)
    pos = broker.open(PLAN, 100.0, now=1)
    state.positions["AAAUSDT"] = pos
    pos.best_price = 101.5
    ex.reject_stops = 2  # new stop and old stop both rejected
    closed = broker.sync({"AAAUSDT": 101.5}, now=2)
    assert len(closed) == 1 and closed[0].reason == "stop_loss"
    assert ex.positions == {}
