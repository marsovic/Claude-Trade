"""Interactive Brokers through IB Gateway / TWS with ``ib_async``.

IBKR explicitly supports automated trading through its API, which makes it
the safest choice legally. At 100 EUR it is expensive, though:

* the TWS API trades WHOLE shares only (fractions are app/FIX only),
* fixed pricing on US stocks: 0.005 USD/share, min 1 USD, max 1 % of value,
* market data subscriptions need ~500 USD of equity: prices come from the
  bot's data provider, IBKR is used for execution only,
* the Gateway needs a weekly manual (2FA) login.

``pip install ib_async`` and run IB Gateway (paper port 4002, live 4001).
"""

from __future__ import annotations

import logging

from invest.brokers.base import (
    AccountInfo,
    BrokerError,
    BrokerPosition,
    LiveBroker,
    OrderRequest,
    OrderUpdate,
)

log = logging.getLogger(__name__)
STATUS = {"PendingSubmit": "pending", "PreSubmitted": "pending", "Submitted": "pending",
          "ApiPending": "pending", "PendingCancel": "pending", "Filled": "filled",
          "Cancelled": "cancelled", "ApiCancelled": "cancelled", "Inactive": "rejected"}


class IbkrBroker(LiveBroker):
    name = "ibkr"
    resting_stops = True
    fractional = False

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 17,
                 etp_contract: str = "", ib=None, fx=None):
        try:
            import ib_async
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BrokerError("IBKR needs `pip install ib_async` and a running IB Gateway") \
                from exc
        self.lib = ib_async
        self.ib = ib or ib_async.IB()
        self.fx = fx or (lambda: 1.10)
        self.etp_contract = etp_contract  # "SYMBOL:EXCHANGE:CURRENCY", e.g. "QQQ3:LSEETF:USD"
        self._contracts: dict[str, object] = {}
        if not self.ib.isConnected():
            self.ib.connect(host, port, clientId=client_id, timeout=15)
        self.ib.reqMarketDataType(3)  # delayed data is free; we only need executions

    # ------------------------------------------------------------- contracts
    def _contract(self, symbol: str):
        if symbol in self._contracts:
            return self._contracts[symbol]
        if symbol.startswith("ETP:"):
            if not self.etp_contract:
                return None
            sym, exchange, currency = self.etp_contract.split(":")
            contract = self.lib.Stock(sym, exchange, currency)
        else:
            contract = self.lib.Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        contract = qualified[0] if qualified else None
        self._contracts[symbol] = contract
        return contract

    def resolve(self, symbol: str) -> str | None:
        return symbol if self._contract(symbol) is not None else None

    # --------------------------------------------------------------- account
    def account(self) -> AccountInfo:
        values = {(v.tag, v.currency): v.value for v in self.ib.accountSummary()}

        def pick(tag):
            for cur in ("EUR", "BASE"):
                if (tag, cur) in values:
                    return float(values[(tag, cur)])
            usd = values.get((tag, "USD"))
            return float(usd) / self.fx() if usd is not None else 0.0

        positions = {}
        for item in self.ib.portfolio():
            sym = item.contract.symbol
            positions[sym] = BrokerPosition(ticker=sym, qty=float(item.position),
                                            avg_price=float(item.averageCost),
                                            price=float(item.marketPrice),
                                            currency=item.contract.currency or "USD")
        return AccountInfo(equity_eur=pick("NetLiquidation"), cash_eur=pick("TotalCashValue"),
                           positions=positions)

    # ---------------------------------------------------------------- orders
    def submit(self, order: OrderRequest) -> str:
        contract = self._contract(order.symbol)
        if contract is None:
            raise BrokerError(f"{order.symbol}: contract not found at IBKR")
        qty = int(order.qty)
        if qty < 1:
            raise BrokerError(f"{order.symbol}: IBKR API needs whole shares")
        lib = self.lib
        if order.kind == "LIMIT":
            ib_order = lib.LimitOrder(order.side, qty, round(order.limit_price, 2), tif="DAY")
        elif order.kind == "STOP":
            ib_order = lib.StopOrder(order.side, qty, round(order.stop_price, 2), tif="GTC")
        else:
            ib_order = lib.MarketOrder(order.side, qty, tif="DAY")
        ib_order.orderRef = order.id  # survives reconnects, used to reconcile
        trade = self.ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)

    def _trade(self, order: OrderRequest):
        for trade in self.ib.trades():
            if getattr(trade.order, "orderRef", "") == order.id:
                return trade
        return None

    def cancel(self, broker_id: str) -> None:
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == str(broker_id):
                self.ib.cancelOrder(trade.order)

    def begin_sync(self) -> None:
        self.ib.sleep(0.5)  # let ib_async process pending events

    def order_update(self, order: OrderRequest) -> OrderUpdate:
        trade = self._trade(order)
        if trade is None:
            # Not in this session's trades: look at today's executions.
            fills = [f for f in self.ib.reqExecutions()
                     if getattr(f.execution, "orderRef", "") == order.id]
            if not fills:
                return OrderUpdate("unknown")
            qty = sum(f.execution.shares for f in fills)
            avg = sum(f.execution.shares * f.execution.price for f in fills) / qty
            fees = sum(f.commissionReport.commission for f in fills) / self.fx()
            return OrderUpdate("filled", qty, avg, fees,
                               int(fills[-1].time.timestamp() * 1000))
        status = STATUS.get(trade.orderStatus.status, "unknown")
        fees = sum(f.commissionReport.commission for f in trade.fills) / self.fx()
        when = trade.fills[-1].time if trade.fills else None
        if status == "pending" and trade.orderStatus.filled > 0:
            status = "partially_filled"
        return OrderUpdate(status, filled_qty=float(trade.orderStatus.filled),
                           avg_price=float(trade.orderStatus.avgFillPrice), fees_eur=fees,
                           filled_at=int(when.timestamp() * 1000) if when else 0)

    def preflight(self) -> list[str]:
        return ["IBKR trades whole shares only through the API; commissions (min 1 USD) "
                "weigh heavily on a 100 EUR account."]
