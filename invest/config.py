"""Settings of the investment bot (environment variables / ``.env``).

Every field can be overridden with an environment variable of the same name
in upper case, prefixed with ``INV_`` (e.g. ``INV_RISK_PER_TRADE=0.03``).
Secrets keep their usual provider names (``T212_API_KEY``, ``FINNHUB_API_KEY``...).

Defaults are AGGRESSIVE on purpose (4 % risk per trade, 2-3 concentrated
positions, an optional 3x leveraged ETP sleeve) but the bot still starts in
simulation and refuses real money until explicitly unlocked.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BROKERS = ("sim", "trading212", "ibkr", "alpaca")
ENVIRONMENTS = ("paper", "live")
DATA_PROVIDERS = ("massive", "alpaca", "tiingo", "csv")
LIVE_CONFIRMATION = "I_UNDERSTAND_THE_RISKS"

# Secrets and provider keys: read from their conventional names, not INV_*.
SECRET_ENV = {
    "t212_api_key": "T212_API_KEY",
    "t212_api_secret": "T212_API_SECRET",
    "alpaca_api_key": "ALPACA_API_KEY",
    "alpaca_api_secret": "ALPACA_API_SECRET",
    "massive_api_key": "MASSIVE_API_KEY",
    "tiingo_api_key": "TIINGO_API_KEY",
    "finnhub_api_key": "FINNHUB_API_KEY",
    "alphavantage_api_key": "ALPHAVANTAGE_API_KEY",
    "fred_api_key": "FRED_API_KEY",
    "sec_user_agent": "SEC_USER_AGENT",
}


@dataclass(frozen=True)
class FeeModel:
    """Costs of one order, expressed in the instrument currency except fx."""

    commission_pct: float = 0.0  # of notional
    commission_min: float = 0.0  # per order (instrument currency)
    commission_max_pct: float = 1.0  # cap as a fraction of notional
    commission_per_share: float = 0.0
    fx_fee_pct: float = 0.0  # charged when instrument currency != EUR
    fx_conversion_min_eur: float = 0.0  # one-off (IBKR): not charged per trade

    def cost(self, qty: float, price: float, foreign: bool) -> float:
        notional = abs(qty) * price
        commission = max(self.commission_min,
                         notional * self.commission_pct + abs(qty) * self.commission_per_share)
        commission = min(commission, notional * self.commission_max_pct)
        fx = notional * self.fx_fee_pct if foreign else 0.0
        return commission + fx


# Published retail pricing as researched in Oct 2026 - check before trusting.
FEE_PRESETS = {
    # Trading 212 Invest: no commission, 0.15 % FX fee on USD instruments.
    "trading212": FeeModel(fx_fee_pct=0.0015),
    # IBKR Pro fixed, US stocks: 0.005 USD/share, min 1 USD, max 1 % of value.
    "ibkr": FeeModel(commission_per_share=0.005, commission_min=1.0,
                     commission_max_pct=0.01, fx_conversion_min_eur=2.0),
    # Alpaca: commission-free US stocks (tiny regulatory fees on sells ignored).
    "alpaca": FeeModel(),
    # The simulator mimics Trading 212 unless told otherwise.
    "sim": FeeModel(fx_fee_pct=0.0015),
}


class ConfigError(ValueError):
    pass


@dataclass
class InvestSettings:
    # --- Broker & mode ----------------------------------------------------
    broker: str = "sim"
    environment: str = "paper"
    fee_preset: str = ""  # defaults to the broker's preset
    live_trading: str = ""
    t212_terms_acknowledged: bool = False
    initial_capital: float = 100.0  # EUR
    survival_floor: float = 10.0  # EUR: at or below, the capital is lost
    state_dir: str = "state_invest"
    data_dir: str = "data_invest"

    # --- Data ----------------------------------------------------------------
    data_provider: str = "massive"
    csv_dir: str = ""
    history_days: int = 460  # calendar days of daily bars kept for signals
    benchmark: str = "SPY"
    eurusd_fallback: float = 1.10

    # --- Universe --------------------------------------------------------------
    universe: list[str] = field(default_factory=list)  # empty = curated list
    exclude: list[str] = field(default_factory=list)
    min_price: float = 10.0
    min_dollar_volume: float = 50e6
    min_history: int = 300
    atr_min_pct: float = 1.0
    atr_max_pct: float = 7.0
    max_daily_jump: float = 0.15  # lottery-stock exclusion over 21 days
    min_beta: float = 1.0

    # --- Ranking & the 7 conditions ----------------------------------------------
    top_pct_entry: float = 0.10
    top_pct_hold: float = 0.25
    near_high_min: float = 0.85
    breakout_lookback: int = 20
    volume_factor: float = 1.3
    max_extension: float = 0.25  # close / SMA50 - 1
    rsi_max: float = 80.0
    vix_max_entry: float = 30.0
    vix_term_max: float = 1.0  # VIX / VIX3M
    vix_exit: float = 35.0
    regime_sma: int = 200

    # --- Risk (aggressive) ---------------------------------------------------------
    risk_per_trade: float = 0.04
    slots: int = 3
    slots_small_equity: int = 2
    small_equity: float = 60.0
    max_weight: float = 0.40
    max_open_risk: float = 0.12
    max_gross: float = 1.0  # no margin
    atr_period: int = 22
    stop_atr_mult: float = 3.0
    regime_off_stop_mult: float = 2.0
    stop_min_pct: float = 0.04
    stop_max_pct: float = 0.20
    pyramid: bool = True
    pyramid_r: float = 1.5
    pyramid_fraction: float = 0.5
    time_stop_days: int = 30
    time_stop_min_r: float = 0.5
    rebalance_weekday: int = 4  # Friday close
    dd_level1: float = 0.20
    dd_level2: float = 0.35
    weekly_loss_pause: float = 0.15
    pause_days: int = 5
    max_entry_gap: float = 0.03  # buy limit = close * (1 + gap)
    max_cost_r: float = 0.15  # skip trades whose round trip costs more than this x R
    slippage_bps: float = 5.0
    fractional: bool = True  # broker allows fractional quantities

    # --- 3x leveraged ETP sleeve -------------------------------------------------------
    etp_enabled: bool = True
    etp_underlying: str = "QQQ"
    etp_leverage: float = 3.0
    etp_weight: float = 0.33
    etp_annual_cost: float = 0.0075
    etp_rv_max: float = 0.20
    etp_rv_exit: float = 0.25
    etp_vix_max: float = 22.0
    etp_vix_exit: float = 28.0
    etp_broker_ticker: str = ""  # real brokers: instrument to trade (e.g. T212 ticker)
    etp_price_symbol: str = ""  # real brokers: Alpha Vantage symbol for its daily prices
    etp_currency: str = "EUR"  # real brokers: currency of the traded ETP line

    # --- Sentiment / research ------------------------------------------------------
    sentiment_enabled: bool = True
    news_veto: float = -0.25
    news_days: int = 3
    mania_ratio: float = 4.0
    mania_rsi: float = 75.0
    insider_bonus: float = 0.25
    earnings_blackout_days: int = 2
    earnings_exit_below_r: float = 1.0

    # --- Live loop ---------------------------------------------------------------
    poll_seconds: int = 60
    eod_delay_minutes: int = 30  # after the 16:00 ET close before fetching bars

    # --- Secrets (conventional env names, see SECRET_ENV) ---------------------------
    t212_api_key: str = ""
    t212_api_secret: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 0  # 0 = 4002 paper / 4001 live (IB Gateway)
    ibkr_client_id: int = 17
    massive_api_key: str = ""
    tiingo_api_key: str = ""
    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""
    fred_api_key: str = ""
    sec_user_agent: str = ""

    # ------------------------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: str | None = ".env", **overrides) -> InvestSettings:
        if env_file and load_dotenv is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        values = {}
        for f in fields(cls):
            name = SECRET_ENV.get(f.name, "INV_" + f.name.upper())
            raw = os.environ.get(name)
            if raw is None and f.name in ("ibkr_host", "ibkr_port", "ibkr_client_id"):
                raw = os.environ.get(f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f, raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**values)
        settings.validate()
        return settings

    @property
    def fees(self) -> FeeModel:
        return FEE_PRESETS[self.fee_preset or self.broker]

    @property
    def is_live(self) -> bool:
        return self.broker != "sim" and self.environment == "live"

    @property
    def mode(self) -> str:
        return "sim" if self.broker == "sim" else f"{self.broker}-{self.environment}"

    def validate(self) -> None:
        self.broker = self.broker.lower()
        self.environment = self.environment.lower()
        if self.broker not in BROKERS:
            raise ConfigError(f"INV_BROKER must be one of {BROKERS}")
        if self.environment not in ENVIRONMENTS:
            raise ConfigError(f"INV_ENVIRONMENT must be one of {ENVIRONMENTS}")
        if self.data_provider not in DATA_PROVIDERS:
            raise ConfigError(f"INV_DATA_PROVIDER must be one of {DATA_PROVIDERS}")
        if (self.fee_preset or self.broker) not in FEE_PRESETS:
            raise ConfigError(f"unknown fee preset {self.fee_preset!r}")
        if not 0 < self.risk_per_trade <= 0.10:
            raise ConfigError("INV_RISK_PER_TRADE must be in (0, 0.10]")
        if self.initial_capital <= 0 or not 0 <= self.survival_floor < self.initial_capital:
            raise ConfigError("need 0 <= SURVIVAL_FLOOR < INITIAL_CAPITAL")
        if not 1 <= self.slots <= 10 or self.slots_small_equity > self.slots:
            raise ConfigError("INV_SLOTS must be 1..10 and >= INV_SLOTS_SMALL_EQUITY")
        if not 0 < self.max_weight <= 1 or self.max_gross > 1.0:
            raise ConfigError("weights must be in (0, 1]; no margin (INV_MAX_GROSS <= 1)")
        if not 0 < self.top_pct_entry <= self.top_pct_hold <= 1:
            raise ConfigError("need 0 < TOP_PCT_ENTRY <= TOP_PCT_HOLD <= 1")
        if self.etp_currency not in ("EUR", "USD"):
            raise ConfigError("INV_ETP_CURRENCY must be EUR or USD (pick an ETP line quoted "
                              "in one of them; GBP/GBX lines are not supported)")
        if self.etp_leverage not in (1.0, 2.0, 3.0):
            raise ConfigError("INV_ETP_LEVERAGE must be 1, 2 or 3")
        if self.data_provider == "csv" and not self.csv_dir:
            raise ConfigError("INV_DATA_PROVIDER=csv needs INV_CSV_DIR")
        if self.broker == "ibkr":
            self.fractional = False  # the TWS API trades whole shares only
        if self.broker == "trading212" and not (self.t212_api_key and self.t212_api_secret):
            raise ConfigError("Trading 212 needs T212_API_KEY and T212_API_SECRET")
        if self.broker == "alpaca" and not (self.alpaca_api_key and self.alpaca_api_secret):
            raise ConfigError("Alpaca needs ALPACA_API_KEY and ALPACA_API_SECRET")
        if self.is_live and self.live_trading != LIVE_CONFIRMATION:
            raise ConfigError(
                "Real-money trading is locked. Set INV_LIVE_TRADING="
                f"{LIVE_CONFIRMATION} (or --i-understand-the-risks) only after "
                "backtests and several weeks of paper trading.")
        if self.broker == "trading212" and self.is_live and not self.t212_terms_acknowledged:
            raise ConfigError(
                "Trading 212's own API material forbids algorithmic trading with its AI "
                "tool (MiFID II). Read the API Terms, get Trading 212's written OK for "
                "automated orders on your account, then set INV_T212_TERMS_ACKNOWLEDGED=true.")

    def public_dict(self) -> dict:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in SECRET_ENV:
                value = bool(value)
            out[f.name] = value
        out.update({
            "venue_label": venue_label(self),
            "currency_symbol": "€",
            "exposure_label": "invested",
            "mode": self.mode,
            "max_open_positions": self.slots + (1 if self.etp_enabled else 0),
        })
        return out


def etp_symbol(s: InvestSettings) -> str:
    return f"ETP:{s.etp_underlying}x{s.etp_leverage:g}"


def venue_label(s: InvestSettings) -> str:
    names = {"sim": "Simulator (Trading 212 costs)", "trading212": "Trading 212",
             "ibkr": "Interactive Brokers", "alpaca": "Alpaca"}
    env = "" if s.broker == "sim" else f" · {s.environment}"
    return f"US stocks & ETPs · {names[s.broker]}{env}"


def _coerce(f, raw: str):
    default = f.default if f.default_factory is MISSING else f.default_factory()
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [x.strip().upper() for x in raw.split(",") if x.strip()]
    return raw.strip()
