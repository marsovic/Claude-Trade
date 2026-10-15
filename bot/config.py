"""Runtime configuration.

Every setting can be overridden with an environment variable of the same name
in upper case (a ``.env`` file in the working directory is loaded first).
Defaults are deliberately conservative: paper trading, 2 % risk per trade,
leverage capped at x5.
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

try:  # python-dotenv is optional at import time (tests do not need it)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

MODES = ("paper", "testnet", "live")
LIVE_CONFIRMATION = "I_UNDERSTAND_THE_RISKS"

FUTURES_BASE_URLS = {
    # Paper trading reads real market data but never sends an order.
    "paper": "https://fapi.binance.com",
    "testnet": "https://testnet.binancefuture.com",
    "live": "https://fapi.binance.com",
}
SPOT_BASE_URL = "https://api.binance.com"

# Interval string -> seconds, for the Binance kline intervals we support.
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


class ConfigError(ValueError):
    """Raised when the configuration is invalid or unsafe."""


def _csv(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    # --- Mode & capital -------------------------------------------------
    mode: str = "paper"
    initial_capital: float = 50.0
    # Equity at or below this level means the capital is considered lost:
    # the bot closes everything, writes a tombstone and never trades again.
    survival_floor: float = 5.0
    state_dir: str = "state"

    # --- Universe ---------------------------------------------------------
    universe_size: int = 40
    quote_asset: str = "USDT"
    exclude_symbols: list[str] = field(
        default_factory=lambda: ["USDCUSDT", "FDUSDUSDT", "BTCDOMUSDT", "DEFIUSDT"]
    )
    universe_refresh_minutes: int = 60

    # --- Market data --------------------------------------------------------
    interval: str = "15m"
    kline_limit: int = 300
    poll_seconds: int = 20
    scan_delay_seconds: int = 5
    max_workers: int = 6

    # --- Strategy: the 7 conditions -----------------------------------------
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    volume_period: int = 20
    volume_factor: float = 1.3
    atr_period: int = 14
    atr_min_pct: float = 0.15  # % of price; below this the market is too quiet
    atr_max_pct: float = 3.0  # % of price; above this it is too wild

    # --- Risk -------------------------------------------------------------
    risk_per_trade: float = 0.02
    max_leverage: int = 5
    # Leverage = leverage_vol_target / ATR%, clamped to [1, max_leverage].
    leverage_vol_target: float = 3.0
    stop_atr_mult: float = 1.5
    take_profit_r: float = 2.0
    breakeven_r: float = 1.0
    trailing: bool = True
    trail_start_r: float = 1.5
    trail_atr_mult: float = 1.5
    max_open_positions: int = 3
    max_margin_per_trade: float = 0.35  # fraction of equity
    max_total_margin: float = 0.90  # fraction of equity
    max_daily_loss: float = 0.06  # fraction of the day's starting equity
    cooldown_bars: int = 4
    max_hold_bars: int = 96
    maintenance_margin_rate: float = 0.01  # conservative approximation
    fee_rate: float = 0.0005  # Binance futures taker fee
    slippage_bps: float = 2.0

    # --- Sentiment ----------------------------------------------------------
    sentiment_enabled: bool = True
    sentiment_veto: float = 0.25
    sentiment_cache_minutes: int = 15
    weight_fear_greed: float = 1.0
    weight_trending: float = 0.5
    weight_cryptopanic: float = 1.0
    weight_x: float = 1.0
    cryptopanic_token: str = ""
    cryptopanic_url: str = "https://cryptopanic.com/api/developer/v2/posts/"
    x_bearer_token: str = ""
    x_api_url: str = "https://api.x.com/2/tweets/search/recent"
    coingecko_api_key: str = ""

    # --- Exchange credentials -------------------------------------------------
    binance_api_key: str = ""
    binance_api_secret: str = ""
    futures_base_url: str = ""
    spot_base_url: str = SPOT_BASE_URL
    live_trading: str = ""  # must equal LIVE_CONFIRMATION for live mode
    margin_type: str = "ISOLATED"

    # --------------------------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: str | None = ".env", **overrides) -> Settings:
        if env_file and load_dotenv is not None and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        values: dict = {}
        for f in fields(cls):
            raw = os.environ.get(f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f, raw)
        values.update({k: v for k, v in overrides.items() if v is not None})
        settings = cls(**values)
        settings.validate()
        return settings

    @property
    def interval_seconds(self) -> int:
        return INTERVAL_SECONDS[self.interval]

    @property
    def base_url(self) -> str:
        return self.futures_base_url or FUTURES_BASE_URLS[self.mode]

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    def validate(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in MODES:
            raise ConfigError(f"MODE must be one of {MODES}, got {self.mode!r}")
        if self.interval not in INTERVAL_SECONDS:
            raise ConfigError(f"Unsupported INTERVAL {self.interval!r}")
        if not 0 < self.risk_per_trade <= 0.05:
            raise ConfigError("RISK_PER_TRADE must be in (0, 0.05]")
        if not 1 <= self.max_leverage <= 20:
            raise ConfigError("MAX_LEVERAGE must be between 1 and 20")
        if self.initial_capital <= 0:
            raise ConfigError("INITIAL_CAPITAL must be positive")
        if not 0 <= self.survival_floor < self.initial_capital:
            raise ConfigError("SURVIVAL_FLOOR must be in [0, INITIAL_CAPITAL)")
        if self.ema_fast >= self.ema_slow:
            raise ConfigError("EMA_FAST must be smaller than EMA_SLOW")
        if self.kline_limit < self.ema_trend + 10:
            raise ConfigError("KLINE_LIMIT must exceed EMA_TREND by at least 10 bars")
        if self.stop_atr_mult <= 0 or self.take_profit_r <= 0:
            raise ConfigError("STOP_ATR_MULT and TAKE_PROFIT_R must be positive")
        if self.margin_type not in ("ISOLATED", "CROSSED"):
            raise ConfigError("MARGIN_TYPE must be ISOLATED or CROSSED")
        if self.mode in ("testnet", "live") and not (
                self.binance_api_key and self.binance_api_secret):
            raise ConfigError(f"{self.mode} mode needs BINANCE_API_KEY and BINANCE_API_SECRET")
        if self.mode == "live" and self.live_trading != LIVE_CONFIRMATION:
            raise ConfigError(
                "Live trading is locked. Set LIVE_TRADING="
                f"{LIVE_CONFIRMATION} (or pass --i-understand-the-risks) once you "
                "have backtested and paper traded for several weeks."
            )

    def public_dict(self) -> dict:
        """Settings without secrets, safe to show in the dashboard."""
        hidden = {"binance_api_key", "binance_api_secret", "cryptopanic_token",
                  "x_bearer_token", "coingecko_api_key"}
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in hidden:
                value = bool(value)
            out[f.name] = value
        return out


def _coerce(f, raw: str):
    default = f.default if f.default_factory is MISSING else f.default_factory()
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return _csv(raw)
    return raw.strip()
