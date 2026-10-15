"""Tradable universe: liquid, high-beta US stocks (no crypto proxies).

US single stocks are not PRIIPs, so EU retail investors can buy them at any
broker; US-domiciled ETFs (SPY, QQQ, TQQQ...) cannot be bought by EU retail
investors (no KID) and are used here ONLY as signals (regime, ETP underlying).

The curated seed list mixes Nasdaq-100 members and high-beta large/mid caps.
It is filtered every day on data (price, dollar volume, volatility, beta,
history), so stale names drop out by themselves. Membership is today's: a
backtest on it carries survivorship bias (documented in the README).
"""

from __future__ import annotations

CURATED_US = [
    # Mega / large cap tech & communication
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "NFLX", "AMD",
    "ADBE", "CRM", "ORCL", "NOW", "INTU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC",
    "MU", "ADI", "MRVL", "INTC", "SNPS", "CDNS", "PANW", "CRWD", "FTNT", "ZS",
    "DDOG", "NET", "SNOW", "MDB", "TEAM", "WDAY", "ADSK", "PLTR", "APP", "SHOP",
    "UBER", "ABNB", "DASH", "BKNG", "MELI", "PDD", "SPOT", "RBLX", "TTD", "TTWO",
    "EA", "ARM", "ASML", "TSM", "ANET", "SMCI", "DELL", "HPE", "WDC", "STX",
    "TER", "ON", "MCHP", "NXPI", "GFS", "ALAB", "CRDO", "COHR", "LITE", "CIEN",
    "VRT", "CLS", "NBIS", "AXON", "RDDT", "HOOD", "AFRM", "SOFI", "XYZ", "PYPL",
    # Energy transition, power & industrial momentum
    "VST", "CEG", "GEV", "NRG", "TLN", "ETN", "PWR", "URI", "CAT", "DE",
    "GE", "HWM", "RKLB", "ASTS", "FSLR", "ENPH", "FCX", "NEM", "BA", "LMT",
    # Financials with high beta
    "GS", "MS", "KKR", "BX", "APO", "SCHW", "COF", "JPM", "IBKR", "CVNA",
    # Consumer & healthcare high beta
    "LULU", "DECK", "ONON", "CELH", "DKNG", "CMG", "NKE", "RCL", "CCL", "DAL",
    "UAL", "LLY", "NVO", "VRTX", "REGN", "ISRG", "DXCM", "MRNA", "HIMS", "ELF",
    # Large consumer names (usually filtered out by the beta >= 1 rule)
    "COST", "WMT",
]

# Crypto proxies are excluded on purpose: the user wants no crypto exposure.
CRYPTO_PROXIES = {"MSTR", "COIN", "MARA", "RIOT", "CLSK", "HUT", "BITF", "IREN",
                  "CORZ", "GLXY", "BTBT", "CIFR", "WULF", "BMNR", "SBET"}

# US ETFs used only as signals (never traded: PRIIPs).
SIGNAL_ONLY = {"SPY", "QQQ", "SMH", "IWM"}


def build_universe(custom: list[str] | None = None, exclude: list[str] | None = None
                   ) -> list[str]:
    base = list(custom) if custom else list(CURATED_US)
    banned = CRYPTO_PROXIES | SIGNAL_ONLY | {x.upper() for x in (exclude or [])}
    seen, out = set(), []
    for symbol in base:
        sym = symbol.strip().upper()
        if sym and sym not in banned and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def data_symbols(universe: list[str], benchmark: str, etp_underlying: str | None) -> list[str]:
    """Everything we need bars for: the universe plus signal-only symbols."""
    extra = [benchmark] + ([etp_underlying] if etp_underlying else [])
    return list(dict.fromkeys(universe + [x.upper() for x in extra]))
