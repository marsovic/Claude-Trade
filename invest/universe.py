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


# Trade themes: stocks that move together. The 3 slots must not become one bet.
_THEME_TEXT = {
    "ai_chips_hardware": "NVDA AVGO AMD QCOM TXN AMAT LRCX KLAC MU ADI MRVL INTC ARM ASML TSM "
                         "ANET SMCI DELL HPE WDC STX TER ON MCHP NXPI GFS ALAB CRDO COHR LITE "
                         "CIEN VRT CLS NBIS SNPS CDNS",
    "ai_power": "VST CEG GEV NRG TLN ETN PWR",
    "mega_tech": "AAPL MSFT AMZN META GOOGL TSLA NFLX ORCL",
    "software": "ADBE CRM NOW INTU PANW CRWD FTNT ZS DDOG NET SNOW MDB TEAM WDAY ADSK PLTR APP "
                "SHOP",
    "consumer_internet": "UBER ABNB DASH BKNG MELI PDD SPOT RBLX TTD TTWO EA AXON RDDT",
    "fintech": "HOOD AFRM SOFI XYZ PYPL",
    "financials": "GS MS KKR BX APO SCHW COF JPM IBKR",
    "industrial_defense_space": "URI CAT DE GE HWM RKLB ASTS BA LMT",
    "materials_clean": "FSLR ENPH FCX NEM",
    "consumer_travel": "CVNA LULU DECK ONON CELH DKNG CMG NKE RCL CCL DAL UAL ELF COST WMT",
    "healthcare": "LLY NVO VRTX REGN ISRG DXCM MRNA HIMS",
}
THEMES = {k: v.split() for k, v in _THEME_TEXT.items()}
_THEME_OF = {sym: theme for theme, syms in THEMES.items() for sym in syms}


def theme_of(symbol: str) -> str | None:
    """Theme of a symbol, None when unknown (such a stock is never capped)."""
    return _THEME_OF.get(symbol.upper())


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
