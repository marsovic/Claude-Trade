"""French tax helper: an EUR ledger of every fill and a yearly summary.

Foreign brokers (Trading 212, IBKR Ireland, Alpaca) send no French IFU, so
gains must be declared by hand. This ledger computes realised gains in EUR
with the weighted average purchase price (PMP, art. 150-0 D CGI), fees
included, and summarises them per year for form 2074 / boxes 3VG (gain) or
3VH (loss) of the 2042-C.

Reminders (2026 rules, verify with impots.gouv.fr - this is NOT tax advice):
* flat tax (PFU) 31.4 % = 12.8 % income tax + 18.6 % social levies
* losses offset same-nature gains of the year and the next 10 years
* every foreign account must be declared yearly on form 3916-bis
  (1 500 EUR fine per undeclared account)

    python -m invest.tax --state-dir state_invest --year 2026
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PFU_RATE_2026 = 0.314


class TaxLedger:
    def __init__(self, state_dir: str | Path):
        self.path = Path(state_dir) / "tax_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, day: str, symbol: str, side: str, qty: float, price: float,
               currency: str, usd_per_eur: float, gross_eur: float, fees_eur: float,
               cost_basis_eur: float = 0.0, gain_eur: float = 0.0, fill_id: str = "") -> None:
        if fill_id and fill_id in self._ids():
            return  # the same execution is never declared twice
        row = {"fill_id": fill_id, "day": day, "symbol": symbol, "side": side, "qty": qty,
               "price": price,
               "currency": currency, "usd_per_eur": usd_per_eur,
               "gross_eur": round(gross_eur, 6), "fees_eur": round(fees_eur, 6),
               "cost_basis_eur": round(cost_basis_eur, 6), "gain_eur": round(gain_eur, 6)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if fill_id:
            self._ids().add(fill_id)

    def _ids(self) -> set[str]:
        if getattr(self, "_seen", None) is None:
            self._seen = {r.get("fill_id") for r in self.rows() if r.get("fill_id")}
        return self._seen

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def summary(self, year: int) -> dict:
        sales = [r for r in self.rows() if r["side"] == "SELL" and r["day"].startswith(str(year))]
        buys = [r for r in self.rows() if r["side"] == "BUY" and r["day"].startswith(str(year))]
        gains = sum(r["gain_eur"] for r in sales if r["gain_eur"] > 0)
        losses = -sum(r["gain_eur"] for r in sales if r["gain_eur"] < 0)
        net = gains - losses
        per_symbol: dict[str, float] = {}
        for r in sales:
            per_symbol[r["symbol"]] = per_symbol.get(r["symbol"], 0.0) + r["gain_eur"]
        return {
            "year": year,
            "sales": len(sales),
            "buys": len(buys),
            "proceeds_eur": round(sum(r["gross_eur"] - r["fees_eur"] for r in sales), 2),
            "cost_basis_eur": round(sum(r["cost_basis_eur"] for r in sales), 2),
            "gains_eur": round(gains, 2),
            "losses_eur": round(losses, 2),
            "net_eur": round(net, 2),
            "box": "3VG" if net > 0 else "3VH",
            "estimated_pfu_eur": round(max(net, 0) * PFU_RATE_2026, 2),
            "fees_eur": round(sum(r["fees_eur"] for r in sales + buys), 2),
            "per_symbol": {k: round(v, 2) for k, v in sorted(per_symbol.items())},
            "reminder": "Declare the foreign broker account on form 3916-bis every year. "
                        "Not tax advice: check impots.gouv.fr.",
        }

    def export_csv(self, year: int, out: Path) -> Path:
        rows = [r for r in self.rows() if r["day"].startswith(str(year))]
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["day"])
            writer.writeheader()
            writer.writerows(rows)
        return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Yearly French tax summary of the invest bot")
    p.add_argument("--state-dir", default="state_invest")
    p.add_argument("--year", type=int, required=True)
    args = p.parse_args(argv)
    ledger = TaxLedger(args.state_dir)
    summary = ledger.summary(args.year)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = ledger.export_csv(args.year, Path(args.state_dir) / f"tax_{args.year}.csv")
    print(f"Detailed ledger: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
