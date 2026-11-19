"""
Selection-window study: treat the pair-selection rule as a hyperparameter.

Design (train / validation / test):

    2010 ........ 2020 | 2021  2022  2023 | 2024  2025  2026
      selection data   |    validation    |       test

1. VALIDATION: for each selection window S in {1, 3, 5, 10} years, select pairs
   using only data that ends at 2020-12-31. Trade those pairs in 2021-2023 under a
   grid of trading parameters (estimation window N, entry z, exit z, stop, max hold).
   Each trading year re-estimates alpha / beta / spread mean / spread std from the N
   years immediately before that year (walk-forward, no look-ahead).
2. CHOOSE one combination (S, N, entry, exit, stop, hold) using validation results only.
3. TEST: re-run the SAME selection rule (window length S) on data ending at
   2023-12-31, then trade 2024-2026 with the frozen parameters. Run it once.

What is frozen for the test: the rules / hyperparameters. What is NOT frozen: the
pairs (re-selected with data through 2023) and beta (re-estimated every year from
past data only). Both are part of the strategy, not overfitting.

Pair-selection fixes compared with earlier notebooks:
  * Engle-Granger assumes both legs are I(1). Each stock is pre-tested; a stock that
    already looks stationary in levels (range-bound) is excluded, so it cannot create
    "pairs" whose spread is just one stationary leg (e.g. beta ~ 0 cases).
  * Each ticker may appear in at most MAX_PAIRS_PER_TICKER selected pairs, so one
    "hub" stock cannot dominate the portfolio.
  * BH-FDR is applied to every tested pair in the window. If fewer than MIN_PAIRS
    pairs are BH-significant, the list is topped up with the smallest raw p-values and
    every such pair is labelled "Top-p fill (not BH-significant)" so the report stays
    honest about it.

Run from the project folder (recommended, so multiprocessing works on macOS):
    python selection_window_study.py
Outputs go to OUTPUT_DIR. Screening results are checkpointed, so a rerun skips
finished work. Delete the checkpoint files if you change selection settings.
"""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
DATA_FILE = Path("history_download_2010/raw_adjusted_close_2010_present.csv")
OUTPUT_DIR = Path("selection_window_study_outputs")

VALIDATION_CUTOFF = 2020            # selection data for validation ends 2020-12-31
VALIDATION_YEARS = [2021, 2022, 2023]
TEST_CUTOFF = 2023                  # selection data for test ends 2023-12-31
TEST_YEARS = [2024, 2025, 2026]

SELECTION_WINDOWS = [1, 3, 5, 10]   # years of data used for cointegration screening
MIN_COVERAGE = 0.95                 # a stock needs >= 95% non-missing days in the window

# Single-stock I(1) pre-test (ADF). Keep a stock only if its level is NOT stationary
# (p > I1_LEVEL_P) and its first difference IS stationary (p < I1_DIFF_P).
REQUIRE_I1 = True
I1_LEVEL_P = 0.10
I1_DIFF_P = 0.05

FDR_Q = 0.05
MIN_PAIRS = 5                       # top up with smallest raw p if BH gives fewer
TOP_K = 10                          # number of pairs to trade per selection window
MAX_PAIRS_PER_TICKER = 1

# Trading-parameter grid searched on VALIDATION only
ESTIMATION_WINDOWS = [1, 2, 3, 5]   # years of history used to fit alpha/beta/mu/sigma
ENTRY_Z = [1.5, 2.0, 2.5]
EXIT_Z = [0.0, 0.5]
STOP_Z = [None, 3.0]
MAX_HOLD = [None, 60]

TRANSACTION_COST = 0.001            # per side, as a fraction of notional
SLIPPAGE = 0.0
EXECUTION_LAG = 1
MIN_VALIDATION_TRADES = 10          # a config needs this many trades to be chosen
COST_SENSITIVITY = [0.0, 0.0005, 0.001, 0.002, 0.003]   # reported on test, not chosen

N_WORKERS = None                    # None = all CPU cores
CHUNK_SIZE = 500

# ---- Version-2 options (defaults reproduce version 1) ----
# Which pairs are tested: "all" (every pair), "sector" (same GICS sector) or
# "sub_industry" (same GICS sub-industry, e.g. two electric utilities).
PAIR_UNIVERSE = "all"
GICS_FILE = Path("sp500_gics.csv")  # Symbol|GICS Sector|GICS Sub-Industry (current classification)
# Also try "estimation window = selection window" (fit beta on the same span in which
# cointegration was verified). Adds e.g. a 10-year estimation window for the 10Y rule.
EST_SAME_AS_SELECTION = False
# Skip a pair-year when the fitted hedge is degenerate. Dollar hedge = beta * P_B / P_A at
# the end of the estimation window; it must lie in [MIN_DOLLAR_HEDGE, 1 / MIN_DOLLAR_HEDGE].
# None disables. Example: OXY-SWKS 2024 had beta = -0.04, i.e. trading OXY unhedged.
MIN_DOLLAR_HEDGE = None

# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------

def load_prices(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path.resolve()}")
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices.columns = prices.columns.astype(str).str.strip()
    prices = prices.apply(pd.to_numeric, errors="coerce").mask(lambda d: d <= 0)
    return prices


def window_slice(prices: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    return prices.loc[f"{first_year}-01-01":f"{last_year}-12-31"]


def load_gics(path: Path | None = None) -> pd.DataFrame:
    path = GICS_FILE if path is None else path
    if not path.exists():
        raise FileNotFoundError(f"GICS file not found: {path.resolve()}")
    g = pd.read_csv(path, sep="|").drop_duplicates("Symbol").set_index("Symbol")
    g.index = g.index.str.replace(".", "-", regex=False)   # BRK.B -> BRK-B (Yahoo style)
    return g


def universe_column() -> str | None:
    return {"all": None, "sector": "GICS Sector", "sub_industry": "GICS Sub-Industry"}[PAIR_UNIVERSE]


def candidate_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    pairs = list(combinations(tickers, 2))
    col = universe_column()
    if col is None:
        return pairs
    group = load_gics()[col]
    return [(a, b) for a, b in pairs if a in group.index and b in group.index and group[a] == group[b]]


# ----------------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------------

def adf_pvalue(series) -> float:
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(np.asarray(series, dtype=float), autolag="AIC")
    p = getattr(result, "pvalue", None)
    return float(result[1] if p is None else p)


def i1_filter(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in history.columns:
        s = history[ticker].dropna()
        try:
            p_level = adf_pvalue(s)
            p_diff = adf_pvalue(s.diff().dropna())
        except Exception:
            p_level, p_diff = np.nan, np.nan
        rows.append({
            "Ticker": ticker, "ADF p (level)": p_level, "ADF p (diff)": p_diff,
            "I(1)": bool(p_level > I1_LEVEL_P and p_diff < I1_DIFF_P),
        })
    return pd.DataFrame(rows)


# Worker state for multiprocessing (set once per worker process)
_VALUES = None
_COLUMN = None
_MIN_COMMON = None


def _init_worker(values, columns, min_common):
    global _VALUES, _COLUMN, _MIN_COMMON
    warnings.filterwarnings("ignore")
    _VALUES = values
    _COLUMN = {c: i for i, c in enumerate(columns)}
    _MIN_COMMON = min_common


def _coint_chunk(pairs):
    from statsmodels.tsa.stattools import coint
    out = []
    for a, b in pairs:
        x = _VALUES[:, _COLUMN[a]]
        y = _VALUES[:, _COLUMN[b]]
        ok = np.isfinite(x) & np.isfinite(y)
        n = int(ok.sum())
        if n < _MIN_COMMON or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            out.append((a, b, np.nan, np.nan, n, "insufficient_overlap"))
            continue
        try:
            stat, p, _ = coint(x[ok], y[ok], trend="c", autolag="aic")
            out.append((a, b, float(p), float(stat), n, "ok"))
        except Exception as exc:
            out.append((a, b, np.nan, np.nan, n, type(exc).__name__))
    return out


# ----------------------------------------------------------------------------------
# Step 1: pair screening for one (cutoff year, selection window)
# ----------------------------------------------------------------------------------

def screen(prices: pd.DataFrame, cutoff: int, sel_years: int) -> pd.DataFrame:
    """Engle-Granger test on all eligible pairs, with BH-FDR. Checkpointed."""
    from statsmodels.stats.multitest import multipletests

    OUTPUT_DIR.mkdir(exist_ok=True)
    tag = f"cut{cutoff}_{sel_years}Y"
    final_file = OUTPUT_DIR / f"screen_{tag}.csv"
    checkpoint = OUTPUT_DIR / f"screen_{tag}_checkpoint.csv"
    signature_file = OUTPUT_DIR / f"screen_{tag}_signature.json"

    history = window_slice(prices, cutoff - sel_years + 1, cutoff)
    coverage = history.notna().mean()
    tickers = sorted(coverage[coverage >= MIN_COVERAGE].index)

    i1_file = OUTPUT_DIR / f"i1_{tag}.csv"
    if REQUIRE_I1:
        if i1_file.exists():
            i1 = pd.read_csv(i1_file)
        else:
            i1 = i1_filter(history[tickers])
            i1.to_csv(i1_file, index=False)
        tickers = sorted(i1.loc[i1["I(1)"], "Ticker"])

    signature = {
        "cutoff": cutoff, "sel_years": sel_years, "coverage": MIN_COVERAGE,
        "require_i1": REQUIRE_I1, "i1_level_p": I1_LEVEL_P, "i1_diff_p": I1_DIFF_P,
        "tickers": tickers,
    }
    if PAIR_UNIVERSE != "all":
        signature["universe"] = PAIR_UNIVERSE
    if final_file.exists() and signature_file.exists():
        if json.loads(signature_file.read_text()) == signature:
            print(f"[{tag}] using cached screen: {final_file}")
            return pd.read_csv(final_file)
        raise ValueError(f"[{tag}] settings changed; delete {final_file}, "
                         f"{checkpoint} and {signature_file} and rerun")

    if checkpoint.exists():
        if not signature_file.exists() or json.loads(signature_file.read_text()) != signature:
            raise ValueError(f"[{tag}] checkpoint belongs to different settings; delete it")
        done = pd.read_csv(checkpoint)
    else:
        signature_file.write_text(json.dumps(signature))
        done = pd.DataFrame(columns=["Ticker A", "Ticker B", "P Value", "T Statistic",
                                     "N Common", "Status"])

    all_pairs = candidate_pairs(tickers)
    finished = set(zip(done["Ticker A"], done["Ticker B"]))
    todo = [p for p in all_pairs if p not in finished]
    print(f"[{tag}] window {cutoff - sel_years + 1}-{cutoff}: {len(history)} days, "
          f"{len(tickers)} tickers, universe={PAIR_UNIVERSE}, {len(all_pairs):,} pairs, "
          f"{len(todo):,} to test")

    if todo:
        values = history[tickers].to_numpy(dtype=float)
        min_common = int(len(history) * MIN_COVERAGE)
        chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
        start = time.monotonic()
        tested = len(finished)
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker,
                                 initargs=(values, tickers, min_common)) as pool:
            for rows in pool.map(_coint_chunk, chunks):
                chunk = pd.DataFrame(rows, columns=done.columns)
                chunk.to_csv(checkpoint, mode="a", header=not checkpoint.exists(), index=False)
                tested += len(chunk)
                print(f"[{tag}] {tested:,}/{len(all_pairs):,} "
                      f"elapsed {time.monotonic() - start:.0f}s", flush=True)

    if not checkpoint.exists():
        done.to_csv(checkpoint, index=False)
    result = pd.read_csv(checkpoint)
    if len(result) != len(all_pairs) or result.duplicated(["Ticker A", "Ticker B"]).any():
        raise RuntimeError(f"[{tag}] checkpoint incomplete or duplicated")
    ok = result["Status"].eq("ok") & result["P Value"].notna()
    p_all = np.where(ok, result["P Value"].fillna(1.0), 1.0)   # failed tests count as p=1
    reject, adjusted, _, _ = multipletests(p_all, alpha=FDR_Q, method="fdr_bh")
    result["BH Adjusted P"] = adjusted
    result["BH Significant"] = reject & ok.to_numpy()
    result = result.sort_values("P Value").reset_index(drop=True)
    result.to_csv(final_file, index=False)
    print(f"[{tag}] BH-significant pairs: {int(result['BH Significant'].sum())}")
    return result


def pick_pairs(screened: pd.DataFrame) -> pd.DataFrame:
    """BH-significant pairs first; top up with smallest raw p. Unique tickers."""
    usage: dict[str, int] = {}
    picked = []

    def try_add(row, basis):
        a, b = row["Ticker A"], row["Ticker B"]
        if usage.get(a, 0) >= MAX_PAIRS_PER_TICKER or usage.get(b, 0) >= MAX_PAIRS_PER_TICKER:
            return
        usage[a] = usage.get(a, 0) + 1
        usage[b] = usage.get(b, 0) + 1
        picked.append({**row.to_dict(), "Selection Basis": basis})

    valid = screened.loc[screened["P Value"].notna()].sort_values("P Value")
    for _, row in valid.loc[valid["BH Significant"]].iterrows():
        if len(picked) >= TOP_K:
            break
        try_add(row, "BH-significant")
    if len(picked) < MIN_PAIRS:
        for _, row in valid.loc[~valid["BH Significant"]].iterrows():
            if len(picked) >= TOP_K:
                break
            try_add(row, "Top-p fill (not BH-significant)")
    out = pd.DataFrame(picked)
    if len(out) and GICS_FILE.exists():
        g = load_gics()
        for side in ("A", "B"):
            out[f"Sector {side}"] = out[f"Ticker {side}"].map(g["GICS Sector"])
            out[f"Sub-Industry {side}"] = out[f"Ticker {side}"].map(g["GICS Sub-Industry"])
    return out


# ----------------------------------------------------------------------------------
# Step 2: walk-forward trading of a fixed pair list
# ----------------------------------------------------------------------------------

def dollar_hedge(train: pd.DataFrame, a: str, b: str) -> float:
    """beta * P_B / P_A at the last training day (scale-free size of the hedge leg)."""
    x = train[b].to_numpy(dtype=float)
    y = train[a].to_numpy(dtype=float)
    beta = np.linalg.lstsq(np.c_[np.ones_like(x), x], y, rcond=None)[0][1]
    return float(beta * x[-1] / y[-1])


def run_strategy(prices, pairs, years, est_years, entry, exit_, stop, hold, cost=None):
    """Each year: fit on the est_years before it, trade that year. Equal-weight pairs.

    A pair's daily return = daily net P&L / that pair-year's starting capital
    (|A| + |beta| * B at the first test day, as defined in the engine).
    """
    from pairs_backtest_engine import backtest_pair

    cost = TRANSACTION_COST if cost is None else cost
    returns, trades, pair_rows = [], [], []
    for year in years:
        test_all = window_slice(prices, year, year)
        train_all = window_slice(prices, year - est_years, year - 1)
        for a, b in pairs:
            tr = train_all[[a, b]].dropna()
            ts = test_all[[a, b]].dropna()
            if len(tr) < max(60, int(0.8 * 252 * est_years)) or len(ts) < 20:
                pair_rows.append({"Year": year, "Pair": f"{a}-{b}", "Status": "skipped: data"})
                continue
            if MIN_DOLLAR_HEDGE is not None:
                h = dollar_hedge(tr, a, b)
                if not (MIN_DOLLAR_HEDGE <= h <= 1 / MIN_DOLLAR_HEDGE):
                    pair_rows.append({"Year": year, "Pair": f"{a}-{b}",
                                      "Status": f"skipped: weak hedge ({h:.2f})"})
                    continue
            cfg = {
                "ticker_a": a, "ticker_b": b, "entry_z": entry, "exit_z": exit_,
                "stop_loss_z": stop, "max_holding_days": hold, "hedge_lookback": None,
                "transaction_cost": cost, "slippage": SLIPPAGE,
                "execution_lag": EXECUTION_LAG, "trading_days": 252,
            }
            try:
                res = backtest_pair(cfg, tr, ts)
            except ValueError as exc:
                pair_rows.append({"Year": year, "Pair": f"{a}-{b}", "Status": f"skipped: {exc}"})
                continue
            d = res["daily_results"]
            capital = float(d["Equity"].iloc[0] - d["Net P&L"].iloc[0])
            returns.append(pd.DataFrame({"Date": d.index, "Pair": f"{a}-{b}",
                                         "Return": d["Net P&L"].to_numpy() / capital}))
            log = res["trade_log"]
            if len(log):
                log = log.assign(Year=year, Pair=f"{a}-{b}",
                                 **{"Contribution (%)": log["Net P&L"] / capital * 100})
                trades.append(log)
            m = res["metrics"]
            pair_rows.append({"Year": year, "Pair": f"{a}-{b}", "Status": "ok",
                              "Return (%)": m["Total Return (%)"], "Trades": m["Trades"],
                              "Turnover": m["Turnover"],
                              "Min Equity / Capital": float(d["Equity"].min() / capital)})

    if returns:
        long = pd.concat(returns, ignore_index=True)
        wide = long.pivot_table(index="Date", columns="Pair", values="Return", aggfunc="sum")
        portfolio = wide.mean(axis=1, skipna=True)
    else:
        portfolio = pd.Series(dtype=float)
    trade_log = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return portfolio, trade_log, pd.DataFrame(pair_rows)


def summarize(portfolio: pd.Series, trade_log: pd.DataFrame, pair_rows: pd.DataFrame) -> dict:
    if portfolio.empty:
        return {"Trades": 0}
    equity = (1 + portfolio).cumprod()
    years = len(portfolio) / 252
    sd = portfolio.std(ddof=1)
    n_trades = len(trade_log)
    contrib = trade_log["Contribution (%)"] if n_trades else pd.Series(dtype=float)
    total_contrib = contrib.sum()
    top5 = contrib.sort_values(ascending=False).head(5).sum()
    ok_rows = pair_rows.loc[pair_rows["Status"].eq("ok")] if len(pair_rows) else pair_rows
    return {
        "Total Return (%)": (equity.iloc[-1] - 1) * 100,
        "Annualized Return (%)": (equity.iloc[-1] ** (1 / years) - 1) * 100 if equity.iloc[-1] > 0 else np.nan,
        "Volatility (%)": sd * np.sqrt(252) * 100,
        "Sharpe": portfolio.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "Max Drawdown (%)": float((equity / equity.cummax() - 1).min() * 100),
        "Trades": n_trades,
        "Win Rate (%)": float((trade_log["Net P&L"] > 0).mean() * 100) if n_trades else np.nan,
        "Avg Trade Return (%)": float(trade_log["Trade Return (%)"].mean()) if n_trades else np.nan,
        "Avg Annual Turnover": float(ok_rows["Turnover"].mean()) if len(ok_rows) else np.nan,
        "Top-5 Trades Share of P&L (%)": float(top5 / total_contrib * 100) if total_contrib > 0 else np.nan,
        "Pair-Years Traded": int(len(ok_rows)),
    }


def grid_combos() -> list[tuple]:
    """(selection window, estimation window, entry, exit, stop, hold) for the validation grid."""
    combos = []
    for s in SELECTION_WINDOWS:
        est = list(ESTIMATION_WINDOWS)
        if EST_SAME_AS_SELECTION and s not in est:
            est.append(s)
        combos += [(s, *c) for c in product(est, ENTRY_Z, EXIT_Z, STOP_Z, MAX_HOLD)]
    return combos


def validation_grid(prices, val_pairs: dict, years=None, cost=None, every=50) -> pd.DataFrame:
    years = VALIDATION_YEARS if years is None else years
    combos = grid_combos()
    rows, start = [], time.monotonic()
    for k, (s, n, entry, exit_, stop, hold) in enumerate(combos, start=1):
        if val_pairs.get(s):
            port, log, pr = run_strategy(prices, val_pairs[s], years, n, entry, exit_, stop, hold, cost=cost)
            rows.append({"Selection Window": s, "Estimation Window": n, "Entry Z": entry,
                         "Exit Z": exit_, "Stop Z": stop, "Max Hold": hold,
                         **summarize(port, log, pr)})
        if every and (k % every == 0 or k == len(combos)):
            print(f"Validation grid {k}/{len(combos)} elapsed {time.monotonic() - start:.0f}s", flush=True)
    return pd.DataFrame(rows)


def choose_params(validation: pd.DataFrame, min_trades=None) -> dict:
    min_trades = MIN_VALIDATION_TRADES if min_trades is None else min_trades
    eligible = validation.loc[validation["Trades"] >= min_trades].dropna(subset=["Sharpe"])
    if eligible.empty:
        raise RuntimeError("No validation config has enough trades; relax MIN_VALIDATION_TRADES")
    best = eligible.sort_values(["Sharpe", "Annualized Return (%)"], ascending=False).iloc[0]
    return dict(
        s=int(best["Selection Window"]), n=int(best["Estimation Window"]),
        entry=float(best["Entry Z"]), exit_=float(best["Exit Z"]),
        stop=None if pd.isna(best["Stop Z"]) else float(best["Stop Z"]),
        hold=None if pd.isna(best["Max Hold"]) else int(best["Max Hold"]),
    )


# ----------------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------------

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)
    prices = load_prices()
    print("Prices:", prices.shape, prices.index.min().date(), "to", prices.index.max().date())

    # ---------- Validation: select with data through 2020, trade 2021-2023 ----------
    val_selected = {}
    for s in SELECTION_WINDOWS:
        chosen = pick_pairs(screen(prices, VALIDATION_CUTOFF, s))
        chosen.to_csv(OUTPUT_DIR / f"validation_pairs_{s}Y.csv", index=False)
        val_selected[s] = chosen
        n_bh = int((chosen["Selection Basis"] == "BH-significant").sum()) if len(chosen) else 0
        print(f"Validation {s}Y: {len(chosen)} pairs ({n_bh} BH-significant)")

    val_pairs = {s: list(zip(df["Ticker A"], df["Ticker B"])) if len(df) else []
                 for s, df in val_selected.items()}
    validation = validation_grid(prices, val_pairs)
    validation.to_csv(OUTPUT_DIR / "validation_grid.csv", index=False)

    # Robust view: how does each selection window do across ALL trading parameters?
    by_window = validation.groupby("Selection Window").agg(
        Configs=("Sharpe", "size"),
        Median_Sharpe=("Sharpe", "median"),
        Share_Sharpe_Positive=("Sharpe", lambda x: float((x > 0).mean())),
        Median_Ann_Return=("Annualized Return (%)", "median"),
        Median_Trades=("Trades", "median"),
    ).reset_index()
    by_window.to_csv(OUTPUT_DIR / "validation_by_selection_window.csv", index=False)
    print("\nValidation summary by selection window (all parameter combos):")
    print(by_window.round(3).to_string(index=False))

    params = choose_params(validation)
    (OUTPUT_DIR / "chosen_parameters.json").write_text(json.dumps(params, indent=2))
    print("\nChosen on validation (frozen for test):", params)

    # Entry/exit sensitivity heatmap on validation, other parameters at chosen values
    sens = validation.loc[
        (validation["Selection Window"] == params["s"])
        & (validation["Estimation Window"] == params["n"])
        & (validation["Stop Z"].fillna(-1) == (params["stop"] if params["stop"] is not None else -1))
        & (validation["Max Hold"].fillna(-1) == (params["hold"] if params["hold"] is not None else -1))
    ].pivot(index="Entry Z", columns="Exit Z", values="Sharpe")
    sens.to_csv(OUTPUT_DIR / "validation_entry_exit_sharpe.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sens.to_numpy(dtype=float), cmap="RdBu", vmin=-2, vmax=2)
    ax.set_xticks(range(len(sens.columns)), [str(c) for c in sens.columns])
    ax.set_yticks(range(len(sens.index)), [str(i) for i in sens.index])
    for i in range(sens.shape[0]):
        for j in range(sens.shape[1]):
            ax.text(j, i, f"{sens.iat[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Exit z"); ax.set_ylabel("Entry z")
    ax.set_title("Validation Sharpe (2021-2023)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_entry_exit_heatmap.png", dpi=150); plt.close(fig)

    # ---------- Test: same rule, data through 2023, trade 2024-2026 once ----------
    test_pairs_df = pick_pairs(screen(prices, TEST_CUTOFF, params["s"]))
    test_pairs_df.to_csv(OUTPUT_DIR / "test_pairs.csv", index=False)
    test_pairs = list(zip(test_pairs_df["Ticker A"], test_pairs_df["Ticker B"]))
    print(f"\nTest pairs ({params['s']}Y window ending {TEST_CUTOFF}):")
    print(test_pairs_df[["Ticker A", "Ticker B", "P Value", "BH Adjusted P", "Selection Basis"]].to_string(index=False))

    val_pairs = list(zip(val_selected[params["s"]]["Ticker A"], val_selected[params["s"]]["Ticker B"]))
    run_args = (params["n"], params["entry"], params["exit_"], params["stop"], params["hold"])
    val_port, val_log, val_pr = run_strategy(prices, val_pairs, VALIDATION_YEARS, *run_args)
    test_port, test_log, test_pr = run_strategy(prices, test_pairs, TEST_YEARS, *run_args)
    test_log.to_csv(OUTPUT_DIR / "test_trade_log.csv", index=False)
    test_pr.to_csv(OUTPUT_DIR / "test_pair_years.csv", index=False)

    is_oos = pd.DataFrame([
        {"Period": f"Validation {VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]} (in-sample for parameter choice)",
         **summarize(val_port, val_log, val_pr)},
        {"Period": f"Test {TEST_YEARS[0]}-{TEST_YEARS[-1]} (out-of-sample)",
         **summarize(test_port, test_log, test_pr)},
    ])
    is_oos.to_csv(OUTPUT_DIR / "in_sample_vs_out_of_sample.csv", index=False)
    print("\nIn-sample vs out-of-sample:")
    print(is_oos.round(3).T.to_string())

    # Per-year test results
    yearly = test_port.groupby(test_port.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    yearly.rename("Return (%)").to_csv(OUTPUT_DIR / "test_yearly_returns.csv")
    print("\nTest return by year (%):"); print(yearly.round(2).to_string())

    # Equity curve: validation then test
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot((1 + val_port).cumprod(), label="Validation 2021-2023")
    ax.plot((1 + test_port).cumprod(), label="Test 2024-2026 (out-of-sample)")
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_title(f"Equal-weight pairs portfolio, selection {params['s']}Y, estimation {params['n']}Y")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "portfolio_equity_curve.png", dpi=150); plt.close(fig)

    # Transaction-cost sensitivity on test (reporting only; nothing is chosen here)
    cost_rows = []
    for c in COST_SENSITIVITY:
        p, l, r = run_strategy(prices, test_pairs, TEST_YEARS, *run_args, cost=c)
        cost_rows.append({"Cost per Side (bp)": c * 1e4, **summarize(p, l, r)})
    cost_table = pd.DataFrame(cost_rows)
    cost_table.to_csv(OUTPUT_DIR / "test_cost_sensitivity.csv", index=False)
    print("\nTest cost sensitivity:")
    print(cost_table[["Cost per Side (bp)", "Annualized Return (%)", "Sharpe", "Trades"]].round(3).to_string(index=False))

    # Do the test pairs stay cointegrated in the test period itself?
    from statsmodels.tsa.stattools import coint
    persist = []
    test_window = window_slice(prices, TEST_YEARS[0], TEST_YEARS[-1])
    for a, b in test_pairs:
        d = test_window[[a, b]].dropna()
        p = float(coint(d[a], d[b], trend="c", autolag="aic")[1]) if len(d) > 100 else np.nan
        persist.append({"Pair": f"{a}-{b}", "Test-period EG p-value": p, "Still p<0.05": p < 0.05})
    persist = pd.DataFrame(persist)
    persist.to_csv(OUTPUT_DIR / "test_cointegration_persistence.csv", index=False)
    print("\nCointegration persistence in test period:"); print(persist.round(4).to_string(index=False))
    print(f"\nAll outputs saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
