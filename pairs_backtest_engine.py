"""Shared, date-safe pairs-trading backtest engine.

The engine estimates parameters from model_train, then simulates trades on
backtest_data with signals at close and execution at a later close. It does not
perform cointegration pair discovery; that is an upstream training-only step.

Permanent re-entry rule: after any protective close (stop loss or maximum
holding period), stay flat until z crosses the zero/mean level from the
stopped side. A subsequent new threshold excursion may then open a trade.
Natural mean-reversion closes do not impose this extra lock.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def backtest_pair(config, train_data, test_data):
    a = config["ticker_a"]
    b = config["ticker_b"]
    entry_z = float(config["entry_z"])
    exit_z = float(config["exit_z"])
    max_days = config.get("max_holding_days")
    stop_z = config.get("stop_loss_z")
    lookback = config.get("hedge_lookback")
    cost_rate = float(config.get("transaction_cost", 0)) + float(config.get("slippage", 0))
    lag = config.get("execution_lag", 1)
    trading_days = float(config.get("trading_days", 252))

    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("execution_lag must be an integer >= 1")
    if entry_z <= exit_z or exit_z < 0:
        raise ValueError("Require entry_z > exit_z >= 0")
    if stop_z is not None and stop_z <= entry_z:
        raise ValueError("stop_loss_z must exceed entry_z")
    if max_days is not None and (not isinstance(max_days, int) or max_days < lag + 1):
        raise ValueError("max_holding_days must be an integer >= execution_lag + 1")
    if lookback is not None and (not isinstance(lookback, int) or lookback < 30):
        raise ValueError("hedge_lookback must be None or an integer >= 30")
    if cost_rate < 0 or trading_days <= 0:
        raise ValueError("Costs must be nonnegative and trading_days positive")

    training = train_data[[a, b]].copy().sort_index()
    testing = test_data[[a, b]].copy().sort_index()
    if len(training) < 30 or len(testing) < 3:
        raise ValueError("Need >= 30 training and >= 3 backtest observations")
    if not training.index.is_unique or not testing.index.is_unique:
        raise ValueError("Dates must be unique")
    if training.index.max() >= testing.index.min():
        raise ValueError("Training dates must all precede backtest dates")
    if training.isna().any().any() or testing.isna().any().any():
        raise ValueError("Price data must not contain missing values")
    if not np.isfinite(training.to_numpy()).all() or not np.isfinite(testing.to_numpy()).all():
        raise ValueError("Price data must be finite")
    if (training <= 0).any().any() or (testing <= 0).any().any():
        raise ValueError("Prices must be positive")

    train_fit = sm.OLS(training[a], sm.add_constant(training[b])).fit()
    alpha0 = float(train_fit.params["const"])
    beta0 = float(train_fit.params[b])
    train_spread = training[a] - alpha0 - beta0 * training[b]
    mu0 = float(train_spread.mean())
    sd0 = float(train_spread.std())
    if sd0 <= 0 or not np.isfinite(sd0):
        raise ValueError("Training spread variance must be positive")

    y = testing[a]
    x = testing[b]
    z_values = []
    beta_values = []
    if lookback is None:
        z_values = ((y - alpha0 - beta0 * x - mu0) / sd0).tolist()
        beta_values = [beta0] * len(testing)
    else:
        history = pd.concat([training, testing]).sort_index()
        for date in testing.index:
            window = history.loc[:date].tail(lookback)
            if len(window) < 30 or window[b].std() == 0:
                z_values.append(np.nan)
                beta_values.append(np.nan)
                continue
            m = sm.OLS(window[a], sm.add_constant(window[b])).fit()
            alpha = float(m.params["const"])
            beta = float(m.params[b])
            r = window[a] - alpha - beta * window[b]
            sd = float(r.std())
            z_values.append(float((r.iloc[-1] - r.mean()) / sd) if sd > 0 else np.nan)
            beta_values.append(beta)

    z = pd.Series(z_values, index=testing.index)
    betas = pd.Series(beta_values, index=testing.index)
    dates = testing.index
    n = len(testing)
    capital = float(y.iloc[0] + abs(beta0) * x.iloc[0])
    pos = 0
    pending = None
    entry_i = None
    entry_beta = None
    entry_cost = 0.0
    entry_notional = None
    gross = np.zeros(n)
    costs = np.zeros(n)
    positions = np.zeros(n, dtype=int)
    actual_betas = np.zeros(n)
    trades = []
    reentry_locked = False
    locked_direction = 0
    reentry_unlocks = []

    for i, date in enumerate(dates):
        if i > 0 and pos != 0:
            gross[i] = pos * (
                (float(y.iloc[i]) - float(y.iloc[i - 1]))
                - entry_beta * (float(x.iloc[i]) - float(x.iloc[i - 1]))
            )

        executed = False
        close_reason = None
        if pending is not None and pending[0] == i:
            if pending[1] == "open":
                pos = pending[2]
                entry_i = i
                entry_beta = pending[3]
                entry_notional = float(y.iloc[i] + abs(entry_beta) * x.iloc[i])
                entry_cost = entry_notional * cost_rate
                costs[i] += entry_cost
            else:
                close_reason = pending[2]
            pending = None
            executed = True

        if i == n - 1 and pos != 0:
            close_reason = "End of backtest"

        if close_reason is not None and pos != 0:
            exit_notional = float(y.iloc[i] + abs(entry_beta) * x.iloc[i])
            exit_cost = exit_notional * cost_rate
            costs[i] += exit_cost
            trade_gross = pos * (
                (float(y.iloc[i]) - float(y.iloc[entry_i]))
                - entry_beta * (float(x.iloc[i]) - float(x.iloc[entry_i]))
            )
            trade_net = trade_gross - entry_cost - exit_cost
            trades.append({
                "Entry Date": dates[entry_i],
                "Exit Date": date,
                "Direction": pos,
                "Hedge Ratio": entry_beta,
                "Holding Days": i - entry_i,
                "Gross P&L": trade_gross,
                "Transaction Cost": entry_cost + exit_cost,
                "Net P&L": trade_net,
                "Trade Return (%)": trade_net / entry_notional * 100,
                "Exit Reason": close_reason,
            })
            if close_reason in ("Stop loss", "Max holding"):
                reentry_locked = True
                locked_direction = pos
            pos = 0
            entry_i = None
            entry_beta = None
        elif not executed and pending is None and i + lag < n:
            zi = float(z.iloc[i])
            if pos == 0:
                if reentry_locked:
                    if np.isfinite(zi) and (
                        (locked_direction == 1 and zi >= 0)
                        or (locked_direction == -1 and zi <= 0)
                    ):
                        reentry_locked = False
                        locked_direction = 0
                        reentry_unlocks.append(date)
                elif i + lag < n - 1 and np.isfinite(zi) and np.isfinite(betas.iloc[i]):
                    if zi < -entry_z:
                        pending = (i + lag, "open", 1, float(betas.iloc[i]))
                    elif zi > entry_z:
                        pending = (i + lag, "open", -1, float(betas.iloc[i]))
            elif np.isfinite(zi):
                reason = None
                if pos == 1 and zi >= -exit_z:
                    reason = "Mean reversion"
                elif pos == -1 and zi <= exit_z:
                    reason = "Mean reversion"
                if stop_z is not None:
                    if pos == 1 and zi <= -stop_z:
                        reason = "Stop loss"
                    elif pos == -1 and zi >= stop_z:
                        reason = "Stop loss"
                if max_days is not None and i - entry_i + lag >= max_days:
                    if reason is None:
                        reason = "Max holding"
                if reason is not None:
                    pending = (i + lag, "close", reason)
            elif pos != 0 and max_days is not None and i - entry_i + lag >= max_days:
                pending = (i + lag, "close", "Max holding")

        positions[i] = pos
        actual_betas[i] = entry_beta if pos else 0

    net = gross - costs
    equity = capital + np.cumsum(net)
    daily_return = np.full(n, np.nan)
    previous_equity = np.r_[capital, equity[:-1]]
    valid = previous_equity > 0
    daily_return[valid] = net[valid] / previous_equity[valid]
    if np.any(equity <= 0):
        daily_return[:] = np.nan

    max_dd = float(np.min(equity / np.maximum.accumulate(np.r_[capital, equity])[1:] - 1))
    total_return = float(equity[-1] / capital - 1)
    annual_return = ((equity[-1] / capital) ** (trading_days / (n - 1)) - 1) if equity[-1] > 0 else np.nan
    vol_daily = float(np.std(daily_return, ddof=1)) if np.isfinite(daily_return).all() else np.nan
    vol = vol_daily * np.sqrt(trading_days)
    sharpe = float(np.mean(daily_return) / vol_daily * np.sqrt(trading_days)) if vol_daily > 0 else np.nan
    trade_log = pd.DataFrame(trades)
    win_rate = float((trade_log["Net P&L"] > 0).mean() * 100) if trades else np.nan
    avg_holding = float(trade_log["Holding Days"].mean()) if trades else np.nan
    avg_trade_return = float(trade_log["Trade Return (%)"].mean()) if trades else np.nan
    turnover = float(sum(
        (float(y.loc[row["Entry Date"]]) + abs(row["Hedge Ratio"]) * float(x.loc[row["Entry Date"]]))
        + (float(y.loc[row["Exit Date"]]) + abs(row["Hedge Ratio"]) * float(x.loc[row["Exit Date"]]))
        for row in trades
    ) / capital)
    metrics = {
        "Pair": f"{a}-{b}",
        "Total Return (%)": total_return * 100,
        "Annualized Return (%)": annual_return * 100,
        "Volatility (%)": vol * 100,
        "Sharpe": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Trades": len(trades),
        "Win Rate (%)": win_rate,
        "Avg Holding Days": avg_holding,
        "Avg Trade Return (%)": avg_trade_return,
        "Turnover": turnover,
        "Net P&L": float(net.sum()),
    }
    daily_results = pd.DataFrame({
        "Z-score": z,
        "Signal Hedge Ratio": betas,
        "Held Hedge Ratio": actual_betas,
        "Position": positions,
        "Gross P&L": gross,
        "Transaction Cost": costs,
        "Net P&L": net,
        "Daily Return": daily_return,
        "Equity": equity,
    }, index=dates)
    if trades and not np.isclose(trade_log["Net P&L"].sum(), net.sum(), atol=1e-7):
        raise AssertionError("Trade log and daily P&L disagree")
    return {
        "metrics": metrics,
        "trade_log": trade_log,
        "daily_results": daily_results,
        "reentry_unlock_dates": reentry_unlocks,
    }
