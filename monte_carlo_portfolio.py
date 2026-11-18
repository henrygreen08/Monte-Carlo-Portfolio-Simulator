import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

TRADING_DAYS = 252


def get_prices(tickers, period="3y") -> pd.DataFrame:
    """Download daily closing prices; one column per ticker."""
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                       threads=False)
    if data.empty:
        raise ValueError("No data downloaded. Check the ticker symbols.")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise ValueError(f"No data for: {', '.join(missing)}")
    return close[tickers].dropna()


def estimate_parameters(prices: pd.DataFrame):
    """Daily mean log returns (vector), covariance matrix, and log returns."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean().values
    cov = log_returns.cov().values
    return mu, cov, log_returns


def fit_degrees_of_freedom(log_returns: pd.DataFrame) -> float:
    """
    Fit a Student's t to each stock's standardized returns and take the
    median degrees of freedom. Lower df = fatter tails (normal ~ df of infinity).
    """
    dfs = []
    for col in log_returns:
        z = (log_returns[col] - log_returns[col].mean()) / log_returns[col].std()
        df, _, _ = stats.t.fit(z, floc=0)
        dfs.append(df)
    return float(np.clip(np.median(dfs), 2.5, 100))


def simulate_portfolio(mu, cov, weights, days, n_sims,
                       dist="t", df=5.0, initial_value=10_000, seed=42):
    """
    Returns:
      portfolio_paths: (days + 1, n_sims)        dollar value of the portfolio
      asset_growth:    (days + 1, n_sims, n_assets)  growth of $1 in each stock
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)

    L = np.linalg.cholesky(cov)

    z = rng.standard_normal((days, n_sims, n_assets))

    if dist == "t":
        chi2 = rng.chisquare(df, size=(days, n_sims, 1))
        z = z / np.sqrt(chi2 / df)
        z = z * np.sqrt((df - 2) / df)

    correlated_shocks = z @ L.T
    daily_log_returns = mu + correlated_shocks

    cum = np.cumsum(daily_log_returns, axis=0)
    cum = np.concatenate([np.zeros((1, n_sims, n_assets)), cum], axis=0)
    asset_growth = np.exp(cum)

    portfolio_paths = initial_value * (asset_growth @ weights)
    return portfolio_paths, asset_growth


def risk_summary(paths, confidences=(0.95, 0.99)):
    """
    VaR = loss you exceed only (1 - confidence) of the time.
    CVaR = average loss in that worst slice.
    A negative VaR means even the bad scenarios ended with a gain.
    """
    start = paths[0, 0]
    final = paths[-1]
    returns = final / start - 1
    summary = {
        "Start value": start,
        "Mean final value": final.mean(),
        "Median final value": np.median(final),
        "5th percentile": np.percentile(final, 5),
        "95th percentile": np.percentile(final, 95),
        "Probability of loss": (final < start).mean(),
    }
    for c in confidences:
        cutoff = np.percentile(returns, (1 - c) * 100)
        summary[f"VaR ({c:.0%})"] = -cutoff
        summary[f"CVaR ({c:.0%})"] = -returns[returns <= cutoff].mean()
    return summary


def print_summary(title, summary):
    print(f"\n--- {title} ---")
    for k, v in summary.items():
        if "Probability" in k or "VaR" in k:
            print(f"{k:22s} {v:8.2%}")
        else:
            print(f"{k:22s} ${v:,.2f}")


def plot_results(paths_t, paths_norm, log_returns, label):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    start = paths_t[0, 0]

    ax = axes[0]
    ax.plot(paths_t[:, :100], linewidth=0.6, alpha=0.6)
    ax.axhline(start, color="black", linestyle="--", label="Start value")
    ax.set_title(f"{label}: 100 simulated paths (t-distribution)")
    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()

    ax = axes[1]
    bins = np.linspace(min(paths_t[-1].min(), paths_norm[-1].min()),
                       np.percentile(np.r_[paths_t[-1], paths_norm[-1]], 99.5), 80)
    ax.hist(paths_norm[-1], bins=bins, alpha=0.5, label="Normal")
    ax.hist(paths_t[-1], bins=bins, alpha=0.5, label="Student's t")
    ax.axvline(start, color="black", linestyle="--", label="Start value")
    ax.set_title("Final values: normal vs fat-tailed")
    ax.set_xlabel("Portfolio value ($)")
    ax.set_ylabel("Number of simulations")
    ax.legend()

    ax = axes[2]
    corr = log_returns.corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Historical correlation of daily returns")
    fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(f"{label.replace(' ', '_')}_monte_carlo.png", dpi=150)
    plt.show()


def ask_inputs():
    tickers = [t.strip().upper() for t in
               input("Tickers, comma-separated (e.g. AAPL, MSFT, JPM): ").split(",") if t.strip()]
    if not tickers:
        raise ValueError("Enter at least one ticker.")

    raw = input(f"Weights for {', '.join(tickers)} (e.g. 0.5, 0.3, 0.2) "
                "or press Enter for equal weights: ").strip()
    if raw:
        weights = np.array([float(w) for w in raw.split(",")])
        if len(weights) != len(tickers):
            raise ValueError("Need one weight per ticker.")
        weights = weights / weights.sum()
    else:
        weights = np.full(len(tickers), 1 / len(tickers))

    initial = float(input("Starting investment in $ (default 10000): ") or 10_000)
    days = int(input("Trading days to simulate (default 252 = 1 year): ") or 252)
    n_sims = int(input("Number of simulations (default 10000): ") or 10_000)
    return tickers, weights, initial, days, n_sims


def main():
    tickers, weights, initial, days, n_sims = ask_inputs()

    prices = get_prices(tickers)
    mu, cov, log_returns = estimate_parameters(prices)
    df = fit_degrees_of_freedom(log_returns)

    print("\nPortfolio:")
    for t, w, m, v in zip(tickers, weights, mu, np.sqrt(np.diag(cov))):
        print(f"  {t:6s} weight {w:6.1%}   annual drift {m*TRADING_DAYS:7.2%}"
              f"   annual volatility {v*np.sqrt(TRADING_DAYS):6.2%}")
    print(f"Fitted Student's t degrees of freedom: {df:.1f} "
          "(lower = fatter tails; a normal is like infinity)")

    paths_t, _ = simulate_portfolio(mu, cov, weights, days, n_sims, dist="t", df=df,
                                    initial_value=initial)
    paths_n, _ = simulate_portfolio(mu, cov, weights, days, n_sims, dist="normal",
                                    initial_value=initial)

    print_summary(f"Student's t, after {days} days", risk_summary(paths_t))
    print_summary(f"Normal (for comparison), after {days} days", risk_summary(paths_n))

    plot_results(paths_t, paths_n, log_returns, "+".join(tickers))


if __name__ == "__main__":
    main()