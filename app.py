"""
Website for the Monte Carlo portfolio simulator.

All the maths lives in monte_carlo_portfolio.py (unchanged).
This file only handles the page: inputs, charts and wording.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from monte_carlo_portfolio import (
    TRADING_DAYS,
    estimate_parameters,
    fit_degrees_of_freedom,
    get_prices,
    risk_summary,
    simulate_portfolio,
)

# ------------------------------------------------------------------ design tokens

MAX_STOCKS = 5
CELL_BUDGET = 6_000_000  # days x simulations x stocks; keeps the free server fast

HORIZONS = {"1 day": 1, "1 week": 5, "1 month": 21, "3 months": 63, "1 year": 252}

st.set_page_config(
    page_title="Portfolio risk simulator",
    page_icon=":material/monitoring:",
    layout="wide",
)

# Two palettes. Streamlit's own theme (buttons, inputs, background) comes from
# .streamlit/config.toml; these cover our custom styling and the charts.
# Keep "brand" in sync with primaryColor in config.toml.
PALETTES = {
    "light": dict(brand="#0F766E", line_strong="#115E59", band="15,118,110", loss="#B42318",
                  ink="#1C1917", muted="#57534E", line="#E4E4E0", neutral="#A8A29E",
                  surface="#FFFFFF", skeleton="#ECECE9", bank="#B45309"),
    "dark":  dict(brand="#2DD4BF", line_strong="#5EEAD4", band="45,212,191", loss="#F97066",
                  ink="#F5F5F4", muted="#A8A29E", line="#33312E", neutral="#78716C",
                  surface="#1E1D1B", skeleton="#2A2826", bank="#FBBF24"),
}
_mode = getattr(st.context.theme, "type", None) or "light"
P = PALETTES.get(_mode, PALETTES["light"])
BRAND, BRAND_DARK, LOSS = P["brand"], P["line_strong"], P["loss"]
INK, MUTED, LINE, NEUTRAL, SURFACE = P["ink"], P["muted"], P["line"], P["neutral"], P["surface"]

st.markdown(
    f"""
<style>
:root {{
  --brand: {BRAND};
  --ink: {INK};
  --muted: {MUTED};
  --line: {LINE};
  --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px; --s7: 48px;
}}
.block-container {{ padding-top: var(--s7); padding-bottom: var(--s7); max-width: 1200px; }}
h1 {{ font-size: 40px !important; font-weight: 700 !important; letter-spacing: -0.02em;
      color: var(--ink); padding-bottom: var(--s2) !important; }}
h2 {{ font-size: 24px !important; font-weight: 700 !important; color: var(--ink); }}
h3 {{ font-size: 18px !important; font-weight: 600 !important; color: var(--ink); }}
.lead {{ font-size: 18px; line-height: 1.55; color: var(--muted); max-width: 64ch;
         margin-bottom: var(--s6); }}
.step {{ font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
         color: var(--brand); margin: var(--s5) 0 var(--s1); }}
.step:first-child {{ margin-top: 0; }}
.step-title {{ font-size: 18px; font-weight: 600; color: var(--ink); margin-bottom: var(--s3); }}
.body {{ font-size: 16px; line-height: 1.6; color: var(--muted); max-width: 68ch; }}
.fine {{ font-size: 14px; line-height: 1.5; color: var(--muted); }}
[data-testid="stMetricValue"] {{ font-size: 28px; font-weight: 700; }}
[data-testid="stMetricLabel"] p {{ font-size: 14px; color: var(--muted); }}

/* empty state */
.empty {{ border: 1px dashed var(--line); border-radius: 6px; padding: var(--s6);
          background: {SURFACE}; }}
.empty h3 {{ margin: 0 0 var(--s2); }}
.empty ul {{ margin: var(--s4) 0 0; padding-left: var(--s5); color: var(--muted); line-height: 1.8; }}

/* loading skeleton, same shape as the results */
@keyframes pulse {{ 0% {{opacity: 1}} 50% {{opacity: .45}} 100% {{opacity: 1}} }}
.sk {{ background: {P["skeleton"]}; border-radius: 6px; animation: pulse 1.4s ease-in-out infinite; }}
.sk-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--s4); margin-bottom: var(--s5); }}
.sk-metric {{ height: 84px; }}
.sk-title {{ height: 24px; width: 40%; margin-bottom: var(--s4); }}
.sk-chart {{ height: 380px; }}
.sk-note {{ font-size: 14px; color: var(--muted); margin-bottom: var(--s4); }}
@media (max-width: 640px) {{ .sk-row {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ helpers
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers: tuple) -> pd.DataFrame:
    return get_prices(list(tickers))


def parse_tickers(raw: str) -> list:
    seen, out = set(), []
    for t in raw.replace(" ", ",").split(","):
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x:.1%}"


def chart_layout(fig, height=400, y_title=None, x_title=None):
    fig.update_layout(
        template="plotly_dark" if _mode == "dark" else "simple_white",
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        font=dict(family="Source Sans Pro, sans-serif", size=14, color=MUTED),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0, title=None),
        hoverlabel=dict(bgcolor=SURFACE, font_size=13, font_color=INK),
    )
    fig.update_xaxes(title=x_title, gridcolor=LINE, showgrid=False, linecolor=LINE)
    fig.update_yaxes(title=y_title, gridcolor=LINE, showgrid=True, linecolor=LINE)
    return fig


SKELETON = """
<div class="sk-note">Downloading price history and running the simulation&hellip;</div>
<div class="sk-row">
  <div class="sk sk-metric"></div><div class="sk sk-metric"></div>
  <div class="sk sk-metric"></div><div class="sk sk-metric"></div>
</div>
<div class="sk sk-title"></div>
<div class="sk sk-chart"></div>
"""

EMPTY = """
<div class="empty">
  <h3>Your results will show up here</h3>
  <div class="body">Fill in the four steps, then select <b>Run simulation</b>.
  It takes a few seconds. You'll get:</div>
  <ul>
    <li>The range of values your portfolio could reach, and the most likely one</li>
    <li>Your chance of losing money, and how much a bad case could cost</li>
    <li>A side-by-side of realistic fat-tailed risk and textbook bell-curve risk</li>
    <li>How closely your stocks move together</li>
    <li>What the same money would earn in a savings account instead</li>
  </ul>
</div>
"""


def run_simulation(tickers, weights, initial, days, n_sims):
    """Calls the unchanged model and keeps only what the page needs (saves memory)."""
    prices = load_prices(tuple(tickers))
    if len(prices) < 60:
        raise ValueError("short_history")
    mu, cov, log_returns = estimate_parameters(prices)
    df = fit_degrees_of_freedom(log_returns)

    paths_t, _ = simulate_portfolio(mu, cov, weights, days, n_sims, dist="t", df=df,
                                    initial_value=initial)
    summary_t = risk_summary(paths_t)
    bands = np.percentile(paths_t, [5, 25, 50, 75, 95], axis=1)
    sample = paths_t[:, :25].copy()
    final_t = paths_t[-1].copy()
    del paths_t

    paths_n, _ = simulate_portfolio(mu, cov, weights, days, n_sims, dist="normal",
                                    initial_value=initial)
    summary_n = risk_summary(paths_n)
    final_n = paths_n[-1].copy()
    del paths_n

    stocks = pd.DataFrame({
        "Stock": tickers,
        "Weight": weights,
        "Last price": prices.iloc[-1].values,
        "Yearly trend (drift)": mu * TRADING_DAYS,
        "Yearly volatility": np.sqrt(np.diag(cov)) * np.sqrt(TRADING_DAYS),
    })
    return dict(
        tickers=tickers, weights=weights, initial=initial, days=days, n_sims=n_sims,
        df=df, summary_t=summary_t, summary_n=summary_n, bands=bands, sample=sample,
        final_t=final_t, final_n=final_n, corr=log_returns.corr(), stocks=stocks,
        history=(prices.index[0], prices.index[-1], len(prices)),
    )


# ------------------------------------------------------------------ header
st.title("Portfolio risk simulator")
st.markdown(
    '<p class="lead">Pick a few stocks and an amount to invest. We use their real price '
    "history to play out thousands of possible futures, then show you the likely range "
    "of outcomes and how bad a bad stretch could get.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 2], gap="large")

# ------------------------------------------------------------------ inputs
with left:
    with st.container(border=True):
        st.markdown('<div class="step">Step 1</div><div class="step-title">Choose your stocks</div>',
                    unsafe_allow_html=True)
        raw = st.text_input(
            "Ticker symbols, separated by commas",
            value="AAPL, MSFT, JPM",
            placeholder="AAPL, MSFT, JPM",
            help="Use the symbol from Yahoo Finance. Canadian stocks end in .TO, "
                 "for example RY.TO for Royal Bank. ETFs like SPY or XIU.TO work too.",
        )
        tickers = parse_tickers(raw)
        if len(tickers) > MAX_STOCKS:
            st.caption(f"Only the first {MAX_STOCKS} are used: {', '.join(tickers[:MAX_STOCKS])}.")
            tickers = tickers[:MAX_STOCKS]

        st.markdown('<div class="step">Step 2</div><div class="step-title">Split your money</div>',
                    unsafe_allow_html=True)
        weights_pct = []
        if tickers:
            st.markdown('<div class="fine">What percentage goes into each stock.</div>',
                        unsafe_allow_html=True)
            equal = round(100 / len(tickers), 1)
            cols = st.columns(min(len(tickers), 3))
            for i, t in enumerate(tickers):
                with cols[i % len(cols)]:
                    weights_pct.append(st.number_input(
                        f"{t} (%)", min_value=0.0, max_value=100.0, value=equal,
                        step=5.0, format="%.1f", key=f"w_{t}_{len(tickers)}",
                    ))
            total = sum(weights_pct)
            if total == 0:
                st.caption(":material/error: Give at least one stock more than 0%.")
            elif abs(total - 100) > 0.5:
                st.caption(f"These add up to {total:.0f}%. We'll scale them so they add up to 100%.")
        else:
            st.markdown('<div class="fine">Add at least one ticker in step 1 first.</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="step">Step 3</div><div class="step-title">Set up the simulation</div>',
                    unsafe_allow_html=True)
        initial = st.number_input("Amount to invest ($)", min_value=100, max_value=10_000_000,
                                  value=10_000, step=1_000)
        horizon = st.segmented_control(
            "How far ahead to look", list(HORIZONS), default="1 year",
            help="Trading days, not calendar days: 1 week is 5, 1 year is 252. "
                 "Banks usually measure risk over 1 day to 2 weeks.",
        ) or "1 year"
        n_sims = st.select_slider(
            "Number of simulated futures", options=[1_000, 2_500, 5_000, 10_000], value=5_000,
            help="More futures give smoother, more reliable numbers but take longer.",
        )

        days = HORIZONS[horizon]
        max_sims = max(1_000, (CELL_BUDGET // (days * max(len(tickers), 1))) // 500 * 500)
        n_sims_used = min(n_sims, max_sims)
        if n_sims_used < n_sims:
            st.caption(f"With {len(tickers)} stocks over {horizon}, we'll run "
                       f"{n_sims_used:,} futures to keep this free site fast.")

        st.markdown('<div class="step">Step 4</div><div class="step-title">Compare with a savings account</div>',
                    unsafe_allow_html=True)
        bank_rate = st.number_input(
            "Savings account interest rate (% per year)", min_value=0.0, max_value=20.0,
            value=2.5, step=0.25, format="%.2f",
            help="Big banks' regular savings accounts often pay under 1%. Online "
                 "high-interest savings accounts often pay about 2% to 3%. Watch out "
                 "for short-term promo rates. Changing this updates the results right away.",
        )

        can_run = bool(tickers) and sum(weights_pct) > 0
        run = st.button("Run simulation", type="primary", icon=":material/play_arrow:",
                        disabled=not can_run, use_container_width=True)

# ------------------------------------------------------------------ results
with right:
    if run:
        w = np.array(weights_pct, dtype=float)
        w = w / w.sum()
        placeholder = st.empty()
        placeholder.markdown(SKELETON, unsafe_allow_html=True)
        st.session_state.pop("error", None)
        try:
            st.session_state.results = run_simulation(tickers, w, float(initial), days, n_sims_used)
            st.session_state.results["horizon"] = horizon
        except ValueError as e:
            msg = str(e)
            st.session_state.pop("results", None)
            if msg.startswith("No data for:"):
                bad = msg.split(":", 1)[1].strip()
                st.session_state.error = (
                    f"We couldn't find price data for **{bad}**. Check the symbol on "
                    "Yahoo Finance. Canadian stocks need .TO at the end, like RY.TO.")
            elif msg == "short_history":
                st.session_state.error = (
                    "These stocks share fewer than 60 days of price history, which is too "
                    "little to measure risk. Try stocks that have been listed longer.")
            else:
                st.session_state.error = (
                    "We couldn't download any prices for those symbols. Check the spelling "
                    "and try again.")
        except np.linalg.LinAlgError:
            st.session_state.pop("results", None)
            st.session_state.error = (
                "Two of these stocks move almost exactly together (for example, two share "
                "classes of the same company), so the model can't tell them apart. "
                "Remove one and run it again.")
        except Exception:
            st.session_state.pop("results", None)
            st.session_state.error = (
                "We couldn't reach Yahoo Finance just now. Wait a minute, then select "
                "Run simulation again.")
        placeholder.empty()

    if "error" in st.session_state:
        st.error(st.session_state.error, icon=":material/error:")

    res = st.session_state.get("results")
    if res is None:
        if "error" not in st.session_state:
            st.markdown(EMPTY, unsafe_allow_html=True)
    else:
        st_t, st_n = res["summary_t"], res["summary_n"]
        start = res["initial"]
        h_start, h_end, h_days = res["history"]

        stale = (res["tickers"] != tickers or res["days"] != days
                 or res["initial"] != float(initial))
        if stale:
            st.caption(":material/info: You've changed the inputs. Select Run simulation "
                       "to update these results.")

        st.header(f"What could happen over {res['horizon']}")
        st.markdown(
            f'<div class="fine">{res["n_sims"]:,} simulated futures for '
            f'{", ".join(res["tickers"])}, based on {h_days:,} trading days of prices '
            f'({h_start:%b %Y} to {h_end:%b %Y}).</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        median = st_t["Median final value"]
        var95, cvar95 = st_t["VaR (95%)"], st_t["CVaR (95%)"]

        def loss_text(r):
            return (f"−{money(start * r)}" if r > 0 else f"+{money(-start * r)}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Most likely value", money(median),
                  delta=f"{median / start - 1:+.1%}",
                  help="The median: half the simulated futures end above this, half below.")
        m2.metric("Chance of a loss", pct(st_t["Probability of loss"]),
                  help="Share of simulated futures that end below what you put in.")
        m3.metric("Bad case (1 in 20)", loss_text(var95),
                  delta=f"{-var95:+.1%}", delta_color="normal",
                  help="Value at Risk (95%). In 1 out of 20 futures, you'd lose at least this much.")
        m4.metric("Worst 5% average", loss_text(cvar95),
                  delta=f"{-cvar95:+.1%}", delta_color="normal",
                  help="Conditional Value at Risk (95%). If you land in the worst 1 in 20, "
                       "this is the average result.")

        # -- savings account comparison (recomputed instantly when the rate changes)
        bank_curve = start * (1 + bank_rate / 100) ** (np.arange(res["days"] + 1) / TRADING_DAYS)
        bank_final = bank_curve[-1]
        beat = (res["final_t"] > bank_final).mean()
        gap = median - bank_final
        with st.container(border=True):
            st.markdown(
                f'<div class="step-title">Compared with a savings account at {bank_rate:.2f}%</div>',
                unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.metric("Savings account value", money(bank_final),
                      delta=f"+{money(bank_final - start)} interest", delta_color="off",
                      help="Interest compounds, so you also earn interest on past interest. "
                           "This amount is guaranteed as long as the rate stays the same.")
            b2.metric("Chance stocks do better", pct(beat),
                      help="Share of simulated futures where the portfolio ends above the "
                           "savings account.")
            b3.metric("Most likely difference", f"{'+' if gap >= 0 else '−'}{money(abs(gap))}",
                      delta="stocks ahead" if gap >= 0 else "savings ahead", delta_color="off",
                      help="Most likely portfolio value minus the savings account value.")
            bad_value = start * (1 - var95)
            if bad_value < bank_final:
                bad_line = (f'but in the bad case (1 in 20) you\'d end up '
                            f'<b>{money(bank_final - bad_value)}</b> behind the bank. That extra '
                            f'risk is the price of a chance at a bigger return.')
            else:
                bad_line = ('and even the bad case (1 in 20) ends ahead of the bank. Remember '
                            'this is based on the past 3 years, which may have been unusually good.')
            st.markdown(
                f'<div class="fine">The savings account is the safe choice: you know what '
                f'you\'ll get. Stocks beat it in <b>{beat:.1%}</b> of the simulated futures, '
                f'{bad_line}</div>', unsafe_allow_html=True)

        tab_paths, tab_tails, tab_corr, tab_stocks = st.tabs(
            ["Range of outcomes", "Fat tails vs bell curve", "How your stocks move together",
             "Stock details"])

        # -- fan chart
        with tab_paths:
            b = res["bands"]
            x = np.arange(b.shape[1])
            fig = go.Figure()
            for path in res["sample"].T:
                fig.add_trace(go.Scatter(x=x, y=path, mode="lines", hoverinfo="skip",
                                         line=dict(color=NEUTRAL, width=0.6), opacity=0.35,
                                         showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=b[4], line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=b[0], fill="tonexty", line=dict(width=0),
                                     fillcolor=f"rgba({P['band']},0.14)", name="90% of futures",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=b[3], line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=b[1], fill="tonexty", line=dict(width=0),
                                     fillcolor=f"rgba({P['band']},0.30)", name="Middle 50%",
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=b[2], line=dict(color=BRAND_DARK, width=2.5),
                                     name="Most likely", hovertemplate="Day %{x}: $%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=x, y=bank_curve, line=dict(color=P["bank"], width=2, dash="dot"),
                                     name=f"Savings account ({bank_rate:.2f}%)",
                                     hovertemplate="Day %{x}: $%{y:,.0f}<extra>Savings</extra>"))
            fig.add_hline(y=start, line=dict(color=INK, dash="dash", width=1),
                          annotation_text="What you put in", annotation_position="bottom right")
            chart_layout(fig, 420, y_title="Portfolio value ($)", x_title="Trading days from today")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div class="fine">The solid line is the most likely path. The shaded bands '
                f'show where 50% and 90% of the simulated futures landed. 90% of the time, '
                f'your {money(start)} ends between <b>{money(st_t["5th percentile"])}</b> '
                f'and <b>{money(st_t["95th percentile"])}</b>.</div>',
                unsafe_allow_html=True)

        # -- t vs normal
        with tab_tails:
            ft, fn = res["final_t"], res["final_n"]
            lo = min(np.percentile(ft, 0.2), np.percentile(fn, 0.2))
            hi = max(np.percentile(ft, 99.8), np.percentile(fn, 99.8))
            bins = dict(start=lo, end=hi, size=(hi - lo) / 70)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ft, xbins=bins, name="Fat tails (Student's t)",
                                       marker=dict(color=BRAND, line=dict(width=0)),
                                       opacity=0.55))
            edges = np.arange(lo, hi + bins["size"], bins["size"])
            counts, _ = np.histogram(fn, bins=edges)
            fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=counts,
                                     line=dict(color=INK, width=1.5, shape="hvh"),
                                     name="Bell curve (normal)", hoverinfo="skip"))
            fig.add_vline(x=start, line=dict(color=INK, dash="dash", width=1))
            fig.add_vline(x=bank_final, line=dict(color=P["bank"], dash="dot", width=2),
                          annotation_text="Savings account", annotation_position="top right",
                          annotation_font_color=P["bank"])
            fig.update_layout(bargap=0.04)
            chart_layout(fig, 360, y_title="Number of futures", x_title="Final value ($)")
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            rows = ["VaR (95%)", "CVaR (95%)", "VaR (99%)", "CVaR (99%)"]
            labels = ["Bad case (1 in 20)", "Average of worst 5%",
                      "Very bad case (1 in 100)", "Average of worst 1%"]
            table = pd.DataFrame({
                "Risk measure": labels,
                "Fat tails": [loss_text(st_t[r]) for r in rows],
                "Bell curve": [loss_text(st_n[r]) for r in rows],
            })
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.markdown(
                f'<div class="fine">Real stock returns have more extreme days than a bell '
                f'curve predicts. We fitted a Student\'s t-distribution to your stocks and got '
                f'<b>{res["df"]:.1f} degrees of freedom</b> (lower means more extreme days). '
                f'The gap between the two models is biggest for short horizons and rare '
                f'losses. Over a full year the daily extremes average out, so the two '
                f'look similar.</div>', unsafe_allow_html=True)

        # -- correlation
        with tab_corr:
            corr = res["corr"]
            if len(corr) < 2:
                st.markdown('<div class="body">Add a second stock in step 1 to see how '
                            'your stocks move together.</div>', unsafe_allow_html=True)
            else:
                fig = go.Figure(go.Heatmap(
                    z=corr.values, x=list(corr.columns), y=list(corr.index),
                    zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}",
                    colorscale=[[0, LOSS], [0.5, SURFACE], [1, BRAND]],
                    hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>",
                    colorbar=dict(thickness=12, outlinewidth=0)))
                chart_layout(fig, 360)
                fig.update_yaxes(autorange="reversed", showgrid=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    '<div class="fine">1 means two stocks always move in the same direction, '
                    '0 means they move independently, and −1 means they move in opposite '
                    'directions. Lower numbers mean more diversification: when one stock '
                    'falls, the others are less likely to fall with it.</div>',
                    unsafe_allow_html=True)

        # -- per-stock
        with tab_stocks:
            st.dataframe(
                res["stocks"], hide_index=True, use_container_width=True,
                column_config={
                    "Weight": st.column_config.NumberColumn(format="percent"),
                    "Last price": st.column_config.NumberColumn(format="dollar"),
                    "Yearly trend (drift)": st.column_config.NumberColumn(format="percent"),
                    "Yearly volatility": st.column_config.NumberColumn(format="percent"),
                })
            st.markdown(
                '<div class="fine"><b>Drift</b> is the average yearly growth in the price '
                'history. <b>Volatility</b> is how much the price swings: higher means a '
                'bumpier ride.</div>', unsafe_allow_html=True)

    st.write("")
    st.caption(":material/dark_mode: Want dark mode? Open the ⋮ menu at the top right, "
               "select **Settings**, and choose Dark. By default the site matches your device.")
    with st.expander("How the simulation works"):
        st.markdown(
            """
1. **Download prices.** We get about 3 years of daily closing prices from Yahoo Finance.
2. **Measure each stock.** From the daily changes we work out each stock's average trend,
   how much it swings, and how closely it moves with the others.
3. **Play out the future.** Each simulated day, every stock moves by its trend plus a
   random shock. The shocks are linked so stocks that moved together in the past still do
   (this uses a *Cholesky decomposition*). They're also drawn from a *Student's
   t-distribution*, so extreme days happen as often as they do in real markets.
4. **Repeat thousands of times** and look at the spread of results.

**Limits:** the model assumes the future will look like the last 3 years, and it never
rebalances your portfolio. It's a way to understand risk, not a price forecast and not
financial advice.
"""
        )
