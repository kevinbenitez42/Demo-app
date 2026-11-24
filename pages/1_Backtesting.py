"""Mobile interface for a separate adaptation of the research backtesting notebook."""
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from backtest_engine import STRATEGIES, run_backtest

st.set_page_config(page_title="Backtesting | Quant Pocket", page_icon="📈", layout="centered")
st.markdown("""<style>
.block-container {max-width:880px;padding-top:1.5rem;padding-bottom:3rem;}
@media(max-width:600px){.block-container{padding-left:1rem;padding-right:1rem;}h1{font-size:2rem!important;}}
</style>""", unsafe_allow_html=True)
st.page_link("home.py", label="Home", icon="🏠")
st.caption("QUANT POCKET / NOTEBOOK LAB")
st.title("Test a strategy.")
st.write("Compare buy and hold, a moving-average crossover, and inverse-volatility sizing using daily market prices.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker, period):
    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True, timeout=20)
    if frame.empty or "Close" not in frame:
        raise ValueError("No price history returned. Check the ticker, try another symbol, or upload a CSV below.")
    close = frame["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

with st.form("backtest_settings"):
    ticker = st.text_input("Ticker", value="NFLX", max_chars=20, help="Examples: NFLX, SPY, AAPL, MSFT, ^GSPC")
    history = st.selectbox("History", ["1 year", "3 years", "5 years", "10 years", "20 years"], index=2)
    short_window = st.number_input("Short moving average (trading days)", min_value=2, max_value=500, value=21)
    long_window = st.number_input("Long moving average (trading days)", min_value=3, max_value=1000, value=200)
    with st.expander("Use your own price data"):
        upload = st.file_uploader("Optional CSV with Date and Close columns", type=["csv"])
        st.caption("An uploaded CSV replaces the market-data download. Its rows determine the test period.")
    submitted = st.form_submit_button("Run backtest", type="primary", width="stretch")

if submitted:
    ticker = ticker.strip().upper()
    if short_window >= long_window:
        st.error("Choose a short moving average smaller than the long moving average.")
    elif upload is None and not re.fullmatch(r"[A-Z0-9.^=\-]{1,20}", ticker):
        st.error("Enter a valid market ticker, such as NFLX or SPY.")
    else:
        try:
            with st.spinner("Loading prices and calculating strategies…"):
                if upload is not None:
                    uploaded = pd.read_csv(upload)
                    columns = {str(c).strip().lower(): c for c in uploaded.columns}
                    if not {"date", "close"}.issubset(columns):
                        raise ValueError("CSV must contain Date and Close columns.")
                    dates = pd.to_datetime(uploaded[columns["date"]], errors="coerce", utc=True)
                    if dates.isna().any():
                        raise ValueError("CSV contains invalid dates.")
                    close = pd.Series(uploaded[columns["close"]].to_numpy(), index=pd.DatetimeIndex(dates).tz_localize(None))
                    source = "Uploaded CSV (daily prices assumed; adjustments depend on your file)"
                    label = ticker or "Uploaded prices"
                else:
                    period = {"1 year":"1y", "3 years":"3y", "5 years":"5y", "10 years":"10y", "20 years":"20y"}[history]
                    close = load_prices(ticker, period)
                    source = "Yahoo Finance · adjusted daily close · cached up to 1 hour"
                    label = ticker
                frame, summary = run_backtest(close, int(short_window), int(long_window))
                st.session_state["backtest_result"] = (frame, summary, label, source, int(short_window), int(long_window))
        except Exception as exc:
            st.error(f"Could not run backtest: {exc}")
            st.caption("Your last successful result, if any, remains below. No simulated prices are substituted.")

result = st.session_state.get("backtest_result")
if result is None:
    st.info("Choose a ticker and tap Run backtest. These results use market prices, rather than the simulated data on the home page.")
else:
    frame, summary, label, source, used_short, used_long = result
    st.subheader(f"{label} · {used_short}/{used_long}-day crossover")
    st.caption(f"{frame.index[0]:%b %d, %Y} – {frame.index[-1]:%b %d, %Y} · {len(frame):,} observations")
    st.caption(source)
    selected = st.selectbox("Strategy details", list(STRATEGIES), index=1)
    metrics = summary.loc[selected]
    c1, c2 = st.columns(2)
    c1.metric("Total return", f"{metrics['Total return (%)']:+.2f}%")
    c2.metric("Max drawdown", f"{metrics['Max drawdown (%)']:.2f}%")
    st.caption(f"Annualized volatility: {metrics['Annualized volatility (%)']:.2f}%")
    view = st.radio("Chart view", ["Growth of $1", "Drawdown", "Price & averages"], horizontal=True)
    fig = go.Figure()
    colors = ["#94a3b8", "#55d6be", "#60a5fa"]
    if view == "Price & averages":
        for column, name, color in [("Close",label,"#e6edf7"),("short_ma",f"{used_short}-day MA","#55d6be"),("long_ma",f"{used_long}-day MA","#60a5fa")]:
            fig.add_scatter(x=frame.index,y=frame[column],name=name,line=dict(color=color,width=2))
        axis = "Price"
    else:
        for (name, (_, index_column)), color in zip(STRATEGIES.items(), colors):
            values = frame[index_column]
            if view == "Drawdown":
                values = (values / values.cummax() - 1) * 100
            fig.add_scatter(x=frame.index,y=values,name=name,line=dict(color=color,width=2.3))
        axis = "Growth of $1" if view == "Growth of $1" else "Drawdown (%)"
    fig.update_layout(height=390,template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8,r=8,t=10,b=8),yaxis_title=axis,hovermode="x unified",dragmode=False,
                      legend=dict(orientation="h",y=-0.22,x=0,font=dict(size=12)))
    fig.update_yaxes(fixedrange=True)
    st.plotly_chart(fig,width="stretch",config={"displayModeBar":False,"scrollZoom":False})
    st.subheader("Compare all three")
    st.dataframe(summary.round(2), width="stretch")
    st.download_button("Download backtest results",frame.to_csv().encode(),file_name=f"{label}_backtest.csv",mime="text/csv")

with st.expander("Method & assumptions"):
    st.write("The crossover holds the asset when its short moving average exceeds its long average; otherwise it holds cash. Positions are shifted by one daily bar before calculating returns. It stays in cash while the long average warms up.")
    st.write("Inverse-volatility exposure uses 21-day annualized volatility, scales its reciprocal by its expanding historical mean, clips exposure to 0–150%, and shifts it by one daily bar. This is a separate strategy, not a filter on the crossover.")
    st.write("This simple notebook model uses close-to-close returns. It excludes transaction costs, slippage, financing costs, and interest on cash. Daily data and 252 trading days per year are assumed. Results are historical and depend on the selected window.")
    st.markdown("Adapted into separate app files from [Backtesting.ipynb](https://github.com/kevinbenitez42/investment-research/blob/main/Notebooks/Single%20Asset%20Profile/Pricing/Backtesting.ipynb). The original notebook is untouched.")
