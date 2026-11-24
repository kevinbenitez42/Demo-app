"""Run with: python -m streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Quant Pocket", page_icon="📈", layout="centered")
st.markdown("""<style>
.block-container {max-width: 880px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {background: #182438; border: 1px solid #2a3b54;
border-radius: 14px; padding: 16px;}
@media(max-width:600px) {
 .block-container {padding-left: 1rem; padding-right: 1rem;}
 h1 {font-size: 2rem !important;}
 [data-testid="stMetric"] {padding: 12px;}
}
</style>""", unsafe_allow_html=True)

ASSETS = {
    "S&P 500": ("SPX", 5200, 0.17, 12),
    "Apple": ("AAPL", 185, 0.25, 24),
    "Microsoft": ("MSFT", 410, 0.22, 36),
    "Nvidia": ("NVDA", 110, 0.40, 48),
}

@st.cache_data
def sample_prices(asset):
    symbol, initial, volatility, seed = ASSETS[asset]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1261)
    returns = rng.normal(0.08 / 252, volatility / np.sqrt(252), len(dates))
    return pd.Series(initial * np.exp(np.cumsum(returns)), index=dates, name=symbol)

st.caption("QUANT POCKET / RESEARCH DEMO")
st.title("Your market, at a glance.")
st.info("Demo mode · All prices and metrics are simulated, not live market data.")
asset = st.selectbox("Asset", list(ASSETS))
period = st.radio("Timeframe", ["1M", "3M", "6M", "1Y", "5Y"], index=3, horizontal=True)
all_prices = sample_prices(asset)
days = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "5Y": 1260}[period]
prices = all_prices.iloc[-(days + 1):]
returns = prices.pct_change().dropna()
change = prices.iloc[-1] / prices.iloc[0] - 1
volatility = returns.std() * np.sqrt(252)
drawdown = prices / prices.cummax() - 1
symbol = ASSETS[asset][0]

st.subheader(f"{asset} · {symbol}")
left, right = st.columns(2)
left.metric("Latest sample level", f"{prices.iloc[-1]:,.2f}")
right.metric(f"{period} return", f"{change:+.2%}")
left.metric("Annualized volatility", f"{volatility:.2%}")
right.metric("Maximum drawdown", f"{drawdown.min():.2%}")

view = st.radio("Chart", ["Price", "Growth of $100", "Drawdown"], horizontal=True)
if view == "Price":
    values, axis = prices, "Index points" if symbol == "SPX" else "Sample price (USD)"
elif view == "Growth of $100":
    values, axis = prices / prices.iloc[0] * 100, "Value (USD)"
else:
    values, axis = drawdown * 100, "Drawdown (%)"
color = "#fb7185" if view == "Drawdown" else "#55d6be"
fig = go.Figure(go.Scatter(x=values.index, y=values, mode="lines", line=dict(color=color, width=2.5),
                          hovertemplate="%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>"))
fig.update_layout(height=350, margin=dict(l=8, r=8, t=12, b=8),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(color="#d9e5f5"), yaxis_title=axis,
                  dragmode=False, hovermode="x unified")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#26364d", fixedrange=True)
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})
st.caption(f"{prices.index[0]:%b %d, %Y} – {prices.index[-1]:%b %d, %Y} · Simulated business-day observations")

with st.expander("Recent observations"):
    st.dataframe(pd.DataFrame({"Sample level": prices, "Daily return (%)": prices.pct_change() * 100})
                 .tail(10).sort_index(ascending=False).round(2), width="stretch")

st.download_button("Download sample prices", prices.rename("simulated_close").to_csv().encode(),
                   file_name=f"{symbol}_simulated_{period}.csv", mime="text/csv")
with st.expander("How this connects to your notebooks"):
    st.write("The next step is to replace the sample price function with a reusable function from one of your notebooks. Your Python calculations run on the computer; this page displays the results on your phone.")
    st.caption("Return is measured over the selected window. Volatility uses 252 trading days per year. Maximum drawdown is the largest decline from a prior peak within that window. Growth of $100 excludes dividends and fees.")
