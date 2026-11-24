"""Quant Pocket application entrypoint with explicit page registration."""
import streamlit as st

st.set_page_config(page_title="Quant Pocket", page_icon="📈", layout="centered")
page = st.navigation([
    st.Page("home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Backtesting.py", title="Backtesting", icon="🧪", url_path="backtesting"),
], position="top")
page.run()
