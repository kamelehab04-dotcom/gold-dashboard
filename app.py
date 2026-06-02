import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gold Analysis", layout="wide")

st.title("🥇 Gold Price Analysis Dashboard")
st.markdown("---")

gold = yf.Ticker("GC=F")
df = gold.history(period="3d", interval="1h")
df.columns = [col.lower() for col in df.columns]
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['rsi'] = ta.rsi(df['close'], length=14)
price = df['close'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("💰 Gold Price", f"${price:.2f}")
col2.metric("📈 RSI", f"{df['rsi'].iloc[-1]:.1f}")
col3.metric("📊 Trend", "Bullish" if price > df['ema20'].iloc[-1] else "Bearish")

fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Gold"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name="EMA 20", line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['ema50'], name="EMA 50", line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color='purple')), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig, use_container_width=True)

if df['rsi'].iloc[-1] < 35:
    st.success("🟢 BUY SIGNAL - Oversold")
elif df['rsi'].iloc[-1] > 65:
    st.error("🔴 SELL SIGNAL - Overbought")
else:
    st.warning("🟡 WAIT - Neutral")
