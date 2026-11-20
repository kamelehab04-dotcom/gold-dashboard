# ============================================================
# BLACK PYRAMID v2010.4 — BTC HIGH-ACCURACY EDITION
# Institutional Analysis Terminal · Bitcoin Only
# ============================================================

import os
import base64
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
APP_VERSION = "v2010.4-BTC"
SYMBOL      = "BTC-USD"
PAIR_NAME   = "BTC/USD"

PROFILE = {
    "atr_period": 14, "rsi_period": 14, "rsi_ob": 80, "rsi_os": 20,
    "bb_period": 50, "bb_std": 2.3, "atr_sl": 2.00, "atr_trail": 1.50,
    "swing_order": 4, "structure_lookback": 250,
    "confidence_threshold": 76, "confirmation_threshold": 70,
}

MODE = "balanced"
_GATE_PRESETS = {
    "strict":   {"gate": 80.0, "a_plus": 85.0, "a": 80.0, "b": 72.0, "c": 65.0, "gap": 15},
    "balanced": {"gate": 75.0, "a_plus": 82.0, "a": 75.0, "b": 68.0, "c": 60.0, "gap": 12},
    "relaxed":  {"gate": 72.0, "a_plus": 80.0, "a": 72.0, "b": 65.0, "c": 55.0, "gap": 10},
}
_p = _GATE_PRESETS[MODE]
A_PLUS_MIN, A_MIN, B_MIN, C_MIN = _p["a_plus"], _p["a"], _p["b"], _p["c"]
MIN_SIGNAL_GAP      = _p["gap"]
MIN_EMIT_CONFIDENCE = _p["gate"]

SL_MIN_DIST_ATR = 0.25
SL_MAX_DIST_ATR = 1.50
SL_BUFFER_ATR   = 0.20

ML_MIN_TRADES = 30
ML_TEST_SIZE  = 0.25

ML_FEATURE_NAMES = [
    "structure","trend","momentum","volume","context",
    "mtf_conf","confirmation","raw_conf","regime_enc","session_enc",
    "mtf_match","weekly_match","htf_match","session_match",
    "funding_rate","funding_bias","oi_change","dominance",
    "ema200_dist","atr_ratio","bb_position","vwap_dist",
    "rsi_val","macd_hist_norm","body_ratio","wick_ratio",
    "consec_bars","candles_since_bos","divergence_flag","vol_ratio",
]

KILL_ZONES = {
    "Asia Pump":    (0, 4),
    "London Open":  (7, 10),
    "NY Open":      (12, 15),
    "London Close": (15, 17),
    "US Close":     (20, 23),
}

# ============================================================
# UTIL
# ============================================================
def safe_float(v, default=np.nan):
    try:
        x = float(v); return x if np.isfinite(x) else default
    except Exception: return default

def clamp(v, lo, hi): return max(lo, min(hi, v))

def fmt_price(v):
    if v is None or not np.isfinite(safe_float(v)): return "N/A"
    return f"${float(v):,.2f}"

def fmt_pct(v):
    if v is None or not np.isfinite(safe_float(v)): return "N/A"
    return f"{float(v):+.2f}%"

def trend_icon(state):
    return {"BULLISH":"🟢","TREND_BULLISH":"🟢","BEARISH":"🔴",
            "TREND_BEARISH":"🔴","NEUTRAL":"🟡","RANGE":"🟡",
            "COMPRESSION":"🔵"}.get(state, "⚪")

def init_state():
    defaults = {
        "strict_filters": False, "trade_journal": [], "ml_model": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ============================================================
# SESSION / MARKET
# ============================================================
def get_current_session_info():
    now = datetime.now(timezone.utc); h = now.hour
    sessions = {"Asian":(0,7,"🌏","طوكيو/سيدني"), "London":(7,12,"🇬🇧","لندن"),
                "Overlap":(12,16,"🔥","London/NY Overlap"), "NY":(16,21,"🇺🇸","نيويورك"),
                "After-Hours":(21,24,"🌙","بعد الإغلاق")}
    cur, ic, dsc = "After-Hours", "🌙", "بعد الإغلاق"
    for name,(s,e,i,d) in sessions.items():
        if s <= h < e: cur, ic, dsc = name, i, d; break
    return {"session":cur, "icon":ic, "desc":dsc, "time_utc": f"{h:02d}:{now.minute:02d} UTC"}

def in_kill_zone():
    h = datetime.now(timezone.utc).hour
    for n,(s,e) in KILL_ZONES.items():
        if s <= h < e: return True, f"{n} 🎯"
    return False, "خارج Kill Zones"

# ============================================================
# DATA LAYER
# ============================================================
def normalize_ohlcv(df, min_rows=50):
    if df is None or df.empty: return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    rename = {c: str(c).lower() for c in df.columns if str(c).lower() in ["open","high","low","close","volume","vol"]}
    df = df.rename(columns=rename)
    if "vol" in df.columns: df = df.rename(columns={"vol": "volume"})
    req = ["open","high","low","close"]
    if any(c not in df.columns for c in req): return None
    if "volume" not in df.columns: df["volume"] = 0.0
    for c in req + ["volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[req + ["volume"]].dropna(subset=req)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) >= min_rows else None

@st.cache_data(ttl=60, show_spinner=False)
def get_btc_ohlcv(period="6mo", interval="4h"):
    try:
        return normalize_ohlcv(yf.download(SYMBOL, period=period, interval=interval, progress=False))
    except Exception: return None

@st.cache_data(ttl=30, show_spinner=False)
def get_btc_spot():
    try:
        df = normalize_ohlcv(yf.download(SYMBOL, period="5d", interval="1h", progress=False), min_rows=2)
        if df is not None and len(df) >= 2:
            lt = float(df["close"].iloc[-1]); pc = float(df["close"].iloc[-2])
            return lt, ((lt - pc) / pc * 100) if pc else 0.0
    except Exception: pass
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def get_macro_ohlcv(sym, period="6mo", interval="1d"):
    try:
        return normalize_ohlcv(yf.download(sym, period=period, interval=interval, progress=False))
    except Exception: return None

# ============================================================
# CRYPTO FUNDAMENTALS
# ============================================================
@st.cache_data(ttl=180, show_spinner=False)
def get_funding_rate():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": "BTCUSDT"}, timeout=8)
        d = r.json()
        fr = float(d.get("lastFundingRate", 0)) * 100
        return {"rate": fr, "state": ("HOT_LONG" if fr > 0.05 else "LONG" if fr > 0.02 else "HOT_SHORT" if fr < -0.02 else "BALANCED")}
    except Exception: return {"rate": 0.0, "state": "UNKNOWN"}

@st.cache_data(ttl=180, show_spinner=False)
def get_open_interest():
    try:
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist", params={"symbol":"BTCUSDT","period":"4h","limit":24}, timeout=8)
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            now = float(data[-1]["sumOpenInterestValue"]); prev = float(data[0]["sumOpenInterestValue"])
            chg = (now - prev) / prev * 100 if prev else 0.0
            return {"oi_usd": now, "change_24h": chg}
    except Exception: pass
    return {"oi_usd": None, "change_24h": 0.0}

@st.cache_data(ttl=600, show_spinner=False)
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception: return {"value": 50, "label": "Neutral"}

@st.cache_data(ttl=600, show_spinner=False)
def get_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception: return None

@st.cache_data(ttl=600, show_spinner=False)
def get_btc_macro():
    out = {"dxy_bias": "NEUTRAL", "vix_state": "UNKNOWN", "vix_value": None}
    try:
        btc = get_btc_ohlcv("6mo", "1d")
        if btc is None: return out
        for key, sym in {"dxy":"DX-Y.NYB", "vix":"^VIX"}.items():
            df = get_macro_ohlcv(sym)
            if df is None or len(df) < 40: continue
            if key == "dxy":
                e50 = float(df["close"].ewm(span=50).mean().iloc[-1])
                lp = float(df["close"].iloc[-1])
                out["dxy_bias"] = ("BULLISH" if lp > e50 else "BEARISH")
            if key == "vix":
                v = float(df["close"].iloc[-1])
                out["vix_value"] = v
                out["vix_state"] = ("HIGH FEAR" if v > 25 else "COMPLACENT" if v < 14 else "NORMAL")
    except Exception: pass
    return out

def get_btc_fundamental_score(funding, oi, fg, dominance, macro):
    bb = sb = 0.0; det = []
    fr = funding.get("rate", 0.0)
    if fr > 0.05: sb += 18; det.append({"factor": "Funding عالي", "value": f"{fr:.3f}%"})
    elif fr < -0.02: bb += 18; det.append({"factor": "Funding سلبي", "value": f"{fr:.3f}%"})
    
    fgv = fg.get("value", 50)
    if fgv >= 75: sb += 22; det.append({"factor": "Extreme Greed", "value": str(fgv)})
    elif fgv <= 25: bb += 22; det.append({"factor": "Extreme Fear", "value": str(fgv)})
    
    if macro.get("dxy_bias") == "BULLISH": sb += 8; det.append({"factor": "DXY صاعد", "value": "ضغط"})
    elif macro.get("dxy_bias") == "BEARISH": bb += 8; det.append({"factor": "DXY هابط", "value": "دعم"})
    
    return clamp(bb, 0, 100), clamp(sb, 0, 100), det

# ============================================================
# TECHNICAL INDICATORS & SMC
# ============================================================
def calc_rsi(series, period=14):
    d = series.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean(); al = l.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(50).clip(0, 100)

def calc_atr(df, period=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calc_macd(s, f=12, sl=26, sig=9):
    ef = s.ewm(span=f, adjust=False).mean(); es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es; sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def calc_bollinger(s, p=20, std=2.0):
    m = s.rolling(p).mean(); d = s.rolling(p).std()
    return m + std*d, m, m - std*d

def calc_rolling_vwap(df, period=20):
    tp = (df["high"]+df["low"]+df["close"])/3; v = df["volume"].clip(lower=0)
    return (tp*v).rolling(period).sum() / v.rolling(period).sum().replace(0, np.nan)

def find_confirmed_swings(df, order=3):
    out = df.copy(); w = 2*order+1
    rm = out["high"].rolling(w, min_periods=w).max(); rn = out["low"].rolling(w, min_periods=w).min()
    out["swing_high"] = ((out["high"] >= rm - 1e-12) & rm.notna()).shift(order).fillna(False).astype(bool)
    out["swing_low"]  = ((out["low"] <= rn + 1e-12) & rn.notna()).shift(order).fillna(False).astype(bool)
    out["swing_high_price"] = out["high"].shift(order).where(out["swing_high"], np.nan)
    out["swing_low_price"]  = out["low"].shift(order).where(out["swing_low"], np.nan)
    return out

def detect_bos_mss(df):
    out = df.copy(); n = len(out)
    bbu = np.zeros(n, dtype=bool); bbe = np.zeros(n, dtype=bool)
    mbu = np.zeros(n, dtype=bool); mbe = np.zeros(n, dtype=bool)
    sh = out["swing_high"].values; sl = out["swing_low"].values
    shp = out["swing_high_price"].values; slp = out["swing_low_price"].values
    ca = out["close"].values
    st = "RANGE"; lh = np.nan; ll = np.nan
    for i in range(n):
        if i > 0:
            if sh[i-1] and np.isfinite(shp[i-1]): lh = shp[i-1]
            if sl[i-1] and np.isfinite(slp[i-1]): ll = slp[i-1]
        if i < 10: continue
        c = ca[i]
        if np.isfinite(lh) and c > lh:
            bbu[i] = True
            if st == "BEARISH": mbu[i] = True
            st = "BULLISH"; lh = np.nan
        elif np.isfinite(ll) and c < ll:
            bbe[i] = True
            if st == "BULLISH": mbe[i] = True
            st = "BEARISH"; ll = np.nan
    out["bos_bullish"] = bbu; out["bos_bearish"] = bbe
    out["mss_bullish"] = mbu; out["mss_bearish"] = mbe
    return out

def detect_liquidity_sweeps(df, tol_atr=0.10):
    out = df.copy(); t = out["atr"] * tol_atr
    ph = out["high"].rolling(3).max().shift(1); pl = out["low"].rolling(3).min().shift(1)
    out["liquidity_sweep_bearish"] = ((out["high"] > ph + t) & (out["close"] < ph)).fillna(False)
    out["liquidity_sweep_bullish"] = ((out["low"] < pl - t) & (out["close"] > pl)).fillna(False)
    return out

def detect_fvg(df):
    out = df.copy()
    out["fvg_bullish"] = (out["low"] > out["high"].shift(2)).fillna(False)
    out["fvg_bearish"] = (out["high"] < out["low"].shift(2)).fillna(False)
    out["fvg_bull_low"]  = out["high"].shift(2).where(out["fvg_bullish"], np.nan)
    out["fvg_bull_high"] = out["low"].where(out["fvg_bullish"], np.nan)
    out["fvg_bear_low"]  = out["high"].where(out["fvg_bearish"], np.nan)
    out["fvg_bear_high"] = out["low"].shift(2).where(out["fvg_bearish"], np.nan)
    return out

def detect_order_blocks(df):
    out = df.copy()
    body = (out["close"] - out["open"]).abs()
    strong = body >= 1.20 * out["atr"]
    po = out["open"].shift(1); pc = out["close"].shift(1)
    ph = out["high"].shift(1); pl = out["low"].shift(1)
    out["order_block_bullish"] = (strong & (pc < po) & (out["close"] > ph)).fillna(False)
    out["order_block_bearish"] = (strong & (pc > po) & (out["close"] < pl)).fillna(False)
    out["ob_low"] = pl.where(out["order_block_bullish"] | out["order_block_bearish"], np.nan)
    out["ob_high"] = ph.where(out["order_block_bullish"] | out["order_block_bearish"], np.nan)
    return out

def add_premium_discount(df, lb=50):
    out = df.copy()
    sh = out["high"].rolling(lb).max().shift(1); sl = out["low"].rolling(lb).min().shift(1)
    m = (sh + sl) / 2
    out["in_discount"] = out["close"] < m; out["in_premium"] = out["close"] > m
    return out

def structure_state(df):
    h = df.index[df["swing_high"]].tolist(); l = df.index[df["swing_low"]].tolist()
    if len(h) < 2 or len(l) < 2: return {"state":"RANGE", "bullish": False, "bearish": False}
    hh = df.loc[h[-1], "swing_high_price"] > df.loc[h[-2], "swing_high_price"]
    hl = df.loc[l[-1], "swing_low_price"] > df.loc[l[-2], "swing_low_price"]
    lh = df.loc[h[-1], "swing_high_price"] < df.loc[h[-2], "swing_high_price"]
    ll = df.loc[l[-1], "swing_low_price"] < df.loc[l[-2], "swing_low_price"]
    if hh and hl: return {"state":"BULLISH", "bullish": True, "bearish": False}
    if lh and ll: return {"state":"BEARISH", "bullish": False, "bearish": True}
    return {"state":"RANGE", "bullish": False, "bearish": False}

def build_features(df):
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["rsi"] = calc_rsi(out["close"])
    out["atr"] = calc_atr(out)
    out["macd"], out["macd_signal"], out["macd_histogram"] = calc_macd(out["close"])
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = calc_bollinger(out["close"], PROFILE["bb_period"], PROFILE["bb_std"])
    out["vwap"] = calc_rolling_vwap(out)
    
    out = find_confirmed_swings(out, PROFILE["swing_order"])
    out = detect_bos_mss(out)
    out = detect_liquidity_sweeps(out)
    out = detect_fvg(out)
    out = detect_order_blocks(out)
    out = add_premium_discount(out, 100)
    return out

# ============================================================
# LOGIC & SCORING
# ============================================================
def directional_score(df, fund_buy, fund_sell, macro):
    l = df.iloc[-1]
    sc = {"BUY": {"structure":0,"trend":0,"momentum":0,"volume":0,"context":0},
          "SELL": {"structure":0,"trend":0,"momentum":0,"volume":0,"context":0}}
    
    s = structure_state(df)
    if s["bullish"]: sc["BUY"]["structure"] += 45
    if s["bearish"]: sc["SELL"]["structure"] += 45
    if df.iloc[-5:]["bos_bullish"].any(): sc["BUY"]["structure"] += 35
    if df.iloc[-5:]["bos_bearish"].any(): sc["SELL"]["structure"] += 35
    if l["in_discount"]: sc["BUY"]["structure"] += 10
    if l["in_premium"]:  sc["SELL"]["structure"] += 10
    
    if l["ema20"] > l["ema50"]: sc["BUY"]["trend"] += 50
    if l["ema20"] < l["ema50"]: sc["SELL"]["trend"] += 50
    
    if l["rsi"] > 50: sc["BUY"]["momentum"] += 40
    if l["rsi"] < 50: sc["SELL"]["momentum"] += 40
    if l["macd_histogram"] > 0: sc["BUY"]["momentum"] += 40
    if l["macd_histogram"] < 0: sc["SELL"]["momentum"] += 40
    
    if l["close"] > l["vwap"]: sc["BUY"]["volume"] += 50
    if l["close"] < l["vwap"]: sc["SELL"]["volume"] += 50
    
    sc["BUY"]["context"] += fund_buy * 0.6; sc["SELL"]["context"] += fund_sell * 0.6
    if macro.get("dxy_bias") == "BEARISH": sc["BUY"]["context"] += 15
    if macro.get("dxy_bias") == "BULLISH": sc["SELL"]["context"] += 15
    
    w = {"structure":0.28,"trend":0.18,"momentum":0.14,"volume":0.14,"context":0.26}
    tb = sum(clamp(sc["BUY"][p],0,100)*w[p] for p in w)
    ts = sum(clamp(sc["SELL"][p],0,100)*w[p] for p in w)
    return {"buy": clamp(tb,0,100), "sell": clamp(ts,0,100), "pillars": sc}

def calculate_ict_sl_tp(df, direction, current_price, atr):
    if not np.isfinite(atr) or atr <= 0: return None
    all_levels = []
    
    # 1. Inducement / Liquidity
    work = df.iloc[-50:]
    if direction == "BUY":
        sl_idx = work.index[work["swing_low"]].tolist()
        if sl_idx: all_levels.append((work.loc[sl_idx[-1], "swing_low_price"], "Swing Low", "swing"))
    else:
        sh_idx = work.index[work["swing_high"]].tolist()
        if sh_idx: all_levels.append((work.loc[sh_idx[-1], "swing_high_price"], "Swing High", "swing"))

    # 2. Order Blocks
    obs = df[df["order_block_bullish"] if direction == "BUY" else df["order_block_bearish"]].iloc[-5:]
    for _, row in obs.iterrows():
        lvl = row["ob_low"] if direction == "BUY" else row["ob_high"]
        if np.isfinite(lvl): all_levels.append((lvl, "Order Block", "ob"))

    # 3. FVGs
    fvgs = df[df["fvg_bullish"] if direction == "BUY" else df["fvg_bearish"]].iloc[-5:]
    for _, row in fvgs.iterrows():
        lvl = row["fvg_bull_low"] if direction == "BUY" else row["fvg_bear_high"]
        if np.isfinite(lvl): all_levels.append((lvl, "FVG", "fvg"))

    # فلترة المستويات لتكون صالحة للاستوب
    if direction == "BUY":
        candidates = [(l,n,t) for l,n,t in all_levels if np.isfinite(l) and l < current_price]
        candidates.sort(key=lambda x: current_price - x[0]) # الأقرب
    else:
        candidates = [(l,n,t) for l,n,t in all_levels if np.isfinite(l) and l > current_price]
        candidates.sort(key=lambda x: x[0] - current_price)
        
    if candidates:
        sl_level, sl_reason, sl_type = candidates[0]
        sl = sl_level - (atr * SL_BUFFER_ATR) if direction == "BUY" else sl_level + (atr * SL_BUFFER_ATR)
    else:
        # Fallback if no SMC level is found
        sl = current_price - (atr * 1.0) if direction == "BUY" else current_price + (atr * 1.0)
        sl_reason = "ATR Fallback"

    risk = abs(current_price - sl)
    
    # --- التعديل الجوهري: فرض حدود ATR الصارمة لمنع الاستوب الكبير جداً أو الصغير جداً ---
    if risk < SL_MIN_DIST_ATR * atr:
        sl = current_price - SL_MIN_DIST_ATR * atr if direction == "BUY" else current_price + SL_MIN_DIST_ATR * atr
        risk = abs(current_price - sl); sl_reason = "ATR Min Cap"
    if risk > SL_MAX_DIST_ATR * atr:
        sl = current_price - SL_MAX_DIST_ATR * atr if direction == "BUY" else current_price + SL_MAX_DIST_ATR * atr
        risk = abs(current_price - sl); sl_reason = "ATR Max Cap (Fixed)"

    # الأهداف (بناءً على مضاعفات المخاطرة R)
    tp1 = current_price + (risk * 1.0) if direction == "BUY" else current_price - (risk * 1.0)
    tp2 = current_price + (risk * 2.0) if direction == "BUY" else current_price - (risk * 2.0)
    tp3 = current_price + (risk * 3.0) if direction == "BUY" else current_price - (risk * 3.0)

    return {
        "entry": current_price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk": risk, "rr1": 1.0, "rr2": 2.0, "rr3": 3.0, "sl_reason": sl_reason
    }

# ============================================================
# ANALYSIS PIPELINE
# ============================================================
def analyze_btc():
    df = get_btc_ohlcv("6mo", "4h")
    if df is None or len(df) < 100: return {"error": "فشل جلب بيانات BTC"}
    x = build_features(df)
    
    funding = get_funding_rate(); oi = get_open_interest(); fg = get_fear_greed()
    dom = get_btc_dominance(); macro = get_btc_macro()
    
    fund_buy, fund_sell, fund_det = get_btc_fundamental_score(funding, oi, fg, dom, macro)
    dir_score = directional_score(x, fund_buy, fund_sell, macro)
    
    price = float(x["close"].iloc[-1]); atr = float(x["atr"].iloc[-1])
    direction = "BUY" if dir_score["buy"] > dir_score["sell"] else "SELL"
    conf = max(dir_score["buy"], dir_score["sell"])
    gap = abs(dir_score["buy"] - dir_score["sell"])
    
    ict = calculate_ict_sl_tp(x, direction, price, atr)
    
    emit = (conf >= MIN_EMIT_CONFIDENCE and gap >= MIN_SIGNAL_GAP and ict is not None)
    
    grade = "A+" if conf >= A_PLUS_MIN else "A" if conf >= A_MIN else "B" if conf >= B_MIN else "C" if conf >= C_MIN else "D"

    return {
        "symbol": SYMBOL, "direction": direction, "signal": direction,
        "confidence": conf, "confidence_gap": gap, "trade_grade": grade,
        "pillars": dir_score["pillars"], "ict": ict, "emit": emit,
        "funding": funding, "open_interest": oi, "fear_greed": fg,
        "dominance": dom, "macro": macro, "fund_details": fund_det,
        "price": price, "df": x, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "spot": get_btc_spot()
    }

# ============================================================
# UI STYLING & COMPONENTS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap');
.stApp { background: radial-gradient(ellipse at top, #1a1428 0%, #0a0a12 60%, #050508 100%); color: #e8e8f0; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #100c1a 0%, #08060f 100%); border-right: 1px solid rgba(247,147,26,0.15); }
.bp-header { background: linear-gradient(135deg, rgba(247,147,26,0.15), rgba(255,193,7,0.05)); border: 1px solid rgba(247,147,26,0.3); border-radius: 18px; padding: 22px 30px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(247,147,26,0.1); }
.bp-header h1 { margin: 0; font-size: 26px; font-weight: 800; background: linear-gradient(90deg, #f7931a, #ffd54f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px; }
.bp-header .sub { color: #8a8aa0; font-size: 13px; margin-top: 4px; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }
.metric-card { background: linear-gradient(145deg, rgba(30,25,45,0.9), rgba(15,12,22,0.9)); border: 1px solid rgba(247,147,26,0.2); border-radius: 14px; padding: 16px 18px; position: relative; overflow: hidden; }
.metric-card:hover { border-color: rgba(247,147,26,0.5); box-shadow: 0 8px 24px rgba(247,147,26,0.15); }
.metric-label { color: #8a8aa0; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.metric-value { color: #fff; font-size: 22px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.metric-delta { font-size: 12px; margin-top: 4px; font-weight: 600; }
.delta-up { color: #26a69a; } .delta-down { color: #ef5350; } .delta-neutral { color: #8a8aa0; }
.signal-card { border-radius: 20px; padding: 28px 32px; margin-bottom: 20px; position: relative; border: 2px solid; }
.signal-buy { background: linear-gradient(135deg, rgba(38,166,154,0.18), rgba(38,166,154,0.03)); border-color: rgba(38,166,154,0.5); }
.signal-sell { background: linear-gradient(135deg, rgba(239,83,80,0.18), rgba(239,83,80,0.03)); border-color: rgba(239,83,80,0.5); }
.signal-wait { background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.02)); border-color: rgba(255,193,7,0.4); }
.signal-title { font-size: 13px; font-weight: 700; color: #8a8aa0; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.signal-main { font-size: 44px; font-weight: 800; font-family: 'JetBrains Mono', monospace; letter-spacing: 2px; margin-bottom: 4px; }
.grade-badge { display: inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 800; font-size: 15px; margin-left: 14px; vertical-align: middle; }
.grade-Aplus { background: linear-gradient(135deg, #ffd54f, #f7931a); color: #1a1428; }
.grade-A { background: #26a69a; color: #fff; }
.grade-B { background: #2196f3; color: #fff; }
.grade-C { background: #ff9800; color: #fff; }
.grade-D { background: #757575; color: #fff; }
.conf-bar { height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; margin-top: 12px; }
.conf-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f7931a, #ffd54f); }
.info-box { background: rgba(30,25,45,0.6); border: 1px solid rgba(255,255,255,0.08); border-left: 3px solid #f7931a; border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; font-size: 13px; }
.stButton > button { background: linear-gradient(135deg, #f7931a, #ffd54f); color: #1a1428; border: none; border-radius: 10px; font-weight: 700; padding: 10px 22px; }
</style>
"""

def render_header():
    st.markdown(f"""<div class="bp-header"><div><h1>₿ BLACK PYRAMID · BTC TERMINAL</h1><div class="sub">Institutional Analysis · {APP_VERSION} · {SYMBOL}</div></div><div style="text-align:right;"><div class="sub">🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div></div></div>""", unsafe_allow_html=True)

def metric_card(label, value, delta=None, delta_dir="neutral"):
    delta_html = f'<div class="metric-delta delta-{delta_dir}">{delta}</div>' if delta else ""
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{delta_html}</div>""", unsafe_allow_html=True)

def render_metrics_row(result):
    spot = result.get("spot", (None, None)); price = spot[0] or result.get("price"); chg = spot[1]
    funding = result.get("funding", {}); oi = result.get("open_interest", {})
    fg = result.get("fear_greed", {}); dom = result.get("dominance")
    
    cols = st.columns(6)
    with cols[0]:
        metric_card("BTC PRICE", fmt_price(price), fmt_pct(chg), "up" if (chg or 0) > 0 else "down")
    with cols[1]:
        metric_card("FUNDING", f"{funding.get('rate', 0):.4f}%", funding.get("state",""), "down" if funding.get('rate',0)>0.02 else "up")
    with cols[2]:
        metric_card("OPEN INTEREST", f"${(oi.get('oi_usd') or 0)/1e9:.1f}B", fmt_pct(oi.get("change_24h",0)), "up" if oi.get("change_24h",0)>0 else "down")
    with cols[3]:
        metric_card("FEAR & GREED", str(fg.get("value", 50)), fg.get("label",""), "neutral")
    with cols[4]:
        metric_card("BTC DOMINANCE", f"{dom:.1f}%" if dom else "N/A")
    with cols[5]:
        vix = result.get("macro", {}).get("vix_value")
        metric_card("VIX", f"{vix:.1f}" if vix else "N/A", result.get("macro", {}).get("vix_state",""), "neutral")

def render_signal_card(result):
    direction = result["signal"]; grade = result["trade_grade"]; conf = result["confidence"]; emit = result["emit"]
    css = "signal-wait" if not emit else ("signal-buy" if direction == "BUY" else "signal-sell")
    color = "#ffd54f" if not emit else ("#26a69a" if direction == "BUY" else "#ef5350")
    main = f"{'▲' if direction=='BUY' else '▼'} {direction}" if emit else "⏸ WAIT"
    
    st.markdown(f"""<div class="signal-card {css}"><div class="signal-title">{'إشارة مُصدَرة' if emit else 'لا إشارة — انتظار'}</div><div class="signal-main" style="color:{color};">{main}<span class="grade-badge grade-{grade.replace('+','plus')}">{grade}</span></div><div style="color:#8a8aa0; font-size:14px; margin-top:6px;">الثقة: <b style="color:#fff;">{conf:.1f}%</b> · فجوة: <b style="color:#fff;">{result['confidence_gap']:.1f}</b></div><div class="conf-bar"><div class="conf-fill" style="width:{conf}%;"></div></div></div>""", unsafe_allow_html=True)

    if emit and result.get("ict"):
        ict = result["ict"]; cols = st.columns(5)
        labels = [("ENTRY", ict["entry"], "#ffd54f"), ("SL", ict["sl"], "#ef5350"), ("TP1", ict["tp1"], "#26a69a"), ("TP2", ict["tp2"], "#26a69a"), ("TP3", ict["tp3"], "#26a69a")]
        for c, (lbl, val, col) in zip(cols, labels):
            with c: st.markdown(f"""<div class="metric-card" style="border-left:3px solid {col};"><div class="metric-label">{lbl}</div><div class="metric-value" style="color:{col};">{fmt_price(val)}</div></div>""", unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title="Black Pyramid · BTC", page_icon="₿", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    render_header()

    with st.sidebar:
        st.markdown("### 🎯 Kill Zones")
        kz_in, kz_msg = in_kill_zone()
        st.success(kz_msg) if kz_in else st.info(kz_msg)
        
        st.markdown("---")
        st.markdown("### 🤖 ML Model")
        if not SKLEARN_AVAILABLE:
            st.error("مكتبة scikit-learn غير مثبتة على السيرفر.\nأضف `scikit-learn` إلى ملف `requirements.txt`.")
        else:
            st.success("المكتبات جاهزة لتدريب النموذج.")
            
    with st.spinner("🔍 تحليل BTC..."):
        result = analyze_btc()

    if "error" in result:
        st.error(f"❌ {result['error']}"); return

    render_metrics_row(result)
    render_signal_card(result)

    tabs = st.tabs(["📈 Chart", "📊 Pillars", "💰 Fundamentals"])
    with tabs[0]:
        df = result["df"].iloc[-150:]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="BTC"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20", line=dict(color="#f7931a")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50", line=dict(color="#2196f3")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(color="#9c27b0")), row=2, col=1)
        
        ict = result.get("ict")
        if ict:
            for key, color, label in [("sl","#ef5350","SL"), ("tp1","#26a69a","TP1"), ("tp2","#26a69a","TP2")]:
                if ict.get(key): fig.add_hline(y=ict[key], line_dash="dash", line_color=color, annotation_text=f"{label} {fmt_price(ict[key])}", row=1, col=1)
                
        fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,20,0.4)")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        dp = result["pillars"].get(result["signal"], {})
        fig = go.Figure(go.Bar(x=list(dp.values()), y=list(dp.keys()), orientation="h", marker_color="#f7931a"))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,20,0.4)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[2]:
        st.markdown("### 📋 تفاصيل العوامل الأساسية")
        for d in result["fund_details"]:
            st.markdown(f"""<div class="info-box">🔹 <b>{d['factor']}</b>: {d['value']}</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()