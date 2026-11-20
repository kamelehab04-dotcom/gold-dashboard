# ============================================================
# BLACK PYRAMID v2010.4 — BTC HIGH-ACCURACY EDITION
# Institutional Analysis Terminal · Bitcoin Only
# ============================================================

import os, base64, logging, warnings, math, time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import concurrent.futures

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
    from sklearn.model_selection import train_test_split, GridSearchCV, \
        RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, \
        log_loss, f1_score, precision_score, recall_score
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
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

APP_VERSION = "v2010.4-BTC"
SYMBOL      = "BTC-USD"
PAIR_NAME   = "BTC/USD"
ASSET_TYPE  = "crypto"

PROFILE = {
    "atr_period": 14, "rsi_period": 14, "rsi_ob": 80, "rsi_os": 20,
    "bb_period": 50, "bb_std": 2.3, "atr_sl": 2.00, "atr_trail": 1.50,
    "swing_order": 4, "structure_lookback": 250,
    "confidence_threshold": 76, "confirmation_threshold": 70,
}

MODE = "balanced"
MAX_STRONG_TRADES = 4

_GATE_PRESETS = {
    "strict":   {"gate": 80.0, "a_plus": 85.0, "a": 80.0, "b": 72.0,
                 "c": 65.0, "gap": 15},
    "balanced": {"gate": 75.0, "a_plus": 82.0, "a": 75.0, "b": 68.0,
                 "c": 60.0, "gap": 12},
    "relaxed":  {"gate": 72.0, "a_plus": 80.0, "a": 72.0, "b": 65.0,
                 "c": 55.0, "gap": 10},
}
_p = _GATE_PRESETS[MODE]
A_PLUS_MIN, A_MIN, B_MIN, C_MIN = _p["a_plus"], _p["a"], _p["b"], _p["c"]
MIN_SIGNAL_GAP      = _p["gap"]
MIN_EMIT_CONFIDENCE = _p["gate"]
BTC_CONFIDENCE_GATE = 70.0

PENALTY_MTF_AGAINST    = 8.0
PENALTY_RANGE_REGIME   = 6.0
PENALTY_WEAK_CANDLE    = 5.0
PENALTY_WEEKLY_AGAINST = 4.0
MAX_SOFT_PENALTY       = 8.0
SOFT_PENALTY_TOP_N     = 3

MIN_RR_TP1, MIN_RR_TP2, MIN_RR_TP3 = 0.80, 1.30, 1.90
SL_MIN_DIST_ATR = 0.25
SL_MAX_DIST_ATR = 1.50
SL_BUFFER_ATR   = 0.20

CORRELATION_WINDOW_DAYS = 30
SENTIMENT_EXTREME_LONG  = 75
SENTIMENT_EXTREME_SHORT = 25

ML_MIN_TRADES = 30
ML_TEST_SIZE  = 0.25
ML_FEATURE_SELECTION_THRESHOLD = 0.02
ML_CV_FOLDS = 5
TUNING_MODE = "randomized"
TUNING_N_ITER = 20
TUNING_CV_FOLDS = 5
TUNING_SCORING  = "roc_auc"
TUNING_N_JOBS   = -1

ML_FEATURE_NAMES = [
    "structure","trend","momentum","volume","context",
    "mtf_conf","confirmation","raw_conf","regime_enc","session_enc",
    "mtf_match","weekly_match","htf_match","session_match",
    "funding_rate","funding_bias","oi_change","dominance",
    "ema200_dist","atr_ratio","bb_position","vwap_dist",
    "rsi_val","macd_hist_norm","body_ratio","wick_ratio",
    "consec_bars","candles_since_bos","divergence_flag","vol_ratio",
]

PARAM_GRIDS = {
    "LogisticRegression": {"C":[0.01,0.1,0.5,1.0,5.0,10.0],
                           "solver":["lbfgs","liblinear"],
                           "class_weight":[None,"balanced"]},
    "GradientBoosting": {"n_estimators":[80,150,250],"max_depth":[3,4,5,6],
                          "learning_rate":[0.03,0.05,0.08,0.10],
                          "subsample":[0.8,0.9,1.0],
                          "min_samples_leaf":[1,3,5]},
    "RandomForest": {"n_estimators":[100,200,300],"max_depth":[4,6,8,None],
                      "min_samples_split":[2,5,10],
                      "class_weight":[None,"balanced"]},
    "XGBoost": {"n_estimators":[100,200,300],"max_depth":[3,4,5,6],
                 "learning_rate":[0.03,0.05,0.08,0.10],
                 "subsample":[0.8,0.9,1.0],
                 "colsample_bytree":[0.8,0.9,1.0]},
}

KILL_ZONES = {
    "Asia Pump":    (0, 4),
    "London Open":  (7, 10),
    "NY Open":      (12, 15),
    "London Close": (15, 17),
    "US Close":     (20, 23),
}

MACRO_SYMBOLS = {"dxy":"DX-Y.NYB","vix":"^VIX","spx":"^GSPC"}

LOGO_CANDIDATES = ["file_000000005cb4824697f509df31f2168a.png",
                    "logo.png","assets/logo.png","static/logo.png"]

# ============================================================
# UTIL
# ============================================================

def safe_float(v, default=np.nan):
    try:
        x = float(v); return x if np.isfinite(x) else default
    except Exception: return default

def clamp(v, lo, hi): return max(lo, min(hi, v))

def safe_bool(v):
    try: return bool(v) and not pd.isna(v)
    except Exception: return False

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

def img_to_base64(path):
    try:
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    except Exception: return None

def load_logo_b64():
    for p in LOGO_CANDIDATES:
        if Path(p).exists():
            b64 = img_to_base64(p)
            if b64: return b64
    return None

def get_secret(name, default=""):
    try:
        if name in st.session_state:
            v = st.session_state[name]
            if v is not None: return str(v)
    except Exception: pass
    try:
        v = st.secrets.get(name, None)
        if v is not None: return str(v)
    except Exception: pass
    return os.getenv(name, default)

def init_state():
    defaults = {
        "all_signals": None, "analyzing_all": False, "analysis_time": None,
        "backtest_results": None, "strict_filters": False,
        "show_calendar_today": False, "show_session_info": False,
        "show_market_status": False, "trade_journal": [],
        "ml_model": None, "show_details": False,
        "strong_trade_result": None, "btc_result": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ============================================================
# SESSION / MARKET
# ============================================================

def get_current_session_info():
    now = datetime.now(timezone.utc); h = now.hour
    sessions = {"Asian":(0,7,"🌏","طوكيو/سيدني"),
                "London":(7,12,"🇬🇧","لندن"),
                "Overlap":(12,16,"🔥","London/NY Overlap"),
                "NY":(16,21,"🇺🇸","نيويورك"),
                "After-Hours":(21,24,"🌙","بعد الإغلاق")}
    cur, ic, dsc = "After-Hours", "🌙", "بعد الإغلاق"
    for name,(s,e,i,d) in sessions.items():
        if s <= h < e: cur, ic, dsc = name, i, d; break
    kz = "خارج Kill Zones"
    for n,(s,e) in KILL_ZONES.items():
        if s <= h < e: kz = f"داخل {n} ✅"; break
    return {"session":cur, "icon":ic, "desc":dsc,
            "time_utc": f"{h:02d}:{now.minute:02d} UTC",
            "kill_zone": kz, "hour": h}

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
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open": rename[c] = "open"
        elif lc == "high": rename[c] = "high"
        elif lc == "low": rename[c] = "low"
        elif lc == "close": rename[c] = "close"
        elif lc in ("volume","vol"): rename[c] = "volume"
    df = df.rename(columns=rename)
    req = ["open","high","low","close"]
    if any(c not in df.columns for c in req): return None
    if "volume" not in df.columns: df["volume"] = 0.0
    for c in req + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[req + ["volume"]].dropna(subset=req)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) >= min_rows else None

@st.cache_data(ttl=60, show_spinner=False)
def get_btc_ohlcv(period="6mo", interval="4h"):
    try:
        return normalize_ohlcv(yf.download(SYMBOL, period=period,
            interval=interval, auto_adjust=False, progress=False,
            threads=False))
    except Exception: return None

@st.cache_data(ttl=30, show_spinner=False)
def get_btc_spot():
    try:
        df = normalize_ohlcv(yf.download(SYMBOL, period="5d",
            interval="1h", auto_adjust=False, progress=False,
            threads=False), min_rows=2)
        if df is not None and len(df) >= 2:
            lt = float(df["close"].iloc[-1])
            pc = float(df["close"].iloc[-2])
            return lt, ((lt - pc) / pc * 100) if pc else 0.0
    except Exception: pass
    try:
        df = get_btc_ohlcv("3mo", "4h")
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1]), 0.0
    except Exception: pass
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def get_macro_ohlcv(sym, period="6mo", interval="1d"):
    try:
        return normalize_ohlcv(yf.download(sym, period=period,
            interval=interval, auto_adjust=False, progress=False,
            threads=False))
    except Exception: return None

# ============================================================
# CRYPTO FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=180, show_spinner=False)
def get_funding_rate():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                        params={"symbol": "BTCUSDT"}, timeout=8)
        d = r.json()
        fr = float(d.get("lastFundingRate", 0)) * 100
        return {"rate": fr, "mark_price": float(d.get("markPrice", 0)),
                "state": ("HOT_LONG"  if fr >  0.05 else
                          "LONG"       if fr >  0.02 else
                          "HOT_SHORT"  if fr < -0.02 else "BALANCED")}
    except Exception:
        return {"rate": 0.0, "state": "UNKNOWN"}

@st.cache_data(ttl=180, show_spinner=False)
def get_open_interest():
    try:
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
                        params={"symbol":"BTCUSDT","period":"4h","limit":24},
                        timeout=8)
        data = r.json()
        if isinstance(data, list) and len(data) > 1:
            now = float(data[-1]["sumOpenInterestValue"])
            prev = float(data[0]["sumOpenInterestValue"])
            chg = (now - prev) / prev * 100 if prev else 0.0
            return {"oi_usd": now, "change_24h": chg,
                    "state": ("EXPANDING" if chg > 5 else
                              "CONTRACTING" if chg < -5 else "FLAT")}
    except Exception: pass
    return {"oi_usd": None, "change_24h": 0.0, "state": "UNKNOWN"}

@st.cache_data(ttl=600, show_spinner=False)
def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "label": d["value_classification"]}
    except Exception:
        return {"value": 50, "label": "Neutral"}

@st.cache_data(ttl=600, show_spinner=False)
def get_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=8)
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception: return None

@st.cache_data(ttl=600, show_spinner=False)
def get_btc_macro():
    out = {"dxy_corr": None, "vix_corr": None, "spx_corr": None,
           "dxy_bias": "NEUTRAL", "vix_state": "UNKNOWN", "vix_value": None}
    try:
        btc = get_btc_ohlcv("6mo", "1d")
        if btc is None or len(btc) < 40: return out
        br = btc["close"].pct_change().dropna()
        for key, sym in MACRO_SYMBOLS.items():
            df = get_macro_ohlcv(sym)
            if df is None or len(df) < 40: continue
            mr = df["close"].pct_change().dropna()
            al = pd.concat([br, mr], axis=1, join="inner").dropna()
            if len(al) >= 30:
                c = float(al.iloc[-30:, 0].corr(al.iloc[-30:, 1]))
                if np.isfinite(c): out[f"{key}_corr"] = c
            if key == "dxy":
                e50 = float(df["close"].ewm(span=50).mean().iloc[-1])
                lp = float(df["close"].iloc[-1])
                out["dxy_bias"] = ("BULLISH" if lp > e50 else
                                    "BEARISH" if lp < e50 else "NEUTRAL")
            if key == "vix":
                v = float(df["close"].iloc[-1])
                out["vix_value"] = v
                out["vix_state"] = ("HIGH FEAR" if v > 25 else
                                    "ELEVATED" if v > 20 else
                                    "COMPLACENT" if v < 14 else "NORMAL")
    except Exception: pass
    return out

def get_btc_fundamental_score(funding=None, oi=None, fg=None,
                              dominance=None, macro=None):
    if funding is None: funding = get_funding_rate()
    if oi is None: oi = get_open_interest()
    if fg is None: fg = get_fear_greed()
    if dominance is None: dominance = get_btc_dominance()
    if macro is None: macro = get_btc_macro()

    bb = sb = 0.0
    det = []

    fr = funding.get("rate", 0.0)
    if fr > 0.05:
        sb += 18; det.append({"factor":"Funding عالي جداً",
                              "value": f"{fr:.3f}% → ضغط بيع"})
    elif fr > 0.02:
        sb += 8; det.append({"factor":"Funding مرتفع", "value": f"{fr:.3f}%"})
    elif fr < -0.02:
        bb += 18; det.append({"factor":"Funding سلبي",
                              "value": f"{fr:.3f}% → Short squeeze"})

    oic = oi.get("change_24h", 0.0)
    if oic > 5.0:
        det.append({"factor":"OI يتوسع", "value": f"+{oic:.1f}%"})
    elif oic < -5.0:
        det.append({"factor":"OI ينكمش", "value": f"{oic:.1f}%"})

    fgv = fg.get("value", 50)
    if fgv >= 75:
        sb += 22; det.append({"factor":"Extreme Greed", "value": str(fgv)})
    elif fgv >= 60:
        sb += 10; det.append({"factor":"Greed", "value": str(fgv)})
    elif fgv <= 25:
        bb += 22; det.append({"factor":"Extreme Fear", "value": str(fgv)})
    elif fgv <= 40:
        bb += 10; det.append({"factor":"Fear", "value": str(fgv)})

    if dominance is not None:
        if dominance > 55:
            bb += 8; det.append({"factor":"Dominance عالية",
                                 "value": f"{dominance:.1f}%"})
        elif dominance < 45:
            sb += 5; det.append({"factor":"Dominance منخفضة",
                                 "value": f"{dominance:.1f}%"})

    vs = macro.get("vix_state", "")
    if vs == "HIGH FEAR":
        sb += 12; det.append({"factor":"VIX خوف مرتفع", "value": vs})
    elif vs == "COMPLACENT":
        bb += 6; det.append({"factor":"VIX هادئ", "value": vs})

    if macro.get("dxy_bias") == "BULLISH":
        sb += 8; det.append({"factor":"DXY صاعد", "value":"ضغط على BTC"})
    elif macro.get("dxy_bias") == "BEARISH":
        bb += 8; det.append({"factor":"DXY هابط", "value":"دعم BTC"})

    return clamp(bb, 0, 100), clamp(sb, 0, 100), det

# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calc_rsi(series, period=14):
    d = series.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    al = l.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    r = 100 - (100/(1+rs))
    r = r.where(al > 0, 100.0); r = r.where(ag > 0, 0.0)
    return r.fillna(50).clip(0, 100)

def calc_atr(df, period=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def calc_macd(s, f=12, sl=26, sig=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    m = ef - es; sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def calc_bollinger(s, p=20, std=2.0):
    m = s.rolling(p).mean(); d = s.rolling(p).std()
    return m + std*d, m, m - std*d

def calc_chaikin(df, period=21):
    hl = (df["high"]-df["low"]).replace(0, np.nan)
    mult = ((df["close"]-df["low"]) - (df["high"]-df["close"])) / hl
    mv = mult.fillna(0) * df["volume"]
    return (mv.rolling(period).sum() /
            df["volume"].rolling(period).sum().replace(0, np.nan)).fillna(0)

def calc_rolling_vwap(df, period=20):
    tp = (df["high"]+df["low"]+df["close"])/3
    v = df["volume"].clip(lower=0); pv = tp*v
    cpv = pv.rolling(period, min_periods=period).sum()
    cv = v.rolling(period, min_periods=period).sum()
    vw = cpv / cv.replace(0, np.nan)
    fb = tp.rolling(period, min_periods=period).mean()
    return vw.fillna(fb)

def calc_session_vwap(df):
    w = df.copy()
    dt = w.index.normalize() if w.index.tz is None \
         else w.index.tz_convert("UTC").normalize()
    tp = (w["high"]+w["low"]+w["close"])/3
    pv = tp * w["volume"].clip(lower=0)
    cpv = pv.groupby(dt).cumsum(); cv = w["volume"].clip(lower=0).groupby(dt).cumsum()
    vw = cpv / cv.replace(0, np.nan)
    fb = tp.groupby(dt).expanding().mean().reset_index(level=0, drop=True)
    return vw.fillna(fb)

# ============================================================
# SMC / ICT
# ============================================================

def find_confirmed_swings(df, order=3):
    out = df.copy(); w = 2*order+1
    if len(out) < w:
        for c in ["swing_high","swing_low"]: out[c] = False
        for c in ["swing_high_price","swing_low_price"]: out[c] = np.nan
        return out
    rm = out["high"].rolling(w, min_periods=w).max()
    rn = out["low"].rolling(w, min_periods=w).min()
    im = (out["high"] >= rm - 1e-12) & rm.notna()
    inn = (out["low"] <= rn + 1e-12) & rn.notna()
    out["swing_high"] = im.shift(order).fillna(False).astype(bool)
    out["swing_low"]  = inn.shift(order).fillna(False).astype(bool)
    out["swing_high_price"] = out["high"].shift(order).where(out["swing_high"], np.nan)
    out["swing_low_price"]  = out["low"].shift(order).where(out["swing_low"], np.nan)
    return out

def get_last_two_swings(df, kind="high"):
    c = "swing_high" if kind == "high" else "swing_low"
    pc = "swing_high_price" if kind == "high" else "swing_low_price"
    if c not in df.columns: return None
    p = df.index[df[c]].tolist()
    if len(p) < 2: return None
    if pc in df.columns:
        v = df.loc[p, pc].fillna(df.loc[p, "high" if kind=="high" else "low"]).tolist()
    else:
        v = df.loc[p, "high" if kind=="high" else "low"].tolist()
    return [(p[-2], float(v[-2])), (p[-1], float(v[-1]))]

def structure_state(df):
    h = get_last_two_swings(df, "high"); l = get_last_two_swings(df, "low")
    bu = be = False; s = "RANGE"
    if h and l:
        hh = h[-1][1] > h[-2][1]; hl = l[-1][1] > l[-2][1]
        lh = h[-1][1] < h[-2][1]; ll = l[-1][1] < l[-2][1]
        if hh and hl: bu = True; s = "BULLISH"
        elif lh and ll: be = True; s = "BEARISH"
    return {"state": s, "bullish": bu, "bearish": be, "highs": h, "lows": l}

def detect_bos_mss(df):
    out = df.copy(); n = len(out)
    bbu = np.zeros(n, dtype=bool); bbe = np.zeros(n, dtype=bool)
    mbu = np.zeros(n, dtype=bool); mbe = np.zeros(n, dtype=bool)
    if n == 0:
        out["bos_bullish"] = bbu; out["bos_bearish"] = bbe
        out["mss_bullish"] = mbu; out["mss_bearish"] = mbe
        return out
    sh = out["swing_high"].values; sl = out["swing_low"].values
    shp = out["swing_high_price"].values if "swing_high_price" in out.columns \
          else out["high"].values
    slp = out["swing_low_price"].values if "swing_low_price" in out.columns \
          else out["low"].values
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
    ph = out["high"].rolling(3).max().shift(1)
    pl = out["low"].rolling(3).min().shift(1)
    out["liquidity_sweep_bearish"] = ((out["high"] > ph + t) & (out["close"] < ph)).fillna(False)
    out["liquidity_sweep_bullish"] = ((out["low"] < pl - t) & (out["close"] > pl)).fillna(False)
    return out

def detect_fvg(df):
    out = df.copy()
    bf = (out["low"] > out["high"].shift(2)).fillna(False)
    be = (out["high"] < out["low"].shift(2)).fillna(False)
    out["fvg_bullish"] = bf; out["fvg_bearish"] = be
    out["fvg_bull_low"]  = out["high"].shift(2).where(bf, np.nan)
    out["fvg_bull_high"] = out["low"].where(bf, np.nan)
    out["fvg_bear_low"]  = out["high"].where(be, np.nan)
    out["fvg_bear_high"] = out["low"].shift(2).where(be, np.nan)
    return out

def detect_order_blocks(df):
    out = df.copy()
    body = (out["close"] - out["open"]).abs()
    strong = body >= 1.20 * out["atr"]
    po = out["open"].shift(1); pc = out["close"].shift(1)
    ph = out["high"].shift(1); pl = out["low"].shift(1)
    bbo = (strong & (pc < po) & (out["close"] > ph)).fillna(False)
    beo = (strong & (pc > po) & (out["close"] < pl)).fillna(False)
    out["order_block_bullish"] = bbo
    out["order_block_bearish"] = beo
    valid = bbo | beo
    out["ob_low"] = pl.where(valid, np.nan)
    out["ob_high"] = ph.where(valid, np.nan)
    return out

def add_premium_discount(df, lb=50):
    out = df.copy()
    sh = out["high"].rolling(lb).max().shift(1)
    sl = out["low"].rolling(lb).min().shift(1)
    m = (sh + sl) / 2
    out["range_high"] = sh; out["range_low"] = sl; out["premium_mid"] = m
    out["in_discount"] = out["close"] < m
    out["in_premium"]  = out["close"] > m
    return out

def analyze_smc(df, profile):
    out = df.copy()
    out = find_confirmed_swings(out, profile["swing_order"])
    out = detect_bos_mss(out)
    out = detect_liquidity_sweeps(out)
    out = detect_fvg(out)
    out = detect_order_blocks(out)
    out = add_premium_discount(out, min(profile["structure_lookback"], 100))
    hi = out.index[out["swing_high"]]; lo = out.index[out["swing_low"]]
    out["bsl"] = np.nan; out["ssl"] = np.nan
    if len(hi):
        v = safe_float(out.loc[hi[-1], "swing_high_price"],
                        safe_float(out.loc[hi[-1], "high"]))
        if np.isfinite(v): out["bsl"] = v
    if len(lo):
        v = safe_float(out.loc[lo[-1], "swing_low_price"],
                        safe_float(out.loc[lo[-1], "low"]))
        if np.isfinite(v): out["ssl"] = v
    return out

def candle_confirmation(df, direction):
    if len(df) < 3: return False
    l = df.iloc[-1]
    b = abs(l["close"] - l["open"]); rng = max(l["high"] - l["low"], 1e-12)
    if direction == "BUY":
        return bool(l["close"] > l["open"] and
                    (l["close"]-l["low"])/rng >= 0.60 and b/rng >= 0.35)
    return bool(l["close"] < l["open"] and
                (l["high"]-l["close"])/rng >= 0.60 and b/rng >= 0.35)

def detect_divergence(df):
    if len(df) < 30: return None
    w = find_confirmed_swings(df, 3)
    lo = w.index[w["swing_low"]].tolist()
    hi = w.index[w["swing_high"]].tolist()
    if len(lo) >= 2:
        p1, p2 = lo[-2], lo[-1]
        lp1 = safe_float(w.loc[p1, "swing_low_price"], safe_float(w.loc[p1, "low"]))
        lp2 = safe_float(w.loc[p2, "swing_low_price"], safe_float(w.loc[p2, "low"]))
        if (np.isfinite(lp1) and np.isfinite(lp2) and
            lp2 < lp1 and w.loc[p2,"rsi"] > w.loc[p1,"rsi"]):
            return "BULLISH"
    if len(hi) >= 2:
        p1, p2 = hi[-2], hi[-1]
        hp1 = safe_float(w.loc[p1, "swing_high_price"], safe_float(w.loc[p1, "high"]))
        hp2 = safe_float(w.loc[p2, "swing_high_price"], safe_float(w.loc[p2, "high"]))
        if (np.isfinite(hp1) and np.isfinite(hp2) and
            hp2 > hp1 and w.loc[p2,"rsi"] < w.loc[p1,"rsi"]):
            return "BEARISH"
    return None

def smc_quality(df):
    if df is None or len(df) < 5: return 0.0, []
    l = df.iloc[-1]; sc = 0.0; rr = []
    checks = [("bos_bullish","bos_bearish",20,"BOS"),
              ("mss_bullish","mss_bearish",20,"MSS"),
              ("liquidity_sweep_bullish","liquidity_sweep_bearish",20,"Liquidity"),
              ("order_block_bullish","order_block_bearish",15,"OB"),
              ("fvg_bullish","fvg_bearish",10,"FVG"),
              ("in_discount","in_premium",15,"Prem/Disc")]
    for a,b,pts,lab in checks:
        if safe_bool(l.get(a)) or safe_bool(l.get(b)):
            sc += pts; rr.append(lab)
    return min(sc, 100.0), rr

# ============================================================
# FEATURE BUILDING
# ============================================================

def build_features(df, profile, use_rvwap=True):
    out = df.copy()
    out["ema20"]  = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"]  = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["rsi"]    = calc_rsi(out["close"], profile["rsi_period"])
    out["atr"]    = calc_atr(out, profile["atr_period"])
    out["macd"], out["macd_signal"], out["macd_histogram"] = calc_macd(out["close"])
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = calc_bollinger(
        out["close"], profile["bb_period"], profile["bb_std"])
    out["chaikin_mf"] = calc_chaikin(out, 21)
    try: bh = (out.index[-1] - out.index[-2]).total_seconds()/3600
    except Exception: bh = 4.0
    if use_rvwap and bh >= 3.5:
        out["vwap"] = calc_rolling_vwap(out, 20); out["vwap_type"] = "rolling"
    else:
        out["vwap"] = calc_session_vwap(out); out["vwap_type"] = "session"
    out = analyze_smc(out, profile)
    return out

def timeframe_bias(df):
    if df is None or len(df) < 80: return "NEUTRAL", 0
    x = build_features(df, PROFILE); l = x.iloc[-1]
    bu = be = 0.0
    if l["ema20"] > l["ema50"]: bu += 1
    elif l["ema20"] < l["ema50"]: be += 1
    if l["ema50"] > l["ema200"]: bu += 1
    elif l["ema50"] < l["ema200"]: be += 1
    if l["macd_histogram"] > 0: bu += 1
    elif l["macd_histogram"] < 0: be += 1
    r = safe_float(l.get("rsi"), 50)
    if r >= 55: bu += 1
    elif r <= 45: be += 1
    s = structure_state(x)
    if s["bullish"]: bu += 2
    elif s["bearish"]: be += 2
    if safe_bool(l.get("bos_bullish")) or safe_bool(l.get("mss_bullish")): bu += 1
    if safe_bool(l.get("bos_bearish")) or safe_bool(l.get("mss_bearish")): be += 1
    if bu > be + 0.75: return "BULLISH", int(round(min(10, 10*bu/max(bu+be,1))))
    if be > bu + 0.75: return "BEARISH", int(round(min(10, 10*be/max(bu+be,1))))
    return "NEUTRAL", 0

@st.cache_data(ttl=120, show_spinner=False)
def get_mtf_analysis():
    frames = {"1D":("1y","1d",4.0), "4H":("6mo","4h",3.0),
              "1H":("3mo","1h",2.0), "15M":("30d","15m",1.0)}
    rs = {}; bu = be = tot = 0.0
    for n,(p,i,w) in frames.items():
        f = get_btc_ohlcv(p, i)
        b, s = timeframe_bias(f)
        rs[n] = {"bias": b, "strength": s, "weight": w}
        c = w * (s/10.0)
        if b == "BULLISH": bu += c
        elif b == "BEARISH": be += c
        tot += w
    fi = ("BULLISH" if bu > be*1.20 and bu > 1.5 else
          "BEARISH" if be > bu*1.20 and be > 1.5 else "NEUTRAL")
    cf = clamp(50 + abs(bu-be)/max(tot,1)*50, 50, 95)
    return fi, cf, rs

def detect_regime(df):
    l = df.iloc[-1]
    atr = safe_float(l["atr"], np.nan)
    price = safe_float(l["close"], np.nan)
    if not np.isfinite(atr) or atr <= 0 or price <= 0:
        return "UNKNOWN", 50.0
    ema_spread = l["ema20"] - l["ema50"]
    ts = abs(ema_spread)/atr
    if ts >= 0.5:
        return ("TREND_BULLISH" if ema_spread > 0 else "TREND_BEARISH"), 75
    try:
        bw = (df["bb_upper"]-df["bb_lower"])/df["close"]
        bwm = bw.rolling(100).mean().iloc[-1]
        bws = bw.rolling(100).std().iloc[-1]
        if (np.isfinite(bws) and bws > 0 and
            (bw.iloc[-1]-bwm)/bws < -0.7):
            return "COMPRESSION", 65
    except Exception: pass
    return "RANGE", 55

def volatility_regime_filter(df):
    a = df["atr"].dropna() if "atr" in df.columns else pd.Series()
    if len(a) < 100: return True, "بيانات قليلة", "NORMAL"
    ca = a.iloc[-1]; am = a.rolling(100).mean().iloc[-1]
    ast = a.rolling(100).std().iloc[-1]
    if not np.isfinite(ast) or ast == 0: return True, "ATR ثابت", "NORMAL"
    z = (ca-am)/ast
    if z < -1.0: return False, f"سوق ميت (Z={z:.1f})", "DEAD"
    if z > 2.5: return False, f"تقلب مفرط (Z={z:.1f})", "CHAOS"
    if 1.5 < z <= 2.5: return True, f"تقلب مرتفع", "HIGH"
    if -1.0 <= z < -0.3: return True, f"تقلب منخفض", "LOW"
    return True, f"طبيعي (Z={z:.1f})", "NORMAL"

# ============================================================
# SCORING
# ============================================================

PILLAR_WEIGHTS = {"structure":0.28,"trend":0.18,"momentum":0.14,
                  "volume":0.14,"context":0.26}

def directional_score(df, fund_buy=0.0, fund_sell=0.0, macro=None):
    l = df.iloc[-1]
    sc = {"BUY":{k:0.0 for k in PILLAR_WEIGHTS},
          "SELL":{k:0.0 for k in PILLAR_WEIGHTS}}
    rs = []
    s = structure_state(df)
    if s["bullish"]: sc["BUY"]["structure"] += 45; rs.append("هيكل صاعد")
    elif s["bearish"]: sc["SELL"]["structure"] += 45; rs.append("هيكل هابط")
    recent = df.iloc[-5:]
    if recent["bos_bullish"].any(): sc["BUY"]["structure"] += 35; rs.append("BOS صاعد")
    if recent["bos_bearish"].any(): sc["SELL"]["structure"] += 35; rs.append("BOS هابط")
    if recent["mss_bullish"].any(): sc["BUY"]["structure"] += 20; rs.append("MSS صاعد")
    if recent["mss_bearish"].any(): sc["SELL"]["structure"] += 20; rs.append("MSS هابط")
    if recent["liquidity_sweep_bullish"].any(): sc["BUY"]["structure"] += 25
    if recent["liquidity_sweep_bearish"].any(): sc["SELL"]["structure"] += 25
    if recent["order_block_bullish"].any(): sc["BUY"]["structure"] += 15
    if recent["order_block_bearish"].any(): sc["SELL"]["structure"] += 15
    if recent["fvg_bullish"].any(): sc["BUY"]["structure"] += 10
    if recent["fvg_bearish"].any(): sc["SELL"]["structure"] += 10
    if safe_bool(l.get("in_discount")): sc["BUY"]["structure"] += 10
    if safe_bool(l.get("in_premium")):  sc["SELL"]["structure"] += 10
    if l["ema20"] > l["ema50"] > l["ema200"]: sc["BUY"]["trend"] += 80
    elif l["ema20"] < l["ema50"] < l["ema200"]: sc["SELL"]["trend"] += 80
    else:
        if l["ema20"] > l["ema50"]: sc["BUY"]["trend"] += 45
        elif l["ema20"] < l["ema50"]: sc["SELL"]["trend"] += 45
    if l["close"] > l["ema20"] > l["ema50"] > l["ema200"]: sc["BUY"]["trend"] += 20
    elif l["close"] < l["ema20"] < l["ema50"] < l["ema200"]: sc["SELL"]["trend"] += 20
    r = safe_float(l["rsi"], 50)
    if r >= 55: sc["BUY"]["momentum"] += 45
    elif r <= 45: sc["SELL"]["momentum"] += 45
    if l["macd"] > l["macd_signal"] and l["macd_histogram"] > 0:
        sc["BUY"]["momentum"] += 45
    elif l["macd"] < l["macd_signal"] and l["macd_histogram"] < 0:
        sc["SELL"]["momentum"] += 45
    if l["chaikin_mf"] > 0.10: sc["BUY"]["momentum"] += 10
    elif l["chaikin_mf"] < -0.10: sc["SELL"]["momentum"] += 10
    if l["close"] > l["vwap"]: sc["BUY"]["volume"] += 45
    elif l["close"] < l["vwap"]: sc["SELL"]["volume"] += 45
    if l["chaikin_mf"] > 0: sc["BUY"]["volume"] += 35
    elif l["chaikin_mf"] < 0: sc["SELL"]["volume"] += 35
    va = df["volume"].rolling(20).mean().iloc[-1]
    if va and np.isfinite(va) and l["volume"] > va:
        if l["close"] > l["open"]: sc["BUY"]["volume"] += 20
        elif l["close"] < l["open"]: sc["SELL"]["volume"] += 20
    # fundamental context
    sc["BUY"]["context"]  += fund_buy * 0.6
    sc["SELL"]["context"] += fund_sell * 0.6
    if fund_buy > 20: rs.append("أساسي BUY")
    if fund_sell > 20: rs.append("أساسي SELL")
    # macro
    if macro:
        if macro.get("dxy_bias") == "BEARISH": sc["BUY"]["context"] += 12
        elif macro.get("dxy_bias") == "BULLISH": sc["SELL"]["context"] += 12
        if macro.get("vix_state") == "HIGH FEAR": sc["SELL"]["context"] += 8
    reg, _ = detect_regime(df)
    if reg == "TREND_BULLISH": sc["BUY"]["context"] += 15
    elif reg == "TREND_BEARISH": sc["SELL"]["context"] += 15
    tb = ts = 0.0
    for p,w in PILLAR_WEIGHTS.items():
        sc["BUY"][p] = clamp(sc["BUY"][p], 0, 100)
        sc["SELL"][p] = clamp(sc["SELL"][p], 0, 100)
        tb += sc["BUY"][p]*w; ts += sc["SELL"][p]*w
    dv = detect_divergence(df)
    if dv == "BULLISH": tb += 3
    elif dv == "BEARISH": ts += 3
    return {"buy": clamp(tb,0,100), "sell": clamp(ts,0,100),
            "pillars": sc, "reasons": rs, "regime": reg, "divergence": dv}

def confirmation_gate(df, direction, pillar_scores, regime,
                     mtf_bias="NEUTRAL", mtf_conf=50.0,
                     weekly_bias_val="NEUTRAL"):
    if df is None or len(df) < 30:
        return False, 0.0, ["بيانات قليلة"], ["DATA"]
    l = df.iloc[-1]; sc = 0.0; rs = []; bl = []
    if (direction=="BUY" and regime=="TREND_BULLISH") or \
       (direction=="SELL" and regime=="TREND_BEARISH"):
        sc += 20; rs.append("Regime متوافق")
    elif regime == "RANGE": sc += 15; rs.append("Range")
    elif regime == "COMPRESSION": sc += 12; rs.append("Compression")
    else: bl.append("Regime غير متوافق")
    if (direction=="BUY" and mtf_bias=="BULLISH") or \
       (direction=="SELL" and mtf_bias=="BEARISH"):
        sc += 25 * min(mtf_conf/95, 1); rs.append("MTF متوافق")
    elif mtf_bias != "NEUTRAL": bl.append("MTF ضد الاتجاه")
    if (direction=="BUY" and weekly_bias_val=="BULLISH") or \
       (direction=="SELL" and weekly_bias_val=="BEARISH"):
        sc += 15; rs.append("Weekly متوافق")
    elif weekly_bias_val != "NEUTRAL": bl.append("Weekly ضد الاتجاه")
    eo = (direction=="BUY" and l["ema20"] > l["ema50"]) or \
         (direction=="SELL" and l["ema20"] < l["ema50"])
    if eo: sc += 15; rs.append("EMA alignment")
    else: bl.append("EMA غير مؤيد")
    mo = (direction=="BUY" and l["macd_histogram"] > 0) or \
         (direction=="SELL" and l["macd_histogram"] < 0)
    if mo: sc += 15; rs.append("MACD مؤيد")
    else: bl.append("Momentum غير مؤيد")
    if candle_confirmation(df, direction): sc += 10; rs.append("Candle")
    smc_s, smc_r = smc_quality(df)
    if smc_s >= 40: sc += 15; rs.extend(smc_r[:3])
    elif smc_s < 20: bl.append("SMC ضعيف")
    own = sum(float(v) for v in pillar_scores.get(direction, {}).values())
    opp = sum(float(v) for v in pillar_scores.get(
        "SELL" if direction=="BUY" else "BUY", {}).values())
    if own < 180: bl.append("أدلة ضعيفة")
    if opp > own*0.85: bl.append("تعارض قوي")
    else: sc += 5
    hard = any(x in bl for x in ("MTF ضد الاتجاه","Regime غير متوافق",
                                  "Weekly ضد الاتجاه"))
    th = max(55, MIN_EMIT_CONFIDENCE - 10)
    ok = sc >= th and not hard and len(bl) <= 3
    return ok, clamp(sc,0,100), rs, bl

def compute_soft_penalty(items, strict=False):
    if not items or not strict: return 0.0, []
    si = sorted([i for i in items if i[1]>0],
                key=lambda x: x[1], reverse=True)[:SOFT_PENALTY_TOP_N]
    w = [1.0, 0.5, 0.25]
    tot = sum(p*wt for (_,p),wt in zip(si, w))
    return min(tot, MAX_SOFT_PENALTY), si

def weighted_confluence(pillar_data, direction):
    w = {"structure":30,"trend":20,"context":20,"momentum":15,"volume":15}
    t = 0.0
    for p,wt in w.items():
        raw = safe_float(pillar_data.get(direction, {}).get(p), 0)
        s = clamp(raw, 0, 100)
        if s >= 70: t += wt
        elif s >= 50: t += wt*0.5
        elif s < 30: t -= wt*0.3
    return clamp(t, 0, 100)

# ============================================================
# SL / TP / ICT ENTRY
# ============================================================

def latest_structure_levels(df):
    lo = df.index[df["swing_low"]].tolist()
    hi = df.index[df["swing_high"]].tolist()
    sl = np.nan; sh = np.nan
    if lo:
        pc = "swing_low_price" if "swing_low_price" in df.columns else "low"
        sl = safe_float(df.loc[lo[-1], pc], safe_float(df.loc[lo[-1], "low"]))
    if hi:
        pc = "swing_high_price" if "swing_high_price" in df.columns else "high"
        sh = safe_float(df.loc[hi[-1], pc], safe_float(df.loc[hi[-1], "high"]))
    return sl, sh

def collect_sr_levels(df, lookback=200):
    lv = []
    w = df.iloc[-lookback:] if len(df) > lookback else df
    if "swing_high" in w.columns:
        pc = "swing_high_price" if "swing_high_price" in w.columns else "high"
        for i in w.index[w["swing_high"]][-5:]:
            v = safe_float(w.loc[i, pc], safe_float(w.loc[i, "high"]))
            if np.isfinite(v): lv.append(v)
    if "swing_low" in w.columns:
        pc = "swing_low_price" if "swing_low_price" in w.columns else "low"
        for i in w.index[w["swing_low"]][-5:]:
            v = safe_float(w.loc[i, pc], safe_float(w.loc[i, "low"]))
            if np.isfinite(v): lv.append(v)
    if len(w) >= 20:
        lv.append(float(w["high"].iloc[-20:].max()))
        lv.append(float(w["low"].iloc[-20:].min()))
    return sorted({l for l in lv if np.isfinite(l) and l > 0})

def detect_inducement(df, direction, lookback=50):
    if len(df) < 20: return None
    work = df.iloc[-lookback:]
    if direction == "BUY":
        sl_idx = work.index[work["swing_low"]].tolist()
        sh_idx = work.index[work["swing_high"]].tolist()
        if not sl_idx or not sh_idx: return None
        prior = [s for s in sl_idx if s < sh_idx[-1]]
        if not prior: return None
        pc = "swing_low_price" if "swing_low_price" in work.columns else "low"
        return safe_float(work.loc[prior[-1], pc], safe_float(work.loc[prior[-1], "low"]))
    sl_idx = work.index[work["swing_low"]].tolist()
    sh_idx = work.index[work["swing_high"]].tolist()
    if not sl_idx or not sh_idx: return None
    prior = [s for s in sh_idx if s < sl_idx[-1]]
    if not prior: return None
    pc = "swing_high_price" if "swing_high_price" in work.columns else "high"
    return safe_float(work.loc[prior[-1], pc], safe_float(work.loc[prior[-1], "high"]))

def find_nearest_liquidity(df, current_price, direction, lookback=100):
    if len(df) < 10: return []
    work = df.iloc[-lookback:] if len(df) > lookback else df
    levels = []
    if direction == "BUY":
        pc = "swing_high_price" if "swing_high_price" in work.columns else "high"
        for idx in work.index[work["swing_high"]][-5:]:
            lvl = safe_float(work.loc[idx, pc], safe_float(work.loc[idx, "high"]))
            if np.isfinite(lvl) and lvl > current_price:
                levels.append({"name":"BSL", "level": lvl})
    else:
        pc = "swing_low_price" if "swing_low_price" in work.columns else "low"
        for idx in work.index[work["swing_low"]][-5:]:
            lvl = safe_float(work.loc[idx, pc], safe_float(work.loc[idx, "low"]))
            if np.isfinite(lvl) and lvl < current_price:
                levels.append({"name":"SSL", "level": lvl})
    levels.sort(key=lambda x: x["level"], reverse=(direction=="SELL"))
    return levels[:5]

def find_nearest_fvg(df, current_price, direction, lookback=100):
    work = df.iloc[-lookback:] if len(df) > lookback else df
    fvgs = []
    if direction == "BUY":
        for idx in work.index[work["fvg_bearish"]]:
            lo = safe_float(work.loc[idx, "fvg_bear_low"], np.nan)
            hi = safe_float(work.loc[idx, "fvg_bear_high"], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and lo > current_price:
                fvgs.append({"name":"Bear FVG","low":lo,"high":hi,"mid":(lo+hi)/2})
    else:
        for idx in work.index[work["fvg_bullish"]]:
            lo = safe_float(work.loc[idx, "fvg_bull_low"], np.nan)
            hi = safe_float(work.loc[idx, "fvg_bull_high"], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi < current_price:
                fvgs.append({"name":"Bull FVG","low":lo,"high":hi,"mid":(lo+hi)/2})
    fvgs.sort(key=lambda x: x["mid"], reverse=(direction=="SELL"))
    return fvgs[:3]

def find_order_block_levels(df, current_price, direction, lookback=100):
    work = df.iloc[-lookback:] if len(df) > lookback else df
    obs = []
    if direction == "BUY":
        for idx in work.index[work["order_block_bullish"]][-5:]:
            lo = safe_float(work.loc[idx, "ob_low"], np.nan)
            hi = safe_float(work.loc[idx, "ob_high"], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi < current_price:
                obs.append({"name":"Bull OB","low":lo,"high":hi,"mid":(lo+hi)/2})
    else:
        for idx in work.index[work["order_block_bearish"]][-5:]:
            lo = safe_float(work.loc[idx, "ob_low"], np.nan)
            hi = safe_float(work.loc[idx, "ob_high"], np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and lo > current_price:
                obs.append({"name":"Bear OB","low":lo,"high":hi,"mid":(lo+hi)/2})
    obs.sort(key=lambda x: x["mid"], reverse=(direction=="SELL"))
    return obs[:3]

def calculate_ict_sl_tp(df, direction, current_price, profile):
    atr = safe_float(df["atr"].iloc[-1], np.nan)
    if not np.isfinite(atr) or atr <= 0: return None
    all_levels = []
    idm = detect_inducement(df, direction)
    if idm is not None and np.isfinite(idm):
        all_levels.append((idm, "IDM (ICT)", "ict"))
    for ob in find_order_block_levels(df, current_price, direction):
        if direction == "BUY": all_levels.append((ob["low"], f"{ob['name']} Low", "ob"))
        else: all_levels.append((ob["high"], f"{ob['name']} High", "ob"))
    for fv in find_nearest_fvg(df, current_price, direction):
        if direction == "BUY": all_levels.append((fv["low"], f"{fv['name']} Bottom", "fvg"))
        else: all_levels.append((fv["high"], f"{fv['name']} Top", "fvg"))
    for lq in find_nearest_liquidity(df, current_price, direction):
        all_levels.append((lq["level"], lq["name"], "liq"))
    sw_lo, sw_hi = latest_structure_levels(df)
    if np.isfinite(sw_lo): all_levels.append((sw_lo, "Swing Low", "swing"))
    if np.isfinite(sw_hi): all_levels.append((sw_hi, "Swing High", "swing"))

    # Filter to valid SL side
    if direction == "BUY":
        sl_candidates = [(lvl,name,t) for lvl,name,t in all_levels
                         if np.isfinite(lvl) and lvl < current_price]
        if not sl_candidates: return None
        sl_candidates.sort(key=lambda x: current_price - x[0])
        sl_level, sl_reason, sl_type = sl_candidates[0]
    else:
        sl_candidates = [(lvl,name,t) for lvl,name,t in all_levels
                         if np.isfinite(lvl) and lvl > current_price]
        if not sl_candidates: return None
        sl_candidates.sort(key=lambda x: x[0] - current_price)
        sl_level, sl_reason, sl_type = sl_candidates[0]

    buffer = atr * SL_BUFFER_ATR
    if direction == "BUY": sl = sl_level - buffer
    else: sl = sl_level + buffer

    risk = abs(current_price - sl)
    if risk < SL_MIN_DIST_ATR*atr:
        sl = current_price - SL_MIN_DIST_ATR*atr if direction=="BUY" \
             else current_price + SL_MIN_DIST_ATR*atr
        risk = abs(current_price - sl); sl_reason = "ATR min"
    if risk > SL_MAX_DIST_ATR*atr:
        sl = current_price - SL_MAX_DIST_ATR*atr if direction=="BUY" \
             else current_price + SL_MAX_DIST_ATR*atr
        risk = abs(current_price - sl); sl_reason = "ATR max"

    # TPs — liquidity-based
    liq_targets = find_nearest_liquidity(df, current_price, direction, lookback=200)
    if direction == "BUY":
        liq_targets = [l for l in liq_targets if l["level"] > current_price]
        liq_targets.sort(key=lambda x: x["level"])
    else:
        liq_targets = [l for l in liq_targets if l["level"] < current_price]
        liq_targets.sort(key=lambda x: x["level"], reverse=True)

    tp1 = tp2 = tp3 = None; tp_notes = []
    if direction == "BUY":
        if len(liq_targets) >= 1: tp1 = liq_targets[0]["level"]; tp_notes.append(liq_targets[0]["name"])
        else: tp1 = current_price + 1.0*risk
        tp2 = liq_targets[1]["level"] if len(liq_targets)>=2 else current_price + 2.0*risk
        tp3 = liq_targets[2]["level"] if len(liq_targets)>=3 else current_price + 3.0*risk
    else:
        if len(liq_targets) >= 1: tp1 = liq_targets[0]["level"]; tp_notes.append(liq_targets[0]["name"])
        else: tp1 = current_price - 1.0*risk
        tp2 = liq_targets[1]["level"] if len(liq_targets)>=2 else current_price - 2.0*risk
        tp3 = liq_targets[2]["level"] if len(liq_targets)>=3 else current_price - 3.0*risk

    def rr(tp):
        if tp is None or risk <= 0: return 0.0
        return abs(tp-current_price)/risk

    return {"entry": current_price, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "risk": risk, "rr1": rr(tp1), "rr2": rr(tp2), "rr3": rr(tp3),
            "sl_reason": sl_reason, "sl_type": sl_type,
            "tp_notes": tp_notes}

# ============================================================
# ML
# ============================================================

def extract_ml_features(result, direction):
    pillars = result.get("pillars", {}); dp = pillars.get(direction, {})
    df = result.get("df")
    rm = {"RANGE":0,"TREND_BULLISH":1,"TREND_BEARISH":2,"COMPRESSION":3,"UNKNOWN":0}
    regime = result.get("regime", "RANGE")
    try:
        si = get_current_session_info()
        sm = {"Asian":0,"London":1,"Overlap":2,"NY":3,"After-Hours":4,
              "Asia Pump":0,"US Close":4}
        se = sm.get(si["session"], 0)
    except Exception: se = 0
    mtf = result.get("mtf_bias","NEUTRAL")
    m_match = 1 if ((direction=="BUY" and mtf=="BULLISH") or
                    (direction=="SELL" and mtf=="BEARISH")) else 0
    wk = result.get("weekly_bias","NEUTRAL")
    w_match = 1 if ((direction=="BUY" and wk=="BULLISH") or
                    (direction=="SELL" and wk=="BEARISH")) else 0
    htf_m = sess_m = 0
    flt = result.get("filter_results", {})
    if "HTF Zone" in flt:
        z = flt["HTF Zone"].get("zone", "UNKNOWN")
        htf_m = 1 if z in ("DISCOUNT","PREMIUM","MID") else 0
    if "Session" in flt:
        sess_m = 1 if flt["Session"].get("pass", True) else 0
    ed=0.0; ar=1.0; bp=50.0; vd=0.0; rv=50.0; mh=0.0
    br=0.5; wr=1.0; cb=0; csb=50; vr=1.0
    if df is not None and len(df) >= 200:
        try:
            last = df.iloc[-1]
            p = safe_float(last.get("close"),0)
            e200 = safe_float(last.get("ema200"),0)
            atr = safe_float(last.get("atr"),1)
            if e200 > 0 and atr > 0: ed = (p-e200)/atr
            a_s = df["atr"].dropna()
            if len(a_s) >= 100:
                am = a_s.rolling(100).mean().iloc[-1]
                if am > 0: ar = atr/am
            bu = safe_float(last.get("bb_upper"),0)
            bl = safe_float(last.get("bb_lower"),0)
            if bu > bl > 0: bp = (p-bl)/(bu-bl)*100
            vw = safe_float(last.get("vwap"),0)
            if vw > 0: vd = (p-vw)/vw*100
            rv = safe_float(last.get("rsi"),50)
            mh_v = safe_float(last.get("macd_histogram"),0)
            if atr > 0: mh = mh_v/atr
            body = abs(float(last["close"]) - float(last["open"]))
            rng = max(float(last["high"]) - float(last["low"]), 1e-9)
            br = body/rng
            uw = float(last["high"]) - max(float(last["close"]), float(last["open"]))
            lw = min(float(last["close"]), float(last["open"])) - float(last["low"])
            wr = (uw+0.001)/(lw+0.001)
            dsg = 1 if float(last["close"]) > float(last["open"]) else -1
            cb = 0
            for k in range(1, min(11, len(df))):
                c = float(df["close"].iloc[-k]); o = float(df["open"].iloc[-k])
                s = 1 if c > o else -1
                if s == dsg: cb += 1
                else: break
            if "bos_bullish" in df.columns and "bos_bearish" in df.columns:
                ba = (df["bos_bullish"] | df["bos_bearish"])
                bi = ba[ba].index
                if len(bi) > 0: csb = len(df) - 1 - df.index.get_loc(bi[-1])
            va = df["volume"].rolling(20).mean().iloc[-1]
            if va > 0: vr = float(last["volume"])/va
        except Exception: pass
    div = result.get("divergence"); dfg = 0
    if div == "BULLISH" and direction == "BUY": dfg = 1
    elif div == "BEARISH" and direction == "SELL": dfg = 1
    elif div in ("BULLISH","BEARISH"): dfg = -1

    funding = result.get("funding", {})
    oi = result.get("open_interest", {})
    dom = result.get("dominance", 0.0) or 0.0
    fr_val = float(funding.get("rate", 0.0))
    fr_bias = (1 if fr_val < -0.02 else -1 if fr_val > 0.05 else 0)
    oi_chg = float(oi.get("change_24h", 0.0))

    return [float(dp.get("structure",0)), float(dp.get("trend",0)),
            float(dp.get("momentum",0)), float(dp.get("volume",0)),
            float(dp.get("context",0)), float(result.get("mtf_conf",50)),
            float(result.get("confirmation_score",0)),
            float(result.get("raw_confidence",0)), float(rm.get(regime,0)),
            float(se), float(m_match), float(w_match), float(htf_m), float(sess_m),
            fr_val, float(fr_bias), oi_chg, float(dom),
            float(ed), float(ar), float(bp), float(vd),
            float(rv), float(mh), float(br), float(wr), float(cb),
            float(csb), float(dfg), float(vr)]

def tune_model(model, param_grid, X, y, mode="randomized", n_iter=20, cv_folds=5):
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    except Exception: cv = cv_folds
    try:
        if mode == "grid":
            tuner = GridSearchCV(model, param_grid, cv=cv, scoring=TUNING_SCORING,
                                 n_jobs=TUNING_N_JOBS, verbose=0, refit=True)
        else:
            tuner = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv,
                                        scoring=TUNING_SCORING, n_jobs=TUNING_N_JOBS,
                                        verbose=0, random_state=42, refit=True)
        tuner.fit(X, y)
        return tuner.best_estimator_, tuner.best_params_, float(tuner.best_score_)
    except Exception:
        return model, {}, 0.0

def train_ml_model(trades_data, use_tuning=False):
    if not SKLEARN_AVAILABLE: return {"error": "sklearn غير مثبت"}
    if len(trades_data) < ML_MIN_TRADES:
        return {"error": f"بيانات غير كافية ({len(trades_data)}/{ML_MIN_TRADES})"}
    X = np.array([t["features"] for t in trades_data], dtype=float)
    y = np.array([t["label"] for t in trades_data], dtype=int)
    if len(np.unique(y)) < 2: return {"error":"كل الصفقات من نفس الفئة"}
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
            test_size=ML_TEST_SIZE, random_state=42, stratify=y)
        Xtr_raw, Xte_raw, _, _ = train_test_split(X, y, test_size=ML_TEST_SIZE,
                                                   random_state=42, stratify=y)
    except Exception:
        X_train = X_test = X_scaled; Xtr_raw = Xte_raw = X
        y_train = y_test = y
    res = {}; trep = {}

    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        if use_tuning:
            lr, bp, bcv = tune_model(lr, PARAM_GRIDS["LogisticRegression"],
                                     X_train, y_train, mode=TUNING_MODE,
                                     n_iter=TUNING_N_ITER)
            trep["LR"] = {"best_params":bp,"best_cv_score":bcv}
        lr.fit(X_train, y_train)
        yp = lr.predict(X_test); ypr = lr.predict_proba(X_test)[:,1]
        auc = float(roc_auc_score(y_test, ypr)) if len(np.unique(y_test))>1 else 0.5
        res["LR"] = {"model":lr,"scaler":scaler,"auc":auc,
                     "accuracy":float(accuracy_score(y_test,yp)),
                     "f1":float(f1_score(y_test,yp,zero_division=0)),
                     "needs_scaling":True}
    except Exception as e: res["LR"] = {"error":str(e)}

    try:
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                        learning_rate=0.05, random_state=42)
        gb.fit(Xtr_raw, y_train)
        yp = gb.predict(Xte_raw); ypr = gb.predict_proba(Xte_raw)[:,1]
        auc = float(roc_auc_score(y_test, ypr)) if len(np.unique(y_test))>1 else 0.5
        res["GB"] = {"model":gb,"scaler":None,"auc":auc,
                     "accuracy":float(accuracy_score(y_test,yp)),
                     "f1":float(f1_score(y_test,yp,zero_division=0)),
                     "needs_scaling":False,
                     "importance":dict(zip(ML_FEATURE_NAMES, gb.feature_importances_.tolist()))}
    except Exception as e: res["GB"] = {"error":str(e)}

    try:
        rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                     random_state=42, n_jobs=-1)
        rf.fit(Xtr_raw, y_train)
        yp = rf.predict(Xte_raw); ypr = rf.predict_proba(Xte_raw)[:,1]
        auc = float(roc_auc_score(y_test, ypr)) if len(np.unique(y_test))>1 else 0.5
        res["RF"] = {"model":rf,"scaler":None,"auc":auc,
                     "accuracy":float(accuracy_score(y_test,yp)),
                     "f1":float(f1_score(y_test,yp,zero_division=0)),
                     "needs_scaling":False,
                     "importance":dict(zip(ML_FEATURE_NAMES, rf.feature_importances_.tolist()))}
    except Exception as e: res["RF"] = {"error":str(e)}

    if XGBOOST_AVAILABLE:
        try:
            xm = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    random_state=42, use_label_encoder=False,
                                    eval_metric="logloss", verbosity=0)
            xm.fit(Xtr_raw, y_train)
            yp = xm.predict(Xte_raw); ypr = xm.predict_proba(Xte_raw)[:,1]
            auc = float(roc_auc_score(y_test, ypr)) if len(np.unique(y_test))>1 else 0.5
            res["XGB"] = {"model":xm,"scaler":None,"auc":auc,
                          "accuracy":float(accuracy_score(y_test,yp)),
                          "f1":float(f1_score(y_test,yp,zero_division=0)),
                          "needs_scaling":False,
                          "importance":dict(zip(ML_FEATURE_NAMES, xm.feature_importances_.tolist()))}
        except Exception as e: res["XGB"] = {"error":str(e)}

    valid = {k:v for k,v in res.items() if "auc" in v}
    if not valid: return {"error":"كل النماذج فشلت"}
    best_name = max(valid.keys(), key=lambda k: valid[k]["auc"])
    best = valid[best_name]
    fs = None
    if "importance" in best:
        imp = best["importance"]; ti = sum(imp.values()) or 1.0
        kept = [n for n,i in imp.items() if i/ti >= ML_FEATURE_SELECTION_THRESHOLD]
        dropped = [n for n in imp.keys() if n not in kept]
        fs = {"kept":kept,"dropped":dropped,"kept_count":len(kept)}
    iss = dict(sorted(best.get("importance", {}).items(),
                      key=lambda x: x[1], reverse=True))
    ta = sum(iss.values()) or 1.0
    cw = {k: round(v/ta*100, 2) for k,v in iss.items()}
    return {"model":best["model"], "scaler":best.get("scaler"),
            "needs_scaling":best.get("needs_scaling", False),
            "model_name":best_name, "auc":best["auc"],
            "accuracy":best["accuracy"], "f1":best["f1"],
            "importance":iss, "calibrated_weights":cw,
            "all_models":{k:{"auc":v.get("auc",0),
                             "accuracy":v.get("accuracy",0),
                             "f1":v.get("f1",0),
                             "error":v.get("error")}
                          for k,v in res.items()},
            "feature_selection":fs, "tuned":use_tuning,
            "n_train":len(X_train), "n_test":len(X_test),
            "n_total":len(trades_data),
            "trained_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

def predict_win_probability(ml_pack, result, direction):
    if not ml_pack or "model" not in ml_pack: return None
    try:
        feats = extract_ml_features(result, direction)
        X = np.array([feats], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if ml_pack.get("needs_scaling") and ml_pack.get("scaler") is not None:
            X = ml_pack["scaler"].transform(X)
        return float(ml_pack["model"].predict_proba(X)[0, 1])
    except Exception: return None

def ml_adjusted_confidence(result, direction, ml_pack):
    if ml_pack is None or "model" not in ml_pack:
        return result.get("confidence", 0), None, "no_model"
    prob = predict_win_probability(ml_pack, result, direction)
    if prob is None: return result.get("confidence", 0), None, "predict_failed"
    fc = result.get("confidence", 50); auc = ml_pack.get("auc", 0.5)
    if auc >= 0.75: mw = 0.75
    elif auc >= 0.70: mw = 0.65
    elif auc >= 0.65: mw = 0.55
    elif auc >= 0.60: mw = 0.45
    elif auc >= 0.55: mw = 0.30
    else: mw = 0.15
    return clamp(mw*(prob*100)+(1-mw)*fc, 0, 99), prob, "applied"

def monte_carlo_simulation(trades_r, n_sims=1000, n_trades=None):
    if not trades_r or len(trades_r) < 5: return None
    if n_trades is None: n_trades = len(trades_r)
    rng = np.random.default_rng(42)
    fe = np.zeros(n_sims); md = np.zeros(n_sims)
    for i in range(n_sims):
        s = rng.choice(trades_r, size=n_trades, replace=True)
        ec = np.cumsum(s); pk = np.maximum.accumulate(ec); dd = pk - ec
        fe[i] = ec[-1]; md[i] = dd.max() if len(dd) > 0 else 0
    return {"mean_final":float(np.mean(fe)), "median_final":float(np.median(fe)),
            "p5_final":float(np.percentile(fe,5)), "p95_final":float(np.percentile(fe,95)),
            "prob_profit":float((fe>0).mean()*100),
            "mean_dd":float(np.mean(md)), "p95_dd":float(np.percentile(md,95)),
            "worst_dd":float(np.max(md)), "n_sims":n_sims, "n_trades":n_trades}

# ============================================================
# SENTIMENT
# ============================================================

def estimate_retail_sentiment(df):
    if df is None or len(df) < 50:
        return {"long_pct":50,"short_pct":50,"bias":"NEUTRAL",
                "contrarian":"NEUTRAL","strength":0}
    last = df.iloc[-1]; ls = 50.0
    rsi = safe_float(last.get("rsi"), 50)
    if rsi >= 70: ls += 20
    elif rsi >= 60: ls += 10
    elif rsi <= 30: ls -= 20
    elif rsi <= 40: ls -= 10
    e200 = safe_float(last.get("ema200"), 0)
    p = safe_float(last.get("close"), 0)
    atr = safe_float(last.get("atr"), 1)
    if e200 > 0 and atr > 0:
        d = (p-e200)/atr
        if d > 3: ls += 15
        elif d > 1.5: ls += 8
        elif d < -3: ls -= 15
        elif d < -1.5: ls -= 8
    if len(df) >= 5:
        up = (df["close"].iloc[-5:].diff() > 0).sum()
        ls += (up - 2.5)*3
    st_obj = structure_state(df)
    if st_obj["bullish"]: ls += 8
    elif st_obj["bearish"]: ls -= 8
    ls = clamp(ls, 5, 95); ss = 100 - ls
    if ls >= SENTIMENT_EXTREME_LONG:
        bias, ct, stv = "EXTREME_LONG", "SELL", ls-SENTIMENT_EXTREME_LONG
    elif ls <= SENTIMENT_EXTREME_SHORT:
        bias, ct, stv = "EXTREME_SHORT", "BUY", SENTIMENT_EXTREME_SHORT-ls
    elif ls >= 60: bias, ct, stv = "CROWD_LONG", "SELL", ls-60
    elif ls <= 40: bias, ct, stv = "CROWD_SHORT", "BUY", 40-ls
    else: bias, ct, stv = "NEUTRAL", "NEUTRAL", 0
    return {"long_pct":ls, "short_pct":ss, "bias":bias,
            "contrarian":ct, "strength":stv}

# ============================================================
# ANALYSIS PIPELINE
# ============================================================

def analyze_btc():
    df = get_btc_ohlcv("6mo", "4h")
    if df is None or len(df) < 100:
        return {"error": "فشل جلب بيانات BTC"}
    x = build_features(df, PROFILE)

    funding = get_funding_rate()
    oi = get_open_interest()
    fg = get_fear_greed()
    dom = get_btc_dominance()
    macro = get_btc_macro()
    fund_buy, fund_sell, fund_det = get_btc_fundamental_score(
        funding=funding, oi=oi, fg=fg, dominance=dom, macro=macro)

    dir_score = directional_score(x, fund_buy=fund_buy,
                                  fund_sell=fund_sell, macro=macro)

    mtf_bias, mtf_conf, mtf_details = get_mtf_analysis()
    wb, _ = weekly_bias_from(x)

    price = float(x["close"].iloc[-1])
    direction = "BUY" if dir_score["buy"] > dir_score["sell"] else "SELL"
    raw_conf = max(dir_score["buy"], dir_score["sell"])
    conf_gap = abs(dir_score["buy"] - dir_score["sell"])

    gate_ok, gate_score, gate_rs, gate_bl = confirmation_gate(
        x, direction, dir_score["pillars"], dir_score["regime"],
        mtf_bias=mtf_bias, mtf_conf=mtf_conf, weekly_bias_val=wb)

    ict = calculate_ict_sl_tp(x, direction, price, PROFILE)

    # Final confidence
    final_conf = raw_conf
    if gate_score: final_conf = (raw_conf*0.65 + gate_score*0.35)
    penalty = 0.0
    if mtf_bias not in ("BULLISH","BEARISH") and mtf_bias != "NEUTRAL": penalty += PENALTY_MTF_AGAINST
    if dir_score["regime"] == "RANGE": penalty += PENALTY_RANGE_REGIME
    if wb not in ("BULLISH","BEARISH","NEUTRAL"): penalty += PENALTY_WEEKLY_AGAINST
    final_conf = clamp(final_conf - penalty*0.3, 0, 99)

    # Grade
    if final_conf >= A_PLUS_MIN: grade = "A+"
    elif final_conf >= A_MIN: grade = "A"
    elif final_conf >= B_MIN: grade = "B"
    elif final_conf >= C_MIN: grade = "C"
    else: grade = "D"

    # Filters
    in_kz, kz_msg = in_kill_zone()
    vol_ok, vol_msg, vol_label = volatility_regime_filter(x)

    emit = (final_conf >= MIN_EMIT_CONFIDENCE and conf_gap >= MIN_SIGNAL_GAP
            and gate_ok and vol_ok and ict is not None)

    result = {
        "pair_name": PAIR_NAME, "symbol": SYMBOL,
        "direction": direction, "signal": direction,
        "confidence": final_conf, "raw_confidence": raw_conf,
        "confirmation_score": gate_score, "confidence_gap": conf_gap,
        "trade_grade": grade, "regime": dir_score["regime"],
        "pillars": dir_score["pillars"],
        "buy_score": dir_score["buy"], "sell_score": dir_score["sell"],
        "reasons": dir_score["reasons"],
        "divergence": dir_score["divergence"],
        "mtf_bias": mtf_bias, "mtf_conf": mtf_conf, "mtf_details": mtf_details,
        "weekly_bias": wb,
        "funding": funding, "open_interest": oi,
        "fear_greed": fg, "dominance": dom, "macro": macro,
        "fund_buy": fund_buy, "fund_sell": fund_sell, "fund_details": fund_det,
        "gate_pass": gate_ok, "gate_reasons": gate_rs, "gate_blockers": gate_bl,
        "filter_results": {
            "Session": {"pass": in_kz, "msg": kz_msg},
            "Volatility": {"pass": vol_ok, "msg": vol_msg, "label": vol_label},
        },
        "ict": ict,
        "in_kill_zone": in_kz, "kz_msg": kz_msg,
        "emit": emit,
        "price": price,
        "df": x,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "spot": get_btc_spot(),
    }

    # Sentiment
    sentiment = estimate_retail_sentiment(x)
    result["sentiment"] = sentiment
    return result

def weekly_bias_from(x):
    try:
        d = get_btc_ohlcv("2y", "1wk")
        if d is None or len(d) < 30: return "NEUTRAL", 0
        w = build_features(d, PROFILE)
        if len(w) < 5: return "NEUTRAL", 0
        l = w.iloc[-1]; bu = be = 0
        if l["ema20"] > l["ema50"]: bu += 1
        elif l["ema20"] < l["ema50"]: be += 1
        if l["macd"] > l["macd_signal"]: bu += 1
        elif l["macd"] < l["macd_signal"]: be += 1
        s = structure_state(w)
        if s["bullish"]: bu += 2
        elif s["bearish"]: be += 2
        if bu > be: return "BULLISH", bu
        if be > bu: return "BEARISH", be
        return "NEUTRAL", 0
    except Exception: return "NEUTRAL", 0

# ============================================================
# CHARTS
# ============================================================

def make_main_chart(df, result, n_bars=150):
    w = df.iloc[-n_bars:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.75, 0.25],
                        subplot_titles=("", "RSI"))
    fig.add_trace(go.Candlestick(
        x=w.index, open=w["open"], high=w["high"],
        low=w["low"], close=w["close"], name="BTC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["ema20"], name="EMA20",
        line=dict(color="#f7931a", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["ema50"], name="EMA50",
        line=dict(color="#2196f3", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["ema200"], name="EMA200",
        line=dict(color="#e91e63", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["bb_upper"], name="BB Upper",
        line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot")),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["bb_lower"], name="BB Lower",
        line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,150,0.05)"), row=1, col=1)

    ict = result.get("ict")
    if ict:
        for key, color, label in [("entry","#ffd54f","Entry"),
                                   ("sl","#ef5350","SL"),
                                   ("tp1","#26a69a","TP1"),
                                   ("tp2","#26a69a","TP2"),
                                   ("tp3","#26a69a","TP3")]:
            if ict.get(key):
                fig.add_hline(y=ict[key], line_dash="dash",
                              line_color=color, line_width=1.2,
                              annotation_text=f"{label} {fmt_price(ict[key])}",
                              annotation_position="right",
                              row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w["rsi"], name="RSI",
        line=dict(color="#9c27b0", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)",
                  row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)",
                  row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=560,
        margin=dict(l=10,r=80,t=30,b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,20,0.4)",
        font=dict(color="#e0e0e0"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig

def make_pillars_chart(pillars, direction):
    dp = pillars.get(direction, {})
    names = list(dp.keys()); vals = [dp[n] for n in names]
    labels_ar = {"structure":"الهيكل","trend":"الاتجاه",
                 "momentum":"الزخم","volume":"السيولة","context":"السياق"}
    fig = go.Figure(go.Bar(
        x=vals, y=[labels_ar.get(n,n) for n in names],
        orientation="h", marker=dict(
            color=vals, colorscale=[[0,"#ef5350"],[0.5,"#ffd54f"],[1,"#26a69a"]],
            cmin=0, cmax=100),
        text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.update_layout(
        template="plotly_dark", height=260,
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(range=[0,110], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,20,0.4)",
        font=dict(color="#e0e0e0"))
    return fig

def make_fund_chart(fund_details):
    if not fund_details: return None
    labels = [d["factor"] for d in fund_details]
    vals = []
    for d in fund_details:
        v = d.get("value", "")
        # extract leading number if any
        try:
            n = float("".join(c for c in str(v) if c.isdigit() or c in ".-")[:8])
        except Exception: n = 0
        vals.append(abs(n) if n else 1)
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                            marker_color="#f7931a",
                            text=[str(d.get("value","")) for d in fund_details],
                            textposition="outside"))
    fig.update_layout(
        template="plotly_dark", height=max(200, len(labels)*32+40),
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,15,20,0.4)",
        font=dict(color="#e0e0e0"))
    return fig

# ============================================================
# UI STYLING
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {
    background: radial-gradient(ellipse at top, #1a1428 0%, #0a0a12 60%, #050508 100%);
    color: #e8e8f0;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #100c1a 0%, #08060f 100%);
    border-right: 1px solid rgba(247,147,26,0.15);
}

.bp-header {
    background: linear-gradient(135deg, rgba(247,147,26,0.15), rgba(255,193,7,0.05));
    border: 1px solid rgba(247,147,26,0.3);
    border-radius: 18px;
    padding: 22px 30px;
    margin-bottom: 20px;
    display: flex; justify-content: space-between; align-items: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(247,147,26,0.1);
}
.bp-header h1 {
    margin: 0;
    font-size: 26px; font-weight: 800;
    background: linear-gradient(90deg, #f7931a, #ffd54f);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}
.bp-header .sub { color: #8a8aa0; font-size: 13px; margin-top: 4px; }

.metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 18px;
}
.metric-card {
    background: linear-gradient(145deg, rgba(30,25,45,0.9), rgba(15,12,22,0.9));
    border: 1px solid rgba(247,147,26,0.2);
    border-radius: 14px;
    padding: 16px 18px;
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(247,147,26,0.5);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(247,147,26,0.15);
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #f7931a, transparent);
    opacity: 0.6;
}
.metric-label { color: #8a8aa0; font-size: 11px; font-weight: 600;
                text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.metric-value { color: #fff; font-size: 22px; font-weight: 800;
                font-family: 'JetBrains Mono', monospace; }
.metric-delta { font-size: 12px; margin-top: 4px; font-weight: 600; }
.delta-up { color: #26a69a; }
.delta-down { color: #ef5350; }
.delta-neutral { color: #8a8aa0; }

.signal-card {
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative; overflow: hidden;
    border: 2px solid;
}
.signal-buy {
    background: linear-gradient(135deg, rgba(38,166,154,0.18), rgba(38,166,154,0.03));
    border-color: rgba(38,166,154,0.5);
    box-shadow: 0 8px 40px rgba(38,166,154,0.15);
}
.signal-sell {
    background: linear-gradient(135deg, rgba(239,83,80,0.18), rgba(239,83,80,0.03));
    border-color: rgba(239,83,80,0.5);
    box-shadow: 0 8px 40px rgba(239,83,80,0.15);
}
.signal-wait {
    background: linear-gradient(135deg, rgba(255,193,7,0.15), rgba(255,193,7,0.02));
    border-color: rgba(255,193,7,0.4);
    box-shadow: 0 8px 40px rgba(255,193,7,0.1);
}
.signal-title {
    font-size: 13px; font-weight: 700; color: #8a8aa0;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;
}
.signal-main {
    font-size: 44px; font-weight: 800; font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px; margin-bottom: 4px;
}
.grade-badge {
    display: inline-block; padding: 6px 18px; border-radius: 999px;
    font-weight: 800; font-size: 15px; letter-spacing: 1px;
    margin-left: 14px; vertical-align: middle;
}
.grade-Aplus { background: linear-gradient(135deg, #ffd54f, #f7931a);
               color: #1a1428; }
.grade-A { background: #26a69a; color: #fff; }
.grade-B { background: #2196f3; color: #fff; }
.grade-C { background: #ff9800; color: #fff; }
.grade-D { background: #757575; color: #fff; }

.conf-bar {
    height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08);
    overflow: hidden; margin-top: 12px;
}
.conf-fill {
    height: 100%; border-radius: 999px;
    transition: width 0.8s ease;
    background: linear-gradient(90deg, #f7931a, #ffd54f);
}

.info-box {
    background: rgba(30,25,45,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 3px solid #f7931a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 13px;
}
.info-box.good { border-left-color: #26a69a; }
.info-box.warn { border-left-color: #ff9800; }
.info-box.bad  { border-left-color: #ef5350; }

.tag {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 600; margin: 2px;
    background: rgba(247,147,26,0.15); color: #f7931a;
    border: 1px solid rgba(247,147,26,0.3);
}
.tag-good { background: rgba(38,166,154,0.15); color: #26a69a;
            border-color: rgba(38,166,154,0.3); }
.tag-bad  { background: rgba(239,83,80,0.15); color: #ef5350;
            border-color: rgba(239,83,80,0.3); }

.stButton > button {
    background: linear-gradient(135deg, #f7931a, #ffd54f);
    color: #1a1428; border: none; border-radius: 10px;
    font-weight: 700; padding: 10px 22px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(247,147,26,0.4);
}

h2, h3 { color: #e8e8f0 !important; }
.stExpander { border-color: rgba(247,147,26,0.2) !important; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #f7931a !important;
    border-bottom-color: #f7931a !important;
}
</style>
"""

# ============================================================
# UI COMPONENTS
# ============================================================

def render_header():
    logo = load_logo_b64()
    logo_html = (f'<img src="data:image/png;base64,{logo}" '
                 f'style="height:56px; border-radius:12px; margin-right:16px;">'
                 if logo else "")
    st.markdown(f"""
    <div class="bp-header">
        <div style="display:flex; align-items:center;">
            {logo_html}
            <div>
                <h1>₿ BLACK PYRAMID · BTC TERMINAL</h1>
                <div class="sub">Institutional Analysis · {APP_VERSION} · {SYMBOL}</div>
            </div>
        </div>
        <div style="text-align:right;">
            <div class="sub">🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}</div>
            <div class="sub" style="margin-top:2px;">{get_current_session_info()['session']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None, delta_dir="neutral"):
    delta_html = ""
    if delta is not None:
        cls = {"up":"delta-up","down":"delta-down"}.get(delta_dir, "delta-neutral")
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_metrics_row(result):
    spot = result.get("spot", (None, None))
    price = spot[0] or result.get("price")
    chg = spot[1] if spot[1] is not None else None

    funding = result.get("funding", {})
    oi = result.get("open_interest", {})
    fg = result.get("fear_greed", {})
    dom = result.get("dominance")

    cols = st.columns(6)
    with cols[0]:
        d = fmt_pct(chg) if chg is not None else None
        dd = "up" if (chg or 0) > 0 else "down" if (chg or 0) < 0 else "neutral"
        metric_card("BTC PRICE", fmt_price(price), d, dd)
    with cols[1]:
        fr = funding.get("rate", 0)
        st_label = {"HOT_LONG":"🔥 Longs","LONG":"↑","HOT_SHORT":"❄️ Shorts",
                    "BALANCED":"·","UNKNOWN":"?"}.get(funding.get("state","?"),"")
        metric_card("FUNDING", f"{fr:.4f}%", st_label,
                    "up" if fr < -0.02 else "down" if fr > 0.02 else "neutral")
    with cols[2]:
        oic = oi.get("change_24h", 0)
        v = f"${oi['oi_usd']/1e9:.1f}B" if oi.get("oi_usd") else "N/A"
        metric_card("OPEN INTEREST", v, fmt_pct(oic),
                    "up" if oic > 0 else "down")
    with cols[3]:
        fgv = fg.get("value", 50)
        dd = "up" if fgv > 60 else "down" if fgv < 40 else "neutral"
        metric_card("FEAR & GREED", str(fgv), fg.get("label",""),
                    dd)
    with cols[4]:
        dsv = f"{dom:.1f}%" if dom else "N/A"
        metric_card("BTC DOMINANCE", dsv, None)
    with cols[5]:
        mac = result.get("macro", {})
        vs = mac.get("vix_state", "?")
        vv = f"{mac['vix_value']:.1f}" if mac.get("vix_value") else "N/A"
        metric_card("VIX", vv, vs.replace("HIGH FEAR","خوف").replace("COMPLACENT","هدوء"),
                    "down" if "HIGH" in vs else "up" if "COMPLACENT" in vs else "neutral")

def render_signal_card(result):
    direction = result["signal"]
    grade = result["trade_grade"]
    conf = result["confidence"]
    emit = result["emit"]

    css = {"BUY":"signal-buy","SELL":"signal-sell"}.get(
        direction, "signal-wait") if emit else "signal-wait"
    grade_cls = f"grade-{grade.replace('+','plus')}"
    arrow = "▲" if direction == "BUY" else "▼"
    color = "#26a69a" if direction == "BUY" else "#ef5350"
    title = ("إشارة مُصدَرة" if emit else "لا إشارة — انتظار")
    main = f"{arrow} {direction}" if emit else "⏸ WAIT"

    st.markdown(f"""
    <div class="signal-card {css}">
        <div class="signal-title">{title}</div>
        <div class="signal-main" style="color:{color};">
            {main}
            <span class="grade-badge {grade_cls}">{grade}</span>
        </div>
        <div style="color:#8a8aa0; font-size:14px; margin-top:6px;">
            الثقة: <b style="color:#fff;">{conf:.1f}%</b> ·
            فجوة: <b style="color:#fff;">{result['confidence_gap']:.1f}</b> ·
            نظام: <b style="color:#fff;">{result['regime']}</b>
        </div>
        <div class="conf-bar">
            <div class="conf-fill" style="width:{conf}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if emit and result.get("ict"):
        ict = result["ict"]
        cols = st.columns(5)
        labels = [("ENTRY", ict["entry"], "#ffd54f"),
                  ("SL", ict["sl"], "#ef5350"),
                  ("TP1", ict["tp1"], "#26a69a"),
                  ("TP2", ict["tp2"], "#26a69a"),
                  ("TP3", ict["tp3"], "#26a69a")]
        for c, (lbl, val, color) in zip(cols, labels):
            with c:
                st.markdown(f"""
                <div class="metric-card" style="border-left:3px solid {color};">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="color:{color};">{fmt_price(val)}</div>
                </div>
                """, unsafe_allow_html=True)

def render_pillars_tab(result):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 📊 تحليل Pillars")
        fig = make_pillars_chart(result["pillars"], result["signal"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### 🎯 الأسباب")
        reasons = result["reasons"] or []
        if reasons:
            for r in reasons[:10]:
                st.markdown(f'<span class="tag tag-good">{r}</span>',
                            unsafe_allow_html=True)
        else:
            st.info("لا أسباب قوية")
        st.markdown("### 🚪 بوابة التأكيد")
        if result["gate_pass"]:
            st.markdown(f'<div class="info-box good">✅ البوابة مفتوحة '
                        f'(score: {result["confirmation_score"]:.0f})</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box warn">⚠️ البوابة مغلقة '
                        f'(score: {result["confirmation_score"]:.0f})</div>',
                        unsafe_allow_html=True)
        for b in result.get("gate_blockers", []):
            st.markdown(f'<span class="tag tag-bad">{b}</span>',
                        unsafe_allow_html=True)

def render_fund_tab(result):
    st.markdown("### 💰 التحليل الأساسي (Crypto Metrics)")
    fg = result["fear_greed"]
    funding = result["funding"]
    oi = result["open_interest"]
    dom = result["dominance"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("FEAR & GREED", str(fg["value"]), fg["label"],
                    "up" if fg["value"]>60 else "down" if fg["value"]<40 else "neutral")
    with c2:
        metric_card("FUNDING", f"{funding['rate']:.4f}%",
                    funding["state"].replace("_"," ").title(),
                    "down" if funding['rate']>0.02 else "up" if funding['rate']<-0.02 else "neutral")
    with c3:
        oiv = f"${oi['oi_usd']/1e9:.1f}B" if oi.get("oi_usd") else "N/A"
        metric_card("OPEN INTEREST", oiv, fmt_pct(oi["change_24h"]),
                    "up" if oi["change_24h"]>0 else "down")
    with c4:
        dv = f"{dom:.1f}%" if dom else "N/A"
        metric_card("BTC DOMINANCE", dv)

    st.markdown("### 📋 تفاصيل العوامل")
    fig = make_fund_chart(result["fund_details"])
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🌍 السياق الماكرو")
    mac = result["macro"]
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        dxy = mac.get("dxy_bias", "NEUTRAL")
        ic = trend_icon("BULLISH" if dxy=="BULLISH" else "BEARISH" if dxy=="BEARISH" else "NEUTRAL")
        st.markdown(f'<div class="info-box">{ic} <b>DXY</b>: {dxy}<br>'
                    f'Corr: {mac.get("dxy_corr", 0):.2f}</div>',
                    unsafe_allow_html=True)
    with mc2:
        vs = mac.get("vix_state", "UNKNOWN")
        st.markdown(f'<div class="info-box">📊 <b>VIX</b>: {mac.get("vix_value","?")}<br>'
                    f'State: {vs}<br>Corr: {mac.get("vix_corr", 0):.2f}</div>',
                    unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="info-box">📈 <b>SPX Corr</b>: '
                    f'{mac.get("spx_corr", 0):.2f}</div>',
                    unsafe_allow_html=True)

def render_mtf_tab(result):
    st.markdown("### 🕐 تحليل Multi-Timeframe")
    details = result["mtf_details"]
    cols = st.columns(len(details))
    for (tf, d), c in zip(details.items(), cols):
        with c:
            ic = trend_icon(d["bias"])
            color = "#26a69a" if d["bias"]=="BULLISH" else \
                    "#ef5350" if d["bias"]=="BEARISH" else "#8a8aa0"
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="metric-label">{tf}</div>
                <div style="font-size:32px;">{ic}</div>
                <div style="color:{color}; font-weight:700;">{d['bias']}</div>
                <div style="color:#8a8aa0; font-size:11px;">قوة: {d['strength']}/10</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"**الإجماع:** {result['mtf_bias']} · ثقة: {result['mtf_conf']:.1f}%")
    st.markdown(f"**Weekly Bias:** {result['weekly_bias']}")

def render_chart_tab(result):
    df = result["df"]
    fig = make_main_chart(df, result)
    st.plotly_chart(fig, use_container_width=True)

def render_ml_tab(result):
    st.markdown("### 🤖 تحليل Machine Learning")
    ml = st.session_state.get("ml_model")
    if not ml or "model" not in ml:
        st.info("لم يتم تدريب نموذج ML بعد. اذهب إلى تبويب ML في الشريط الجانبي.")
        return
    st.markdown(f"""
    <div class="info-box good">
        <b>النموذج:</b> {ml['model_name']} ·
        <b>AUC:</b> {ml['auc']:.3f} ·
        <b>Acc:</b> {ml['accuracy']:.3f} ·
        <b>F1:</b> {ml['f1']:.3f}<br>
        <b>عدد الميزات:</b> {len(ML_FEATURE_NAMES)} ·
        <b>Train:</b> {ml['n_train']} · <b>Test:</b> {ml['n_test']}
    </div>
    """, unsafe_allow_html=True)
    conf_adj, prob, status = ml_adjusted_confidence(result, result["signal"], ml)
    if prob is not None:
        st.markdown(f"**احتمالية الفوز (ML):** {prob*100:.1f}%")
        st.markdown(f"**الثقة المعدلة:** {conf_adj:.1f}% (الحالة: {status})")

# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(page_title="Black Pyramid · BTC",
                       page_icon="₿", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()

    render_header()

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")
        strict = st.toggle("فلاتر مشددة", value=st.session_state.strict_filters,
                           key="strict_filters")
        st.markdown("---")
        st.markdown("### 🎯 Kill Zones")
        kz_in, kz_msg = in_kill_zone()
        if kz_in:
            st.success(kz_msg)
        else:
            st.info(kz_msg)
        st.markdown("---")
        st.markdown("### 🤖 ML Model")
        if st.button("📚 تدريب ML (يحتاج بيانات)", use_container_width=True):
            st.warning("جمع البيانات التاريخية... قد يستغرق دقائق.")
            trades = collect_historical_trades(limit=200)
            if "error" in trades:
                st.error(trades["error"])
            else:
                with st.spinner(f"تدريب على {len(trades)} صفقة..."):
                    ml = train_ml_model(trades, use_tuning=False)
                if "error" in ml:
                    st.error(ml["error"])
                else:
                    st.session_state.ml_model = ml
                    st.success(f"✅ AUC: {ml['auc']:.3f}")
        ml = st.session_state.get("ml_model")
        if ml and "model" in ml:
            st.markdown(f"**AUC:** {ml['auc']:.3f}<br>"
                        f"**Acc:** {ml['accuracy']:.3f}<br>"
                        f"**Model:** {ml['model_name']}",
                        unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 📝 سجل التداول")
        journal = st.session_state.get("trade_journal", [])
        st.markdown(f"**{len(journal)}** صفقة مسجلة")
        st.markdown("---")
        st.caption(f"{APP_VERSION} · Built for BTC")

    # ── Analysis ────────────────────────────────────────
    with st.spinner("🔍 تحليل BTC..."):
        result = analyze_btc()

    if "error" in result:
        st.error(f"❌ {result['error']}")
        return

    st.session_state.btc_result = result
    st.session_state.analysis_time = result["timestamp"]

    # ── Metrics Row ────────────────────────────────────
    render_metrics_row(result)

    # ── Signal Card ────────────────────────────────────
    render_signal_card(result)

    # ── Tabs ───────────────────────────────────────────
    tabs = st.tabs(["📈 Chart", "📊 Pillars", "💰 Fundamentals",
                    "🕐 MTF", "🤖 ML", "📉 Monte Carlo"])

    with tabs[0]:
        render_chart_tab(result)
    with tabs[1]:
        render_pillars_tab(result)
    with tabs[2]:
        render_fund_tab(result)
    with tabs[3]:
        render_mtf_tab(result)
    with tabs[4]:
        render_ml_tab(result)
    with tabs[5]:
        render_monte_carlo_tab()

    # ── Footer ─────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; color:#5a5a72; font-size:11px;
                margin-top:40px; padding:20px 0;
                border-top:1px solid rgba(255,255,255,0.05);">
        Black Pyramid · {APP_VERSION} · BTC Only ·
        آخر تحديث: {result['timestamp']} ·
        ⚠️ تحليل آلي — ليس نصيحة استثمارية
    </div>
    """, unsafe_allow_html=True)


def render_monte_carlo_tab():
    st.markdown("### 📉 محاكاة Monte Carlo")
    ml = st.session_state.get("ml_model")
    if not ml or not ml.get("_trades"):
        st.info("درّب نموذج ML أولاً — يحتوي على عوائد الصفقات التاريخية.")
        return
    mc = monte_carlo_simulation(ml["_trades"], n_sims=1000)
    if not mc: return
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("PROFIT PROB", f"{mc['prob_profit']:.1f}%",
                         "up" if mc['prob_profit']>50 else "down")
    with c2: metric_card("MEAN RETURN", f"{mc['mean_final']:.2f}R",
                         "up" if mc['mean_final']>0 else "down")
    with c3: metric_card("P95 DD", f"{mc['p95_dd']:.2f}R", "down")
    with c4: metric_card("MEDIAN", f"{mc['median_final']:.2f}R")


def collect_historical_trades(limit=200):
    """
    جمع صفقات تاريخية من بيانات 4h — يستخدم للـ ML training.
    """
    try:
        df = get_btc_ohlcv("2y", "4h")
        if df is None or len(df) < 200:
            return {"error": "بيانات تاريخية غير كافية"}
        x = build_features(df, PROFILE)
        trades = []
        step = max(1, len(x) // limit)
        for i in range(100, len(x) - 20, step):
            window = x.iloc[:i+1]
            try:
                res = _quick_result_from_window(window)
                if res is None: continue
                d = res["signal"]
                future = x.iloc[i:i+20]
                if len(future) < 5: continue
                entry = float(future["close"].iloc[0])
                atr = float(future["atr"].iloc[0])
                if atr <= 0: continue
                tp = entry + 2*atr if d == "BUY" else entry - 2*atr
                sl = entry - atr if d == "BUY" else entry + atr
                outcome = None
                for _, row in future.iterrows():
                    if d == "BUY":
                        if row["low"] <= sl: outcome = 0; break
                        if row["high"] >= tp: outcome = 1; break
                    else:
                        if row["high"] >= sl: outcome = 0; break
                        if row["low"] <= tp: outcome = 1; break
                if outcome is None: continue
                feats = extract_ml_features(res, d)
                trades.append({"features": feats, "label": outcome,
                               "direction": d})
            except Exception:
                continue
        if len(trades) < ML_MIN_TRADES:
            return {"error": f"صفقات كافية للتدريب ({len(trades)}/{ML_MIN_TRADES})"}
        return trades
    except Exception as e:
        return {"error": str(e)}


def _quick_result_from_window(window):
    """توليد نتيجة مبسطة داخل نافذة بيانات للتدريب."""
    try:
        dir_sc = directional_score(window)
        d = "BUY" if dir_sc["buy"] > dir_sc["sell"] else "SELL"
        conf = max(dir_sc["buy"], dir_sc["sell"])
        return {
            "signal": d, "confidence": conf,
            "raw_confidence": conf, "confirmation_score": 0,
            "mtf_conf": 50, "mtf_bias": "NEUTRAL",
            "weekly_bias": "NEUTRAL", "regime": dir_sc["regime"],
            "pillars": dir_sc["pillars"],
            "divergence": dir_sc["divergence"],
            "pair_name": PAIR_NAME, "df": window,
            "funding": {"rate": 0.0}, "open_interest": {"change_24h": 0.0},
            "dominance": 0.0, "filter_results": {},
        }
    except Exception:
        return None


if __name__ == "__main__":
    main()