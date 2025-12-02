import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# Optional loaders
try:
    from tensorflow.keras.models import load_model as keras_load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(page_title="Room Cancellation Predictor", layout="wide", page_icon="🏨")

# --------------- CSS (hero widened + subtitle animation + UI polish) ---------------
st.markdown(
    """
    <style>
    :root {
      --bg-gradient: linear-gradient(180deg,#f3f7fb 0%, #eaf1f8 40%, #ffffff 100%);
      --card-bg: #ffffff;
      --muted: #64748b;
      --accent-1: #2563eb;
      --accent-2: #7c3aed;
      --input-border: #d6e1ee;
      --spin-bg: #edf2fb;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-gradient) !important;
    }
    .stApp, .stApp * { color: #0f172a !important; }

    /* WIDE HERO: spans across the content area with big visual presence */
    .wide-hero {
        width: 100%;
        display: flex;
        gap: 20px;
        align-items: center;
        justify-content: flex-start;
        background: linear-gradient(90deg, rgba(255,255,255,0.95), rgba(255,255,255,0.98));
        padding: 28px 34px;
        border-radius: 18px;
        box-shadow: 0 14px 45px rgba(2,6,23,0.06);
        margin-bottom: 22px;
        overflow: hidden;
    }

    .wide-hero .hero-left {
        display: flex;
        align-items: center;
        gap: 18px;
        min-width: 420px;
    }

    .wide-hero .logo {
        width:110px;
        height:110px;
        border-radius:16px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:48px;
        background: linear-gradient(180deg,#ffffff,#f6fbff);
        box-shadow: 0 12px 34px rgba(15,23,42,0.06);
    }

    .wide-hero .title-wrap {
        display:flex;
        flex-direction:column;
        gap:6px;
    }

    .wide-hero h1 {
        margin:0;
        font-size:34px;
        font-weight:800;
        letter-spacing: -0.3px;
    }

    /* Subtitle: subtle slide-up + fade animation */
    @keyframes slideFadeUp {
      0% { transform: translateY(12px); opacity: 0; }
      100% { transform: translateY(0px); opacity: 1; }
    }
    .wide-hero .subtitle {
        margin:0;
        color: var(--muted);
        font-size:15px;
        animation: slideFadeUp 700ms ease-out 250ms both;
    }

    /* Hero right area: action buttons / quick stats */
    .wide-hero .hero-right {
        margin-left: auto;
        display:flex;
        gap:12px;
        align-items:center;
    }
    .hero-btn {
        padding:10px 18px;
        border-radius:10px;
        font-weight:700;
        background: linear-gradient(90deg,var(--accent-1),var(--accent-2));
        color:#fff;
        border:none;
        box-shadow: 0 8px 24px rgba(37,99,235,0.12);
    }

    /* KPI row */
    .kpi-row { display:flex; gap:14px; margin-bottom:18px; }
    .kpi {
        background: var(--card-bg);
        padding:12px 16px;
        border-radius:12px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.04);
        flex:1;
        min-width:140px;
    }
    .kpi .label { color:var(--muted); font-size:13px; }
    .kpi .value { font-weight:800; font-size:20px; margin-top:6px; }

    /* Panels */
    .panel {
        background: var(--card-bg) !important;
        padding: 18px !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 26px rgba(15,23,42,0.04) !important;
        margin-bottom: 18px !important;
    }

    /* Inputs */
    input[type="text"], input[type="number"], textarea, select, .stTextInput>div>input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid var(--input-border) !important;
        padding: 10px 12px !important;
    }
    div[role="spinbutton"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid var(--input-border) !important;
        display:flex !important;
        align-items:center !important;
    }
    div[role="spinbutton"] button {
        background-color: var(--spin-bg) !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        padding: 6px 12px !important;
        border: none !important;
        margin-left: 6px !important;
        margin-right: 6px !important;
    }
    div[role="spinbutton"] button:hover { background-color: #e2eaf9 !important; }

    div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid var(--input-border) !important;
        padding: 8px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,var(--accent-1),var(--accent-2)) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        border: none !important;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 14px 36px rgba(79,70,229,0.14) !important; }

    /* Result badge */
    .result-badge { padding:10px 14px; border-radius:10px; font-weight:700; display:inline-block; }

    /* Signature */
    .signature { text-align:center; margin-top:28px; }
    .signature .name {
        font-family: 'Brush Script MT','Satisfy',cursive;
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        padding: 10px 28px;
        border-radius: 999px;
        background: linear-gradient(90deg,#ffffff,#f6fbff);
        box-shadow: 0 10px 30px rgba(0,0,0,0.06);
        display:inline-block;
    }

    /* Ensure JSON/code blocks are white */
    pre, code, .stCodeBlock, .stJson {
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        padding: 12px !important;
        border: 1px solid #e6edf3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Paths ----------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"

# ---------------- Loaders ----------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            st.warning("Preprocessor failed to load.")
    return None

@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    lower = path.lower()
    if lower.endswith(".h5") and KERAS_AVAILABLE:
        try:
            return ("keras", keras_load_model(path))
        except Exception:
            pass
    if JOBLIB_AVAILABLE:
        try:
            return ("sklearn", joblib.load(path))
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            return ("pickle", pickle.load(f))
    except Exception:
        return None

# ---------------- Heuristic ----------------
def heuristic_predict(row):
    score = 0.0
    score += 0.45 * (row.get("lead_time", 0) / 365.0)
    score += 0.25 * (1 if row.get("previous_cancellations", 0) > 0 else 0)
    score += 0.15 * (1 if row.get("deposit_type") == "No Deposit" else 0)
    score += 0.15 * (1 if row.get("booking_changes", 0) > 2 else 0)
    return float(min(max(score, 0.0), 0.99))

# ---------------- Hero (wider + animated subtitle) ----------------
col1, col2 = st.columns([1, 4], gap="small")
with col1:
    st.markdown('<div class="wide-hero" style="align-items:center;"><div class="hero-left"><div class="logo">🏩</div></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        """
        <div class="wide-hero" style="padding-left:8px;">
            <div style="display:flex; flex-direction:column; gap:6px;">
                <div class="title-wrap">
                    <h1>Room Cancellation Predictor</h1>
                    <p class="subtitle">A clear, modern UI — connect your model or use the built-in heuristic to estimate cancellation risk.</p>
                </div>
            </div>
            <div class="hero-right" style="margin-left:auto;">
                <button class="hero-btn" onclick="">Quick demo</button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- KPI row ----------------
lead_sample = 30
k1, k2, k3 = st.columns([1,1,1], gap="small")
with k1:
    st.markdown('<div class="kpi"><div class="label">Sample bookings</div><div class="value">300</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="label">Avg lead time</div><div class="value">{lead_sample} days</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi"><div class="label">Model status</div><div class="value">' + ('Loaded' if os.path.exists(MODEL_PATH) else 'Heuristic') + '</div></div>', unsafe_allow_html=True)

# ---------------- Layout - form & results ----------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Booking details")
    st.markdown('<p class="small-note">Complete the booking information — labels and spacing are optimized for clarity.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=2000, value=30)
            stays_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=30, value=0)
            stays_week_nights = st.number_input("Week nights", min_value=0, max_value=365, value=2)
        with r1c2:
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0)
            booking_changes = st.number_input("Booking changes", min_value=0, max_value=50, value=0)
            deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
        g1, g2 = st.columns(2)
        with g1:
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with g2:
            market_segment = st.selectbox("Market segment", ["Direct", "Online TA", "Offline TA/TO", "Groups", "Corporate", "Complementary", "Aviation"])

        st.markdown("")  # spacing
        submit = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Inputs summary")
    feature_row = {
        "lead_time": int(lead_time),
        "stays_weekend_nights": int(stays_weekend_nights),
        "stays_week_nights": int(stays_week_nights),
        "adults": int(adults),
        "children": int(children),
        "previous_cancellations": int(previous_cancellations),
        "booking_changes": int(booking_changes),
        "deposit_type": deposit_type,
        "market_segment": market_segment,
    }
    df_inputs = pd.DataFrame(list(feature_row.items()), columns=["Field", "Value"])
    st.table(df_inputs)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction")
    st.markdown('<p class="small-note">Prediction results and quick actions.</p>', unsafe_allow_html=True)
    metric_placeholder = st.empty()
    badge_placeholder = st.empty()
    gauge_placeholder = st.empty()
    download_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction logic ----------------
if submit:
    preproc = load_preprocessor()
    model_info = load_trained_model()

    prob = None
    if preproc is not None and model_info is not None:
        try:
            mtype, model_obj = model_info
            X = pd.DataFrame([feature_row])
            try:
                Xp = preproc.transform(X)
            except Exception:
                Xp = X
            if mtype == "keras":
                preds = model_obj.predict(Xp)
                prob = float(preds[0][1]) if (hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] > 1) else float(preds[0])
            else:
                try:
                    prob = float(model_obj.predict_proba(Xp)[0][1])
                except Exception:
                    pred = model_obj.predict(Xp)[0]
                    prob = 1.0 if int(pred) == 1 else 0.0
        except Exception as e:
            st.sidebar.error(f"Model inference error: {e}")
            prob = heuristic_predict(feature_row)
    else:
        prob = heuristic_predict(feature_row)

    label = "Cancelled" if prob >= 0.5 else "Not cancelled"

    # Update right column
    metric_placeholder.metric("Cancellation probability", f"{prob:.2%}")

    if label == "Cancelled":
        badge_placeholder.markdown("<div class='result-badge' style='background:#fff5f5;color:#b91c1c'>Prediction: CANCELLED</div>", unsafe_allow_html=True)
    else:
        badge_placeholder.markdown("<div class='result-badge' style='background:#f0fdf4;color:#047857'>Prediction: NOT CANCELLED</div>", unsafe_allow_html=True)

    fig = go.Figure(go.Pie(values=[prob, 1-prob], hole=0.66, marker_colors=["#ef4444","#10b981"], textinfo="none"))
    fig.update_layout(showlegend=False, margin=dict(t=6,b=6,l=6,r=6), height=260,
                      annotations=[dict(text=f"{prob:.0%}", x=0.5, y=0.5, font_size=26, showarrow=False)])
    gauge_placeholder.plotly_chart(fig, use_container_width=True)

    result_csv = pd.DataFrame([feature_row]).assign(prediction_prob=prob)
    download_placeholder.download_button("Download result CSV", data=result_csv.to_csv(index=False).encode("utf-8"),
                                         file_name="prediction_result.csv", mime="text/csv")

# ---------------- Signature ----------------
st.markdown(
    """
    <div class="signature">
        <div class="name">Created by Venky &amp; Subba Reddy</div>
    </div>
    """,
    unsafe_allow_html=True,
)
