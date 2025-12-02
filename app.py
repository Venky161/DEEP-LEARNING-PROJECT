# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go

# Optional model loaders
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


# ------------------ Page config ------------------
st.set_page_config(page_title="Room Cancellation Predictor",
                   layout="wide",
                   page_icon="🏨")

# ------------------ Sidebar Controls ------------------
st.sidebar.title("⚙ Settings")
dark_mode = st.sidebar.checkbox("Dark mode", value=False, help="Toggle a dark theme for the app visuals (inputs remain visible).")
st.sidebar.markdown("---")
st.sidebar.markdown("**Header logo (optional)**")
uploaded_logo = st.sidebar.file_uploader("Upload logo (png/jpg, optional)", type=["png", "jpg", "jpeg"])
st.sidebar.markdown("---")
st.sidebar.write("Model present:", os.path.exists("/mnt/data/Hotel reservatiosn.h5"))
st.sidebar.write("Preprocessor present:", os.path.exists("/mnt/data/preprocessor.pkl"))


# ------------------ CSS (switches light/dark + floating animation) ------------------
# Base CSS (light theme)
css_light = """
<style>
:root{
  --page-bg: linear-gradient(180deg,#eef4f9 0%, #e6eef6 40%, #f9fbfd 100%);
  --card-bg: #ffffff;
  --text: #0f172a;
  --muted: #475569;
  --input-bg: #ffffff;
  --input-border: #d1d9e6;
  --spin-bg: #e9eef5;
  --accent1: #2563eb;
  --accent2: #4f46e5;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--page-bg) !important;
}
.stApp, .stApp * { color: var(--text) !important; }

/* floating hero animation */
@keyframes floaty {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
  100% { transform: translateY(0px); }
}
.hero {
    background: var(--card-bg);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 18px;
    animation: floaty 6s ease-in-out infinite;
}
.hero h1 { margin: 0; font-size: 28px; }
.small-note { color: var(--muted); margin-top:6px; }

/* panels */
.panel {
    background: var(--card-bg) !important;
    padding: 18px !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 22px rgba(15,23,42,0.05) !important;
    margin-bottom: 18px !important;
}

/* inputs */
input[type="text"], input[type="number"], textarea, select, .stTextInput>div>input {
    background-color: var(--input-bg) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}

/* spinbutton (– / +) */
div[role="spinbutton"] {
    background-color: var(--input-bg) !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}
div[role="spinbutton"] button {
    background-color: var(--spin-bg) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    border: none !important;
}
div[role="spinbutton"] button:hover { background-color: #d8e0ea !important; }

/* dropdown */
div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
    background-color: var(--input-bg) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg,var(--accent1),var(--accent2)) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    border-radius: 10px !important;
    border: none !important;
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(37,99,235,0.15); }

/* signature */
.signature { text-align: center; margin-top: 32px; }
.signature .name {
    font-family: 'Brush Script MT','Satisfy',cursive;
    font-size: 36px;
    font-weight: 800;
    padding: 10px 26px;
    border-radius: 999px;
    color: var(--text);
    background: linear-gradient(90deg,#ffffff,#f7faff);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    display: inline-block;
}

/* ensure code/json blocks appear white */
pre, code, .stCodeBlock, .stJson {
    background: #ffffff !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid #e6edf3 !important;
}
</style>
"""

# Dark CSS (keeps inputs readable)
css_dark = """
<style>
:root{
  --page-bg: linear-gradient(180deg,#0b1220 0%, #0f1724 40%, #161a26 100%);
  --card-bg: #0f1724;
  --text: #e6eef6;
  --muted: #cbd5e1;
  --input-bg: #ffffff;        /* keep inputs white for readability */
  --input-border: #2b3946;
  --spin-bg: #f1f5f9;
  --accent1: #2563eb;
  --accent2: #7c3aed;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--page-bg) !important;
}
.stApp, .stApp * { color: var(--text) !important; }

/* floating hero animation */
@keyframes floaty {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
  100% { transform: translateY(0px); }
}
.hero {
    background: var(--card-bg);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    margin-bottom: 18px;
    animation: floaty 6s ease-in-out infinite;
}
.hero h1 { margin: 0; font-size: 28px; color: var(--text); }
.small-note { color: var(--muted); margin-top:6px; }

/* panels */
.panel {
    background: linear-gradient(180deg,#0f1724,#0b1220) !important;
    padding: 18px !important;
    border-radius: 12px !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.5) !important;
    margin-bottom: 18px !important;
}

/* inputs (kept white for legibility) */
input[type="text"], input[type="number"], textarea, select, .stTextInput>div>input {
    background-color: var(--input-bg) !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}

/* spinbutton */
div[role="spinbutton"] {
    background-color: var(--input-bg) !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}
div[role="spinbutton"] button {
    background-color: var(--spin-bg) !important;
    color: #0f172a !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    border: none !important;
}
div[role="spinbutton"] button:hover { background-color: #e0e7ef !important; }

/* dropdown */
div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
    background-color: var(--input-bg) !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid var(--input-border) !important;
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg,var(--accent1),var(--accent2)) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    border-radius: 10px !important;
    border: none !important;
}

/* signature */
.signature { text-align: center; margin-top: 32px; }
.signature .name {
    font-family: 'Brush Script MT','Satisfy',cursive;
    font-size: 36px;
    font-weight: 800;
    padding: 10px 26px;
    border-radius: 999px;
    color: var(--text);
    background: linear-gradient(90deg,#0f1724,#111827);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    display: inline-block;
}

/* code blocks - keep white for readability */
pre, code, .stCodeBlock, .stJson {
    background: #ffffff !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    padding: 12px !important;
    border: 1px solid #273241 !important;
}
</style>
"""

# Inject the CSS according to toggle
if dark_mode:
    st.markdown(css_dark, unsafe_allow_html=True)
else:
    st.markdown(css_light, unsafe_allow_html=True)


# ------------------ Paths ------------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"


# ------------------ Loaders ------------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            st.sidebar.warning("Preprocessor failed to load properly.")
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


# ------------------ Heuristic predictor ------------------
def heuristic_predict(row):
    score = 0.45 * (row.get("lead_time", 0) / 365.0)
    score += 0.25 * (1 if row.get("previous_cancellations", 0) > 0 else 0)
    score += 0.15 * (1 if row.get("deposit_type") == "No Deposit" else 0)
    score += 0.15 * (1 if row.get("booking_changes", 0) > 2 else 0)
    return float(min(max(score, 0.0), 0.99))


# ------------------ Header: logo + title ------------------
logo_html = ""
if uploaded_logo is not None:
    try:
        # read uploaded bytes and display
        logo_bytes = uploaded_logo.getvalue()
        encoded = "data:image/png;base64," + (logo_bytes.encode("base64") if hasattr(logo_bytes, "encode") else "")
        # Fallback: Streamlit will display via st.image below instead, so keep logo_html minimal
        logo_html = ""
    except Exception:
        logo_html = ""
# Show header with either uploaded logo (st.image) or a default hotel SVG + title
col1, col2 = st.columns([1, 9])
with col1:
    if uploaded_logo is not None:
        try:
            st.image(uploaded_logo, width=84)
        except Exception:
            st.markdown("<div style='font-size:40px'>🏨</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:40px'>🏨</div>", unsafe_allow_html=True)
with col2:
    st.markdown(
        """
        <div class="hero">
            <h1>Room Cancellation Predictor</h1>
            <p class="small-note">Use your model or the built-in heuristic to estimate cancellation risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------ Layout: form & results ------------------
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill the fields and press Predict.</p>', unsafe_allow_html=True)

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
            market_segment = st.selectbox("Market segment",
                                         ["Direct", "Online TA", "Offline TA/TO", "Groups", "Corporate", "Complementary", "Aviation"])

        submit = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

    # Present inputs neatly (table)
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

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction")
    st.markdown('<p class="small-note">Results appear here after you predict.</p>', unsafe_allow_html=True)
    # placeholders
    result_metric = st.empty()
    result_label = st.empty()
    result_gauge = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ Prediction logic ------------------
if submit:
    preproc = st.session_state.get("_uploaded_preproc", None) or load_preprocessor()
    model_info = st.session_state.get("_uploaded_model", None) or load_trained_model()

    prob = None
    if preproc is not None and model_info is not None:
        try:
            mtype, model_obj = model_info
            X = pd.DataFrame([feature_row])
            try:
                X_proc = preproc.transform(X)
            except Exception:
                X_proc = X
            if mtype == "keras":
                preds = model_obj.predict(X_proc)
                prob = float(preds[0][1]) if (hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[1] > 1) else float(preds[0])
            else:
                try:
                    prob = float(model_obj.predict_proba(X_proc)[0][1])
                except Exception:
                    pred = model_obj.predict(X_proc)[0]
                    prob = 1.0 if int(pred) == 1 else 0.0
        except Exception as e:
            st.sidebar.error(f"Model inference failed: {e}")
            prob = heuristic_predict(feature_row)
    else:
        prob = heuristic_predict(feature_row)

    label = "Cancelled" if prob >= 0.5 else "Not cancelled"

    # update result widgets
    result_metric.metric("Cancellation probability", f"{prob:.2%}")

    if label == "Cancelled":
        result_label.markdown("<div style='padding:8px;border-radius:8px;background:#fff6f6;color:#9b1c1c;font-weight:700'>Prediction: Cancelled</div>", unsafe_allow_html=True)
    else:
        result_label.markdown("<div style='padding:8px;border-radius:8px;background:#f6fffb;color:#0b8235;font-weight:700'>Prediction: Not cancelled</div>", unsafe_allow_html=True)

    # donut gauge
    fig = go.Figure(go.Pie(
        values=[prob, 1-prob],
        hole=0.65,
        marker_colors=["#ef4444", "#10b981"],
        hoverinfo="label+percent",
        textinfo="none"
    ))
    fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                      height=260,
                      annotations=[dict(text=f"{prob:.0%}", x=0.5,y=0.5, font_size=28, showarrow=False)])
    result_gauge.plotly_chart(fig, use_container_width=True)


# ------------------ Footer signature ------------------
st.markdown(
    """
    <div class="signature">
        <div class="name">Created by Venky &amp; Subba Reddy</div>
    </div>
    """,
    unsafe_allow_html=True,
)
