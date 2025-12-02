import streamlit as st
import pandas as pd
import pickle
import os
import json

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

# ---------------- Page ----------------
st.set_page_config(page_title="Room Cancellation Predictor", layout="centered", page_icon="🏨")

# ---------------- CSS: enforce readable UI & code blocks ----------------
st.markdown(
    """
    <style>
    /* Page gradient */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e9f0f6 0%, #dce8ef 30%, #f6f9fb 100%) !important;
    }

    /* Ensure default text is dark and legible */
    .stApp, .stApp * {
        color: #0f172a !important;
    }

    /* Main card */
    .main-card {
        background: #ffffff !important;
        padding: 22px 26px !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06) !important;
        margin-bottom: 22px !important;
    }
    .small-note { color: #4b5563 !important; }

    /* Tabs selected */
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(37,99,235,0.08) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#2563eb,#4f46e5) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.6rem !important;
        font-weight: 700 !important;
        border: none !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(79,70,229,0.18) !important;
    }

    /* Force input backgrounds and text to be readable (overrides dark theme) */
    input[type="text"], input[type="number"], textarea, select, .stTextInput>div>input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* number input (spinbutton) fix */
    div[role="spinbutton"] input[type="number"], input[type="number"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* dropdowns / select */
    div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }

    /* fix dark slider or other dark widget backgrounds */
    div[data-baseweb="slider"], .stSlider {
        background: transparent !important;
    }

    /* Ensure labels and headings appear dark */
    label, .stMarkdown, .css-16idsys, .css-1fv8s86 {
        color: #0f172a !important;
    }

    /* Make st.metric boxes readable */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff,#fbfdff) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        border: 1px solid #eef3f7 !important;
        color: #0f172a !important;
    }

    /* Code / JSON display: force white background and dark text */
    pre, code, .stCodeBlock, .stJson {
        background: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        padding: 12px !important;
        border: 1px solid #e6edf3 !important;
        overflow: auto !important;
    }

    /* Signature styling (big cursive bold) */
    @font-face {
        font-family: 'SatisfyFallback';
        src: local('Satisfy'), local('Brush Script MT'), local('Lucida Handwriting');
    }
    .footer-tag {
        text-align: center;
        margin-top: 28px;
    }
    .footer-name {
        font-family: 'SatisfyFallback', cursive;
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        background: linear-gradient(90deg,#fff,#f7fbff);
        box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    }

    /* Keep sidebar readable */
    [data-testid="stSidebar"] { background: #ffffff !important; color: #0f172a !important; box-shadow: 0 6px 20px rgba(2,6,23,0.04) inset; }

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
        except Exception as e:
            st.warning(f"Could not load preprocessor: {e}")
    return None

@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    lower = path.lower()
    # keras
    if lower.endswith(".h5") and KERAS_AVAILABLE:
        try:
            return ("keras", keras_load_model(path))
        except Exception as e:
            st.warning(f"Could not load keras model: {e}")
    # joblib/sklearn
    if JOBLIB_AVAILABLE:
        try:
            return ("sklearn", joblib.load(path))
        except Exception:
            pass
    # pickle fallback
    try:
        with open(path, "rb") as f:
            return ("pickle", pickle.load(f))
    except Exception as e:
        st.warning(f"Could not load model: {e}")
    return None

# ---------------- Heuristic ----------------
def heuristic_predict(row):
    score = 0.0
    score += 0.45 * (row.get("lead_time", 0) / 365.0)
    score += 0.25 * (1 if row.get("previous_cancellations", 0) > 0 else 0)
    score += 0.15 * (1 if row.get("deposit_type") == "No Deposit" else 0)
    score += 0.15 * (1 if row.get("booking_changes", 0) > 2 else 0)
    prob = float(min(max(score, 0.0), 0.99))
    return prob

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ How to use")
st.sidebar.markdown(
    """
1. Fill the booking details  
2. Click **Predict**  
3. View the probability & label

(Optional) Place your model files in `/mnt/data`:
- `preprocessor.pkl`  
- `Hotel reservatiosn.h5`
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Model present:", os.path.exists(MODEL_PATH))
st.sidebar.write("Preprocessor present:", os.path.exists(PREPROC_PATH))

# allow user uploads
st.sidebar.markdown("**Upload (optional)**")
uploaded_preproc = st.sidebar.file_uploader("Upload preprocessor (.pkl)", type=["pkl"])
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.joblib/.h5)", type=["pkl","joblib","h5","keras"])

if uploaded_preproc is not None:
    try:
        uploaded_preproc.seek(0)
        st.session_state["_uploaded_preproc"] = pickle.load(uploaded_preproc)
        st.sidebar.success("Preprocessor loaded for this session")
    except Exception as e:
        st.sidebar.error(f"Preprocessor upload failed: {e}")

if uploaded_model is not None:
    try:
        # handle keras .h5 by writing temp file if needed
        if uploaded_model.name.lower().endswith(".h5") and KERAS_AVAILABLE:
            tmp_path = "/tmp/_streamlit_uploaded_model.h5"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.session_state["_uploaded_model"] = ("keras", keras_load_model(tmp_path))
        else:
            # joblib or pickle
            if JOBLIB_AVAILABLE:
                uploaded_model.seek(0)
                st.session_state["_uploaded_model"] = ("sklearn", joblib.load(uploaded_model))
            else:
                uploaded_model.seek(0)
                st.session_state["_uploaded_model"] = ("pickle", pickle.load(uploaded_model))
        st.sidebar.success("Model loaded for this session")
    except Exception as e:
        st.sidebar.error(f"Model upload failed: {e}")

# ---------------- Header ----------------
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin:0 0 6px 0;">🏨 Room Cancellation Predictor</h1>
        <p class="small-note" style="margin-top:0;">Estimate the probability that a booking will be cancelled. Connect your model or use the built-in heuristic.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Tabs ----------------
tabs = st.tabs(["🔮 Prediction", "ℹ About"])

with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Booking details")
    st.markdown('<p class="small-note">Enter booking details and click Predict.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=2000, value=30)
            stays_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=30, value=0)
            stays_week_nights = st.number_input("Week nights", min_value=0, max_value=365, value=2)
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with c2:
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0)
            booking_changes = st.number_input("Booking changes", min_value=0, max_value=50, value=0)
            deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
            market_segment = st.selectbox("Market segment",
                                         ["Direct", "Online TA", "Offline TA/TO", "Groups", "Corporate", "Complementary", "Aviation"])
        submit = st.form_submit_button("Predict")

    if submit:
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

        # Instead of st.json (which used dark code theme), render JSON inside a styled white code block
        st.markdown("### Inputs")
        json_pretty = json.dumps(feature_row, indent=2)
        st.markdown(f"<pre style='background:#ffffff; color:#0f172a; padding:12px; border-radius:8px; border:1px solid #e6edf3'>{json_pretty}</pre>", unsafe_allow_html=True)

        # load preprocessor & model (uploaded session take precedence)
        preproc = st.session_state.get("_uploaded_preproc", None) or load_preprocessor()
        model_info = st.session_state.get("_uploaded_model", None) or load_trained_model()

        prob = None
        if preproc is not None and model_info is not None:
            try:
                mtype, model_obj = model_info
                st.success("✅ Loaded preprocessor & model; running prediction...")
                X = pd.DataFrame([feature_row])
                try:
                    X_proc = preproc.transform(X)
                except Exception:
                    # If transform fails, try passing X as-is to model
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
                st.error(f"Model error: {e}")
                st.warning("Falling back to heuristic.")
                prob = heuristic_predict(feature_row)
        else:
            st.warning("No preprocessor/model found; using heuristic predictor.")
            prob = heuristic_predict(feature_row)

        label = "Cancelled" if prob >= 0.5 else "Not cancelled"

        st.markdown("### Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Probability of cancellation", f"{prob:.2%}")
        with c2:
            if label == "Cancelled":
                st.error(label)
            else:
                st.success(label)

        prog = int(min(max(prob * 100, 0), 100))
        st.markdown("#### Risk level")
        st.progress(prog)

    st.markdown("</div>", unsafe_allow_html=True)


with tabs[1]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("About")
    st.write(
        """
        Demo app that predicts whether a hotel booking will be cancelled.
        Upload your preprocessor and model to `/mnt/data` or use the upload widgets.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Signature (big cursive badge) ----------------
st.markdown(
    """
    <div class="footer-tag">
        <div class="footer-name">Created by Venky &amp; Subba Reddy</div>
    </div>
    """,
    unsafe_allow_html=True,
)
