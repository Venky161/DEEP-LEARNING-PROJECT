import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Optional imports for different model types
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
st.set_page_config(page_title="Room Cancellation Predictor", layout="centered", page_icon="🏨")

# ---------------- Safe, readable CSS ----------------
# This CSS aims to avoid "white-out" by enforcing readable dark text inside cards
# while keeping a pleasant page gradient. It also styles inputs/buttons consistently.
st.markdown(
    """
    <style>
    /* Page gradient */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e9f0f6 0%, #dce8ef 30%, #f6f9fb 100%) !important;
    }

    /* Make sure the main App text is dark and legible */
    .stApp, .stApp * {
        color: #0f172a !important;               /* dark slate text for legibility */
    }

    /* Main white card */
    .main-card {
        background: #ffffff !important;
        color: #0f172a !important;
        padding: 22px 26px !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06) !important;
        margin-bottom: 22px !important;
    }

    /* Titles inside cards */
    .main-card h1, .main-card h2, .main-card h3 {
        color: #0f172a !important;
    }
    .small-note { color: #4b5563 !important; } /* muted description text */

    /* Tabs visual */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #475569 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin-right: 8px !important;
    }
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

    /* Inputs - make their backgrounds white and text dark */
    input, textarea, select {
        background: #ffffff !important;
        color: #0f172a !important;
    }

    /* Number input / spin buttons style (works for most Streamlit versions) */
    div[role="spinbutton"] input[type="number"], input[type="number"] {
        background: #11182710 !important;  /* very light tint */
        color: #0f172a !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        border: 1px solid #e6edf3 !important;
    }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff,#fbfdff) !important;
        border-radius: 10px !important;
        padding: 12px !important;
        border: 1px solid #eef3f7 !important;
        color: #0f172a !important;
    }

    /* Ensure markdown headings are visible */
    .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3, h1, h2, h3 {
        color: #0f172a !important;
    }

    /* Small note style */
    .small-note { font-size: 0.95rem; color: #526270 !important; }

    /* Make the sidebar readable too */
    [data-testid="stSidebar"] { background: #ffffff !important; color: #0f172a !important; box-shadow: 0 6px 20px rgba(2,6,23,0.04) inset; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Paths ----------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"   # adjust if your model filename differs
PREPROC_PATH = "/mnt/data/preprocessor.pkl"    # adjust if your preprocessor filename differs

# ---------------- Helpers ----------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                pre = pickle.load(f)
            return pre
        except Exception as e:
            st.warning(f"Couldn't load preprocessor from {path}: {e}")
            return None
    return None

@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    lower = path.lower()
    if lower.endswith(".h5") or lower.endswith(".keras"):
        if KERAS_AVAILABLE:
            try:
                m = keras_load_model(path)
                return ("keras", m)
            except Exception as e:
                st.warning(f"Found .h5 file but couldn't load Keras model: {e}")
        else:
            st.warning("Keras/TensorFlow not available; .h5 cannot be loaded here.")
            return None
    if JOBLIB_AVAILABLE:
        try:
            m = joblib.load(path)
            return ("sklearn", m)
        except Exception:
            pass
    try:
        with open(path, 'rb') as f:
            m = pickle.load(f)
        return ("pickle", m)
    except Exception as e:
        st.warning(f"Couldn't load model from {path}: {e}")
        return None

# Fallback heuristic predictor
def heuristic_predict(row):
    score = 0.0
    score += 0.45 * (row.get('lead_time', 0) / 365.0)
    score += 0.25 * (1 if row.get('previous_cancellations', 0) > 0 else 0)
    score += 0.15 * (1 if row.get('deposit_type') == 'No Deposit' else 0)
    score += 0.15 * (1 if row.get('booking_changes', 0) > 2 else 0)
    prob = float(min(max(score, 0.0), 0.99))
    return prob

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ How to use")
st.sidebar.markdown(
    """
1. Enter booking details on the main screen  
2. Click **Predict**  
3. See the probability and label (Cancelled / Not cancelled)

(Optional) Place model files into `/mnt/data`:
- `preprocessor.pkl`  
- `Hotel reservatiosn.h5`  
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Model present:", os.path.exists(MODEL_PATH))
st.sidebar.write("Preprocessor present:", os.path.exists(PREPROC_PATH))

# Allow uploading files from browser
st.sidebar.markdown("---")
st.sidebar.markdown("**Upload (optional)**")
uploaded_preproc = st.sidebar.file_uploader("Upload preprocessor (.pkl)", type=["pkl"], key="up_preproc")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.joblib/.h5)", type=["pkl","joblib","h5","keras"], key="up_model")

if uploaded_preproc is not None:
    try:
        preproc_obj = pickle.load(uploaded_preproc)
        st.sidebar.success("Preprocessor uploaded (session).")
        st.session_state["_uploaded_preproc"] = preproc_obj
    except Exception as e:
        st.sidebar.error(f"Could not load preprocessor: {e}")

if uploaded_model is not None:
    try:
        # handle keras h5 specially by writing to tmp if needed
        if uploaded_model.name.lower().endswith(".h5") and KERAS_AVAILABLE:
            tmp = "/tmp/_uploaded_model.h5"
            with open(tmp, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_loaded = ("keras", keras_load_model(tmp))
        else:
            try:
                # try joblib if available
                loaded = joblib.load(uploaded_model) if JOBLIB_AVAILABLE else None
                if loaded is not None:
                    model_loaded = ("sklearn", loaded)
                else:
                    uploaded_model.seek(0)
                    model_loaded = ("pickle", pickle.load(uploaded_model))
            except Exception:
                uploaded_model.seek(0)
                model_loaded = ("pickle", pickle.load(uploaded_model))
        st.session_state["_uploaded_model"] = model_loaded
        st.sidebar.success("Model uploaded (session).")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")

# ---------------- Main UI ----------------
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:6px;">🏨 Room Cancellation Predictor</h1>
        <p class="small-note">A clear, simple UI to estimate whether a booking may be cancelled. Use it for demo or connect your real model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["🔮 Prediction", "ℹ About"])

with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill the booking details below and press Predict.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Stay")
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=2000, value=30)
            stays_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=30, value=0)
            stays_week_nights = st.number_input("Week nights", min_value=0, max_value=365, value=2)

            st.markdown("#### Guests")
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)

        with col2:
            st.markdown("#### History")
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0)
            booking_changes = st.number_input("Booking changes", min_value=0, max_value=50, value=0)

            st.markdown("#### Commercial")
            deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
            market_segment = st.selectbox("Market segment", ["Direct", "Online TA", "Offline TA/TO", "Groups", "Corporate", "Complementary", "Aviation"])

        submitted = st.form_submit_button("Predict")

    if submitted:
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

        st.markdown("### Inputs")
        st.json(feature_row)

        # load preproc / model (uploaded session-state preferred)
        preproc = st.session_state.get("_uploaded_preproc", None) or load_preprocessor()
        model_info = st.session_state.get("_uploaded_model", None) or load_trained_model()

        if preproc is not None and model_info is not None:
            try:
                mtype, model_obj = model_info
                st.success("✅ Loaded preprocessor & model; running prediction.")
                X = pd.DataFrame([feature_row])
                try:
                    X_proc = preproc.transform(X)
                except Exception:
                    X_proc = preproc.transform(X)

                if mtype == "keras":
                    preds = model_obj.predict(X_proc)
                    prob = float(preds[0][1]) if (preds.ndim == 2 and preds.shape[1] > 1) else float(preds[0])
                else:
                    try:
                        prob = float(model_obj.predict_proba(X_proc)[0][1])
                    except Exception:
                        pred = model_obj.predict(X_proc)[0]
                        prob = 1.0 if pred == 1 else 0.0
                label = "Cancelled" if prob >= 0.5 else "Not cancelled"
            except Exception as e:
                st.error(f"Model error: {e}")
                st.warning("Falling back to heuristic.")
                prob = heuristic_predict(feature_row)
                label = "Cancelled" if prob >= 0.5 else "Not cancelled"
        else:
            st.warning("No preprocessor/model available; using heuristic predictor.")
            prob = heuristic_predict(feature_row)
            label = "Cancelled" if prob >= 0.5 else "Not cancelled"

        # Results
        st.markdown("### Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Probability of cancellation", f"{prob:.2%}")
        with c2:
            if label == "Cancelled":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

        # Progress bar (0-100)
        prog = int(min(max(prob * 100, 0), 100))
        st.markdown("#### Risk level")
        st.progress(prog)

       st.markdown("<p class='small-note'>This is a demo. In production you'd include many more features and validation.</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("About")
    st.write(
        """
        Demo app for hotel booking cancellation prediction.
        If you want a branded/polished theme I can:
        - Add hotel icon + header image
        - Replace metric with a donut/gauge visualization
        - Convert the UI to a darker theme or compact dashboard
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

