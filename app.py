# app.py
import streamlit as st
import pandas as pd
import numpy as np
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

# ---------------- Page ----------------
st.set_page_config(page_title="Room Cancellation Predictor", layout="wide", page_icon="🏨")

# ---------------- CSS: clean readable UI ----------------
st.markdown(
    """
    <style>
    /* Page gradient */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg,#f0f4f8 0%, #e6eef6 40%, #f9fbfd 100%) !important;
    }

    /* Ensure default text is dark and legible */
    .stApp, .stApp * {
        color: #0f172a !important;
    }

    /* Header card */
    .hero {
        background: linear-gradient(180deg,#ffffff,#fbfdff) !important;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 8px 30px rgba(15,23,42,0.06);
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-size: 28px; }
    .hero p { margin: 6px 0 0 0; color:#475569; }

    /* Panel card */
    .panel {
        background: #ffffff !important;
        padding: 18px !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 22px rgba(15,23,42,0.05) !important;
        margin-bottom: 18px !important;
    }

    /* Input control visibility: force white background & dark text */
    input[type="text"], input[type="number"], textarea, select, .stTextInput>div>input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid #e6edf3 !important;
    }
    div[role="spinbutton"] input[type="number"], input[type="number"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        padding: 10px 12px !important;
        border-radius: 8px !important;
        border: 1px solid #e6edf3 !important;
    }
    div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 8px !important;
        border: 1px solid #e6edf3 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#2563eb,#4f46e5) !important;
        color: #fff !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 700 !important;
        border: none !important;
    }
    .stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 30px rgba(37,99,235,0.12) !important; }

    /* Metric box */
    [data-testid="stMetric"] {
        background: linear-gradient(180deg,#ffffff,#fbfdff) !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px solid #eef3f7 !important;
    }

    /* Clear code/json block style (white background) */
    .inputs-pre {
        background: #ffffff; color: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #e6edf3;
    }

    /* Signature: big cursive bold */
    .signature {
        text-align: center;
        margin-top: 28px;
    }
    .signature .name {
        font-family: 'Brush Script MT', 'Satisfy', cursive;
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        padding: 10px 26px;
        border-radius: 999px;
        background: linear-gradient(90deg,#fff,#f7fbff);
        box-shadow: 0 10px 30px rgba(15,23,42,0.06);
        display: inline-block;
    }

    /* small-note style */
    .small-note { color:#475569; margin:0; }

    /* sidebar readability */
    [data-testid="stSidebar"] { background: #ffffff !important; color: #0f172a !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Paths ----------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"

# ---------------- Load helpers ----------------
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
    if lower.endswith(".h5") and KERAS_AVAILABLE:
        try:
            return ("keras", keras_load_model(path))
        except Exception as e:
            st.warning(f"Could not load keras model: {e}")
    if JOBLIB_AVAILABLE:
        try:
            return ("sklearn", joblib.load(path))
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            return ("pickle", pickle.load(f))
    except Exception as e:
        st.warning(f"Could not load model: {e}")
    return None

# ---------------- Heuristic predictor ----------------
def heuristic_predict(row):
    score = 0.0
    score += 0.45 * (row.get("lead_time", 0) / 365.0)
    score += 0.25 * (1 if row.get("previous_cancellations", 0) > 0 else 0)
    score += 0.15 * (1 if row.get("deposit_type") == "No Deposit" else 0)
    score += 0.15 * (1 if row.get("booking_changes", 0) > 2 else 0)
    return float(min(max(score, 0.0), 0.99))

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ How to use")
st.sidebar.markdown("Enter booking details and press **Predict**.")
st.sidebar.markdown("---")
st.sidebar.write("Model file exists:", os.path.exists(MODEL_PATH))
st.sidebar.write("Preprocessor exists:", os.path.exists(PREPROC_PATH))
st.sidebar.markdown("---")
st.sidebar.markdown("Upload (optional):")
uploaded_preproc = st.sidebar.file_uploader("Upload preprocessor (.pkl)", type=["pkl"])
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.joblib/.h5)", type=["pkl", "joblib", "h5", "keras"])

# handle uploads into session state
if uploaded_preproc is not None:
    try:
        uploaded_preproc.seek(0)
        st.session_state["_uploaded_preproc"] = pickle.load(uploaded_preproc)
        st.sidebar.success("Preprocessor loaded for session")
    except Exception as e:
        st.sidebar.error(f"Preprocessor load error: {e}")

if uploaded_model is not None:
    try:
        if uploaded_model.name.lower().endswith(".h5") and KERAS_AVAILABLE:
            tmp = "/tmp/_uploaded_model.h5"
            with open(tmp, "wb") as f:
                f.write(uploaded_model.getbuffer())
            st.session_state["_uploaded_model"] = ("keras", keras_load_model(tmp))
        else:
            if JOBLIB_AVAILABLE:
                uploaded_model.seek(0)
                st.session_state["_uploaded_model"] = ("sklearn", joblib.load(uploaded_model))
            else:
                uploaded_model.seek(0)
                st.session_state["_uploaded_model"] = ("pickle", pickle.load(uploaded_model))
        st.sidebar.success("Model loaded for session")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")

# ---------------- Header ----------------
st.markdown(
    """
    <div class="hero">
      <h1>🏨 Room Cancellation Predictor</h1>
      <p class="small-note">A clean, simple interface to estimate booking cancellation risk — use a trained model or the built-in heuristic.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Layout: form on left, results on right ----------------
left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill the fields below and press Predict.</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        # clean grouped layout
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=2000, value=30)
            stays_weekend_nights = st.number_input("Weekend nights", min_value=0, max_value=30, value=0)
            stays_week_nights = st.number_input("Week nights", min_value=0, max_value=365, value=2)
        with r1c2:
            previous_cancellations = st.number_input("Previous cancellations", min_value=0, max_value=50, value=0)
            booking_changes = st.number_input("Booking changes", min_value=0, max_value=50, value=0)
            deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
        # row for guests and segment
        g1, g2 = st.columns(2)
        with g1:
            adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
            children = st.number_input("Children", min_value=0, max_value=10, value=0)
        with g2:
            market_segment = st.selectbox("Market segment", ["Direct", "Online TA", "Offline TA/TO", "Groups", "Corporate", "Complementary", "Aviation"])

        st.markdown("")  # spacing
        submit = st.form_submit_button("Predict", help="Click to run prediction")

    st.markdown("</div>", unsafe_allow_html=True)

    # Inputs summary as a clear table (not dark JSON)
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
    # present as 2-column table for clarity
    df_inputs = pd.DataFrame(list(feature_row.items()), columns=["Field", "Value"])
    st.table(df_inputs)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction")
    st.markdown('<p class="small-note">Model status and results appear here.</p>', unsafe_allow_html=True)

    # show model presence
    model_present = os.path.exists(MODEL_PATH)
    preproc_present = os.path.exists(PREPROC_PATH)
    st.write("Model file:", "✅ present" if model_present else "❌ not present")
    st.write("Preprocessor:", "✅ present" if preproc_present else "❌ not present")

    # placeholder metric & gauge (will update after prediction)
    prob_display = st.empty()
    gauge_display = st.empty()
    label_display = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction logic (runs when submit pressed) ----------------
if submit:
    # try to use uploaded session preproc/model first
    preproc = st.session_state.get("_uploaded_preproc", None) or load_preprocessor()
    model_info = st.session_state.get("_uploaded_model", None) or load_trained_model()

    prob = None
    # If both preprocessor and model present, attempt to use them safely
    if preproc is not None and model_info is not None:
        try:
            mtype, model_obj = model_info
            X = pd.DataFrame([feature_row])
            # attempt transform if available
            try:
                X_proc = preproc.transform(X)
            except Exception:
                X_proc = X  # fallback
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
            st.sidebar.error(f"Model error: {e}")
            prob = heuristic_predict(feature_row)
    else:
        prob = heuristic_predict(feature_row)

    label = "Cancelled" if prob >= 0.5 else "Not cancelled"

    # Update right column displays
    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Prediction result")

        # Metric
        st.metric(label="Cancellation probability", value=f"{prob:.2%}")

        # colored label
        if label == "Cancelled":
            st.markdown("<div style='padding:8px;border-radius:8px;background:#fff6f6;color:#9b1c1c;font-weight:700'>Prediction: Cancelled</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:8px;border-radius:8px;background:#f6fffb;color:#0b8235;font-weight:700'>Prediction: Not cancelled</div>", unsafe_allow_html=True)

        # Donut gauge via Plotly
        fig = go.Figure(
            data=[
                go.Pie(
                    values=[prob, 1 - prob],
                    labels=["Cancelled", "Not cancelled"],
                    hole=0.65,
                    marker_colors=["#ef4444", "#10b981"],
                    hoverinfo="label+percent",
                    textinfo="none",
                )
            ]
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=240,
            annotations=[
                dict(text=f"{prob:.0%}", x=0.5, y=0.5, font_size=28, showarrow=False, font=dict(color="#0f172a", family="Arial"))
            ],
        )
        gauge_display.plotly_chart(fig, use_container_width=True)

        # small explanation
        st.markdown("<p class='small-note'>This score is based on the inputs. In production you'd use richer feature sets and model validation.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer signature ----------------
st.markdown(
    """
    <div class="signature">
      <div class="name">Created by Venky &amp; Subba Reddy</div>
    </div>
    """,
    unsafe_allow_html=True,
)
