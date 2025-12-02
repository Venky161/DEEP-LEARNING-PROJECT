import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Optional imports
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


# ---------------- Streamlit Page Setup ----------------
st.set_page_config(
    page_title="Room Cancellation Predictor",
    layout="centered",
    page_icon="🏨"
)


# ---------------- FIX UI + MAKE INPUTS VISIBLE ----------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #e9f0f6 0%, #dce8ef 30%, #f6f9fb 100%) !important;
}

/* MAIN TEXT */
.stApp, .stApp * {
    color: #0f172a !important;
}

/* MAIN CARD */
.main-card {
    background: #ffffff !important;
    padding: 22px 26px !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 30px rgba(15,23,42,0.06) !important;
    margin-bottom: 22px !important;
}

.small-note { color: #4b5563 !important; }

/* TABS */
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #2563eb !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.08) !important;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#4f46e5) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 700 !important;
    border: none !important;
}

/* INPUT FIX (DARK MODE OVERRIDE) */
input[type="text"], input[type="number"], textarea, select {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Fix black background in number boxes */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown fix */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Border styling */
input, select, textarea {
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Footer */
.footer-tag {
    text-align: center;
    font-size: 15px;
    font-weight: 600;
    margin-top: 40px;
    color: #334155;
}

</style>
""", unsafe_allow_html=True)


# ---------------- FILE PATHS ----------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"


# ---------------- LOADERS ----------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading preprocessor: {e}")
    return None


@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None

    # keras model
    if path.lower().endswith(".h5") and KERAS_AVAILABLE:
        try:
            return ("keras", keras_load_model(path))
        except Exception as e:
            st.warning(f"Could not load keras model: {e}")

    # sklearn or pickle
    if JOBLIB_AVAILABLE:
        try:
            return ("sklearn", joblib.load(path))
        except:
            pass

    try:
        with open(path, "rb") as f:
            return ("pickle", pickle.load(f))
    except:
        return None


# ---------------- HEURISTIC MODEL ----------------
def heuristic_predict(row):
    score = 0
    score += 0.45 * (row["lead_time"] / 365)
    score += 0.25 * (1 if row["previous_cancellations"] > 0 else 0)
    score += 0.15 * (1 if row["deposit_type"] == "No Deposit" else 0)
    score += 0.15 * (1 if row["booking_changes"] > 2 else 0)
    return float(max(0, min(score, 0.99)))


# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ How to use")
st.sidebar.write("Enter the details → click **Predict** → get outcome.")
st.sidebar.write("Model present:", os.path.exists(MODEL_PATH))
st.sidebar.write("Preprocessor present:", os.path.exists(PREPROC_PATH))


# ---------------- MAIN HEADER ----------------
st.markdown("""
<div class="main-card">
    <h1>🏨 Room Cancellation Predictor</h1>
    <p class="small-note">Estimate the probability that a hotel room booking will be cancelled.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- TABS ----------------
tabs = st.tabs(["🔮 Prediction", "ℹ About"])


# ---------------- PREDICTION TAB ----------------
with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill these fields to generate a prediction.</p>',
                unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            lead_time = st.number_input("Lead time (days)", 0, 2000, 30)
            stays_weekend = st.number_input("Weekend nights", 0, 30, 0)
            stays_week = st.number_input("Week nights", 0, 365, 2)
            adults = st.number_input("Adults", 0, 10, 2)
            children = st.number_input("Children", 0, 10, 0)

        with col2:
            previous_c = st.number_input("Previous cancellations", 0, 50, 0)
            changes = st.number_input("Booking changes", 0, 50, 0)
            deposit = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
            segment = st.selectbox("Market segment", 
                                   ["Direct", "Online TA", "Offline TA/TO", "Groups",
                                    "Corporate", "Complementary", "Aviation"])

        submit = st.form_submit_button("Predict")

    if submit:
        # prepare row
        row = {
            "lead_time": lead_time,
            "stays_weekend_nights": stays_weekend,
            "stays_week_nights": stays_week,
            "adults": adults,
            "children": children,
            "previous_cancellations": previous_c,
            "booking_changes": changes,
            "deposit_type": deposit,
            "market_segment": segment
        }

        st.write("### Inputs")
        st.json(row)

        # load model
        pre = load_preprocessor()
        model_info = load_trained_model()

        if pre and model_info:
            try:
                mtype, model = model_info
                X = pd.DataFrame([row])
                Xp = pre.transform(X)

                if mtype == "keras":
                    preds = model.predict(Xp)
                    prob = float(preds[0][1]) if preds.ndim == 2 else float(preds[0])
                else:
                    try:
                        prob = float(model.predict_proba(Xp)[0][1])
                    except:
                        prob = float(model.predict(Xp)[0])

            except Exception as e:
                st.error(f"Model error: {e}")
                prob = heuristic_predict(row)
        else:
            prob = heuristic_predict(row)

        label = "Cancelled" if prob >= 0.5 else "Not cancelled"

        st.write("### Result")
        st.metric("Cancellation Probability", f"{prob*100:.2f}%")

        if label == "Cancelled":
            st.error(label)
        else:
            st.success(label)

        st.progress(int(prob * 100))

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- ABOUT TAB ----------------
with tabs[1]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("About this app")
    st.write("""
    This demo predicts whether a hotel booking will be cancelled.
    If a trained model is available, the app uses it.  
    Otherwise it falls back to a simple heuristic.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- SIGNATURE ----------------
st.markdown("""
<div class='footer-tag'>
    Created by <strong>Venky & Subba Reddy</strong>
</div>
""", unsafe_allow_html=True)
