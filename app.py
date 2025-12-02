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


# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Room Cancellation Predictor",
                   layout="wide",
                   page_icon="🏨")


# ------------------ CSS THEME (UI FIXED) ------------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg,#eef4f9 0%, #e6eef6 40%, #f9fbfd 100%) !important;
}

.stApp, .stApp * {
    color: #0f172a !important;
}

/* Main panel */
.panel {
    background: #ffffff !important;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 22px rgba(15,23,42,0.05);
    margin-bottom: 18px;
}

.hero {
    background: #ffffff;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 32px;
}
.small-note {
    color: #475569 !important;
}

/* ---------------- INPUT FIXES ---------------- */

/* All input boxes white */
input[type="text"], input[type="number"], textarea, select,
.stTextInput>div>input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid #d1d9e6 !important;
}

/* Fix for number input spinbutton container */
div[role="spinbutton"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #d1d9e6 !important;
}

/* Fix for + and - buttons */
div[role="spinbutton"] button {
    background-color: #e9eef5 !important;
    color: #0f172a !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
}
div[role="spinbutton"] button:hover {
    background-color: #d8e0ea !important;
}

/* Fix number text inside input */
div[role="spinbutton"] input[type="number"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 6px !important;
    padding-left: 10px !important;
}

/* Dropdown visible */
div[data-baseweb="select"] > div, .stSelectbox>div>div>div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    border: 1px solid #d1d9e6 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#2563eb,#4f46e5) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    border-radius: 10px !important;
    border: none !important;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(37,99,235,0.15);
}

/* Metric */
[data-testid="stMetric"] {
    background: #ffffff;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #eef3f7;
}

/* Table */
table {
    background: #ffffff !important;
    color: #0f172a !important;
}

/* Signature */
.signature {
    text-align: center;
    margin-top: 35px;
}
.signature .name {
    font-family: 'Brush Script MT','Satisfy',cursive;
    font-size: 36px;
    font-weight: 800;
    padding: 10px 30px;
    border-radius: 999px;
    color: #0f172a;
    background: linear-gradient(90deg,#ffffff,#f7faff);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)


# ------------------ PATHS ------------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"


# ------------------ LOADERS ------------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except:
            st.warning("Preprocessor failed to load.")
    return None


@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None

    if path.lower().endswith(".h5") and KERAS_AVAILABLE:
        try:
            return ("keras", keras_load_model(path))
        except:
            pass

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


# ------------------ HEURISTIC ------------------
def heuristic_predict(row):
    score = 0.45*(row["lead_time"]/365) + \
            0.25*(1 if row["previous_cancellations"]>0 else 0) + \
            0.15*(1 if row["deposit_type"]=="No Deposit" else 0) + \
            0.15*(1 if row["booking_changes"]>2 else 0)
    return min(max(score,0),0.99)


# ------------------ HEADER ------------------
st.markdown("""
<div class="hero">
    <h1>🏨 Room Cancellation Predictor</h1>
    <p class="small-note">Simple, clean interface to estimate booking cancellation risk.</p>
</div>
""", unsafe_allow_html=True)


# ------------------ LAYOUT ------------------
col_left, col_right = st.columns([2, 1])

# ------------------ LEFT: FORM ------------------
with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Booking details")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            lead_time = st.number_input("Lead time (days)", 0, 2000, 30)
            weekend = st.number_input("Weekend nights", 0, 30, 0)
            week = st.number_input("Week nights", 0, 365, 2)
        with c2:
            prev_cancel = st.number_input("Previous cancellations", 0, 50, 0)
            changes = st.number_input("Booking changes", 0, 50, 0)
            deposit = st.selectbox("Deposit type", ["No Deposit","Refundable","Non Refund"])

        g1, g2 = st.columns(2)
        with g1:
            adults = st.number_input("Adults", 0, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
        with g2:
            segment = st.selectbox("Market segment",
                                   ["Direct","Online TA","Offline TA/TO","Groups","Corporate","Complementary","Aviation"])

        submit = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ RIGHT: RESULT PANEL ------------------
with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction")

    if submit:
        row = {
            "lead_time": int(lead_time),
            "stays_weekend_nights": int(weekend),
            "stays_week_nights": int(week),
            "adults": int(adults),
            "children": int(children),
            "previous_cancellations": int(prev_cancel),
            "booking_changes": int(changes),
            "deposit_type": deposit,
            "market_segment": segment,
        }

        # Load model or fallback
        pre = load_preprocessor()
        model_info = load_trained_model()

        if pre and model_info:
            try:
                mtype, model = model_info
                X = pd.DataFrame([row])
                try:
                    Xp = pre.transform(X)
                except:
                    Xp = X
                if mtype == "keras":
                    pred = model.predict(Xp)
                    prob = float(pred[0][1] if pred.ndim == 2 else pred[0])
                else:
                    try:
                        prob = float(model.predict_proba(Xp)[0][1])
                    except:
                        prob = float(model.predict(Xp)[0])
            except:
                prob = heuristic_predict(row)
        else:
            prob = heuristic_predict(row)

        label = "Cancelled" if prob >= 0.5 else "Not cancelled"

        # Metric
        st.metric("Cancellation probability", f"{prob:.2%}")

        # Donut gauge
        fig = go.Figure(go.Pie(
            values=[prob, 1-prob],
            hole=0.65,
            marker_colors=["#ef4444", "#10b981"],
            hoverinfo="label+percent",
            textinfo="none"
        ))
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                          height=250,
                          annotations=[dict(text=f"{prob:.0%}", x=0.5,y=0.5,
                                            font_size=28, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

        # Result label
        if label == "Cancelled":
            st.error("Prediction: CANCELLED")
        else:
            st.success("Prediction: NOT CANCELLED")

    else:
        st.info("Fill the form and click Predict.")

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ SIGNATURE ------------------
st.markdown("""
<div class="signature">
    <div class="name">Created by Venky &amp; Subba Reddy</div>
</div>
""", unsafe_allow_html=True)
