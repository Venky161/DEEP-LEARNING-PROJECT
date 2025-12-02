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

# --------- WORKING THEME CSS (Streamlit Safe) ----------
st.markdown("""
<style>

html, body {
    background-color: #eef2f7 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #eef2f7 !important;
    background-image: linear-gradient(135deg, #eef2f7 0%, #d9e2ec 100%);
    padding-top: 18px;
    padding-bottom: 18px;
}

.main-card {
    background: white;
    padding: 20px 25px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.5rem;
}

.small-note {
    font-size: 0.9rem;
    color: #6b7280;
}

.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #2563eb);
    color: white;
    font-weight: 600;
    padding: 0.6rem 2rem;
    border-radius: 10px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #4338ca, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37,99,235,0.3);
}

.stTabs [data-baseweb="tab"] {
    background-color: #f1f5f9 !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #2563eb !important;
    font-weight: 600 !important;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

[data-testid="stMetric"] {
    background: #f9fafb;
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

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
    """
    Attempts to load a model:
    - If file extension is .h5 and TensorFlow available -> Keras model
    - Else try joblib (sklearn) if available
    - Else try pickle
    Returns loaded model or None.
    """
    if not os.path.exists(path):
        return None

    lower = path.lower()
    # Keras .h5
    if lower.endswith(".h5") or lower.endswith(".keras"):
        if KERAS_AVAILABLE:
            try:
                m = keras_load_model(path)
                return ("keras", m)
            except Exception as e:
                st.warning(f"Found .h5 file but couldn't load Keras model: {e}")
        else:
            st.warning("Keras/TensorFlow not available in environment; cannot load .h5 model.")
            return None

    # Try joblib
    if JOBLIB_AVAILABLE:
        try:
            m = joblib.load(path)
            return ("sklearn", m)
        except Exception:
            pass

    # Fallback to pickle
    try:
        with open(path, 'rb') as f:
            m = pickle.load(f)
        # we can't reliably identify type, but return
        return ("pickle", m)
    except Exception as e:
        st.warning(f"Couldn't load model from {path}: {e}")
        return None

# Fallback heuristic predictor (simple and explainable)
def heuristic_predict(row):
    # row is a dict of features from the form
    score = 0.0
    # common-sense rules that increase cancellation risk
    score += 0.45 * (row.get('lead_time', 0) / 365.0)        # long lead-time increases risk
    score += 0.25 * (1 if row.get('previous_cancellations', 0) > 0 else 0)
    score += 0.15 * (1 if row.get('deposit_type') == 'No Deposit' else 0)
    score += 0.15 * (1 if row.get('booking_changes', 0) > 2 else 0)
    # normalize and clamp
    prob = float(min(max(score, 0.0), 0.99))
    return prob

# ---------------- Sidebar ----------------
st.sidebar.title("⚙ How to use")
st.sidebar.markdown(
    """
1. Enter booking details on the main screen  
2. Click **Predict**  
3. See the probability and label (Cancelled / Not cancelled)

**Files (optional)**  
Place these files in `/mnt/data` if you want the real model to be used:  
- `preprocessor.pkl`  
- `Hotel reservatiosn.h5`  
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Model present: ", os.path.exists(MODEL_PATH))
st.sidebar.write("Preprocessor present: ", os.path.exists(PREPROC_PATH))

# Optionally allow upload from UI (useful if you can't put files into /mnt/data)
st.sidebar.markdown("---")
st.sidebar.markdown("**(Optional)** Upload model & preprocessor here")
uploaded_preproc = st.sidebar.file_uploader("Upload preprocessor (pkl)", type=["pkl"], key="up_preproc")
uploaded_model = st.sidebar.file_uploader("Upload model (joblib/pkl/.h5)", type=["pkl","joblib","h5","keras"], key="up_model")

if uploaded_preproc is not None:
    try:
        preproc_obj = pickle.load(uploaded_preproc)
        st.sidebar.success("Preprocessor uploaded (will be used this session).")
        # override loader by caching the uploaded preprocessor in session_state
        st.session_state["_uploaded_preproc"] = preproc_obj
    except Exception as e:
        st.sidebar.error(f"Could not load uploaded preprocessor: {e}")

if uploaded_model is not None:
    # save uploaded model temporarily to a session-like place
    try:
        # For simplicity, load into memory rather than saving to /mnt/data
        if uploaded_model.name.lower().endswith(".h5") and KERAS_AVAILABLE:
            # For Keras .h5 we need to save the uploaded bytes to a temp file to load with keras
            tmp_path = "/tmp/_uploaded_model.h5"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_loaded = ("keras", keras_load_model(tmp_path))
        else:
            # try joblib/pickle
            try:
                model_loaded = ("sklearn_or_pickle", joblib.load(uploaded_model)) if JOBLIB_AVAILABLE else None
            except Exception:
                try:
                    uploaded_model.seek(0)
                    model_loaded = ("pickle", pickle.load(uploaded_model))
                except Exception as e:
                    model_loaded = None
        if model_loaded is not None:
            st.session_state["_uploaded_model"] = model_loaded
            st.sidebar.success("Model uploaded (will be used this session).")
        else:
            st.sidebar.error("Uploaded model could not be loaded.")
    except Exception as e:
        st.sidebar.error(f"Error loading uploaded model: {e}")

# ---------------- Main UI ----------------
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:0.2rem;">🏨 Room Cancellation Predictor</h1>
        <p class="small-note">
            A simple simulation of booking cancellation prediction. Use the form below to estimate risk.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["🔮 Prediction", "ℹ About this app"])

with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill in the booking information to estimate cancellation risk.</p>', unsafe_allow_html=True)

    with st.form('predict_form'):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Stay details")
            lead_time = st.number_input('Lead time (days between booking and arrival)', min_value=0, max_value=2000, value=30)
            stays_weekend_nights = st.number_input('Weekend nights', min_value=0, max_value=30, value=0)
            stays_week_nights = st.number_input('Week nights', min_value=0, max_value=365, value=2)

            st.markdown("#### Guests")
            adults = st.number_input('Adults', min_value=0, max_value=10, value=2)
            children = st.number_input('Children', min_value=0, max_value=10, value=0)

        with col_right:
            st.markdown("#### Booking history")
            previous_cancellations = st.number_input('Previous cancellations', min_value=0, max_value=20, value=0)
            booking_changes = st.number_input('Booking changes', min_value=0, max_value=20, value=0)

            st.markdown("#### Commercial info")
            deposit_type = st.selectbox('Deposit type', ['No Deposit', 'Refundable', 'Non Refund'])
            market_segment = st.selectbox(
                'Market segment',
                ['Direct', 'Online TA', 'Offline TA/TO', 'Groups', 'Corporate', 'Complementary', 'Aviation']
            )

        st.markdown("")
        centered = st.columns([1, 1, 1])
        with centered[1]:
            submitted = st.form_submit_button('Predict')

    if submitted:
        # Prepare feature dict
        feature_row = {
            'lead_time': int(lead_time),
            'stays_weekend_nights': int(stays_weekend_nights),
            'stays_week_nights': int(stays_week_nights),
            'adults': int(adults),
            'children': int(children),
            'previous_cancellations': int(previous_cancellations),
            'booking_changes': int(booking_changes),
            'deposit_type': deposit_type,
            'market_segment': market_segment
        }

        # Display inputs summary
        st.markdown("### Inputs")
        st.json(feature_row)

        # Load preprocessor & model (from session uploaded first, then /mnt/data)
        preproc = None
        model_info = None

        if "_uploaded_preproc" in st.session_state:
            preproc = st.session_state["_uploaded_preproc"]
        else:
            preproc = load_preprocessor()

        if "_uploaded_model" in st.session_state:
            model_info = st.session_state["_uploaded_model"]
        else:
            model_info = load_trained_model()

        prob = None
        label = None

        if preproc is not None and model_info is not None:
            # model_info is a tuple like ("keras", model) or ("sklearn", model)
            try:
                mtype, model_obj = model_info
                st.success('✅ Loaded preprocessor and model. Running prediction...')
                X = pd.DataFrame([feature_row])

                # Attempt transform (some preprocessors expect exact columns)
                try:
                    X_proc = preproc.transform(X)
                except Exception:
                    # If transformer expects array-like or different fmt, try passing df directly
                    X_proc = preproc.transform(X)

                # Keras model
                if mtype == "keras":
                    preds = model_obj.predict(X_proc)
                    # assume classifier gives prob for positive class in column 1 or returns single prob
                    if preds.ndim == 2 and preds.shape[1] > 1:
                        prob = float(preds[0][1])
                    else:
                        prob = float(preds[0])
                else:
                    # sklearn / pickle models - try predict_proba then predict
                    try:
                        prob = float(model_obj.predict_proba(X_proc)[0][1])
                    except Exception:
                        # fallback to predict (0/1)
                        pred = model_obj.predict(X_proc)[0]
                        prob = 1.0 if pred == 1 else 0.0

                label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'
            except Exception as e:
                st.error(f'Error while using your model: {e}')
                st.warning('Falling back to heuristic predictor.')
                prob = heuristic_predict(feature_row)
                label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'
        else:
            st.warning('⚠ Preprocessor or model not found. Using heuristic predictor.')
            prob = heuristic_predict(feature_row)
            label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'

        # --- Result UI ---
        st.markdown("### Result")
        col_metric, col_label = st.columns([1, 1])
        with col_metric:
            st.metric(label='Probability of cancellation', value=f"{prob:.2%}")

        with col_label:
            if label == "Cancelled":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

        st.markdown("#### Risk level")
        # st.progress expects 0-100 for int; convert
        try:
            prog_val = int(min(max(prob * 100, 0), 100))
        except Exception:
            prog_val = 0
        st.progress(prog_val)

        st.markdown(
            """
            <p class="small-note">
            This score is based on the inputs above. In a production system you'd use many more features
            (seasonality, customer history, channel signals, payment info, promotions, etc.).
            </p>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("About this app")
    st.write(
        """
        This mini app demonstrates how a hotel might predict whether a booking will be cancelled.

        - If a trained model and preprocessor are available (either uploaded or present in `/mnt/data`),
          the app will use them.  
        - Otherwise, the app uses a simple, transparent heuristic so you can interact with the UI.

        Notes:
        - Ensure form field names match the columns your preprocessor expects.
        - If your model is sklearn: save it with `joblib.dump(model, 'model.pkl')`.
        - If your model is Keras: save as `.h5`.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
