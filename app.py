import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Try importing keras only when needed (optional)
try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ---------------- Page & Global Style ------------------
st.set_page_config(
    page_title="Room Cancellation Predictor",
    layout="centered",
    page_icon="🏨"
)

# Custom CSS for nicer UI
def inject_css():
    st.markdown(
        """
        <style>
        /* App background */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #fdfbfb 0%, #ebedee 50%, #dde1e7 100%);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #ffffff;
        }

        /* Main card */
        .main-card {
            background-color: #ffffff;
            padding: 1.5rem 1.5rem 1.2rem 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-weight: 600;
            font-size: 1.1rem;
            color: #111827;
            margin-bottom: 0.75rem;
        }

        .small-note {
            font-size: 0.85rem;
            color: #6b7280;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #4f46e5);
            color: white;
            border-radius: 999px;
            padding: 0.5rem 2.5rem;
            border: none;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8, #4338ca);
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);
        }

        /* Metrics tweak */
        [data-testid="stMetric"] {
            background: #f9fafb;
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

# ---------------- Paths ------------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"

# ---------------- Helpers ------------------
@st.cache_data
def load_preprocessor(path=PREPROC_PATH):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            pre = pickle.load(f)
        return pre
    return None

@st.cache_resource
def load_trained_model(path=MODEL_PATH):
    if os.path.exists(path) and KERAS_AVAILABLE:
        try:
            m = load_model(path)
            return m
        except Exception as e:
            st.warning(f"Found model file but couldn't load it: {e}")
            return None
    return None

# Fallback heuristic predictor (simple and explainable)
def heuristic_predict(row):
    # row is a dict of features from the form
    score = 0.0
    # common-sense rules that increase cancellation risk
    score += 0.4 * (row.get('lead_time', 0) / 365.0)        # long lead-time slightly increases risk
    score += 0.3 * (row.get('previous_cancellations', 0) > 0)
    score += 0.2 * (1 if row.get('deposit_type') == 'No Deposit' else 0)
    score += 0.1 * (1 if row.get('booking_changes', 0) > 2 else 0)
    # normalize
    prob = min(max(score, 0.0), 0.99)
    return prob

# ---------------- Sidebar ------------------
st.sidebar.title("⚙ How to use")
st.sidebar.markdown(
    """
1. Enter booking details on the main screen  
2. Click *Predict*  
3. See the probability of cancellation  

*Files for real model*  
- /mnt/data/preprocessor.pkl  
- /mnt/data/Hotel reservatiosn.h5  

If they’re missing, a simple heuristic will be used instead.
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
*Tip:*  
Try changing lead time, deposit type and previous cancellations  
to see how the risk changes.
"""
)

# ---------------- Main Title ------------------
st.markdown(
    """
    <div class="main-card">
        <h1 style="margin-bottom:0.2rem;">🏨 Room Cancellation Predictor</h1>
        <p class="small-note">
            A simple simulation of how hotel booking cancellation prediction works in the real world.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["🔮 Prediction", "ℹ About this app"])

with tabs[0]:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    st.subheader("Booking details")
    st.markdown('<p class="small-note">Fill in the customer and booking information to estimate cancellation risk.</p>', unsafe_allow_html=True)

    with st.form('predict_form'):
        # Grouped layout
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
        # Create a simple feature dict from inputs
        feature_row = {
            'lead_time': lead_time,
            'stays_weekend_nights': stays_weekend_nights,
            'stays_week_nights': stays_week_nights,
            'adults': adults,
            'children': children,
            'previous_cancellations': previous_cancellations,
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'market_segment': market_segment
        }

        # Try loading preprocessor and model
        preproc = load_preprocessor()
        model = load_trained_model()

        if preproc is not None and model is not None:
            st.success('✅ Loaded preprocessor and model. Running prediction...')
            # Convert to a single-row DataFrame (the preprocessor should expect this format)
            X = pd.DataFrame([feature_row])

            try:
                X_proc = preproc.transform(X)
                preds = model.predict(X_proc)
                # If model returns probability for positive class
                if preds.ndim == 2 and preds.shape[1] > 1:
                    prob = float(preds[0][1])
                else:
                    prob = float(preds[0])
                label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'

            except Exception as e:
                st.error(f'Error when transforming or predicting with your model: {e}')
                st.warning('Falling back to heuristic prediction instead.')
                prob = heuristic_predict(feature_row)
                label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'
        else:
            st.warning('⚠ Preprocessor or model not found in /mnt/data. Using simple heuristic predictor.')
            prob = heuristic_predict(feature_row)
            label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'

        # --- Result UI ---
        st.markdown("### Result")

        col_metric, col_label = st.columns([1, 1])

        with col_metric:
            st.metric(
                label='Probability of cancellation',
                value=f"{prob:.2%}"
            )

        with col_label:
            if label == "Cancelled":
                st.error(f"Prediction: *{label}*")
            else:
                st.success(f"Prediction: *{label}*")

        # Progress bar
        st.markdown("#### Risk level")
        st.progress(min(max(prob, 0.0), 1.0))

        # Tiny explanation
        st.markdown(
            """
            <p class="small-note">
            This score is based on the inputs above. In a real system, many more features from the booking, customer history,
            and seasonality would be used.
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
        This mini app shows how a hotel might predict whether a room booking will be cancelled:

        - ✅ If a *trained model* and *preprocessor* are available, it uses them.  
        - 🧮 Otherwise, a *simple heuristic* is used based on:
          - Lead time (how early the booking was made)  
          - Previous cancellations  
          - Deposit type  
          - Number of booking changes  

        *Important:*  
        This is not a production system – it's a learning/demo app to understand
        how such predictions can be integrated into a user interface.
        """
    )
    st.markdown(
        """
        Notes:  
        - This is not real project, this is to mimic what happens in the real world.
        """)
    st.markdown('</div>', unsafe_allow_html=True)
