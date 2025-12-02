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

st.set_page_config(page_title="Room Cancellation Predictor", layout="centered", page_icon="🏨")

# Simple, easy-to-understand Streamlit app to predict whether a hotel room booking will be cancelled.
# Behavior:
# - If a preprocessor (preprocessor.pkl) and a model (Hotel reservatiosn.h5) exist in /mnt/data, the app will load them and use them.
# - If not present, the app runs a clear, deterministic fallback heuristic so you can still interact with the UI.
# How to run:
# 1. Put your preprocessor (pickle) at /mnt/data/preprocessor.pkl and your trained model at /mnt/data/Hotel reservatiosn.h5
# 2. Run: streamlit run app.py

# ---------------- Helpers ------------------
MODEL_PATH = "/mnt/data/Hotel reservatiosn.h5"
PREPROC_PATH = "/mnt/data/preprocessor.pkl"

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

# ---------------- UI ------------------
st.title("Room Cancellation Predictor")
st.write("Use this simple interface to predict whether a customer's booking will be cancelled. Fill the form and press **Predict**.")

st.markdown("---")

with st.form('predict_form'):
    st.subheader('Booking information (fill these)')
    col1, col2 = st.columns(2)
    with col1:
        lead_time = st.number_input('Lead time (days between booking and arrival)', min_value=0, max_value=2000, value=30)
        stays_weekend_nights = st.number_input('Weekend nights', min_value=0, max_value=30, value=0)
        stays_week_nights = st.number_input('Week nights', min_value=0, max_value=365, value=2)
        adults = st.number_input('Adults', min_value=0, max_value=10, value=2)
        children = st.number_input('Children', min_value=0, max_value=10, value=0)
    with col2:
        previous_cancellations = st.number_input('Previous cancellations', min_value=0, max_value=20, value=0)
        booking_changes = st.number_input('Booking changes', min_value=0, max_value=20, value=0)
        deposit_type = st.selectbox('Deposit type', ['No Deposit', 'Refundable', 'Non Refund'])
        market_segment = st.selectbox('Market segment', ['Direct', 'Online TA', 'Offline TA/TO', 'Groups', 'Corporate', 'Complementary', 'Aviation'])

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

    st.write('### Inputs summary')
    st.json(feature_row)

    # Try loading preprocessor and model
    preproc = load_preprocessor()
    model = load_trained_model()

    if preproc is not None and model is not None:
        st.success('Loaded preprocessor and model. Running model prediction...')
        # Convert to a single-row DataFrame (the preprocessor should expect this format)
        X = pd.DataFrame([feature_row])

        # NOTE: Real preprocessor may expect specific column names / dtypes.
        # If your preprocessor was fit on many categorical columns, ensure they exist here.
        try:
            X_proc = preproc.transform(X)
            preds = model.predict(X_proc)
            # If model returns probability for positive class
            if preds.ndim == 2 and preds.shape[1] > 1:
                prob = float(preds[0][1])
            else:
                prob = float(preds[0])
            label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'

            st.write('### Prediction')
            st.metric(label='Probability of cancellation', value=f"{prob:.2f}", delta=None)
            st.info(f'Predicted label: **{label}**')

        except Exception as e:
            st.error(f'Error when transforming or predicting with your model: {e}')
            st.warning('Falling back to heuristic prediction instead.')
            prob = heuristic_predict(feature_row)
            label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'
            st.metric(label='Heuristic probability of cancellation', value=f"{prob:.2f}")
            st.info(f'Heuristic label: **{label}**')

    else:
        st.warning('Preprocessor or model not found in /mnt/data. Using simple heuristic predictor.')
        prob = heuristic_predict(feature_row)
        label = 'Cancelled' if prob >= 0.5 else 'Not cancelled'
        st.metric(label='Heuristic probability of cancellation', value=f"{prob:.2f}")
        st.info(f'Heuristic label: **{label}**')

st.markdown('---')
st.write("""
**Notes:**
- To use your real model, upload `preprocessor.pkl` and `Hotel reservatiosn.h5` to `/mnt/data`.
- Make sure your feature names in the form match the training data columns.
- If your model is not Keras `.h5`, adjust the load function accordingly.
""")



