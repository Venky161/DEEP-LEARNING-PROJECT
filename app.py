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

[data-]()
