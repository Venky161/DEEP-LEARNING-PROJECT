import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import io

# --------------------------------------------------
# Impressive Streamlit App Template
# Single-file app. Drop this into app.py and run:
# streamlit run app.py
# --------------------------------------------------

st.set_page_config(page_title="InsightDash — Impressive UI", layout="wide", page_icon="🚀")

# ---------- Styles (minimal custom CSS for polish) ----------
st.markdown("""
<style>
/* page background */
.stApp {
  background: linear-gradient(180deg, #f8fafc 0%, #ffffff 60%);
  color: #0f172a;
  font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* card look */
.card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(15,23,42,0.06);
  padding: 18px;
}

.hero-title {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 6px;
}

.hero-sub {
  color: #475569;
  margin-top: 0;
  margin-bottom: 12px;
}

.small-muted { color: #94a3b8; font-size: 13px; }

.footer { color: #64748b; font-size: 13px; }

</style>
""", unsafe_allow_html=True)

# ---------- Helper utilities ----------
@st.cache_data
def load_sample_data(rows=200):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "customer_id": np.arange(1, rows+1),
        "age": rng.integers(18, 70, rows),
        "signup_days_ago": rng.integers(1, 1200, rows),
        "spend": np.round(rng.normal(120, 60, rows).clip(5), 2),
        "churn_prob": np.round(rng.random(rows), 2)
    })
    return df

# ---------- Sidebar (controls) ----------
with st.sidebar.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/frontend/static/media/brand/streamlit-mark-color.svg", width=64)
    st.markdown("""
    <h3 style='margin:6px 0 0 0'>InsightDash</h3>
    <p class='small-muted'>Beautiful analytics & customer-facing dashboards</p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    dataset_choice = st.selectbox("Choose action", ["Use sample data", "Upload CSV", "Connect to DB (placeholder)"])
    st.markdown("\n")
    show_kpis = st.checkbox("Show KPIs", value=True)
    show_charts = st.checkbox("Show charts", value=True)
    st.markdown("---")
    st.markdown("<p class='small-muted'>Theme</p>", unsafe_allow_html=True)
    theme = st.radio("Color theme", ["Soft", "Dark (beta)", "Corporate"], index=0)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main layout ----------
header_col1, header_col2 = st.columns([3,1])
with header_col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Deliver delightful dashboards to customers</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Clean, minimal and conversion-focused interface — ready to plug into your ML model or analytics pipeline.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown("<div class='card' style='text-align:center'>", unsafe_allow_html=True)
    st.metric(label="Live Users", value="1,248", delta="+8%")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("\n")

# ---------- Data loading ----------
if dataset_choice == "Use sample data":
    df = load_sample_data(300)
    st.success("Loaded sample dataset (300 rows)")

elif dataset_choice == "Upload CSV":
    uploaded = st.file_uploader("Upload a CSV file", type=["csv", "txt"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            st.error("Could not read the CSV file. Check format.")
            df = None
    else:
        st.info("No file uploaded yet — using sample data preview")
        df = load_sample_data(60)

else:
    st.info("DB connectors are placeholders in this template — drop your connection code here.")
    df = load_sample_data(120)

# ---------- Top KPI cards ----------
if show_kpis and df is not None:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total customers", value=f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with k2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Avg. age", value=f"{df['age'].mean():.1f}" if 'age' in df.columns else "—")
        st.markdown("</div>", unsafe_allow_html=True)
    with k3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Avg. spend", value=f"₹{df['spend'].mean():.2f}" if 'spend' in df.columns else "—")
        st.markdown("</div>", unsafe_allow_html=True)
    with k4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Avg. churn", value=f"{df['churn_prob'].mean():.2f}" if 'churn_prob' in df.columns else "—")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("\n")

# ---------- Main content: data preview and charts ----------
main_left, main_right = st.columns([2,1])
with main_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Data preview")
    if df is not None:
        st.dataframe(df.head(10))

        # Quick filters
        with st.expander("Quick filters"):
            cols = df.select_dtypes(include=["number"]).columns.tolist()
            chosen = st.selectbox("Filter numeric column", [None] + cols)
            if chosen:
                rng = st.slider("Filter range", float(df[chosen].min()), float(df[chosen].max()), (float(df[chosen].min()), float(df[chosen].max())))
                df = df[df[chosen].between(rng[0], rng[1])]
                st.write(f"Filtered to {len(df)} rows")

    else:
        st.info("No data to preview")
    st.markdown("</div>", unsafe_allow_html=True)

with main_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Actions")
    st.write("Use the quick action buttons below to export or visualize the dataset.")
    if st.button("Download sample CSV"):
        tmp = load_sample_data(50)
        csv = tmp.to_csv(index=False).encode('utf-8')
        st.download_button("Click to download", data=csv, file_name="sample_data.csv", mime='text/csv')

    if st.button("Show simple report"):
        st.balloons()
        st.success("Report generated — use the charts panel to the left")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Charts ----------
if show_charts and df is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Charts & insights")
    col1, col2 = st.columns(2)
    with col1:
        # Histogram
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            choice = st.selectbox("Choose numeric for histogram", num_cols, index=0)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(df[choice].dropna(), bins=30)
            ax.set_title(f"Distribution of {choice}")
            st.pyplot(fig)
    with col2:
        if 'age' in df.columns and 'spend' in df.columns:
            fig = px.scatter(df, x='age', y='spend', size='spend', hover_data=['customer_id'] if 'customer_id' in df.columns else None, title='Age vs Spend')
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Predict / Action area (placeholder for ML) ----------
st.markdown("\n")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Customer action center")
with st.form(key='action_form'):
    colA, colB, colC = st.columns(3)
    with colA:
        cid = st.text_input("Customer ID")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    with colB:
        recent_days = st.number_input("Days since signup", min_value=0, max_value=5000, value=120)
        spend = st.number_input("Avg spend", min_value=0.0, format="%.2f", value=100.0)
    with colC:
        engage_score = st.slider("Engagement (0-100)", 0, 100, 50)

    submitted = st.form_submit_button("Suggest action")
    if submitted:
        # Placeholder logic — replace with real model prediction
        score = (100 - engage_score) * 0.01 + (spend / (spend + 100)) * 0.4 + (1 if recent_days < 90 else 0) * 0.2
        recommended = "Send discount & email campaign" if score > 0.5 else "Standard nurture sequence"
        st.success(f"Recommended action: {recommended}")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("\n")
footer_col1, footer_col2 = st.columns([3,1])
with footer_col1:
    st.markdown("<p class='footer'>Built with ❤️ — customize this template for your product, plug in your models or analytics, and ship delightful dashboards to customers.</p>", unsafe_allow_html=True)
with footer_col2:
    st.markdown("<div style='text-align:right'><small class='small-muted'>v1.0 • InsightDash</small></div>", unsafe_allow_html=True)

# ----------------- End -----------------
