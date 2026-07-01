import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="centered",
)

# ── Custom CSS (exact dark UI from screenshot) ────────────────────────────────
st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 540px !important;
    padding: 2.5rem 1.5rem 3rem !important;
    margin: auto;
}

/* ---------- Hero header ---------- */
.hero {
    text-align: center;
    margin-bottom: 2rem;
}
.hero-icon { font-size: 3.2rem; margin-bottom: 0.4rem; }
.hero h1 {
    font-size: 2.3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.2rem 0;
}
.hero p {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.2rem;
}

/* ---------- Divider ---------- */
.divider {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 2rem;
}
.divider hr {
    flex: 1;
    border: none;
    border-top: 1px solid #21262d;
    margin: 0;
}
.divider .dot {
    width: 8px; height: 8px;
    background: #58a6ff;
    border-radius: 50%;
}

/* ---------- Input labels ---------- */
.field-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1rem;
    font-weight: 500;
    color: #e6edf3;
    margin-bottom: 0.5rem;
    margin-top: 1.4rem;
}
.field-label span { font-size: 1.3rem; }

/* ---------- Number input box ---------- */
div[data-testid="stNumberInput"] {
    margin-bottom: 0.25rem;
}
div[data-testid="stNumberInput"] input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 1.1rem !important;
    font-weight: 400 !important;
    padding: 0.85rem 1rem !important;
    height: 3.4rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
    outline: none !important;
}
/* stepper buttons */
div[data-testid="stNumberInput"] button {
    background: #21262d !important;
    border: none !important;
    border-radius: 7px !important;
    color: #8b949e !important;
    font-size: 1.2rem !important;
    width: 2.2rem !important;
    height: 2.2rem !important;
    margin: 0 2px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: #30363d !important;
    color: #e6edf3 !important;
}

/* ---------- Predict button ---------- */
div[data-testid="stButton"] > button {
    width: 100%;
    background: transparent !important;
    border: 2px solid #58a6ff !important;
    border-radius: 10px !important;
    color: #58a6ff !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 0.9rem 1rem !important;
    margin-top: 2rem !important;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: rgba(88,166,255,0.12) !important;
    color: #79c0ff !important;
}

/* ---------- Result box ---------- */
.result-box {
    background: #d4edda;
    border-radius: 12px;
    padding: 1.4rem 1rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-box h2 {
    color: #1a7431;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}

/* ---------- Footer ---------- */
.footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 3rem;
    color: #484f58;
    font-size: 0.8rem;
}
.footer-dot { color: #484f58; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def get_model():
    # Synthetic training data that mirrors the Advertising dataset distribution
    np.random.seed(42)
    n = 200
    TV        = np.random.uniform(0.7, 296.4, n)
    Radio     = np.random.uniform(0.0, 49.6,  n)
    Newspaper = np.random.uniform(0.3, 114.0, n)
    # True relationship (from Advertising dataset regression)
    Sales = 2.939 + 0.04576*TV + 0.18853*Radio - 0.00104*Newspaper \
            + np.random.normal(0, 1.68, n)

    X = np.column_stack([TV, Radio, Newspaper])
    model = LinearRegression()
    model.fit(X, Sales)
    return model

model = get_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">📈</div>
  <h1>Sales Prediction System</h1>
  <p>Predict Product Sales using Advertising Budget</p>
</div>
<div class="divider"><hr/><div class="dot"></div><hr/></div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="field-label"><span>📺</span> TV Advertising Budget</div>', unsafe_allow_html=True)
tv = st.number_input("tv", min_value=0.0, max_value=500.0, value=150.0,
                     step=1.0, format="%.2f", label_visibility="collapsed")

st.markdown('<div class="field-label"><span>📻</span> Radio Advertising Budget</div>', unsafe_allow_html=True)
radio = st.number_input("radio", min_value=0.0, max_value=200.0, value=25.0,
                        step=1.0, format="%.2f", label_visibility="collapsed")

st.markdown('<div class="field-label"><span>📰</span> Newspaper Advertising Budget</div>', unsafe_allow_html=True)
newspaper = st.number_input("newspaper", min_value=0.0, max_value=300.0, value=20.0,
                            step=1.0, format="%.2f", label_visibility="collapsed")

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("📈  Predict Sales"):
    pred = model.predict([[tv, radio, newspaper]])[0]
    st.markdown(f"""
    <div class="result-box">
      <h2>Predicted Sales = {pred:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# # ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="footer">
#   ✦ Built with Streamlit <span class="footer-dot">|</span> Machine Learning Project
# </div>
# """, unsafe_allow_html=True)