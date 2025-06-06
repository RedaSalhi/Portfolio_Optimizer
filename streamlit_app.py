# streamlit_app.py

import streamlit as st

# ✅ Set config FIRST, before any other Streamlit calls
st.set_page_config(page_title="Home Page", page_icon="📈", layout="centered")

# 🔒 Hide sidebar and default Streamlit elements
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.75em 1.5em;
            font-size: 1rem;
            border: none;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            cursor: pointer;
        }

        .centered-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.25em;
        }

        .subtitle {
            text-align: center;
            font-size: 1.25rem;
            color: #555;
            margin-bottom: 2em;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title and subtitle
st.markdown('<div class="centered-title">Modern Portfolio Theory & Value-at-Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose a section to continue</div>', unsafe_allow_html=True)

# 🔘 Navigation buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Portfolio Optimizer"):
        st.switch_page("pages/1_Portfolio_Optimizer.py")

with col2:
    if st.button("About Me"):
        st.switch_page("pages/2_About_Me.py")

with col3:
    if st.button("Value-at-Risk"):
        st.switch_page("pages/3_var_app.py")

