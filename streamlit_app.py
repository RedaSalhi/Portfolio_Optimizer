# streamlit_app.py

import streamlit as st

# Page configuration
st.set_page_config(page_title="Home Page", page_icon="📈", layout="centered")

# Hide sidebar and native headers/footers
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }

        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #1f4e79;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }

        .custom-button {
            display: inline-block;
            padding: 1rem;
            width: 100%;
            text-align: center;
            background-color: #4B8BBE;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            transition: background-color 0.3s ease, transform 0.1s ease;
        }

        .custom-button:hover {
            background-color: #306998;
            transform: scale(1.03);
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-title">Modern Portfolio Theory & Value-at-Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore tools and concepts below</div>', unsafe_allow_html=True)

# Four-column layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Portfolio Optimizer", use_container_width=True):
        st.switch_page("pages/1_Portfolio_Optimizer.py")

with col2:
    if st.button("About Me", use_container_width=True):
        st.switch_page("pages/2_About_Me.py")

with col3:
    if st.button("Value-at-Risk", use_container_width=True):
        st.switch_page("pages/3_var_app.py")

with col4:
    if st.button("Bibliography", use_container_width=True):
        st.switch_page("pages/4_Bibliography.py")

