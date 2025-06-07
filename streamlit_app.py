# streamlit_app.py

import streamlit as st

# Set config FIRST, before any other Streamlit calls
st.set_page_config(page_title="Home Page", page_icon="📈", layout="centered")

# Custom CSS to hide sidebar and native header/footer
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main-button {
            padding: 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            background-color: #4B8BBE;
            color: white;
            border-radius: 8px;
            width: 100%;
            text-align: center;
            border: none;
            transition: background-color 0.3s ease;
        }
        .main-button:hover {
            background-color: #306998;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title and Intro
st.markdown("<h1 style='text-align: center;'>Modern Portfolio Theory & Value-at-Risk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Navigate through the sections below:</p>", unsafe_allow_html=True)

# Button layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("📊 Portfolio Optimizer", use_container_width=True):
        st.switch_page("pages/1_Portfolio_Optimizer.py")

with col2:
    if st.button("👤 About Me", use_container_width=True):
        st.switch_page("pages/2_About_Me.py")

with col3:
    if st.button("📉 Value-at-Risk", use_container_width=True):
        st.switch_page("pages/3_var_app.py")
