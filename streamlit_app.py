# streamlit_app.py

import streamlit as st
import time

# Hide sidebar completely
st.set_page_config(page_title="Reda Salhi's App", layout="centered")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Welcome to My Streamlit App")
st.markdown("Choose a section to continue:")

col1, col2 = st.columns(2)

with col1:
    if st.button("📈 Portfolio Optimizer"):
        st.switch_page("pages/1_Portfolio_Optimizer.py")

with col2:
    if st.button("👤 About Me"):
        st.switch_page("pages/2_About_Me.py")

