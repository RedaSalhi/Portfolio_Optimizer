# streamlit_app.py


import streamlit as st

st.set_page_config(page_title="Reda Salhi's App", layout="centered")

# Hide sidebar & header
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Init state once
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------- Navigation Logic -------- #
if st.session_state.page == "home":
    st.title("🚀 Welcome to My Streamlit App")
    st.markdown("Choose where you'd like to go:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 Portfolio Optimizer"):
            st.session_state.page = "optimizer"
            st.experimental_rerun()

    with col2:
        if st.button("👤 About Me"):
            st.session_state.page = "about"
            st.experimental_rerun()

elif st.session_state.page == "optimizer":
    import page_optimizer

elif st.session_state.page == "about":
    import page_about
