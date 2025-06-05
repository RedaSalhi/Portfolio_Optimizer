# pages/2_About_Me.py

import streamlit as st
import os

st.set_page_config(page_title="About Me", layout="centered")

if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

st.title("👤 About Me")
st.caption("Financial Engineering Student | Quant Researcher")

st.markdown("""
**SALHI Reda**  
🎓 Engineering student at **Centrale Méditerranée**  
📈 Passionate about **mathematics**, **financial markets**, and **economic research**  
🌍 International backpacking experience: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, etc.   
""")

cv_en = "assets/Reda_Salhi_CV_EN.pdf"
cv_fr = "assets/Reda_Salhi_CV_FR.pdf"

if os.path.exists(cv_en):
    with open(cv_en, "rb") as f:
        st.download_button(
            label="📄 Download My CV - English Version",
            data=f,
            file_name="Reda_Salhi_CV_EN.pdf",
            mime="application/pdf"
        )

if os.path.exists(cv_fr):
    with open(cv_fr, "rb") as f:
        st.download_button(
            label="📄 Download My CV - French Version",
            data=f,
            file_name="Reda_Salhi_CV_FR.pdf",
            mime="application/pdf"
        )

st.markdown("---")
st.subheader("🔗 Links")
st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/reda-salhi-195297290/)
- [GitHub](https://github.com/RedaSalhi)
""")

st.markdown("---")
st.subheader("📬 Contact Me")
st.markdown("If you'd like to get in touch, just fill out the form below:")

formsubmit_email = "salhi.reda47@gmail.com"

form_code = f"""
<form action="https://formsubmit.co/{formsubmit_email}" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
    <input type="text" name="name" placeholder="Your Name" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"><br>
    <input type="email" name="email" placeholder="Your Email" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"><br>
    <textarea name="message" placeholder="Your Message" rows="5" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"></textarea><br>
    <button type="submit" style="background-color:#4CAF50;color:white;padding:0.75rem 1.5rem;border:none;cursor:pointer;">
        Send Message
    </button>
</form>
"""
st.markdown(form_code, unsafe_allow_html=True)
