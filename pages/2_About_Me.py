# pages/2_About_Me.py

import streamlit as st
import os

# Page configuration
st.set_page_config(page_title="About Me", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }

        .title {
            text-align: center;
            font-size: 2.3rem;
            color: #1f4e79;
            font-weight: 700;
            margin-top: 1rem;
        }

        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
        }

        .section {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.07);
            margin-bottom: 1.5rem;
        }

        .contact-button {
            background-color: #4CAF50;
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }

        .contact-button:hover {
            background-color: #388e3c;
        }

        .cv-button {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Back button
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")

# Title and subtitle
st.markdown('<div class="title">About Me</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Financial Engineering Student | Quant Researcher</div>', unsafe_allow_html=True)

# Profile section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("""
    **SALHI Reda**  
    - Engineering student at **Centrale Méditerranée**  
    - Passionate about **mathematics**, **financial markets**, and **economic research**.  
    - I also enjoy international backpacking: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, etc.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# CV Downloads
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📄 Curriculum Vitae")
cv_en = "assets/Reda_Salhi_CV_EN.pdf"
cv_fr = "assets/Reda_Salhi_CV_FR.pdf"

if os.path.exists(cv_en):
    with open(cv_en, "rb") as f:
        st.download_button("Download My CV - English Version", f, "Reda_Salhi_CV_EN.pdf", mime="application/pdf", key="cv_en")

if os.path.exists(cv_fr):
    with open(cv_fr, "rb") as f:
        st.download_button("Download My CV - French Version", f, "Reda_Salhi_CV_FR.pdf", mime="application/pdf", key="cv_fr")

st.markdown('</div>', unsafe_allow_html=True)

# Links section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🔗 Online Profiles")
st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/reda-salhi-195297290/)
- [GitHub](https://github.com/RedaSalhi)
""")
st.markdown('</div>', unsafe_allow_html=True)

# Contact form
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("📬 Contact Me")
st.markdown("If you'd like to get in touch, just fill out the form below:")

formsubmit_email = "salhi.reda47@gmail.com"
form_code = f"""
<form action="https://formsubmit.co/{formsubmit_email}" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
    <input type="text" name="name" placeholder="Your Name" required style="width:100%;padding:0.6rem;margin-bottom:0.8rem;border-radius:5px;border:1px solid #ccc;"><br>
    <input type="email" name="email" placeholder="Your Email" required style="width:100%;padding:0.6rem;margin-bottom:0.8rem;border-radius:5px;border:1px solid #ccc;"><br>
    <textarea name="message" placeholder="Your Message" rows="5" required style="width:100%;padding:0.6rem;margin-bottom:0.8rem;border-radius:5px;border:1px solid #ccc;"></textarea><br>
    <button type="submit" class="contact-button">Send Message</button>
</form>
"""
st.markdown(form_code, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
