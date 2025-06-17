# pages/2_About_Me.py

import streamlit as st
import sys
import os

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Page configuration
st.set_page_config(page_title="About Me", layout="centered", page_icon="👨‍💼")

# Enhanced CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .about-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .about-hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }
        
        .about-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        .hero-badges {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }

        .hero-badge {
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Section Cards */
        .section-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin: 2rem 0;
            box-shadow: 0 15px 35px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }

        .section-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            text-align: center;
            justify-content: center;
        }

        /* Profile Section */
        .profile-content {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            font-size: 1.1rem;
            line-height: 1.8;
            color: #2c3e50;
        }

        .profile-content strong {
            color: #667eea;
            font-weight: 600;
        }

        /* CV Download Section */
        .cv-section {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }

        .cv-description {
            color: #6c757d;
            margin-bottom: 2rem;
            text-align: center;
            font-size: 1.1rem;
        }

        /* Links Section */
        .links-container {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }

        .link-item {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            margin-bottom: 1rem;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .link-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }

        .link-item:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        }

        .link-item a {
            color: #495057;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .link-item a:hover {
            color: #667eea;
        }

        /* Contact Form */
        .contact-form-container {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }

        .contact-form {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }

        .contact-form input, .contact-form textarea {
            width: 100%;
            padding: 15px;
            margin-bottom: 20px;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 16px;
            background: white;
            transition: all 0.3s ease;
            color: #495057;
            box-sizing: border-box;
        }

        .contact-form input:focus, .contact-form textarea:focus {
            outline: none;
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.15);
        }

        .contact-form button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: none;
            letter-spacing: 0.5px;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .contact-form button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Skills Section */
        .skills-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }

        .skill-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border-top: 4px solid #667eea;
        }

        .skill-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }

        .skill-card h3 {
            color: #2c3e50;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .skill-card ul {
            list-style: none;
            padding: 0;
        }

        .skill-card li {
            padding: 0.5rem 0;
            color: #495057;
            position: relative;
            padding-left: 1.5rem;
        }

        .skill-card li::before {
            content: "→";
            color: #667eea;
            font-weight: bold;
            position: absolute;
            left: 0;
        }

        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            width: 100%;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Back Button */
        .back-button .stButton > button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            font-size: 1.1rem;
            margin-top: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .about-hero h1 { font-size: 2.5rem; }
            .about-hero p { font-size: 1.1rem; }
            .section-card { padding: 1.5rem; }
            .hero-badges { gap: 0.5rem; }
            .skills-grid { grid-template-columns: 1fr; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="about-hero">
        <h1>👨‍💼 About Me</h1>
        <p>Financial Engineering Student & Quantitative Research Enthusiast</p>
        <div class="hero-badges">
            <span class="hero-badge">Centrale Méditerranée</span>
            <span class="hero-badge">Quantitative Finance</span>
            <span class="hero-badge">Risk Management</span>
            <span class="hero-badge">Mathematical Modeling</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back Button
st.markdown('<div class="back-button">', unsafe_allow_html=True)
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")
st.markdown('</div>', unsafe_allow_html=True)

# Profile Section
st.markdown("""
    <div class="section-card">
        <div class="section-title">🎓 Profile</div>
        <div class="profile-content">
            <strong>SALHI Reda</strong><br><br>
            Engineering student at <strong>Centrale Méditerranée</strong><br>
            Passionate about <strong>mathematics</strong>, <strong>financial markets</strong>, and <strong>quantitative research</strong><br><br>
            Specializing in quantitative finance and derivatives pricing<br>
            Currently developing advanced pricing models and risk management tools<br><br>
            I also enjoy international backpacking: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, and more!
        </div>
    </div>
""", unsafe_allow_html=True)

# CV Downloads Section
st.markdown("""
    <div class="section-card">
        <div class="section-title">📄 Resume Downloads</div>
        <div class="cv-section">
            <p class="cv-description">Download my latest resume in your preferred language:</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# CV download buttons
col1, col2 = st.columns(2)

cv_en = "assets/Reda_Salhi_CV_EN.pdf"
cv_fr = "assets/Reda_Salhi_CV_FR.pdf"

with col1:
    if os.path.exists(cv_en):
        with open(cv_en, "rb") as f:
            st.download_button(
                label="📥 Download CV - English Version",
                data=f,
                file_name="Reda_Salhi_CV_EN.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("📝 English CV - Coming Soon")

with col2:
    if os.path.exists(cv_fr):
        with open(cv_fr, "rb") as f:
            st.download_button(
                label="📥 Download CV - French Version", 
                data=f,
                file_name="Reda_Salhi_CV_FR.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("📝 French CV - Coming Soon")

# Links Section
st.markdown("""
    <div class="section-card">
        <div class="section-title">🔗 Connect With Me</div>
        <div class="links-container">
            <div class="link-item">
                <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">
                    💼 LinkedIn Profile
                </a>
            </div>
            <div class="link-item">
                <a href="https://github.com/RedaSalhi" target="_blank">
                    💻 GitHub Portfolio
                </a>
            </div>
            <div class="link-item">
                <a href="mailto:salhi.reda47@gmail.com">
                    📧 salhi.reda47@gmail.com
                </a>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Contact Form Section
st.markdown("""
    <div class="section-card">
        <div class="section-title">📬 Contact Me</div>
        <div class="contact-form-container">
            <p style="text-align: center; color: #6c757d; margin-bottom: 2rem;">If you'd like to get in touch, just fill out the form below:</p>
        </div>
    </div>
""", unsafe_allow_html=True)

formsubmit_email = "salhi.reda47@gmail.com"

form_code = f"""
<div class="contact-form">
    <form action="https://formsubmit.co/{formsubmit_email}" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_template" value="table">
        <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Your Message" rows="5" required></textarea>
        <button type="submit">Send Message</button>
    </form>
</div>
"""

st.markdown(form_code, unsafe_allow_html=True)

# Skills & Interests Section
st.markdown("""
    <div class="section-card">
        <div class="section-title">🚀 Skills & Expertise</div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="skills-grid">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="skill-card">
            <h3>💻 Technical Skills</h3>
            <ul>
                <li><strong>Programming:</strong> Python, SQL, MATLAB, Excel</li>
                <li><strong>Finance:</strong> Derivatives Pricing, Risk Management, Portfolio Optimization</li>
                <li><strong>Tools:</strong> Streamlit, NumPy, Pandas</li>
                <li><strong>Mathematics:</strong> Stochastic Calculus, Statistics</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="skill-card">
            <h3>🎯 Areas of Interest</h3>
            <ul>
                <li>Quantitative Finance</li>
                <li>Financial Engineering</li>
                <li>Risk Management</li>
                <li>Economic Research</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="skill-card">
            <h3>🔬 Current Focus</h3>
            <ul>
                <li>Derivatives Pricing</li>
                <li>Monte Carlo Simulations</li>
                <li>Interest Rate Models</li>
                <li>Portfolio Optimization</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        Thank you for visiting my profile! Looking forward to connecting with you.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>QuantRisk Analytics</strong> | Reda SALHI</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Financial Engineering • Quantitative Research • Risk Management</p>
    </div>
""", unsafe_allow_html=True)