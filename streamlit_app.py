# streamlit_app.py

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="QuantRisk Analytics", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }
        
        /* Hero Section */
        .hero-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 3rem;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            font-weight: 300;
            opacity: 0.9;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }
        
        .hero-description {
            font-size: 1.1rem;
            opacity: 0.8;
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.7;
        }
        
        /* Feature Cards */
        
        .feature-card {
            background: white;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 16px 48px rgba(0,0,0,0.15);
        }
        
        .card-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            display: block;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .card-description {
            color: #555;
            line-height: 1.6;
            margin-bottom: 2rem;
            font-size: 1rem;
        }
        
        .card-features {
            list-style: none;
            padding: 0;
            margin-bottom: 2rem;
        }
        
        .card-features li {
            padding: 0.3rem 0;
            color: #666;
            font-size: 0.9rem;
        }
        
        .card-features li::before {
            content: "✓";
            color: #27ae60;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        
        /* Custom Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            height: 60px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* Get Started Section */
        .get-started-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 3rem 0 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .get-started-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .get-started-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Stats Section */
        .stats-container {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 16px;
            margin: 3rem 0;
        }
        
        .stat-item {
            padding: 1rem;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 800;
            color: #667eea;
            display: block;
        }
        
        .stat-label {
            color: #666;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .hero-title { font-size: 2.5rem; }
            .hero-subtitle { font-size: 1.2rem; }
            .feature-card { padding: 1.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">QuantRisk Analytics</div>
        <div class="hero-subtitle">Portfolio Management & Risk Assessment Platform</div>
        <div class="hero-description">
            Test the power modern portfolio theory, value-at-risk modeling, and quantitative finance 
            to make informed investment decisions.
        </div>
    </div>
""", unsafe_allow_html=True)

# Stats Section
st.markdown("""
    <div class="stats-container">
""", unsafe_allow_html=True)

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:
    st.markdown("""
        <div class="stat-item">
            <span class="stat-number">4</span>
            <div class="stat-label">VaR Methods</div>
        </div>
    """, unsafe_allow_html=True)

with stat2:
    st.markdown("""
        <div class="stat-item">
            <span class="stat-number">∞</span>
            <div class="stat-label">Asset Classes</div>
        </div>
    """, unsafe_allow_html=True)

with stat3:
    st.markdown("""
        <div class="stat-item">
            <span class="stat-number">Real-time</span>
            <div class="stat-label">Market Data</div>
        </div>
    """, unsafe_allow_html=True)

with stat4:
    st.markdown("""
        <div class="stat-item">
            <span class="stat-number">Academic</span>
            <div class="stat-label">Grade Research</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Get Started Section
st.markdown("""
    <div class="get-started-section">
        <div class="get-started-title">Get Started</div>
        <div class="get-started-subtitle">
            Choose your analytical journey and explore the power of quantitative finance
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Portfolio Optimizer", key="opt_btn", help="Optimize your portfolio using Markowitz theory"):
        st.switch_page("pages/1_Portfolio_Optimizer.py")

with col2:
    if st.button("⚡ Value-at-Risk", key="var_btn", help="Calculate portfolio risk using multiple VaR methods"):
        st.switch_page("pages/3_var_app.py")

with col3:
    if st.button("About Me", key="about_btn", help="Learn about the developer and download CV"):
        st.switch_page("pages/2_About_Me.py")

with col4:
    if st.button("Bibliography", key="bib_btn", help="Access research papers and documentation"):
        st.switch_page("pages/4_Bibliography.py")

st.markdown("</div>", unsafe_allow_html=True)

# Feature Cards using Streamlit columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-card">
            <span class="card-icon">📈</span>
            <div class="card-title">Portfolio Optimizer</div>
            <div class="card-description">
                Implement Markowitz Modern Portfolio Theory with CAPM analysis for optimal asset allocation.
            </div>
            <ul class="card-features">
                <li>Efficient Frontier Visualization</li>
                <li>Risk-Free Asset Integration</li>
                <li>Beta & Alpha Calculations</li>
                <li>S&P 500 Benchmarking</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <span class="card-icon">🎓</span>
            <div class="card-title">Academic Research</div>
            <div class="card-description">
                Access comprehensive documentation and research papers backing every financial model.
            </div>
            <ul class="card-features">
                <li>Peer-Reviewed Methodologies</li>
                <li>Complete Bibliography</li>
                <li>Implementation Details</li>
                <li>Downloadable Research PDF</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <span class="card-icon">⚡</span>
            <div class="card-title">Value-at-Risk Engine</div>
            <div class="card-description">
                Comprehensive risk assessment using multiple VaR methodologies for robust risk management.
            </div>
            <ul class="card-features">
                <li>Parametric VaR (Normal Distribution)</li>
                <li>Monte Carlo Simulation</li>
                <li>Fixed Income PV01 Analysis</li>
                <li>Portfolio-Level Risk Aggregation</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <span class="card-icon">👨‍💼</span>
            <div class="card-title">About the Developer</div>
            <div class="card-description">
                Learn about the financial engineering expertise and academic background behind this platform.
            </div>
            <ul class="card-features">
                <li>Centrale Méditerranée Student</li>
                <li>Financial Engineering Focus</li>
                <li>Quantitative Research Experience</li>
                <li>Professional CV Available</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Add some space before Get Started section
st.markdown("<br>", unsafe_allow_html=True)

# Add spacing before footer
st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>QuantRisk Analytics</strong> | Built with Streamlit & Modern Portfolio Theory</p>
        <p>© 2025 SALHI Reda | Financial Engineering Research</p>
    </div>
""", unsafe_allow_html=True)