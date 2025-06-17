# pages/4_Bibliography.py

import streamlit as st

# Page config
st.set_page_config(page_title="Bibliography & Research", layout="wide", page_icon="🎓")

# Enhanced CSS styling with interactive elements
st.markdown("""
    <style>
        [data-testid="stSidebar"] { display: none !important; }
        header, footer { visibility: hidden; }
        .main { padding-top: 1rem; }

        /* Hero Section */
        .bib-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }

        .bib-hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }

        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .bib-hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.8rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            letter-spacing: -1px;
            position: relative;
            z-index: 1;
        }
        
        .bib-hero p {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 1rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
            position: relative;
            z-index: 1;
        }

        .hero-stats {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 2rem;
            position: relative;
            z-index: 1;
        }

        .hero-stat {
            background: rgba(255,255,255,0.2);
            padding: 1rem 1.5rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 800;
            display: block;
        }

        .stat-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }

        /* Interactive Filter Section */
        .filter-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
        }

        .filter-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            text-align: center;
        }

        .filter-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 2rem;
        }

        .filter-btn {
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }

        .filter-btn:hover, .filter-btn.active {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        /* Reference Cards */
        .reference-category {
            margin: 2rem 0;
        }

        .category-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 15px 15px 0 0;
            font-size: 1.4rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .category-header:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .category-content {
            background: white;
            border-radius: 0 0 15px 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .reference-card {
            padding: 2rem;
            border-bottom: 1px solid #e9ecef;
            transition: all 0.3s ease;
            position: relative;
        }

        .reference-card:last-child {
            border-bottom: none;
        }

        .reference-card:hover {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            transform: translateX(10px);
        }

        .reference-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }

        .reference-authors {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .reference-details {
            color: #6c757d;
            margin-bottom: 1rem;
            font-style: italic;
        }

        .reference-description {
            color: #495057;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .reference-tags {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .tag {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* PDF Download Section */
        .pdf-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin: 3rem 0;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            border: 2px dashed #667eea;
        }

        .pdf-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            display: block;
        }

        .pdf-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        .pdf-description {
            color: #6c757d;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        /* Search Box */
        .search-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }

        .search-box {
            width: 100%;
            padding: 1rem 1.5rem;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .search-box:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        /* Timeline */
        .timeline-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin: 2rem 0;
        }

        .timeline-item {
            display: flex;
            align-items: center;
            margin: 1rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .timeline-item:hover {
            transform: translateX(10px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .timeline-year {
            background: #667eea;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 700;
            min-width: 80px;
            text-align: center;
            margin-right: 1rem;
        }

        .timeline-content {
            flex: 1;
            color: #2c3e50;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
        }

        /* Back Button */
        .back-button .stButton > button {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
        }

        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 1.5rem 3rem;
            border-radius: 15px;
            font-weight: 700;
            font-size: 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        }

        .stDownloadButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(40, 167, 69, 0.6);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .bib-hero h1 { font-size: 2.5rem; }
            .bib-hero p { font-size: 1.1rem; }
            .hero-stats { gap: 1rem; }
            .filter-buttons { gap: 0.5rem; }
            .reference-card { padding: 1.5rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="bib-hero">
        <h1>🎓 Bibliography & Research</h1>
        <p>Comprehensive academic foundations and cutting-edge research in quantitative finance</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="stat-number">15+</span>
                <span class="stat-label">Key References</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">4</span>
                <span class="stat-label">Research Areas</span>
            </div>
            <div class="hero-stat">
                <span class="stat-number">50+</span>
                <span class="stat-label">Years Covered</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Back Button
st.markdown('<div class="back-button">', unsafe_allow_html=True)
if st.button("🔙 Back to Home"):
    st.switch_page("streamlit_app.py")
st.markdown('</div>', unsafe_allow_html=True)

# Interactive Filter Section
st.markdown("""
    <div class="filter-section">
        <div class="filter-title">📚 Browse Research by Category</div>
    </div>
""", unsafe_allow_html=True)

# Category selection
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📈 Portfolio Theory", use_container_width=True):
        st.session_state.selected_category = "portfolio"

with col2:
    if st.button("⚡ Risk Management", use_container_width=True):
        st.session_state.selected_category = "risk"

with col3:
    if st.button("💰 Asset Pricing", use_container_width=True):
        st.session_state.selected_category = "pricing"

with col4:
    if st.button("📊 All References", use_container_width=True):
        st.session_state.selected_category = "all"

# Get selected category
selected_category = st.session_state.get("selected_category", "all")

# Search functionality
st.markdown("""
    <div class="search-container">
        <input type="text" class="search-box" placeholder="🔍 Search references by author, title, or keyword..." id="searchBox">
    </div>
""", unsafe_allow_html=True)

# Timeline of Key Developments
with st.expander("📅 Timeline of Key Financial Theory Developments", expanded=False):
    st.markdown("""
        <div class="timeline-container">
            <div class="timeline-item">
                <div class="timeline-year">1952</div>
                <div class="timeline-content"><strong>Portfolio Selection</strong> - Harry Markowitz introduces Modern Portfolio Theory</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1964</div>
                <div class="timeline-content"><strong>CAPM</strong> - William Sharpe develops Capital Asset Pricing Model</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1973</div>
                <div class="timeline-content"><strong>Black-Scholes</strong> - Option pricing model revolutionizes derivatives</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1976</div>
                <div class="timeline-content"><strong>APT</strong> - Stephen Ross introduces Arbitrage Pricing Theory</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-year">1990s</div>
                <div class="timeline-content"><strong>Value-at-Risk</strong> - VaR becomes standard risk measurement tool</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# References by category
references = {
    "portfolio": [
        {
            "title": "Portfolio Selection",
            "authors": "Markowitz, H.",
            "details": "Journal of Finance, 7(1), 77–91 (1952)",
            "description": "The foundational paper that introduced Modern Portfolio Theory, establishing the mathematical framework for optimal portfolio construction based on mean-variance optimization.",
            "tags": ["Portfolio Theory", "Optimization", "Risk-Return", "Fundamental"]
        },
        {
            "title": "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk",
            "authors": "Sharpe, W. F.",
            "details": "Journal of Finance, 19(3), 425–442 (1964)",
            "description": "Introduction of the Capital Asset Pricing Model (CAPM), providing a framework for pricing risky securities and determining expected returns.",
            "tags": ["CAPM", "Asset Pricing", "Market Equilibrium", "Beta"]
        },
        {
            "title": "The Arbitrage Theory of Capital Asset Pricing",
            "authors": "Ross, S. A.",
            "details": "Journal of Economic Theory, 13(3), 341–360 (1976)",
            "description": "Development of Arbitrage Pricing Theory (APT) as an alternative to CAPM, using multiple factors to explain asset returns.",
            "tags": ["APT", "Multi-factor", "Arbitrage", "Asset Pricing"]
        }
    ],
    "risk": [
        {
            "title": "Value at Risk: The New Benchmark for Managing Financial Risk",
            "authors": "Jorion, P.",
            "details": "McGraw-Hill Education, 3rd Edition (2006)",
            "description": "Comprehensive treatment of Value-at-Risk methodology, covering parametric, historical simulation, and Monte Carlo approaches to risk measurement.",
            "tags": ["VaR", "Risk Management", "Monte Carlo", "Parametric"]
        },
        {
            "title": "Coherent Measures of Risk",
            "authors": "Artzner, P., Delbaen, F., Eber, J. M., & Heath, D.",
            "details": "Mathematical Finance, 9(3), 203–228 (1999)",
            "description": "Fundamental paper establishing the mathematical properties that risk measures should satisfy, leading to the development of conditional VaR (CVaR).",
            "tags": ["Risk Measures", "CVaR", "Coherence", "Mathematical Finance"]
        },
        {
            "title": "An Introduction to Credit Risk Modeling",
            "authors": "Bluhm, C., Overbeck, L., & Wagner, C.",
            "details": "Chapman and Hall/CRC, 2nd Edition (2010)",
            "description": "Comprehensive coverage of credit risk models including structural models, reduced-form models, and portfolio credit risk.",
            "tags": ["Credit Risk", "Default Probability", "Structural Models", "Portfolio"]
        }
    ],
    "pricing": [
        {
            "title": "The Pricing of Options and Corporate Liabilities",
            "authors": "Black, F., & Scholes, M.",
            "details": "Journal of Political Economy, 81(3), 637–654 (1973)",
            "description": "The revolutionary Black-Scholes option pricing model that provided the first complete mathematical framework for valuing European options.",
            "tags": ["Black-Scholes", "Options", "Derivatives", "PDE"]
        },
        {
            "title": "Option Pricing: A Simplified Approach",
            "authors": "Cox, J. C., Ross, S. A., & Rubinstein, M.",
            "details": "Journal of Financial Economics, 7(3), 229–263 (1979)",
            "description": "Introduction of the binomial option pricing model, providing an intuitive and computationally tractable approach to option valuation.",
            "tags": ["Binomial Model", "Options", "Discrete Time", "Numerical Methods"]
        },
        {
            "title": "Interest Rate Models - Theory and Practice",
            "authors": "Brigo, D., & Mercurio, F.",
            "details": "Springer Finance, 2nd Edition (2006)",
            "description": "Comprehensive treatment of interest rate modeling, covering short-rate models, HJM framework, and market models for derivatives pricing.",
            "tags": ["Interest Rates", "HJM", "Short Rate Models", "Fixed Income"]
        }
    ]
}

# Display references based on selected category
if selected_category == "all":
    categories_to_show = ["portfolio", "risk", "pricing"]
    category_names = {
        "portfolio": "📈 Modern Portfolio Theory & Asset Pricing", 
        "risk": "⚡ Risk Management & Measurement",
        "pricing": "💰 Derivatives Pricing & Models"
    }
else:
    categories_to_show = [selected_category]
    category_names = {
        "portfolio": "📈 Modern Portfolio Theory & Asset Pricing",
        "risk": "⚡ Risk Management & Measurement", 
        "pricing": "💰 Derivatives Pricing & Models"
    }

for category in categories_to_show:
    if category in references:
        st.markdown(f"""
            <div class="reference-category">
                <div class="category-header">
                    {category_names[category]}
                </div>
                <div class="category-content">
        """, unsafe_allow_html=True)
        
        for ref in references[category]:
            tags_html = "".join([f'<span class="tag">{tag}</span>' for tag in ref["tags"]])
            
            st.markdown(f"""
                <div class="reference-card">
                    <div class="reference-title">{ref["title"]}</div>
                    <div class="reference-authors">{ref["authors"]}</div>
                    <div class="reference-details">{ref["details"]}</div>
                    <div class="reference-description">{ref["description"]}</div>
                    <div class="reference-tags">{tags_html}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# Additional Research Tools
with st.expander("🔬 Research Tools & Data Sources", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Sources")
        st.markdown("""
        - **Yahoo Finance API** (via `yfinance`) — Equity & ETF price data
        - **FRED (Federal Reserve Economic Data)** — Bond yields & macro indicators  
        - **CoinGecko API** — Cryptocurrency market data
        - **Alpha Vantage** — High-frequency financial data
        - **Quandl** — Alternative and economic datasets
        """)
    
    with col2:
        st.markdown("### 🛠️ Technical Implementation")
        st.markdown("""
        - **Python Libraries**: `numpy`, `pandas`, `scipy`, `matplotlib`, `plotly`
        - **Financial Tools**: `yfinance`, `pandas_datareader`, `quantlib`
        - **Machine Learning**: `scikit-learn`, `tensorflow`
        - **Web Framework**: `streamlit` for interactive applications
        - **Statistical Analysis**: `statsmodels`, `arch` (GARCH models)
        """)

# PDF Download Section
st.markdown("""
    <div class="pdf-section">
        <span class="pdf-icon">📑</span>
        <div class="pdf-title">Complete Research Document</div>
        <div class="pdf-description">
            Download the comprehensive research paper that provides detailed analysis, implementation notes, 
            and empirical results for all VaR methodologies and portfolio optimization techniques used in this platform.
        </div>
    </div>
""", unsafe_allow_html=True)

# Check if PDF exists and provide download
try:
    with open("assets/Value_at_Risk.pdf", "rb") as pdf_file:
        st.download_button(
            label="📥 Download Complete Research Paper (PDF)",
            data=pdf_file,
            file_name="QuantRisk_Analytics_Research_Paper.pdf",
            mime="application/pdf",
            use_container_width=True
        )
except FileNotFoundError:
    st.info("📝 Research paper will be available soon. Currently finalizing the comprehensive analysis.")

# Concepts Covered Section
with st.expander("🎯 Key Concepts Implemented in This Platform", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Risk Management
        - **Parametric VaR** (Normal Distribution)
        - **Monte Carlo Simulation** VaR
        - **Fixed Income VaR** (PV01 Method)
        - **Portfolio-Level Risk Aggregation**
        - **Backtesting & Validation**
        """)
    
    with col2:
        st.markdown("""
        ### Portfolio Optimization
        - **Modern Portfolio Theory**
        - **Efficient Frontier Construction**
        - **Capital Asset Pricing Model (CAPM)**
        - **Risk-Free Asset Integration**
        - **Beta & Alpha Analysis**
        """)
    
    with col3:
        st.markdown("""
        ### Mathematical Methods
        - **Correlation Analysis**
        - **Covariance Matrix Estimation**
        - **Maximum Likelihood Estimation**
        - **Cholesky Decomposition**
        - **Statistical Hypothesis Testing**
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>QuantRisk Analytics</strong> | Academic Research & Implementation</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">© 2025 SALHI Reda | Built on rigorous financial theory and academic research</p>
    </div>
""", unsafe_allow_html=True)

# JavaScript for search functionality
st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const searchBox = document.getElementById('searchBox');
        if (searchBox) {
            searchBox.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                const referenceCards = document.querySelectorAll('.reference-card');
                
                referenceCards.forEach(card => {
                    const text = card.textContent.toLowerCase();
                    if (text.includes(searchTerm)) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
        }
    });
    </script>
""", unsafe_allow_html=True)