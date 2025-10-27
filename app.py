"""
app.py - Fashion E-commerce Sentiment Analysis Dashboard (FULLY FIXED VERSION)
"""
import streamlit as st
import pandas as pd
import numpy as np
import nltk

# Import custom modules
from src.data_loader import DataLoader
from src.preprocessing import TextPreprocessor
from src.model import SentimentModel
from src.visualization import Visualizer
from config.config import (
    PAGE_CONFIG, SENTIMENT_CONFIG, CHAT_RESPONSES,
    EXAMPLE_TEXTS, REVIEWS_CSV, CHART_COLORS
)

# Page configuration with fashion theme
st.set_page_config(
    page_title="Fashion Review Analyzer",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Poppins:wght@300;400;500;600&display=swap');

    /* FIXED: Remove black background completely */
    .stApp {
        background: linear-gradient(135deg, #fef5f8 0%, #ffe8f0 50%, #f8f0f5 100%) !important;
    }

    /* NEW: Sidebar with pastel violet background */
    .css-1d391kg {
        background: linear-gradient(135deg, #f4f0ff 0%, #e8e0ff 50%, #f0e8ff 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f4f0ff 0%, #e8e0ff 50%, #f0e8ff 100%) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(135deg, #f4f0ff 0%, #e8e0ff 50%, #f0e8ff 100%) !important;
    }

    /* Sidebar text colors */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #5a4d6f !important;
    }

    /* Global Styles */
    .main {
        padding: 0rem 1rem;
        background: transparent;
    }

    /* FIXED: Better text contrast for headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #6d3d5f !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.3);
    }

    h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: #7d4d6f !important;
    }

    p, span, div, label {
        font-family: 'Poppins', sans-serif;
        color: #5a3d52;
    }

    /* FIXED: Better tab visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.8rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        color: #7d4d6f;
        border-radius: 10px;
        transition: all 0.3s ease;
        background: white;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #ffc9de 0%, #ffb3d9 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 158, 196, 0.3);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%);
        color: white;
        box-shadow: 0 6px 20px rgba(255, 158, 196, 0.4);
    }

    /* FIXED: Metric Cards with better contrast */
    .metric-card {
        background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(255, 158, 196, 0.35);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
        transition: all 0.5s ease;
    }

    .metric-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 15px 35px rgba(255, 158, 196, 0.45);
    }

    .metric-card:hover::before {
        top: -30%;
        right: -30%;
    }

    /* FIXED: Darker, more visible sentiment colors */
    .positive-sentiment {
        background: linear-gradient(135deg, #6dbfa0 0%, #5aaa8c 100%);
        box-shadow: 0 8px 25px rgba(109, 191, 160, 0.4);
        border-color: rgba(255, 255, 255, 0.4);
    }

    .positive-sentiment:hover {
        box-shadow: 0 15px 35px rgba(109, 191, 160, 0.5);
    }

    .negative-sentiment {
        background: linear-gradient(135deg, #ff8888 0%, #ff6b6b 100%);
        box-shadow: 0 8px 25px rgba(255, 136, 136, 0.4);
        border-color: rgba(255, 255, 255, 0.4);
    }

    .negative-sentiment:hover {
        box-shadow: 0 15px 35px rgba(255, 136, 136, 0.5);
    }

    .neutral-sentiment {
        background: linear-gradient(135deg, #b399e6 0%, #9d7fd9 100%);
        box-shadow: 0 8px 25px rgba(179, 153, 230, 0.4);
        border-color: rgba(255, 255, 255, 0.4);
    }

    .neutral-sentiment:hover {
        box-shadow: 0 15px 35px rgba(179, 153, 230, 0.5);
    }

    /* Fashion Icons */
    .fashion-icon {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        display: block;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        filter: drop-shadow(0 0 8px rgba(255,255,255,0.5));
    }

    /* FIXED: Review Cards with better text contrast */
    .review-card {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 100%);
        padding: 1.3rem;
        border-radius: 15px;
        border-left: 5px solid;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.12);
        color: #3d2635;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .review-card:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 20px rgba(139, 79, 117, 0.18);
    }

    .review-card strong {
        color: #3d2635;
    }

    .positive-border { 
        border-left-color: #5aaa8c;
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf7 100%);
    }

    .negative-border { 
        border-left-color: #ff6b6b;
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
    }

    .neutral-border { 
        border-left-color: #9d7fd9;
        background: linear-gradient(135deg, #ffffff 0%, #faf8ff 100%);
    }

    /* FIXED: Stat Box with better contrast */
    .stat-box {
        padding: 1.8rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);
        background: white;
        transition: all 0.3s ease;
        border: 2px solid #ffe8f0;
    }

    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(139, 79, 117, 0.15);
        border-color: #ffc9de;
    }

    .stat-box p {
        color: #5a3d52;
        margin: 0;
    }

    .stat-box h4 {
        color: #6d3d5f !important;
    }

    /* FIXED: Buttons with better states */
    .stButton>button {
        background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.8rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 158, 196, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #ff7eb3 0%, #ff5e9d 100%);
        box-shadow: 0 6px 18px rgba(255, 158, 196, 0.45);
        transform: translateY(-3px);
    }

    .stButton>button:active {
        transform: translateY(-1px);
    }

    /* Expander - Fashion Card Style */
    div[data-testid="stExpander"] {
        background: white;
        border: 2px solid #ffe8f0;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(139, 79, 117, 0.08);
    }

    div[data-testid="stExpander"]:hover {
        border-color: #ffc9de;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.12);
    }

    /* FIXED: Info boxes with better visibility */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid #ff9ec4;
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 100%);
        color: #5a3d52;
    }

    /* Headers with Fashion Icons */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
        padding: 1.2rem;
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 100%);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);
        border: 2px solid #ffe8f0;
    }

    .section-header:hover {
        box-shadow: 0 6px 20px rgba(139, 79, 117, 0.15);
        border-color: #ffc9de;
    }

    .section-header h2, .section-header h3, .section-header h4 {
        margin: 0;
        color: #6d3d5f !important;
    }

    /* Fashion-themed metric numbers */
    .metric-number {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
    }

    /* Custom Table Styles */
    .custom-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);
        border: 2px solid #ffe8f0;
    }
    .custom-table thead {
        background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%);
    }
    .custom-table thead th {
        padding: 1.2rem 1rem;
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: left;
        border: none;
    }
    .custom-table tbody tr {
        border-bottom: 1px solid #fff5f7;
        transition: all 0.2s ease;
    }
    .custom-table tbody tr:hover {
        background: #fff8fa;
        transform: scale(1.01);
    }
    .custom-table tbody td {
        padding: 1rem;
        color: #5a3d52;
        font-family: 'Poppins', sans-serif;
        font-size: 0.9rem;
        line-height: 1.5;
        border: none;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-align: center;
    }
    .sentiment-positive { background: #e8f5f0; color: #5aaa8c; }
    .sentiment-negative { background: #ffe8e8; color: #ff6b6b; }
    .sentiment-neutral { background: #f0eaf7; color: #9d7fd9; }

    /* FIXED: Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif;
        color: #6d3d5f !important;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: #7d4d6f !important;
        font-weight: 600;
    }

    /* IMPROVED: Better separator lines */
    hr {
        border: none;
        border-top: 2px solid #ffe8f0;
        margin: 2rem 0;
    }

    /* FIXED: Plotly charts background */
    .js-plotly-plot {
        background: white !important;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.08);
    }

    /* FIXED: All default Streamlit text colors */
    .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #5a3d52 !important;
    }

    /* FIXED: File Uploader - Pastel Fashion Theme */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #ffffff 0%, #fff8fa 100%);
        border: 2px solid #ffe8f0;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #ffc9de;
        box-shadow: 0 6px 20px rgba(139, 79, 117, 0.15);
    }

    [data-testid="stFileUploader"] label {
        color: #6d3d5f !important;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stFileUploader"] section {
        border: 2px dashed #ffc9de !important;
        border-radius: 12px;
        background: linear-gradient(135deg, #fff5f7 0%, #ffe8f0 50%, #fff5f7 100%);
        padding: 1.5rem;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #ff9ec4 !important;
        background: linear-gradient(135deg, #ffe8f0 0%, #ffc9de 50%, #ffe8f0 100%);
    }

    [data-testid="stFileUploader"] small {
        color: #9d7a8f !important;
        font-family: 'Poppins', sans-serif;
    }

    /* FIXED: File uploader text - dark and readable */
    [data-testid="stFileUploader"] span {
        color: #3d2635 !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* NEW: Text Area Styling for Tab 5 (Pastel Theme) */
    textarea {
        background: linear-gradient(135deg, #fffaff 0%, #fef5fb 100%) !important;
        border: 2px solid #f2c8e3 !important;
        border-radius: 12px !important;
        color: #5a3d52 !important;
        font-family: 'Poppins', sans-serif !important;
        box-shadow: 0 2px 8px rgba(255, 158, 196, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    textarea:focus {
        outline: none !important;
        border-color: #ff9ec4 !important;
        box-shadow: 0 0 0 4px rgba(255, 158, 196, 0.25) !important;
    }
    
    /* ================================
   STRONG Pastel override for SELECTBOX (copy into your existing <style>)
   Covers Streamlit/baseweb popovers, native selects, role=button/listbox/options
   ================================ */
    
    /* main select "button" */
    [data-testid="stSelectbox"] div[role="button"],
    .stSelectbox div[role="button"],
    [data-baseweb="select"] > div,
    div[role="combobox"] > div[role="button"] {
        background: linear-gradient(135deg, #fffafe 0%, #fff5ff 100%) !important;
        border: 2px solid #eadcff !important;
        color: #6b5a89 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(210,180,255,0.10) !important;
    }
    
    /* caret / svg color */
    [data-baseweb="select"] svg,
    [data-testid="stSelectbox"] svg,
    .stSelectbox svg {
        fill: #cfaef6 !important;
    }
    
    /* focus/active on the controlled "button" */
    [data-testid="stSelectbox"] div[role="button"]:focus-within,
    [data-baseweb="select"] > div:focus-within,
    div[role="combobox"] > div[role="button"]:focus-within {
        border-color: #d7b8ff !important;
        box-shadow: 0 0 0 5px rgba(215,184,255,0.18) !important;
        background: linear-gradient(135deg, #fbf5ff 0%, #fff6fb 100%) !important;
    }
    
    /* LIST / POPOVER that appears when open */
    [data-baseweb="popover"],
    div[role="listbox"],
    ul[role="listbox"],
    .baseweb-popover,
    .baseweb-menu {
        background: linear-gradient(135deg, #fffefe 0%, #fdf7ff 100%) !important;
        border: 2px solid #efe0ff !important;
        box-shadow: 0 10px 30px rgba(200,160,255,0.12) !important;
        border-radius: 12px !important;
    }
    
    /* each option row */
    div[role="option"],
    li[role="option"],
    [role="listbox"] [role="option"] {
        background: transparent !important;
        color: #6b5a89 !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.6rem 0.9rem !important;
        border-radius: 8px !important;
        margin: 0.15rem 0.4rem !important;
    }
    
    /* hover for options */
    div[role="option"]:hover,
    li[role="option"]:hover,
    [role="listbox"] [role="option"]:hover {
        background: linear-gradient(135deg, #f4edff 0%, #fbeefc 100%) !important;
        color: #543e6e !important;
    }
    
    /* selected option */
    [role="option"][aria-selected="true"],
    [role="listbox"] [aria-selected="true"] {
        background: linear-gradient(135deg, #e6d0ff 0%, #f2cfff 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* native HTML select fallback (mobile / some browsers) */
    select,
    select option {
        background: linear-gradient(135deg, #fffafe 0%, #fff5ff 100%) !important;
        color: #6b5a89 !important;
        border: 2px solid #eadcff !important;
    }
    select:focus {
        outline: none !important;
        box-shadow: 0 0 0 5px rgba(215,184,255,0.18) !important;
    }
    
    /* ensure no dark overlay accidentally sits above the dropdown */
    div[role="presentation"],
    div[role="dialog"] {
        background: transparent !important;
    }

    /* FIXED: Streamlit headers that aren't caught by h1, h2, h3 */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #6d3d5f !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess data (cached)"""
    loader = DataLoader(str(REVIEWS_CSV))
    preprocessor = TextPreprocessor()

    try:
        df = loader.load_from_csv()
    except FileNotFoundError:
        st.warning("👗 Data file not found. Creating sample fashion reviews...")
        df = loader.create_sample_csv(str(REVIEWS_CSV))

    df = preprocessor.preprocess_dataframe(df)
    return df, preprocessor


@st.cache_resource
def train_sentiment_model(_df):
    """Train model (cached)"""
    model = SentimentModel()
    metrics = model.train(_df['cleaned_text'], _df['sentiment'])
    return model, metrics


@st.cache_resource
def initialize_visualizer():
    """Initialize visualizer (cached)"""
    # Updated with better contrast colors
    improved_colors = {
        'positive': '#5aaa8c',
        'negative': '#ff6b6b',
        'neutral': '#9d7fd9'
    }
    return Visualizer(color_map=improved_colors)


# Initialize components
df, preprocessor = load_and_prepare_data()
model, metrics = train_sentiment_model(df)
viz = initialize_visualizer()

# IMPROVED Header with Fashion Theme
st.markdown("""
    <div style='text-align: center; padding: 2.5rem 1rem; background: linear-gradient(135deg, #ffffff 0%, #fff0f5 100%); border-radius: 25px; margin-bottom: 2rem; box-shadow: 0 8px 30px rgba(139, 79, 117, 0.15); border: 3px solid #ffe8f0;'>
        <h1 style='font-size: 3.5rem; margin: 0; color: #6d3d5f; font-family: "Playfair Display", serif; text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);'>
            👗 Fashion Review Analyzer
        </h1>
        <p style='font-size: 1.3rem; color: #9d7a8f; margin: 1rem 0 0 0; font-family: "Poppins", sans-serif; font-weight: 500;'>
            ✨ Analyze customer fashion reviews with AI-powered sentiment detection ✨
        </p>
    </div>
""", unsafe_allow_html=True)

# IMPROVED Sidebar with Fashion Theme
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1);'>
            <div style='font-size: 4.5rem; margin-bottom: 0.8rem;'>👗</div>
            <h2 style='color: #6d3d5f; font-family: "Playfair Display", serif; margin: 0;'>Fashion Analytics</h2>
        </div>
    """, unsafe_allow_html=True)

    st.info("""
        **✨ About This Dashboard**

        Analyzing customer reviews from our online fashion boutique using **Machine Learning**.

        **🛍️ Features:**
        - 📊 Real-time sentiment analysis
        - 📈 Model performance metrics
        - 💭 Fashion word trends
        - 🤖 Interactive review analyzer
    """)

    st.markdown("---")
    st.subheader("📊 Store Insights")
    st.metric("Total Reviews", f"{len(df):,}")
    st.metric("AI Accuracy", f"{metrics['accuracy']:.1%}")

    st.markdown("---")
    st.subheader("💾 Data Management")

    if st.button("💾 Download Reviews"):
        csv = df[['text', 'sentiment']].to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="fashion_reviews.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("📤 Upload New Reviews", type=['csv'])
    if uploaded_file:
        st.success("✅ File uploaded! Restart app to load.")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "🔍 Exploration",
    "📈 Performance",
    "💬 Word Trends",
    "🤖 Live Analyzer"
])

# TAB 1: Overview
with tab1:
    st.markdown(
        '<div class="section-header"><span style="font-size: 2rem;">🪄</span><h2>Fashion Store Overview</h2></div>',
        unsafe_allow_html=True)

    # Enhanced Metrics with Fashion Icons
    col1, col2, col3, col4 = st.columns(4)
    sentiment_counts = df['sentiment'].value_counts()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="fashion-icon">🛍️</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{len(df)}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Total Reviews</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card positive-sentiment">
            <span class="fashion-icon">😊</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{sentiment_counts.get('positive', 0)}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Happy Customers</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card negative-sentiment">
            <span class="fashion-icon">😞</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{sentiment_counts.get('negative', 0)}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Need Improvement</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card neutral-sentiment">
            <span class="fashion-icon">😐</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{sentiment_counts.get('neutral', 0)}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Neutral Feedback</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">📊</span><h3>Customer Sentiment Distribution</h3></div>',
            unsafe_allow_html=True)
        fig_pie = viz.create_sentiment_pie_chart(sentiment_counts)
        st.plotly_chart(fig_pie, use_container_width=False)

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">👗</span><h3>Sample Fashion Reviews</h3></div>',
            unsafe_allow_html=True)
        sentiment_filter = st.selectbox("Filter by sentiment:", ["All", "Positive", "Negative", "Neutral"],
                                        key="overview_filter")

        if sentiment_filter == "All":
            sample_df = df.sample(min(5, len(df)))
        else:
            filtered = df[df['sentiment'] == sentiment_filter.lower()]
            sample_df = filtered.sample(min(5, len(filtered))) if len(filtered) > 0 else filtered

        for _, row in sample_df.iterrows():
            sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            border_class = {"positive": "positive-border", "negative": "negative-border", "neutral": "neutral-border"}

            with st.expander(f"{sentiment_emoji[row['sentiment']]} {row['sentiment'].title()} Review"):
                st.markdown(f"""
                <div class="review-card {border_class[row['sentiment']]}">
                    <strong>"{row['text'][:200] + "..." if len(row['text']) > 200 else row['text']}"</strong>
                </div>
                """, unsafe_allow_html=True)

# TAB 2: Data Exploration
with tab2:
    st.markdown(
        '<div class="section-header"><span style="font-size: 2rem;">🔍</span><h2>Fashion Review Exploration</h2></div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">📏</span><h3>Review Length Analysis</h3></div>',
            unsafe_allow_html=True)
        fig_box = viz.create_word_length_boxplot(df)
        st.plotly_chart(fig_box, use_container_width=False)

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">📊</span><h3>Review Statistics</h3></div>',
            unsafe_allow_html=True)
        avg_words = df['word_count'].mean()
        avg_chars = df['char_count'].mean()

        st.metric("👗 Avg Words/Review", f"{avg_words:.0f}")
        st.metric("✏️ Avg Chars/Review", f"{avg_chars:.0f}")
        st.metric("📖 Longest Review", f"{df['word_count'].max()} words")
        st.metric("📄 Shortest Review", f"{df['word_count'].min()} words")

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span style="font-size: 1.5rem;">🔬</span><h3>Text Processing Preview</h3></div>',
        unsafe_allow_html=True)

    sample_indices = df.sample(3).index
    comparison_data = []

    for idx in sample_indices:
        comparison_data.append({
            "👗 Original Review": df.loc[idx, 'text'][:100] + "...",
            "✨ Cleaned Text": df.loc[idx, 'cleaned_text'][:100] + "...",
            "💭 Sentiment": df.loc[idx, 'sentiment'].title()
        })

    # Custom styled dataframe for text processing
    table_html = "<table class='custom-table'><thead><tr>"
    table_html += "<th>👗 Original Review</th><th>✨ Cleaned Text</th><th>💭 Sentiment</th>"
    table_html += "</tr></thead><tbody>"

    for item in comparison_data:
        original_key = list(item.keys())[0]
        cleaned_key = list(item.keys())[1]
        sentiment_key = list(item.keys())[2]

        sentiment = item[sentiment_key].lower()
        badge_class = f"sentiment-{sentiment}"
        table_html += f"<tr><td>{item[original_key]}</td>"
        table_html += f"<td>{item[cleaned_key]}</td>"
        table_html += f"<td><span class='sentiment-badge {badge_class}'>{item[sentiment_key]}</span></td></tr>"

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Additional insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header"><span style="font-size: 1.5rem;">📈</span><h3>Average Review Length by Sentiment</h3></div>',
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    colors = {'positive': '#5aaa8c', 'negative': '#ff6b6b', 'neutral': '#9d7fd9'}
    icons = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}

    for col, sentiment in zip([col1, col2, col3], ['positive', 'negative', 'neutral']):
        with col:
            avg_length = df[df['sentiment'] == sentiment]['word_count'].mean()
            st.markdown(f"""
            <div class="stat-box" style="background: {colors[sentiment]}15; border-left: 5px solid {colors[sentiment]}; border-color: {colors[sentiment]};">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{icons[sentiment]}</div>
                <h4 style="margin: 0; color: {colors[sentiment]}; font-family: 'Playfair Display', serif; font-weight: 700;">{sentiment.title()}</h4>
                <p style="margin: 0.8rem 0 0 0; font-size: 2.5rem; font-weight: 700; color: #6d3d5f;" class="metric-number">{avg_length:.0f}</p>
                <p style="margin: 0.3rem 0 0 0; font-size: 1rem; color: #7d4d6f; font-weight: 600;">average words</p>
            </div>
            """, unsafe_allow_html=True)

# TAB 3: Model Performance
with tab3:
    st.markdown(
        '<div class="section-header"><span style="font-size: 2rem;">📈</span><h2>AI Model Performance</h2></div>',
        unsafe_allow_html=True)

    summary = model.get_metrics_summary()

    # Calculate training accuracy to detect overfitting
    train_score = model.model.score(
        model.vectorizer.transform(df['cleaned_text']),
        df['sentiment']
    )
    test_score = summary['accuracy']

    # Determine model status
    accuracy_diff = train_score - test_score

    if accuracy_diff > 0.15:
        model_status = "⚠️ Overfitting Detected"
        status_color = "linear-gradient(135deg, #ff8888 0%, #ff6b6b 100%)"
        status_explanation = "Training accuracy is significantly higher than test accuracy. Model memorized training data."
    elif test_score < 0.70:
        model_status = "⚠️ Underfitting Detected"
        status_color = "linear-gradient(135deg, #ffd9a0 0%, #ffcc80 100%)"
        status_explanation = "Both accuracies are low. Model is too simple to capture patterns."
    else:
        model_status = "✅ Excellent Model Fit"
        status_color = "linear-gradient(135deg, #6dbfa0 0%, #5aaa8c 100%)"
        status_explanation = "Model generalizes well to new fashion reviews. Balanced performance!"

    # Display model status banner
    st.markdown(f"""
    <div style='background: {status_color}; 
                padding: 2rem; 
                border-radius: 20px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 8px 25px rgba(139, 79, 117, 0.2);
                border: 2px solid rgba(255, 255, 255, 0.3);'>
        <h3 style='margin:0; font-family: "Playfair Display", serif; font-size: 1.8rem;'>{model_status}</h3>
        <p style='margin:0.8rem 0 0 0; font-family: "Poppins", sans-serif; font-size: 1.1rem;'>{status_explanation}</p>
        <p style='margin:1rem 0 0 0; font-size:1rem; font-weight: 600;'>
            🎯 Train Accuracy: {train_score:.2%} | 📊 Test Accuracy: {test_score:.2%} | 📉 Gap: {accuracy_diff:.2%}
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="fashion-icon">🎯</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{summary['accuracy']:.2%}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card positive-sentiment">
            <span class="fashion-icon">📊</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{summary['precision_avg']:.2%}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Avg Precision</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card neutral-sentiment">
            <span class="fashion-icon">📈</span>
            <h2 style="margin:0; font-size:3rem;" class="metric-number">{summary['recall_avg']:.2%}</h2>
            <p style="margin:0; font-size:1.1rem; font-weight: 600;">Avg Recall</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">🎯</span><h3>Confusion Matrix</h3></div>',
            unsafe_allow_html=True)
        cm = metrics['confusion_matrix']
        labels = ['Negative', 'Neutral', 'Positive']
        fig_cm = viz.create_confusion_matrix(cm, labels)
        st.pyplot(fig_cm)

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">📋</span><h3>Classification Report</h3></div>',
            unsafe_allow_html=True)
        report = metrics['classification_report']

        report_data = []
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in report:
                report_data.append({
                    '💭 Sentiment': sentiment.title(),
                    '🎯 Precision': f"{report[sentiment]['precision']:.2%}",
                    '📊 Recall': f"{report[sentiment]['recall']:.2%}",
                    '⭐ F1-Score': f"{report[sentiment]['f1-score']:.2%}",
                    '📈 Support': int(report[sentiment]['support'])
                })

        # Custom styled classification report table
        report_html = "<table class='custom-table'><thead><tr>"
        report_html += "<th>💭 Sentiment</th><th>🎯 Precision</th><th>📊 Recall</th><th>⭐ F1-Score</th><th>📈 Support</th>"
        report_html += "</tr></thead><tbody>"

        for item in report_data:
            sentiment_val = item[list(item.keys())[0]]
            precision_val = item[list(item.keys())[1]]
            recall_val = item[list(item.keys())[2]]
            f1_val = item[list(item.keys())[3]]
            support_val = item[list(item.keys())[4]]

            sentiment = sentiment_val.lower()
            badge_class = f"sentiment-{sentiment}"
            report_html += f"<tr>"
            report_html += f"<td><span class='sentiment-badge {badge_class}'>{sentiment_val}</span></td>"
            report_html += f"<td><strong>{precision_val}</strong></td>"
            report_html += f"<td><strong>{recall_val}</strong></td>"
            report_html += f"<td><strong>{f1_val}</strong></td>"
            report_html += f"<td><strong>{support_val}</strong></td>"
            report_html += "</tr>"

        report_html += "</tbody></table>"
        st.markdown(report_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("""
        **🤖 Model Insights:**
        - TF-IDF vectorization with bigrams
        - Logistic Regression with balanced weights
        - 80/20 train-test split
        """)

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span style="font-size: 1.5rem;">📊</span><h3>Performance Comparison</h3></div>',
        unsafe_allow_html=True)
    fig_perf = viz.create_performance_bar_chart(report)
    st.plotly_chart(fig_perf, use_container_width=False)

# TAB 4: Word Analysis
with tab4:
    st.markdown('<div class="section-header"><span style="font-size: 2rem;">💬</span><h2>Fashion Word Trends</h2></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">☁️</span><h3>Fashion Keywords Cloud</h3></div>',
            unsafe_allow_html=True)
        all_text = " ".join(df['cleaned_text'])
        fig_wc = viz.create_wordcloud(all_text)
        st.pyplot(fig_wc)

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">🎨</span><h3>Filter by Sentiment</h3></div>',
            unsafe_allow_html=True)
        selected_sentiment = st.radio(
            "Select sentiment:",
            ["All", "Positive", "Negative", "Neutral"]
        )

        if selected_sentiment != "All":
            filtered_text = " ".join(df[df['sentiment'] == selected_sentiment.lower()]['cleaned_text'])

            sentiment_colormaps = {
                'Positive': 'Greens',
                'Negative': 'Reds',
                'Neutral': 'Purples'
            }

            fig_wc_filtered = viz.create_wordcloud(
                filtered_text,
                colormap=sentiment_colormaps[selected_sentiment],
                width=800,
                height=400
            )
            st.pyplot(fig_wc_filtered)

        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%); padding: 1.3rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(255, 158, 196, 0.3);'>
            <p style='margin: 0; font-weight: 600; font-size: 1rem;'>💡 <strong>Tip:</strong> Different sentiments reveal unique fashion trends!</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div class="section-header"><span style="font-size: 1.5rem;">📊</span><h3>Top 20 Fashion Keywords</h3></div>',
        unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        all_tokens = nltk.word_tokenize(all_text)
        freq_dist = nltk.FreqDist(all_tokens)
        top_20 = dict(freq_dist.most_common(20))

        fig_bar = viz.create_top_words_bar_chart(top_20)
        st.plotly_chart(fig_bar, use_container_width=False)

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.2rem;">📋</span><h4>Word Frequency</h4></div>',
            unsafe_allow_html=True)

        # Custom styled word frequency table
        freq_html = "<table class='custom-table'><thead><tr>"
        freq_html += "<th style='width: 70%;'>💬 Word</th><th style='width: 30%;'>🔢 Count</th>"
        freq_html += "</tr></thead><tbody>"

        for idx, (word, count) in enumerate(top_20.items()):
            # Gradient colors for visual interest
            opacity = 1 - (idx * 0.03)
            freq_html += f"<tr style='background: rgba(255, 158, 196, {opacity * 0.1});'>"
            freq_html += f"<td><strong>{word}</strong></td>"
            freq_html += f"<td><span style='background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%); "
            freq_html += f"color: white; padding: 0.3rem 0.8rem; border-radius: 15px; "
            freq_html += f"font-weight: 600; display: inline-block;'>{count}</span></td>"
            freq_html += "</tr>"

        freq_html += "</tbody></table>"
        st.markdown(freq_html, unsafe_allow_html=True)

# TAB 5: Chatbot
with tab5:
    st.markdown(
        '<div class="section-header"><span style="font-size: 2rem;">🤖</span><h2>Live Fashion Review Analyzer</h2></div>',
        unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #ff9ec4 0%, #ff7eb3 100%); 
                padding: 2.5rem; 
                border-radius: 25px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 35px rgba(255, 158, 196, 0.35);
                text-align: center;
                border: 3px solid rgba(255, 255, 255, 0.3);'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>👗✨</div>
        <h3 style='margin:0; font-family: "Playfair Display", serif; font-size: 2rem;'>Try Our Fashion Sentiment Analyzer!</h3>
        <p style='margin:1rem 0 0 0; font-family: "Poppins", sans-serif; font-size: 1.15rem; font-weight: 500;'>
            Enter any fashion review to analyze customer sentiment in real-time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "✏️ Enter a fashion review:",
            height=120,
            placeholder="e.g., 'The dress fits perfectly! The fabric is amazing and the color is exactly as shown...'",
            help="Type or paste a customer review about clothing, shoes, or accessories"
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    with col2:
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.2rem;">💡</span><h4>Try Example Reviews</h4></div>',
            unsafe_allow_html=True)

        fashion_examples = [
            ("The dress fits perfectly! Amazing quality and fast shipping. Love the fabric!", "😊"),
            ("Terrible quality. The sweater fell apart after one wash. Very disappointed.", "😞"),
            ("The jeans are okay for the price. Nothing special but they fit fine.", "😐")
        ]

        for i, (example, emoji) in enumerate(fashion_examples):
            if st.button(f"{emoji} Example {i + 1}", key=f"ex_{i}", use_container_width=True):
                user_input = example
                analyze_button = True

    # Process input
    if analyze_button and user_input.strip():
        cleaned = preprocessor.clean_text(user_input)

        if cleaned.strip():
            result = model.predict(cleaned)
            sentiment = result['prediction']

            # Add to history
            st.session_state.chat_history.insert(0, {
                'input': user_input,
                'sentiment': sentiment,
                'probabilities': result['probabilities'],
                'confidence': result['confidence']
            })

    # Display results
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown(
            '<div class="section-header"><span style="font-size: 1.5rem;">💬</span><h3>Analysis Results</h3></div>',
            unsafe_allow_html=True)
        latest = st.session_state.chat_history[0]

        # Improved gradient colors for sentiments
        sentiment_gradients = {
            'positive': 'linear-gradient(135deg, #6dbfa0 0%, #5aaa8c 100%)',
            'negative': 'linear-gradient(135deg, #ff8888 0%, #ff6b6b 100%)',
            'neutral': 'linear-gradient(135deg, #b399e6 0%, #9d7fd9 100%)'
        }

        st.markdown(f"""
        <div style='background: {sentiment_gradients[latest['sentiment']]}; 
                    padding: 2rem; 
                    border-radius: 20px; 
                    color: white; 
                    margin-bottom: 1.5rem;
                    box-shadow: 0 8px 25px rgba(139, 79, 117, 0.2);
                    border: 2px solid rgba(255, 255, 255, 0.3);'>
            <h4 style='margin:0; font-family: "Playfair Display", serif; font-size: 1.5rem;'>👗 Your Fashion Review:</h4>
            <p style='margin:1rem 0 0 0; font-size: 1.15rem; font-family: "Poppins", sans-serif; line-height: 1.6;'>"{latest['input']}"</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            sentiment_icons = {"positive": "😊", "negative": "😞", "neutral": "😐"}

            st.markdown(f"""
            <div style='background: {sentiment_gradients[latest['sentiment']]}; 
                        padding: 3rem; 
                        border-radius: 25px; 
                        text-align: center; 
                        color: white;
                        box-shadow: 0 10px 30px rgba(139, 79, 117, 0.25);
                        border: 3px solid rgba(255, 255, 255, 0.3);'>
                <div style='font-size: 6rem; margin-bottom: 1.5rem;'>
                    {sentiment_icons[latest['sentiment']]}
                </div>
                <h2 style='margin:0; font-size: 2.5rem; font-family: "Playfair Display", serif; font-weight: 700;'>{latest['sentiment'].title()}</h2>
                <p style='margin:0.8rem 0 0 0; font-size: 1.2rem; font-family: "Poppins", sans-serif; font-weight: 600;'>Sentiment Detected</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Fashion-themed responses
            fashion_responses = {
                'positive': [
                    "👗 Wonderful! Your customers love this piece! Keep up the great work!",
                    "✨ Fantastic feedback! This item is clearly a customer favorite!",
                    "🌟 Excellent review! Happy customers mean successful fashion!"
                ],
                'negative': [
                    "😞 We understand the concern. This feedback helps improve our collection.",
                    "💙 Thank you for the honest feedback. We'll work on improving quality.",
                    "🙏 Your feedback is valuable for enhancing our fashion offerings."
                ],
                'neutral': [
                    "🔍 Thanks for the balanced review. Every opinion helps us improve!",
                    "⚖️ Noted! We appreciate your honest perspective on our fashion.",
                    "💭 Thank you for sharing. Constructive feedback shapes better products!"
                ]
            }

            response = np.random.choice(fashion_responses[latest['sentiment']])

            border_colors = {'positive': '#5aaa8c', 'negative': '#ff6b6b', 'neutral': '#9d7fd9'}

            st.markdown(f"""
            <div style='background: white; padding: 1.3rem; border-radius: 12px; border-left: 5px solid {border_colors[latest['sentiment']]}; box-shadow: 0 4px 15px rgba(139, 79, 117, 0.12);'>
                <p style='margin: 0; color: #5a3d52; font-family: "Poppins", sans-serif; font-size: 1rem; line-height: 1.6;'>{response}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div class="section-header"><span style="font-size: 1.2rem;">📊</span><h4>Confidence Breakdown</h4></div>',
                unsafe_allow_html=True)
            fig_prob = viz.create_probability_chart(latest['probabilities'])
            st.plotly_chart(fig_prob, use_container_width=True)

            confidence_label = "🔥 High" if latest['confidence'] > 0.7 else "⚡ Medium" if latest[
                                                                                             'confidence'] > 0.5 else "⚠️ Low"
            st.metric("🎯 Overall Confidence", f"{latest['confidence']:.1%}", confidence_label)

        # Chat history
        if len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown(
                '<div class="section-header"><span style="font-size: 1.5rem;">📜</span><h3>Previous Analyses</h3></div>',
                unsafe_allow_html=True)

            for idx, item in enumerate(st.session_state.chat_history[1:6]):
                sentiment_icons = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                with st.expander(
                        f"{sentiment_icons[item['sentiment']]} {item['sentiment'].title()} Review - \"{item['input'][:50]}...\"",
                        expanded=False
                ):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"""
                        <div style='font-family: "Poppins", sans-serif; color: #5a3d52; line-height: 1.6;'>
                            <strong>👗 Review:</strong> {item['input']}
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        st.metric("🎯 Confidence", f"{item['confidence']:.1%}")
    else:
        st.info("👆 Enter a fashion review above and click 'Analyze Review' to get started!")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-box">
                <div style='font-size: 2.5rem; margin-bottom: 0.8rem;'>🤖</div>
                <p style='margin: 0; color: #6d3d5f; font-weight: 700; font-size: 1.1rem;'>Model Type</p>
                <p style='margin: 0.5rem 0 0 0; color: #9d7a8f; font-size: 1rem;'>Logistic Regression</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-box">
                <div style='font-size: 2.5rem; margin-bottom: 0.8rem;'>📊</div>
                <p style='margin: 0; color: #6d3d5f; font-weight: 700; font-size: 1.1rem;'>Features</p>
                <p style='margin: 0.5rem 0 0 0; color: #9d7a8f; font-size: 1rem;'>TF-IDF</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div style='font-size: 2.5rem; margin-bottom: 0.8rem;'>👗</div>
                <p style='margin: 0; color: #6d3d5f; font-weight: 700; font-size: 1.1rem;'>Training Reviews</p>
                <p style='margin: 0.5rem 0 0 0; color: #9d7a8f; font-size: 1rem;'>{len(df):,}</p>
            </div>
            """, unsafe_allow_html=True)

# IMPROVED Footer with Fashion Theme
st.markdown("---")
precision_avg = summary['precision_avg']
recall_avg = summary['recall_avg']

st.markdown(f"""
<div style='text-align: center; padding: 2.5rem 1rem; background: linear-gradient(135deg, #ffffff 0%, #fff0f5 100%); border-radius: 25px; margin-top: 2rem; box-shadow: 0 8px 30px rgba(139, 79, 117, 0.15); border: 3px solid #ffe8f0;'>
    <div style='font-size: 3rem; margin-bottom: 1rem;'>👗✨</div>
    <p style='font-size: 1.5rem; color: #6d3d5f; margin: 0; font-family: "Playfair Display", serif; font-weight: 700;'>
        Fashion Review Analyzer Dashboard
    </p>
    <p style='color: #9d7a8f; margin: 1rem 0; font-family: "Poppins", sans-serif; font-size: 1.1rem; font-weight: 500;'>
        Powered by Machine Learning | Built with 💖 for Fashion
    </p>
    <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem; flex-wrap: wrap;'>
        <div style='background: white; padding: 1rem 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1); border: 2px solid #ffe8f0;'>
            <span style='color: #6d3d5f; font-weight: 700; font-size: 1rem;'>📊 Accuracy:</span>
            <span style='color: #9d7a8f; margin-left: 0.5rem; font-size: 1.1rem; font-weight: 600;'>{metrics['accuracy']:.1%}</span>
        </div>
        <div style='background: white; padding: 1rem 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1); border: 2px solid #ffe8f0;'>
            <span style='color: #6d3d5f; font-weight: 700; font-size: 1rem;'>📈 Precision:</span>
            <span style='color: #9d7a8f; margin-left: 0.5rem; font-size: 1.1rem; font-weight: 600;'>{precision_avg:.1%}</span>
        </div>
        <div style='background: white; padding: 1rem 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(139, 79, 117, 0.1); border: 2px solid #ffe8f0;'>
            <span style='color: #6d3d5f; font-weight: 700; font-size: 1rem;'>🎯 Recall:</span>
            <span style='color: #9d7a8f; margin-left: 0.5rem; font-size: 1.1rem; font-weight: 600;'>{recall_avg:.1%}</span>
        </div>
    </div>
    <p style='margin-top: 2rem; font-size: 0.95rem; color: #9d7a8f; font-family: "Poppins", sans-serif; font-weight: 500;'>
        🛍️ Streamlit • 🧠 NLTK • 📊 Scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)