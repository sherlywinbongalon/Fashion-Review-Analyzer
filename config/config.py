"""
config.py - Configuration settings for Fashion Review Dashboard (UPDATED COLORS)
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REVIEWS_CSV = DATA_DIR / "reviews.csv"
PREPROCESSED_CSV = DATA_DIR / "preprocessed_reviews.csv"

# Model settings
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 5000,
    'ngram_range': (1, 2)
}

# Streamlit page config - Fashion Theme
PAGE_CONFIG = {
    'page_title': "Fashion Review Analyzer",
    'page_icon': "👗",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# UPDATED: Sentiment colors - Darker for better visibility
SENTIMENT_CONFIG = {
    'positive': {
        'color': '#5aaa8c',  # Darker mint green
        'gradient': 'linear-gradient(135deg, #6dbfa0 0%, #5aaa8c 100%)',
        'emoji': '😊'
    },
    'negative': {
        'color': '#ff6b6b',  # Darker coral red
        'gradient': 'linear-gradient(135deg, #ff8888 0%, #ff6b6b 100%)',
        'emoji': '😞'
    },
    'neutral': {
        'color': '#9d7fd9',  # Darker lavender
        'gradient': 'linear-gradient(135deg, #b399e6 0%, #9d7fd9 100%)',
        'emoji': '😐'
    }
}

# Fashion-themed chat responses
CHAT_RESPONSES = {
    'positive': [
        "👗 Wonderful! Your customers love this piece! Keep up the great work! ✨",
        "🌟 Fantastic feedback! This item is clearly a customer favorite!",
        "💖 Excellent review! Happy customers mean successful fashion trends!"
    ],
    'negative': [
        "😞 We understand the concern. This feedback helps improve our collection.",
        "💙 Thank you for the honest feedback. We'll work on improving quality and fit.",
        "🙏 Your feedback is valuable for enhancing our fashion offerings. We're listening!"
    ],
    'neutral': [
        "🔍 Thanks for the balanced review. Every opinion helps us improve! 👗",
        "⚖️ Noted! We appreciate your honest perspective on our fashion pieces.",
        "💭 Thank you for sharing. Constructive feedback shapes better products!"
    ]
}

# Fashion-themed example texts
EXAMPLE_TEXTS = [
    ("The dress fits perfectly! Amazing quality and fast shipping. Love the fabric and color!", "👗"),
    ("Terrible quality. The sweater fell apart after one wash. Very disappointed with this purchase.", "😞"),
    ("The jeans are okay for the price. Nothing special but they fit fine and look decent.", "😐")
]

# WordCloud settings - UPDATED for pastel theme
WORDCLOUD_CONFIG = {
    'width': 1200,
    'height': 600,
    'background_color': 'white',
    'max_words': 100,
    'relative_scaling': 0.5,
    'min_font_size': 10,
    'colormap': 'RdPu'  # Pastel pink/purple colormap
}

# UPDATED: Visualization colors - Darker Pastel Fashion Palette
CHART_COLORS = {
    'positive': '#5aaa8c',    # Darker mint
    'negative': '#ff6b6b',    # Darker coral
    'neutral': '#9d7fd9',     # Darker lavender
    'primary': '#ff9ec4',     # Medium pink
    'secondary': '#ff7eb3',   # Deeper rose
    'background': '#ffffff',  # White background for charts
    'text': '#5a3d52',        # Darker text for readability
    'grid': '#ffe8f0'         # Light grid lines
}

# Fashion categories (optional - for future enhancements)
FASHION_CATEGORIES = {
    'tops': ['shirt', 'blouse', 'sweater', 'hoodie', 'tank', 'tee', 'cardigan'],
    'bottoms': ['jeans', 'pants', 'skirt', 'shorts', 'trousers', 'leggings'],
    'dresses': ['dress', 'gown', 'maxi', 'mini', 'midi'],
    'outerwear': ['jacket', 'coat', 'blazer', 'parka', 'vest'],
    'footwear': ['shoes', 'boots', 'sandals', 'sneakers', 'heels'],
    'accessories': ['bag', 'belt', 'scarf', 'hat', 'jewelry', 'necklace']
}

# Fashion-themed insights
FASHION_INSIGHTS = {
    'positive_keywords': ['perfect', 'love', 'amazing', 'beautiful', 'comfortable', 'stylish', 'quality'],
    'negative_keywords': ['terrible', 'cheap', 'disappointed', 'poor', 'worst', 'awful', 'uncomfortable'],
    'fit_keywords': ['fit', 'size', 'sizing', 'tight', 'loose', 'small', 'large', 'true to size'],
    'quality_keywords': ['quality', 'fabric', 'material', 'stitching', 'durable', 'lasted']
}