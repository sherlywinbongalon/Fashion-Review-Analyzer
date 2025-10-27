"""
preprocessing.py - Module for text preprocessing
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


class TextPreprocessor:
    """Handle all text preprocessing operations"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self._download_nltk_data()

    @staticmethod
    def _download_nltk_data():
        """Download required NLTK data"""
        required_packages = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]

        for path, package in required_packages:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)

    def clean_text(self, text: str) -> str:
        """Clean and preprocess a single text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)

        # Check if text is empty after cleaning
        if not text.strip():
            return ""

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return " ".join(cleaned_tokens)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Preprocess all texts in a DataFrame"""
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.clean_text)

        # Calculate additional features
        df['word_count'] = df[text_column].str.split().str.len()
        df['char_count'] = df[text_column].str.len()

        return df

    def get_tokens(self, text: str) -> list:
        """Get tokens from cleaned text"""
        return word_tokenize(text)

    def get_word_frequency(self, texts: list, top_n: int = 20) -> dict:
        """Get most frequent words from a list of texts"""
        all_text = " ".join(texts)
        tokens = word_tokenize(all_text)
        freq_dist = nltk.FreqDist(tokens)
        return dict(freq_dist.most_common(top_n))