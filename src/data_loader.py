"""
data_loader.py - Module for loading and managing review data
"""
import pandas as pd
from pathlib import Path


class DataLoader:
    """Handle loading of review data from various sources"""

    def __init__(self, data_path: str = "data/reviews.csv"):
        self.data_path = Path(data_path)

    def load_from_csv(self) -> pd.DataFrame:
        """Load reviews from CSV file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)

        # Validate required columns
        required_cols = ['text', 'sentiment']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        return df

    def load_from_txt(self, txt_path: str) -> pd.DataFrame:
        """
        Load reviews from structured TXT file
        Expected format:
        SENTIMENT: positive
        REVIEW: This is a great product...
        ---
        """
        reviews = []
        sentiments = []

        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by separator
        entries = content.split('---')

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            lines = entry.split('\n')
            sentiment = None
            review = None

            for line in lines:
                if line.startswith('SENTIMENT:'):
                    sentiment = line.replace('SENTIMENT:', '').strip().lower()
                elif line.startswith('REVIEW:'):
                    review = line.replace('REVIEW:', '').strip()

            if sentiment and review:
                sentiments.append(sentiment)
                reviews.append(review)

        return pd.DataFrame({'text': reviews, 'sentiment': sentiments})

    def create_sample_csv(self, output_path: str = "data/reviews.csv"):
        """Create a sample CSV file with reviews"""
        data = {
            'text': [
                "This is the best online clothing store! My order arrived in two days.",
                "I am absolutely in love with the sweater I bought.",
                "Five stars! The customer service team was incredibly helpful.",
                "This is the worst store I have ever shopped at.",
                "The quality is absolutely terrible. Complete scam.",
                "Do not buy from here! The sizing is a joke.",
                "The dress is nice, but the color is much darker than the picture.",
                "The quality of the t-shirt is okay for the price.",
                "Shipping took longer than expected, almost three weeks."
            ],
            'sentiment': ['positive', 'positive', 'positive', 'negative', 'negative', 'negative', 'neutral', 'neutral',
                          'neutral']
        }

        df = pd.DataFrame(data)

        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"Sample CSV created at: {output_path}")
        return df

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to CSV"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to: {output_path}")