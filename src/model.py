"""
model.py - Module for sentiment analysis model with overfitting detection
"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class SentimentModel:
    """Handle model training and prediction with overfitting detection"""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.metrics = {}
        self.X_train_tfidf = None
        self.X_test_tfidf = None
        self.y_train = None
        self.y_test = None

    def train(self, X, y):
        """Train the sentiment analysis model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Store for later use
        self.y_train = y_train
        self.y_test = y_test

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )

        # Transform text to TF-IDF features
        self.X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.X_test_tfidf = self.vectorizer.transform(X_test)

        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
        self.model.fit(self.X_train_tfidf, y_train)

        # Make predictions
        y_pred_train = self.model.predict(self.X_train_tfidf)
        y_pred_test = self.model.predict(self.X_test_tfidf)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        self.metrics = {
            'train_accuracy': train_accuracy,
            'accuracy': test_accuracy,
            'accuracy_gap': train_accuracy - test_accuracy,
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'y_test': y_test,
            'y_pred': y_pred_test
        }

        return self.metrics

    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        # Transform text
        text_tfidf = self.vectorizer.transform([text])

        # Predict
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]

        # Get class names in order
        classes = self.model.classes_

        return {
            'prediction': prediction,
            'probabilities': {
                classes[i]: float(probabilities[i])
                for i in range(len(classes))
            },
            'confidence': float(max(probabilities))
        }

    def diagnose_overfitting(self) -> dict:
        """
        Diagnose if the model is overfitting or underfitting

        Returns:
            dict with diagnosis information including:
            - status: 'overfitting', 'underfitting', or 'good_fit'
            - train_accuracy: training set accuracy
            - test_accuracy: test set accuracy
            - gap: difference between train and test
            - recommendation: what to do to fix the issue
        """
        if not self.metrics:
            raise ValueError("Model not trained. Call train() first.")

        train_acc = self.metrics['train_accuracy']
        test_acc = self.metrics['accuracy']
        gap = self.metrics['accuracy_gap']

        # Determine status
        if gap > 0.15:
            status = 'overfitting'
            recommendation = """
            🛠️ Recommendations to fix OVERFITTING:
            1. Increase regularization: Use smaller C value (e.g., C=0.1)
            2. Reduce max_features in TfidfVectorizer (e.g., 2000 instead of 5000)
            3. Collect more training data
            4. Use cross-validation
            5. Apply feature selection to remove noisy features
            """
        elif test_acc < 0.70:
            status = 'underfitting'
            recommendation = """
            🛠️ Recommendations to fix UNDERFITTING:
            1. Increase max_features (e.g., 10000 instead of 5000)
            2. Add trigrams: ngram_range=(1, 3)
            3. Try more complex models (Random Forest, Neural Networks)
            4. Add more features (sentiment lexicons, embeddings)
            5. Reduce regularization: Use larger C value (e.g., C=10.0)
            6. Train longer: increase max_iter
            """
        else:
            status = 'good_fit'
            recommendation = """
            ✅ Model is well-balanced!
            - Training and test accuracies are close
            - Model generalizes well to unseen data
            - Continue monitoring with new data
            """

        return {
            'status': status,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap,
            'recommendation': recommendation
        }

    def cross_validate(self, X, y, cv: int = 5) -> dict:
        """
        Perform cross-validation to get more reliable performance estimates

        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds

        Returns:
            dict with cross-validation scores
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        # Transform all data
        X_tfidf = self.vectorizer.transform(X)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_tfidf, y, cv=cv, scoring='accuracy')

        return {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'min_cv_score': cv_scores.min(),
            'max_cv_score': cv_scores.max(),
            'interpretation': self._interpret_cv_scores(cv_scores)
        }

    def _interpret_cv_scores(self, cv_scores) -> str:
        """Interpret cross-validation scores"""
        mean = cv_scores.mean()
        std = cv_scores.std()

        if std > 0.1:
            variance_note = "⚠️ High variance detected - possible overfitting"
        else:
            variance_note = "✅ Low variance - model is stable"

        if mean < 0.70:
            performance_note = "⚠️ Low average score - possible underfitting"
        else:
            performance_note = "✅ Good average performance"

        return f"{variance_note}\n{performance_note}"

    def get_per_class_performance(self) -> dict:
        """
        Analyze performance for each sentiment class

        Returns:
            dict with per-class metrics and interpretation
        """
        if not self.metrics:
            raise ValueError("Model not trained. Call train() first.")

        report = self.metrics['classification_report']

        class_metrics = {}
        f1_scores = []

        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in report:
                class_metrics[sentiment] = {
                    'precision': report[sentiment]['precision'],
                    'recall': report[sentiment]['recall'],
                    'f1-score': report[sentiment]['f1-score'],
                    'support': report[sentiment]['support']
                }
                f1_scores.append(report[sentiment]['f1-score'])

        # Check variance in F1-scores
        f1_variance = np.var(f1_scores)
        f1_range = max(f1_scores) - min(f1_scores)

        if f1_range > 0.2:
            balance_note = "⚠️ Large variance in class performance - model may be overfitting to some classes"
        else:
            balance_note = "✅ Balanced performance across classes"

        return {
            'class_metrics': class_metrics,
            'f1_variance': f1_variance,
            'f1_range': f1_range,
            'interpretation': balance_note
        }

    def get_feature_importance(self, top_n: int = 20, sentiment: str = 'positive'):
        """Get most important features for a sentiment"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        feature_names = self.vectorizer.get_feature_names_out()

        # Get class index
        sentiment_idx = list(self.model.classes_).index(sentiment)

        # Get coefficients
        coefficients = self.model.coef_[sentiment_idx]

        # Get top features
        top_indices = np.argsort(coefficients)[-top_n:][::-1]

        return {
            feature_names[i]: float(coefficients[i])
            for i in top_indices
        }

    def get_metrics_summary(self) -> dict:
        """Get a summary of model metrics"""
        if not self.metrics:
            raise ValueError("Model not trained. Call train() first.")

        report = self.metrics['classification_report']

        return {
            'train_accuracy': self.metrics.get('train_accuracy', 0),
            'accuracy': self.metrics['accuracy'],
            'accuracy_gap': self.metrics.get('accuracy_gap', 0),
            'precision_avg': np.mean([
                report[label]['precision']
                for label in ['positive', 'negative', 'neutral']
                if label in report
            ]),
            'recall_avg': np.mean([
                report[label]['recall']
                for label in ['positive', 'negative', 'neutral']
                if label in report
            ]),
            'f1_avg': np.mean([
                report[label]['f1-score']
                for label in ['positive', 'negative', 'neutral']
                if label in report
            ])
        }