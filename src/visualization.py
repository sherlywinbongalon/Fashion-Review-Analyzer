"""
visualization.py - Visualization utilities (UPDATED FOR PASTEL THEME - ALL FIXED)
"""
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    """Create visualizations with pastel fashion theme"""

    def __init__(self, color_map=None):
        """Initialize with pastel color scheme"""
        self.colors = color_map or {
            'positive': '#5aaa8c',
            'negative': '#ff6b6b',
            'neutral': '#9d7fd9'
        }

        # Pastel theme colors
        self.bg_color = '#ffffff'
        self.text_color = '#5a3d52'
        self.grid_color = '#ffe8f0'
        self.title_color = '#6d3d5f'

    def create_sentiment_pie_chart(self, sentiment_counts):
        """Create pastel-themed pie chart"""
        labels = [s.title() for s in sentiment_counts.index]
        values = sentiment_counts.values
        colors_list = [self.colors.get(s, '#cccccc') for s in sentiment_counts.index]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(
                colors=colors_list,
                line=dict(color='white', width=3)
            ),
            textfont=dict(
                size=14,
                color='white',
                family='Poppins, sans-serif'
            ),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            showlegend=False,
            legend=dict(
                font=dict(size=13, color=self.text_color, family='Poppins, sans-serif'),
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            height=400,
            width=1000,
            margin=dict(t=40, b=80, l=40, r=40)
        )

        return fig

    def create_word_length_boxplot(self, df):
        """Create pastel-themed boxplot"""
        fig = go.Figure()

        for sentiment in ['positive', 'negative', 'neutral']:
            data = df[df['sentiment'] == sentiment]['word_count']
            if len(data) > 0:
                fig.add_trace(go.Box(
                    y=data,
                    name=sentiment.title(),
                    marker=dict(
                        color=self.colors[sentiment],
                        line=dict(color=self.colors[sentiment], width=2)
                    ),
                    boxmean='sd',
                    hovertemplate='<b>%{y} words</b><extra></extra>'
                ))

        fig.update_layout(
            yaxis=dict(
                title=dict(
                    text='Number of Words',
                    font=dict(size=14, color=self.text_color, family='Poppins, sans-serif')
                ),
                tickfont=dict(size=12, color=self.text_color, family='Poppins, sans-serif'),
                gridcolor=self.grid_color,
                gridwidth=1
            ),
            xaxis=dict(
                tickfont=dict(size=12, color=self.text_color, family='Poppins, sans-serif')
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            showlegend=False,
            height=400,
            width=1400,
            margin=dict(t=40, b=60, l=60, r=40)
        )

        return fig

    def create_confusion_matrix(self, cm, labels):
        """Create pastel-themed confusion matrix using matplotlib"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Use pastel color palette
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='RdPu',  # Pastel pink colormap
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            linewidths=2,
            linecolor='white',
            ax=ax,
            annot_kws={'size': 14, 'weight': 'bold', 'color': '#5a3d52'}
        )

        # Styling
        ax.set_xlabel('Predicted', fontsize=14, color=self.text_color,
                      fontfamily='sans-serif', weight='bold')
        ax.set_ylabel('Actual', fontsize=14, color=self.text_color,
                      fontfamily='sans-serif', weight='bold')

        # Set tick colors
        ax.tick_params(colors=self.text_color, labelsize=11)

        # Set background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        plt.tight_layout()
        return fig

    def create_performance_bar_chart(self, report):
        """Create pastel-themed performance bar chart"""
        sentiments = []
        precisions = []
        recalls = []
        f1_scores = []

        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in report:
                sentiments.append(sentiment.title())
                precisions.append(report[sentiment]['precision'] * 100)
                recalls.append(report[sentiment]['recall'] * 100)
                f1_scores.append(report[sentiment]['f1-score'] * 100)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Precision',
            x=sentiments,
            y=precisions,
            marker=dict(
                color='#ff9ec4',
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}%' for v in precisions],
            textposition='outside',
            textfont=dict(size=12, color=self.text_color),
            hovertemplate='<b>Precision</b><br>%{y:.1f}%<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Recall',
            x=sentiments,
            y=recalls,
            marker=dict(
                color='#b399e6',
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}%' for v in recalls],
            textposition='outside',
            textfont=dict(size=12, color=self.text_color),
            hovertemplate='<b>Recall</b><br>%{y:.1f}%<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='F1-Score',
            x=sentiments,
            y=f1_scores,
            marker=dict(
                color='#6dbfa0',
                line=dict(color='white', width=2)
            ),
            text=[f'{v:.1f}%' for v in f1_scores],
            textposition='outside',
            textfont=dict(size=12, color=self.text_color),
            hovertemplate='<b>F1-Score</b><br>%{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            xaxis=dict(
                title=dict(
                    text='Sentiment',
                    font=dict(size=14, color=self.text_color, family='Poppins, sans-serif')
                ),
                tickfont=dict(size=12, color=self.text_color, family='Poppins, sans-serif')
            ),
            yaxis=dict(
                title=dict(
                    text='Score (%)',
                    font=dict(size=14, color=self.text_color, family='Poppins, sans-serif')
                ),
                tickfont=dict(size=12, color=self.text_color, family='Poppins, sans-serif'),
                gridcolor=self.grid_color,
                gridwidth=1,
                range=[0, 110]
            ),
            legend=dict(
                font=dict(size=12, color=self.text_color, family='Poppins, sans-serif'),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            height=450,
            width=2250,
            margin=dict(t=80, b=60, l=60, r=40)
        )

        return fig

    def create_wordcloud(self, text, colormap='RdPu', width=1200, height=600):
        """Create pastel-themed word cloud"""
        if not text or not text.strip():
            # Create empty figure if no text
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available',
                    ha='center', va='center', fontsize=20, color=self.text_color)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            return fig

        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap=colormap,  # Pastel colormaps: RdPu, PuRd, YlGnBu, RdYlGn
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            prefer_horizontal=0.7,
            font_path=None,
            contour_width=2,
            contour_color='white'
        ).generate(text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')

        # Set background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        plt.tight_layout(pad=0)
        return fig

    def create_top_words_bar_chart(self, word_freq_dict):
        """Create pastel-themed horizontal bar chart"""
        words = list(word_freq_dict.keys())
        counts = list(word_freq_dict.values())

        # Create gradient colors
        colors_gradient = [
            f'rgb({int(255 - i * 5)}, {int(158 - i * 3)}, {int(196 - i * 4)})'
            for i in range(len(words))
        ]

        fig = go.Figure(go.Bar(
            x=counts,
            y=words,
            orientation='h',
            marker=dict(
                color=colors_gradient,
                line=dict(color='white', width=2)
            ),
            text=counts,
            textposition='outside',
            textfont=dict(size=11, color=self.text_color),
            hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
        ))

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text='Frequency',
                    font=dict(size=14, color=self.text_color, family='Poppins, sans-serif')
                ),
                tickfont=dict(size=11, color=self.text_color, family='Poppins, sans-serif'),
                gridcolor=self.grid_color,
                gridwidth=1
            ),
            yaxis=dict(
                tickfont=dict(size=11, color=self.text_color, family='Poppins, sans-serif'),
                autorange='reversed'
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            height=1450,
            width=1400,
            margin=dict(t=40, b=60, l=100, r=60)
        )

        return fig

    def create_probability_chart(self, probabilities):
        """Create pastel-themed probability bar chart"""
        sentiments = list(probabilities.keys())
        probs = [probabilities[s] * 100 for s in sentiments]

        colors_map = {
            'positive': '#5aaa8c',
            'negative': '#ff6b6b',
            'neutral': '#9d7fd9'
        }
        colors_list = [colors_map.get(s, '#cccccc') for s in sentiments]

        fig = go.Figure(go.Bar(
            x=[s.title() for s in sentiments],
            y=probs,
            marker=dict(
                color=colors_list,
                line=dict(color='white', width=3)
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='outside',
            textfont=dict(size=13, color=self.text_color),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            xaxis=dict(
                tickfont=dict(size=13, color=self.text_color, family='Poppins, sans-serif')
            ),
            yaxis=dict(
                title=dict(
                    text='Confidence (%)',
                    font=dict(size=13, color=self.text_color, family='Poppins, sans-serif')
                ),
                tickfont=dict(size=11, color=self.text_color, family='Poppins, sans-serif'),
                gridcolor=self.grid_color,
                gridwidth=1,
                range=[0, max(probs) * 1.2] if probs else [0, 100]
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            showlegend=False,
            height=350,
            margin=dict(t=40, b=60, l=60, r=40)
        )

        return fig