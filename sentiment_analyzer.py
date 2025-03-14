from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import seaborn as sns
from datetime import datetime
import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.conversation_history = []
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'pleased', 'glad', 'wonderful', 'great', 'good'],
            'anger': ['angry', 'furious', 'annoyed', 'irritated', 'mad', 'hate', 'frustrated', 'rage'],
            'sadness': ['sad', 'unhappy', 'depressed', 'disappointed', 'upset', 'miserable', 'hurt'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'frightened'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'wow', 'unexpected'],
            'neutral': ['okay', 'fine', 'alright', 'neutral', 'normal']
        }

    def analyze_text(self, text):
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # Emotion detection with improved negative emotion handling
        emotions = self.detect_emotions(text.lower())
        
        # Calculate mood score with more weight on VADER
        mood_score = (0.3 * textblob_sentiment + 0.7 * vader_scores['compound'])
        
        # Determine dominant emotion based on both keyword and sentiment
        if not emotions:  # If no emotion keywords found
            if mood_score <= -0.3:
                emotions['anger'] = 0.7
                emotions['sadness'] = 0.3
            elif mood_score >= 0.3:
                emotions['joy'] = 0.7
                emotions['neutral'] = 0.3
            else:
                emotions['neutral'] = 1.0
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Determine conversation tone
        tone = self.determine_tone(text, mood_score, subjectivity)

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'text': text,
            'textblob_sentiment': round(textblob_sentiment, 2),
            'vader_compound': round(vader_scores['compound'], 2),
            'vader_pos': round(vader_scores['pos'], 2),
            'vader_neg': round(vader_scores['neg'], 2),
            'vader_neu': round(vader_scores['neu'], 2),
            'subjectivity': round(subjectivity, 2),
            'mood_score': round(mood_score, 2),
            'dominant_emotion': dominant_emotion,
            'emotions': emotions,
            'tone': tone
        }

    def detect_emotions(self, text):
        emotions = defaultdict(float)
        words = text.split()
        
        # First pass: check for emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in words:
                    emotions[emotion] += 1
        
        # If no emotions detected, use sentiment to infer
        if not emotions:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            if sentiment <= -0.3:
                if any(word in text for word in ['hate', 'angry', 'mad']):
                    emotions['anger'] = 0.8
                else:
                    emotions['sadness'] = 0.8
            elif sentiment >= 0.3:
                emotions['joy'] = 0.8
            else:
                emotions['neutral'] = 1.0
        
        # Normalize emotions
        total = sum(emotions.values()) or 1
        return {k: round(v/total, 2) for k, v in emotions.items()}

    def determine_tone(self, text, mood_score, subjectivity):
        if mood_score > 0.5:
            base_tone = "Very Positive"
        elif mood_score > 0.1:
            base_tone = "Positive"
        elif mood_score < -0.5:
            base_tone = "Very Negative"
        elif mood_score < -0.1:
            base_tone = "Negative"
        else:
            base_tone = "Neutral"

        if subjectivity > 0.7:
            tone_style = "Highly Subjective"
        elif subjectivity < 0.3:
            tone_style = "Objective"
        else:
            tone_style = "Balanced"

        return f"{base_tone}, {tone_style}"

class SentimentVisualizer:
    def __init__(self, master):
        self.master = master
        
        # Define fixed colors for sentiment categories
        self.sentiment_colors = {
            'Positive': '#4CAF50',  # Green
            'Neutral': '#FFC107',   # Yellow
            'Negative': '#FF5252'   # Red
        }
        
        # Initialize matplotlib settings for better performance
        plt.style.use('dark_background')
        matplotlib.rcParams['figure.autolayout'] = True
        matplotlib.rcParams['figure.dpi'] = 100
        
        # Create and configure figures with fixed sizes
        self.sentiment_fig = Figure(figsize=(8, 4))
        self.sentiment_fig.patch.set_facecolor('#2B2B2B')
        
        self.emotion_fig = Figure(figsize=(8, 4))
        self.emotion_fig.patch.set_facecolor('#2B2B2B')
        
        # Initialize canvases as None
        self.sentiment_canvas = None
        self.emotion_canvas = None
        
    def create_sentiment_plot(self, history, frame):
        if not history:
            return
            
        try:
            # Clear the figure but keep it
            self.sentiment_fig.clear()
            
            # Create subplot with specific size
            ax = self.sentiment_fig.add_subplot(111)
            ax.set_facecolor('#1B1B1B')
            
            # Get last 10 entries for clearer visualization
            recent_history = history[-10:]
            timestamps = range(len(recent_history))
            mood_scores = [h['mood_score'] for h in recent_history]
            
            # Plot data with enhanced visibility
            line = ax.plot(timestamps, mood_scores, '-o', color='#4CAF50', linewidth=2, markersize=6)[0]
            
            # Add color regions for sentiment zones
            ax.axhspan(0.1, 1.0, color='#4CAF5044', alpha=0.3)
            ax.axhspan(-0.1, 0.1, color='#FFC10744', alpha=0.3)
            ax.axhspan(-1.0, -0.1, color='#FF525244', alpha=0.3)
            
            # Customize appearance
            ax.grid(True, alpha=0.3, color='white', linestyle='--')
            ax.set_title('Recent Mood Trend', color='white', pad=10, fontsize=12, weight='bold')
            ax.set_xlabel('Messages', color='white', fontsize=10)
            ax.set_ylabel('Mood Score', color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=9)
            
            # Set fixed axis limits
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(-0.5, len(recent_history) - 0.5)
            ax.set_xticks(timestamps)
            
            # Enhance visibility of spines
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(1.5)
            
            # Create or update canvas
            if self.sentiment_canvas is None:
                self.sentiment_canvas = FigureCanvasTkAgg(self.sentiment_fig, frame)
                self.sentiment_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
            else:
                self.sentiment_canvas.draw()
            
        except Exception as e:
            print(f"Error in create_sentiment_plot: {str(e)}")

    def create_emotion_distribution(self, history, frame):
        if not history:
            return
            
        try:
            # Clear the figure but keep it
            self.emotion_fig.clear()
            
            # Create subplot with specific size
            ax = self.emotion_fig.add_subplot(111)
            ax.set_facecolor('#1B1B1B')
            
            # Count sentiments from recent history
            recent_history = history[-10:]
            sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            
            for entry in recent_history:
                mood_score = entry['mood_score']
                if mood_score >= 0.1:
                    sentiment_counts['Positive'] += 1
                elif mood_score <= -0.1:
                    sentiment_counts['Negative'] += 1
                else:
                    sentiment_counts['Neutral'] += 1
            
            # Calculate percentages and create pie chart
            total = sum(sentiment_counts.values())
            if total > 0:
                values = []
                labels = []
                colors = []
                
                for label, count in sentiment_counts.items():
                    if count > 0:
                        values.append(count)
                        labels.append(f"{label}\n({count}/{total})")
                        colors.append(self.sentiment_colors[label])
                
                if values:
                    wedges, texts, autotexts = ax.pie(
                        values,
                        labels=labels,
                        colors=colors,
                        autopct='%1.1f%%',
                        textprops={'color': 'white', 'fontsize': 10, 'weight': 'bold'},
                        pctdistance=0.85,
                        startangle=90
                    )
                    
                    # Enhance text visibility
                    plt.setp(autotexts, size=9, weight="bold")
                    plt.setp(texts, size=9)
                    
                    ax.set_title('Sentiment Distribution', color='white', pad=10, fontsize=12, weight='bold')
                else:
                    ax.text(0.5, 0.5, 'No data available',
                           horizontalalignment='center',
                           verticalalignment='center',
                           color='white',
                           fontsize=10)
            
            # Create or update canvas
            if self.emotion_canvas is None:
                self.emotion_canvas = FigureCanvasTkAgg(self.emotion_fig, frame)
                self.emotion_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
            else:
                self.emotion_canvas.draw()
            
        except Exception as e:
            print(f"Error in create_emotion_distribution: {str(e)}")

    def create_tone_summary(self, history, frame):
        if not history:
            return

        try:
            # Clear existing widgets
            for widget in frame.winfo_children():
                widget.destroy()

            # Calculate metrics from recent history
            recent_history = history[-10:]
            avg_mood = np.mean([h['mood_score'] for h in recent_history])
            
            # Count sentiments
            sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            for entry in recent_history:
                mood_score = entry['mood_score']
                if mood_score >= 0.1:
                    sentiment_counts['Positive'] += 1
                elif mood_score <= -0.1:
                    sentiment_counts['Negative'] += 1
                else:
                    sentiment_counts['Neutral'] += 1
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
            
            # Create summary text
            summary = (
                f"Conversation Analysis (Last 10 Messages):\n\n"
                f"Overall Mood: {self.get_mood_description(avg_mood)}\n"
                f"Average Mood Score: {avg_mood:.2f}\n"
                f"Dominant Sentiment: {dominant_sentiment}\n"
                f"Positive Messages: {sentiment_counts['Positive']}\n"
                f"Neutral Messages: {sentiment_counts['Neutral']}\n"
                f"Negative Messages: {sentiment_counts['Negative']}"
            )
            
            # Create label with custom styling
            label = ctk.CTkLabel(
                frame,
                text=summary,
                font=("Helvetica", 12),
                justify="left",
                anchor="w",
                padx=10,
                pady=5
            )
            label.pack(fill="both", expand=True)
            
        except Exception as e:
            print(f"Error in create_tone_summary: {str(e)}")

    @staticmethod
    def get_mood_description(mood_score):
        if mood_score > 0.5:
            return "Very Positive"
        elif mood_score > 0.1:
            return "Positive"
        elif mood_score < -0.5:
            return "Very Negative"
        elif mood_score < -0.1:
            return "Negative"
        return "Neutral"

class ConversationAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.conversation_history = []
        self.mood_weights = {
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'cheerful'],
            'anger': ['angry', 'frustrated', 'annoyed', 'irritated', 'furious'],
            'sadness': ['sad', 'disappointed', 'unhappy', 'depressed', 'gloomy'],
            'anxiety': ['worried', 'nervous', 'anxious', 'tense', 'stressed'],
            'neutral': ['okay', 'fine', 'normal', 'regular', 'standard']
        }
        
    def analyze_text(self, text):
        """Perform comprehensive sentiment analysis on text"""
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # Determine overall sentiment
        compound_score = vader_scores['compound']
        if compound_score >= 0.05:
            sentiment_category = "Positive"
        elif compound_score <= -0.05:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"
            
        # Analyze mood
        mood = self.analyze_mood(text)
        
        # Get emotion intensity
        emotion_intensity = abs(compound_score)
        
        return {
            'timestamp': None,  # Will be set by caller
            'text': text,
            'sentiment': sentiment_category,
            'sentiment_score': round(compound_score, 2),
            'subjectivity': round(subjectivity, 2),
            'mood': mood,
            'emotion_intensity': round(emotion_intensity, 2),
            'detailed_scores': {
                'positive': round(vader_scores['pos'], 2),
                'negative': round(vader_scores['neg'], 2),
                'neutral': round(vader_scores['neu'], 2)
            }
        }
    
    def analyze_mood(self, text):
        """Analyze the overall mood of the text"""
        words = word_tokenize(text.lower())
        mood_scores = defaultdict(float)
        
        for mood, keywords in self.mood_weights.items():
            for word in words:
                if word in keywords:
                    mood_scores[mood] += 1
                    
        if not mood_scores:
            return "neutral"
        
        return max(mood_scores.items(), key=lambda x: x[1])[0]
    
    def generate_word_cloud(self, figure, texts):
        """Generate a word cloud from conversation texts"""
        if not texts:
            return None
            
        # Clear the figure
        figure.clear()
        
        # Combine all texts
        text = ' '.join(texts)
        
        # Create and generate a word cloud image
        wordcloud = WordCloud(
            width=400, 
            height=200,
            background_color='white',
            max_words=50
        ).generate(text)
        
        # Display the word cloud
        ax = figure.add_subplot(111)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        return figure
    
    def generate_sentiment_graph(self, figure, sentiment_history):
        """Generate a line graph of sentiment over time"""
        if not sentiment_history:
            return None
            
        # Clear the figure
        figure.clear()
        
        # Create sentiment plot
        ax = figure.add_subplot(111)
        sentiments = [x['sentiment_score'] for x in sentiment_history]
        ax.plot(sentiments, marker='o')
        ax.set_title('Sentiment Trend')
        ax.set_ylabel('Sentiment Score')
        ax.set_xlabel('Message Number')
        ax.grid(True)
        
        # Add horizontal lines for sentiment thresholds
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.05, color='red', linestyle='--', alpha=0.3)
        
        return figure
    
    def generate_emotion_distribution(self, figure, sentiment_history):
        """Generate a pie chart of emotion distribution"""
        if not sentiment_history:
            return None
            
        # Clear the figure
        figure.clear()
        
        # Count sentiments
        sentiment_counts = defaultdict(int)
        for entry in sentiment_history:
            sentiment_counts[entry['sentiment']] += 1
            
        # Create pie chart
        ax = figure.add_subplot(111)
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())
        colors = ['green', 'gray', 'red']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Emotion Distribution')
        
        return figure
    
    def get_conversation_summary(self, history):
        """Generate a comprehensive conversation summary"""
        if not history:
            return {
                'overall_mood': 'No conversation data',
                'avg_sentiment': 0,
                'avg_intensity': 0,
                'sentiment_distribution': {},
                'mood_distribution': {}
            }
            
        # Calculate averages
        avg_sentiment = np.mean([entry['sentiment_score'] for entry in history])
        avg_intensity = np.mean([entry['emotion_intensity'] for entry in history])
        
        # Count sentiments and moods
        sentiment_counts = defaultdict(int)
        mood_counts = defaultdict(int)
        
        for entry in history:
            sentiment_counts[entry['sentiment']] += 1
            mood_counts[entry['mood']] += 1
            
        # Determine overall mood
        overall_mood = max(mood_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'overall_mood': overall_mood.capitalize(),
            'avg_sentiment': round(avg_sentiment, 2),
            'avg_intensity': round(avg_intensity, 2),
            'sentiment_distribution': dict(sentiment_counts),
            'mood_distribution': dict(mood_counts)
        } 