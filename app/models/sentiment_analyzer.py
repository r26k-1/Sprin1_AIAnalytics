from transformers import pipeline
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        # Load the sentiment-analysis pipeline
        self.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


    def analyze_with_transformers(self, text):
        # Perform sentiment analysis using Hugging Face Transformers
        results = self.model(text)
        return results

    def analyze_with_textblob(self, text):
        # Perform sentiment analysis using TextBlob
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
