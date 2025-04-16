import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Agent:
    def __init__(self, name):
        self.name = name
    
    def analyze(self, ticker, data, show_reasoning=False):
        # Mock analysis with random but biased sentiment
        sentiment = np.random.choice(['Bullish', 'Neutral', 'Bearish'], 
                                   p=[0.4, 0.3, 0.3])
        confidence = np.random.uniform(0.6, 0.95)
        
        if show_reasoning:
            print(f"\n{self.name}'s Analysis for {ticker}:")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence * 100}%")
            print("Reasoning: Based on current market conditions and analysis.")
        
        return sentiment, confidence

def get_mock_data(ticker, start_date, end_date):
    """Generate mock market data for demonstration"""
    return {
        'price': np.random.uniform(1900, 2100),  # Gold price range
        'volume': int(np.random.uniform(5000000, 15000000)),
        'start_date': start_date,
        'end_date': end_date
    } 