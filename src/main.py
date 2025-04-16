import os
import argparse
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Agent:
    def __init__(self, name):
        self.name = name
    
    def analyze(self, ticker, data, show_reasoning=False):
        # Simulate analysis with some basic metrics
        sentiment = np.random.choice(['Bullish', 'Neutral', 'Bearish'])
        confidence = round(np.random.uniform(0.5, 1.0), 2)
        
        if show_reasoning:
            print(f"\n{self.name}'s Analysis for {ticker}:")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence * 100}%")
            print("Reasoning: Based on current market conditions and analysis.")
        
        return sentiment, confidence

def get_mock_data(ticker, start_date=None, end_date=None):
    # Simulate getting market data
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    return {
        'ticker': ticker,
        'price': round(np.random.uniform(100, 1000), 2),
        'volume': int(np.random.uniform(1000000, 10000000)),
        'start_date': start_date,
        'end_date': end_date
    }

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Hedge Fund')
    parser.add_argument('--ticker', type=str, required=True, help='Comma-separated list of tickers')
    parser.add_argument('--show-reasoning', action='store_true', help='Show agent reasoning')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Check for required API keys
    required_keys = ['OPENAI_API_KEY', 'GROQ_API_KEY', 'ANTHROPIC_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("Warning: The following API keys are missing:")
        for key in missing_keys:
            print(f"- {key}")
        print("\nPlease set these keys in your .env file")
        return
    
    # Initialize agents
    agents = [
        Agent("Warren Buffett"),
        Agent("Charlie Munger"),
        Agent("Peter Lynch"),
        Agent("Bill Ackman"),
        Agent("Michael Burry")
    ]
    
    # Print configuration
    print(f"\n=== AI Hedge Fund Analysis ===")
    print(f"Analyzing tickers: {args.ticker}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")
    if args.show_reasoning:
        print("Agent reasoning will be displayed")
    
    # Analyze each ticker
    tickers = [t.strip() for t in args.ticker.split(',')]
    
    for ticker in tickers:
        print(f"\n=== Analysis for {ticker} ===")
        
        # Get market data
        data = get_mock_data(ticker, args.start_date, args.end_date)
        print(f"Current Price: ${data['price']}")
        print(f"Volume: {data['volume']:,}")
        
        # Get agent analysis
        print("\nAgent Recommendations:")
        print("-" * 50)
        overall_sentiment = []
        
        for agent in agents:
            sentiment, confidence = agent.analyze(ticker, data, args.show_reasoning)
            overall_sentiment.append(sentiment)
            if not args.show_reasoning:
                print(f"{agent.name}: {sentiment} (Confidence: {confidence * 100}%)")
        
        # Calculate consensus
        bullish = overall_sentiment.count('Bullish')
        bearish = overall_sentiment.count('Bearish')
        neutral = overall_sentiment.count('Neutral')
        
        print("\nConsensus:")
        print(f"Bullish: {bullish} agents")
        print(f"Neutral: {neutral} agents")
        print(f"Bearish: {bearish} agents")
        
        consensus = "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutral"
        print(f"\nFinal Recommendation for {ticker}: {consensus}")

if __name__ == "__main__":
    main() 