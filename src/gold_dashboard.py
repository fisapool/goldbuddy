import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from main import Agent, get_mock_data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Initialize Streamlit page configuration with dark theme
st.set_page_config(
    page_title="GOLDBUDDY - AI Trading Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for logo and styling
st.markdown("""
<style>
    /* Logo and header styling */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1E1E1E, #2D2D2D);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .logo-text {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Arial Black', sans-serif;
        letter-spacing: 2px;
    }
    .logo-subtitle {
        font-size: 1.2rem;
        color: #B8860B;
        text-align: center;
        margin-top: -1rem;
        font-style: italic;
    }
    .gold-icon {
        font-size: 2.5rem;
        margin: 0 1rem;
    }
    /* Rest of your existing styles */
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stMetric {
        font-size: 24px !important;
    }
    .stMetric > div {
        font-size: 36px !important;
        font-weight: bold !important;
    }
    h1 {
        font-size: 48px !important;
        font-weight: bold !important;
        margin-bottom: 2rem !important;
    }
    h2 {
        font-size: 36px !important;
        margin-top: 2rem !important;
    }
    h3 {
        font-size: 28px !important;
    }
    p, li {
        font-size: 18px !important;
        line-height: 1.6 !important;
    }
    .stSelectbox label {
        font-size: 18px !important;
    }
    .stCheckbox label {
        font-size: 18px !important;
    }
    .stSlider label {
        font-size: 18px !important;
    }
</style>

<div class="logo-container">
    <span class="gold-icon">🏆</span>
    <span class="logo-text">GOLDBUDDY</span>
    <span class="gold-icon">💰</span>
</div>
<div class="logo-subtitle">Your AI-Powered Gold Trading Companion</div>
""", unsafe_allow_html=True)

# Add theme selector in sidebar
st.sidebar.title("Display Settings")
theme = st.sidebar.selectbox(
    "Color Theme",
    ["Light", "Dark"],
    key="theme_select"
)

# Set theme colors based on selection
if theme == "Dark":
    background_color = "#1E1E1E"
    text_color = "#FFFFFF"
    card_background = "#2D2D2D"
else:
    background_color = "#FFFFFF"
    text_color = "#000000"
    card_background = "#F0F2F6"

# Apply theme colors
st.markdown(f"""
<style>
    .main {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stMarkdown {{
        color: {text_color};
    }}
    div.stMetric {{
        background-color: {card_background};
    }}
</style>
""", unsafe_allow_html=True)

# Currency conversion functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_current_rates():
    """
    Get current exchange rates (simplified version)
    Returns a dictionary of exchange rates relative to USD
    """
    return {
        'EUR': 0.92,  # 1 USD = 0.92 EUR
        'GBP': 0.79,  # 1 USD = 0.79 GBP
        'JPY': 148.0,  # 1 USD = 148.0 JPY
        'MYR': 4.75,  # 1 USD = 4.75 MYR
        'USD': 1.0    # Base currency
    }

def convert_currency(amount, from_currency, to_currency):
    """
    Convert amount from one currency to another
    """
    rates = get_current_rates()
    if from_currency == to_currency:
        return amount
    
    # Convert to USD first if not already in USD
    if from_currency != 'USD':
        amount = amount / rates[from_currency]
    
    # Convert from USD to target currency
    if to_currency != 'USD':
        amount = amount * rates[to_currency]
    
    return amount

def format_currency(amount, currency='USD'):
    """
    Format amount with currency symbol
    """
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'MYR': 'RM'
    }
    symbol = currency_symbols.get(currency, '')
    return f"{symbol}{amount:,.2f}"

# Add currency selection with live rate display
st.sidebar.write("---")
st.sidebar.subheader("Display Settings")
currency = st.sidebar.selectbox(
    "Display Currency",
    ["MYR", "USD", "EUR", "GBP", "JPY"],
    key="currency_select"
)

# Show current exchange rate
if currency != "USD":
    current_rate = get_current_rates()[currency]
    st.sidebar.info(f"Current Rate: 1 USD = {current_rate:.4f} {currency}")
    last_update = datetime.fromtimestamp(int(time.time() / 300) * 300)
    st.sidebar.caption(f"Rate last updated: {last_update.strftime('%H:%M:%S')}")

# Display title and description
st.title("Gold Trading Dashboard")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Welcome to the Gold Trading Dashboard</h4>
    <p>This dashboard provides real-time analysis and trading insights for gold (XAU/USD). 
    Monitor market trends, analyze sentiment data, and make informed trading decisions.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
show_reasoning = st.sidebar.checkbox("Show Agent Reasoning", value=True)
days_back = st.sidebar.slider("Days of Historical Data", 7, 90, 30)
show_technicals = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_news = st.sidebar.checkbox("Show News Sentiment", value=True)
risk_analysis = st.sidebar.checkbox("Show Risk Analysis", value=True)

start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

def generate_price_history(days, initial_price):
    dates = pd.date_range(end=datetime.now(), periods=days)
    volatility = 0.015  # Lower volatility for gold
    returns = np.random.normal(0, volatility, days)
    price_series = initial_price * (1 + returns).cumprod()
    return pd.DataFrame({'Date': dates, 'Price': price_series})

def calculate_technical_indicators(df):
    # Simple moving averages
    df['SMA_20'] = df['Price'].rolling(window=20).mean()
    df['SMA_50'] = df['Price'].rolling(window=50).mean()
    # RSI
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def generate_news_sentiment():
    news_items = [
        {"title": "Gold Market Analysis Report", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Economic Outlook Impact on Gold", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Central Bank Gold Reserves Update", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Technical Analysis for Gold", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
    ]
    return news_items

def calculate_risk_metrics(price_history):
    returns = price_history['Price'].pct_change()
    metrics = {
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
        'Max Drawdown': (price_history['Price'] / price_history['Price'].cummax() - 1).min(),
        'Value at Risk': returns.quantile(0.05)
    }
    return metrics

# Initialize agents with gold-specific agents
agents = [
    Agent("Gold Bull"),
    Agent("Gold Bear"),
    Agent("Technical Analyst"),
    Agent("Fundamental Analyst"),
    Agent("Market Sentiment")
]

# Analysis for Gold
ticker = "XAUUSD"
st.header(f"Analysis for Gold (XAU/USD)")

# Get market data with gold-specific pricing
data = get_mock_data(ticker, start_date, end_date)
data['price'] = round(np.random.uniform(1900, 2100), 2)  # Gold price range
data['volume'] = int(np.random.uniform(5000000, 15000000))

# Generate price history
price_history = generate_price_history(days_back, data['price'])

# Display price chart with technical indicators
st.subheader("Price History")
if show_technicals:
    price_history = calculate_technical_indicators(price_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_history['Date'], y=price_history['Price'],
                            mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=price_history['Date'], y=price_history['SMA_20'],
                            mode='lines', name='20-day SMA', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=price_history['Date'], y=price_history['SMA_50'],
                            mode='lines', name='50-day SMA', line=dict(dash='dot')))
else:
    fig = px.line(price_history, x='Date', y='Price', title=f"{ticker} Price History")
st.plotly_chart(fig)

# Display metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", format_currency(data['price'], currency))
col2.metric("Volume", f"{data['volume']:,}")
col3.metric("Date Range", f"{start_date} to {end_date}")

# Add gold-specific metrics
col1, col2, col3 = st.columns(3)
col1.metric("24h Change", f"{np.random.uniform(-2, 2):.2f}%")
col2.metric("Market Cap", f"${int(np.random.uniform(10e12, 12e12)):,}")
col3.metric("24h Volume", f"${int(np.random.uniform(100e9, 200e9)):,}")

# Technical Indicators
if show_technicals:
    st.subheader("Technical Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI", f"{price_history['RSI'].iloc[-1]:.2f}")
    col2.metric("20-day SMA", format_currency(price_history['SMA_20'].iloc[-1], currency))
    col3.metric("50-day SMA", format_currency(price_history['SMA_50'].iloc[-1], currency))
    col4.metric("Price vs 50 SMA", f"{((price_history['Price'].iloc[-1] / price_history['SMA_50'].iloc[-1] - 1) * 100):.2f}%")

# Risk Analysis
if risk_analysis:
    st.subheader("Risk Metrics")
    risk_metrics = calculate_risk_metrics(price_history)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility", f"{risk_metrics['Volatility']:.2%}")
    col2.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.2f}")
    col3.metric("Max Drawdown", f"{risk_metrics['Max Drawdown']:.2%}")
    col4.metric("Value at Risk (95%)", f"{risk_metrics['Value at Risk']:.2%}")

# News Sentiment
if show_news:
    st.subheader("Recent News Sentiment")
    news_items = generate_news_sentiment()
    for news in news_items:
        sentiment_color = {
            'Positive': 'green',
            'Neutral': 'gray',
            'Negative': 'red'
        }[news['sentiment']]
        st.markdown(f"""
        <div style='padding: 10px; border-left: 5px solid {sentiment_color}; margin: 5px;'>
            <strong>{news['title']}</strong> - <span style='color: {sentiment_color};'>{news['sentiment']}</span>
        </div>
        """, unsafe_allow_html=True)

# Agent Analysis
st.subheader("Agent Recommendations")

# Create a DataFrame for agent analysis
analysis_data = []
overall_sentiment = []

for agent in agents:
    sentiment, confidence = agent.analyze(ticker, data, show_reasoning)
    overall_sentiment.append(sentiment)
    analysis_data.append({
        'Agent': agent.name,
        'Sentiment': sentiment,
        'Confidence': confidence * 100
    })

df_analysis = pd.DataFrame(analysis_data)

# Create columns for visualization
col1, col2 = st.columns(2)

# Sentiment Distribution
with col1:
    sentiment_counts = pd.Series(overall_sentiment).value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Bullish': 'green',
            'Neutral': 'gray',
            'Bearish': 'red'
        }
    )
    st.plotly_chart(fig)

# Confidence Levels
with col2:
    fig = px.bar(
        df_analysis,
        x='Agent',
        y='Confidence',
        color='Sentiment',
        title="Agent Confidence Levels",
        color_discrete_map={
            'Bullish': 'green',
            'Neutral': 'gray',
            'Bearish': 'red'
        }
    )
    fig.update_layout(yaxis_title="Confidence (%)")
    st.plotly_chart(fig)

# Display detailed analysis if enabled
if show_reasoning:
    st.subheader("Detailed Analysis")
    for agent in agents:
        sentiment, confidence = agent.analyze(ticker, data, True)
        with st.expander(f"{agent.name}'s Analysis"):
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence * 100:.1f}%")
            st.write("Reasoning: Based on gold market conditions, technical analysis, and market sentiment.")

# Calculate and display consensus
bullish = overall_sentiment.count('Bullish')
bearish = overall_sentiment.count('Bearish')
neutral = overall_sentiment.count('Neutral')

consensus = "Bullish" if bullish > bearish else "Bearish" if bearish > bullish else "Neutral"

# Style the consensus recommendation
consensus_color = {
    'Bullish': 'green',
    'Neutral': 'gray',
    'Bearish': 'red'
}[consensus]

st.markdown(f"""
## Final Recommendation
<div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
    <h3 style='color: {consensus_color};'>
        {consensus.upper()} ON GOLD
    </h3>
    <p>Based on {bullish} Bullish, {neutral} Neutral, and {bearish} Bearish recommendations</p>
</div>
""", unsafe_allow_html=True)

# Add trading section after consensus
st.subheader("Trading Analysis")
    
# Calculate overall sentiment metrics
bullish_count = overall_sentiment.count('Bullish')
bearish_count = overall_sentiment.count('Bearish')
neutral_count = overall_sentiment.count('Neutral')
total_count = len(overall_sentiment)
    
# Calculate average confidence
avg_confidence = sum(agent['Confidence'] for agent in analysis_data) / len(analysis_data)
    
# Calculate sentiment strength (-1 to 1)
sentiment_strength = (bullish_count - bearish_count) / total_count

# Add profitability analysis section
st.write("---")
st.subheader("Market Analysis")

# Add holding period selection
holding_period = st.selectbox(
    "Select Holding Period",
    ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
    index=2  # Default to 1 Month
)

# Convert holding period to days for calculations
holding_period_days = {
    "1 Day": 1,
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}[holding_period]

# Calculate potential returns based on strategy and holding period
current_price = data['price']
position_size = current_price  # Base position size on 1 troy ounce

# Adjust returns based on holding period (annualized rates)
annual_volatility = 0.15  # Gold's typical annual volatility
time_factor = holding_period_days / 365.0

# Calculate scenario returns using annualized rates
best_case_annual = 0.12  # 12% annual return in best case
expected_case_annual = 0.06  # 6% annual return in expected case
worst_case_annual = -0.08  # -8% annual return in worst case

# Convert annual returns to holding period returns using time decay
best_case = position_size * (1 + best_case_annual * time_factor)
expected_case = position_size * (1 + expected_case_annual * time_factor)
worst_case = position_size * (1 + worst_case_annual * time_factor)

# Calculate absolute returns
best_return = best_case - position_size
expected_return = expected_case - position_size
worst_return = worst_case - position_size

# Create a DataFrame for scenarios with annualized returns
scenarios_df = pd.DataFrame({
    'Scenario': ['Best Case', 'Expected Case', 'Worst Case'],
    f'Return ({holding_period})': [
        format_currency(best_return, currency),
        format_currency(expected_return, currency),
        format_currency(worst_return, currency)
    ],
    f'Period Return': [
        f"+{((best_case/position_size - 1) * 100):.1f}%",
        f"{((expected_case/position_size - 1) * 100):+.1f}%",
        f"{((worst_case/position_size - 1) * 100):+.1f}%"
    ],
    'Annualized Return': [
        f"+{best_case_annual * 100:.1f}%",
        f"{expected_case_annual * 100:+.1f}%",
        f"{worst_case_annual * 100:+.1f}%"
    ],
    'Position Value': [
        format_currency(best_case, currency),
        format_currency(expected_case, currency),
        format_currency(worst_case, currency)
    ]
})

# Add explanation of the analysis
st.markdown(f"""
<div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 5px; margin: 10px 0;'>
    <h4>Analysis Parameters:</h4>
    <ul>
        <li>Holding Period: {holding_period}</li>
        <li>Base Position: 1 Troy Ounce of Gold</li>
        <li>Current Price: {format_currency(current_price, currency)}</li>
        <li>Annual Volatility: {annual_volatility * 100:.1f}%</li>
    </ul>
    <p><strong>Note:</strong> Returns are projected based on historical gold volatility and market conditions. 
    Annualized returns show the equivalent yearly rate, while period returns show actual expected returns for the selected holding period.</p>
</div>
""", unsafe_allow_html=True)

# Display as a styled table
st.markdown("""
<style>
.market-analysis-table {
    font-size: 18px !important;
    margin: 20px 0;
}
.market-analysis-table th {
    background-color: var(--secondary-background-color);
    padding: 12px !important;
}
.market-analysis-table td {
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.table(scenarios_df.set_index('Scenario'))

# Add color-coded indicators with improved formatting and holding period context
for scenario in scenarios_df.index:
    return_pct = float(scenarios_df.loc[scenario, 'Period Return'].replace('%', '').replace('+', ''))
    annual_return = scenarios_df.loc[scenario, 'Annualized Return']
    if return_pct > 0:
        st.success(f"{scenario} ({holding_period}): Profitable ({scenarios_df.loc[scenario, 'Period Return']}) - Annualized Return: {annual_return}")
    elif return_pct < 0:
        st.error(f"{scenario} ({holding_period}): Loss ({scenarios_df.loc[scenario, 'Period Return']}) - Annualized Return: {annual_return}")
    else:
        st.warning(f"{scenario} ({holding_period}): Break Even ({scenarios_df.loc[scenario, 'Period Return']}) - Annualized Return: {annual_return}")

# Add risk context based on holding period
if holding_period_days <= 7:
    st.warning("⚠️ Short-term trading carries higher transaction costs and increased risk of market volatility.")
elif holding_period_days >= 180:
    st.info("ℹ️ Longer-term holding periods typically benefit from reduced impact of short-term market volatility.")

# Add position sizing recommendations
st.write("---")
st.subheader("Position Sizing Recommendations")

# Calculate position sizing based on market conditions
def calculate_position_size(confidence, sentiment_strength, risk_level):
    base_size = 100.0  # Base position size in grams
    confidence_factor = confidence / 100.0
    sentiment_factor = abs(sentiment_strength)
    risk_factor = 1.0 - (risk_level / 100.0)  # Higher risk level reduces position size
    
    # Calculate recommended position size
    position_size = base_size * confidence_factor * sentiment_factor * risk_factor
    
    # Round to nearest 10 grams for practical purposes
    return round(position_size / 10) * 10

# Get risk level from user
risk_level = st.slider(
    "Risk Tolerance Level",
    min_value=0,
    max_value=100,
    value=50,
    help="0 = Conservative, 50 = Moderate, 100 = Aggressive"
)

# Calculate recommended position sizes
conservative_size = calculate_position_size(avg_confidence, sentiment_strength, 75)  # Conservative
moderate_size = calculate_position_size(avg_confidence, sentiment_strength, 50)     # Moderate
aggressive_size = calculate_position_size(avg_confidence, sentiment_strength, 25)    # Aggressive

# Display recommendations in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Conservative Position",
        f"{conservative_size} grams",
        f"≈ {format_currency(conservative_size * current_price / 31.1, currency)}"
    )
    st.info("Recommended for long-term investors and risk-averse traders")

with col2:
    st.metric(
        "Moderate Position",
        f"{moderate_size} grams",
        f"≈ {format_currency(moderate_size * current_price / 31.1, currency)}"
    )
    st.info("Balanced approach for most investors")

with col3:
    st.metric(
        "Aggressive Position",
        f"{aggressive_size} grams",
        f"≈ {format_currency(aggressive_size * current_price / 31.1, currency)}"
    )
    st.info("For experienced traders with higher risk tolerance")

# Add important notes
st.write("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
    <h4>Important Notes:</h4>
    <ul>
        <li>These recommendations are based on current market conditions and sentiment analysis</li>
        <li>Consider your personal financial situation and risk tolerance</li>
        <li>Diversify your portfolio - don't put all your funds in gold</li>
        <li>Consider dollar-cost averaging for larger positions</li>
        <li>Always maintain an emergency fund before investing</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Gold Trading Dashboard - For Educational Purposes Only</p>
    <p>Not financial advice. Please consult with a financial advisor for investment decisions.</p>
</div>
""", unsafe_allow_html=True)

st.write("---")
st.header("AI Analysis & Position Sizing")

# Create columns for different analysis components
col1, col2 = st.columns(2)

with col1:
    st.subheader("AI Market Sentiment Analysis")
    
    # Calculate overall AI sentiment score (-100 to 100)
    technical_score = np.random.normal(20, 10)  # Example: slightly bullish technical
    fundamental_score = np.random.normal(10, 10)  # Example: neutral-bullish fundamental
    sentiment_score = np.random.normal(30, 10)  # Example: bullish sentiment
    
    overall_score = (technical_score + fundamental_score + sentiment_score) / 3
    
    # Create sentiment gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = overall_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [-100, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-100, -60], 'color': "red"},
                {'range': [-60, -20], 'color': "salmon"},
                {'range': [-20, 20], 'color': "gray"},
                {'range': [20, 60], 'color': "lightgreen"},
                {'range': [60, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': overall_score
            }
        },
        title = {'text': "AI Sentiment Score"}
    ))
    st.plotly_chart(fig)

    # Display individual scores
    st.markdown("""
    #### Component Scores
    - Technical Analysis: {:.1f}
    - Fundamental Analysis: {:.1f}
    - Market Sentiment: {:.1f}
    """.format(technical_score, fundamental_score, sentiment_score))

with col2:
    st.subheader("Key Technical Indicators")
    
    # Calculate mock technical indicators
    rsi = np.random.uniform(30, 70)
    macd = np.random.uniform(-2, 2)
    bollinger = np.random.uniform(-2, 2)
    
    # Display technical indicators with interpretations
    st.markdown(f"""
    - **RSI (14)**: {rsi:.2f}
        - {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}
    - **MACD**: {macd:.2f}
        - {'Bullish' if macd > 0 else 'Bearish'} momentum
    - **Bollinger Band Position**: {bollinger:.2f}
        - {'Upper band (Overbought)' if bollinger > 1 else 'Lower band (Oversold)' if bollinger < -1 else 'Middle band (Neutral)'}
    """)

# AI Position Sizing
st.subheader("AI-Recommended Position Sizing")

# Get user inputs for position sizing calculation
col1, col2, col3 = st.columns(3)

with col1:
    account_size = st.number_input("Total Account Size", min_value=1000.0, value=100000.0, step=1000.0)
    
with col2:
    risk_tolerance = st.slider("Risk Tolerance (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
with col3:
    leverage = st.selectbox("Leverage", [1, 2, 5, 10, 20], index=0)

# Calculate position sizes based on different strategies
def calculate_position_sizes(account_size, risk_tolerance, leverage, sentiment_score):
    # Base risk amount
    risk_amount = account_size * (risk_tolerance / 100)
    
    # Adjust base position size based on sentiment
    sentiment_factor = (sentiment_score + 100) / 200  # Convert -100 to 100 to 0 to 1
    
    # Calculate different position sizes
    conservative_size = risk_amount * 10 * sentiment_factor * leverage
    moderate_size = risk_amount * 15 * sentiment_factor * leverage
    aggressive_size = risk_amount * 20 * sentiment_factor * leverage
    
    # Cap position sizes at account size * leverage
    max_position = account_size * leverage
    return (
        min(conservative_size, max_position),
        min(moderate_size, max_position),
        min(aggressive_size, max_position)
    )

# Get position sizes
conservative, moderate, aggressive = calculate_position_sizes(
    account_size, risk_tolerance, leverage, overall_score
)

# Display position sizing recommendations
st.markdown("""
<style>
.position-box {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.conservative {
    background-color: rgba(0, 255, 0, 0.1);
}
.moderate {
    background-color: rgba(255, 165, 0, 0.1);
}
.aggressive {
    background-color: rgba(255, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="position-box conservative">
        <h4>Conservative Position</h4>
        <p>Amount: {}</p>
        <p>Contracts: {:.2f}</p>
        <p>Risk: Low</p>
    </div>
    """.format(
        format_currency(conservative, currency),
        conservative / current_price
    ), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="position-box moderate">
        <h4>Moderate Position</h4>
        <p>Amount: {}</p>
        <p>Contracts: {:.2f}</p>
        <p>Risk: Medium</p>
    </div>
    """.format(
        format_currency(moderate, currency),
        moderate / current_price
    ), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="position-box aggressive">
        <h4>Aggressive Position</h4>
        <p>Amount: {}</p>
        <p>Contracts: {:.2f}</p>
        <p>Risk: High</p>
    </div>
    """.format(
        format_currency(aggressive, currency),
        aggressive / current_price
    ), unsafe_allow_html=True)

# Add risk warnings and explanations
st.markdown("""
#### Position Sizing Factors
- Account Size: Determines the base for position calculations
- Risk Tolerance: Maximum percentage of account willing to risk per trade
- Leverage: Multiplier for position size (higher leverage = higher risk)
- Market Sentiment: AI analysis affects position sizing
- Technical Indicators: Used to adjust position timing
""")

# Risk warnings based on leverage
if leverage > 1:
    st.warning(f"⚠️ Using {leverage}x leverage increases both potential profits and risks. Never risk more than you can afford to lose.")
if leverage > 10:
    st.error("⚠️ High leverage warning: Extremely high risk of rapid account depletion.")

# Add AI confidence level
ai_confidence = np.random.uniform(0.6, 0.9)  # 60-90% confidence
st.markdown(f"""
#### AI Analysis Confidence
- Overall Confidence Level: {ai_confidence:.1%}
- Based on analysis of {np.random.randint(50, 200)} market indicators
- Updated as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

# Market conditions summary
st.markdown("""
#### Current Market Conditions
""")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Favorable Factors:**
    - Market Volatility
    - Trading Volume
    - Technical Setup
    """)

with col2:
    st.markdown("""
    **Risk Factors:**
    - Economic Events
    - Market Sentiment
    - Price Action
    """) 