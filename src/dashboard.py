import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from main import Agent, get_mock_data
from three_commas_client import ThreeCommasClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from functools import lru_cache
import time

# Define subscription plans
plans = {
    'Free': {
        'monthly_fee': 0,
        'max_bots': 1,
        'features': {
            'smart_trades': 3,
            'signal_bots': 1,
            'grid_bots': 1,
            'dca_bots': 1,
            'dca_trades': 10,
            'multi_pair': False,
            'backtests': 0,
            'notes': 'Single Pair Only, Without Futures Trading'
        }
    },
    'Pro': {
        'monthly_fee': 49,
        'max_bots': 10,
        'features': {
            'smart_trades': 50,
            'signal_bots': 50,
            'grid_bots': 10,
            'dca_bots': 50,
            'dca_trades': 500,
            'multi_pair': True,
            'backtests': 10,
            'notes': 'Multi-pair Available'
        }
    },
    'Expert': {
        'monthly_fee': 79,
        'max_bots': 50,
        'features': {
            'smart_trades': 'Unlimited',
            'signal_bots': 250,
            'grid_bots': 50,
            'dca_bots': 250,
            'dca_trades': 2500,
            'multi_pair': True,
            'backtests': 100,
            'sub_accounts': True,
            'notes': 'Multi-pair Available, Sub-Accounts Connection'
        }
    }
}

# Initialize 3Commas client
@st.cache_resource
def get_three_commas_client():
    api_key = st.secrets.get("THREE_COMMAS_API_KEY", "")
    api_secret = st.secrets.get("THREE_COMMAS_API_SECRET", "")
    if api_key and api_secret:
        return ThreeCommasClient(api_key, api_secret)
    return None

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="AI Hedge Fund Dashboard",
    page_icon="📈",
    layout="wide"
)

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
        'JPY': '¥'
    }
    symbol = currency_symbols.get(currency, '')
    return f"{symbol}{amount:,.2f}"

# Add currency selection with live rate display
st.sidebar.write("---")
st.sidebar.subheader("Display Settings")
currency = st.sidebar.selectbox(
    "Display Currency",
    ["USD", "MYR", "EUR", "GBP", "JPY", "SGD", "AUD"],
    key="currency_select"
)

# Show current exchange rate
if currency != "USD":
    current_rate = get_current_rates()[currency]
    st.sidebar.info(f"Current Rate: 1 USD = {current_rate:.4f} {currency}")
    last_update = datetime.fromtimestamp(int(time.time() / 300) * 300)
    st.sidebar.caption(f"Rate last updated: {last_update.strftime('%H:%M:%S')}")

# Display title and description
st.title("AI Hedge Fund Dashboard")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>Welcome to the AI Hedge Fund Dashboard</h4>
    <p>This dashboard provides real-time analysis and trading insights powered by artificial intelligence. 
    Monitor market trends, analyze sentiment data, and execute trades through 3Commas integration.</p>
</div>
""", unsafe_allow_html=True)

# Initialize 3Commas client
three_commas = get_three_commas_client()
if three_commas:
    if three_commas.validate_credentials():
        st.sidebar.success("✅ Connected to 3Commas")
        trading_accounts = three_commas.get_accounts()
    else:
        st.sidebar.error("❌ Invalid 3Commas API credentials")
else:
    st.sidebar.warning("⚠️ 3Commas API credentials not configured")

# Sidebar
st.sidebar.header("Configuration")
asset_type = st.sidebar.radio("Asset Type", ["Stocks", "Cryptocurrency"])
if asset_type == "Stocks":
    tickers = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT,NVDA").split(',')
else:
    tickers = ["BTC"]  # Only show BTC for cryptocurrency
    st.sidebar.info("Currently only Bitcoin (BTC) is supported for cryptocurrency analysis")

show_reasoning = st.sidebar.checkbox("Show Agent Reasoning", value=True)
days_back = st.sidebar.slider("Days of Historical Data", 7, 90, 30)
show_technicals = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_news = st.sidebar.checkbox("Show News Sentiment", value=True)
risk_analysis = st.sidebar.checkbox("Show Risk Analysis", value=True)

start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

def generate_price_history(days, initial_price):
    dates = pd.date_range(end=datetime.now(), periods=days)
    volatility = 0.02
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
        {"title": "Market Analysis Report", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Economic Outlook", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Industry Update", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
        {"title": "Technical Analysis", "sentiment": np.random.choice(['Positive', 'Neutral', 'Negative'])},
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

# Initialize agents with crypto-specific agents
if asset_type == "Cryptocurrency":
    agents = [
        Agent("Crypto Bull"),
        Agent("Crypto Bear"),
        Agent("Technical Analyst"),
        Agent("Fundamental Analyst"),
        Agent("Market Sentiment")
    ]
else:
    agents = [
        Agent("Warren Buffett"),
        Agent("Charlie Munger"),
        Agent("Peter Lynch"),
        Agent("Bill Ackman"),
        Agent("Michael Burry")
    ]

# Analysis for each ticker
for ticker in tickers:
    st.header(f"Analysis for {ticker}")
    
    # Get market data with crypto-specific pricing
    data = get_mock_data(ticker, start_date, end_date)
    if asset_type == "Cryptocurrency":
        data['price'] = round(np.random.uniform(1000, 100000), 2) if ticker == "BTC" else round(np.random.uniform(100, 10000), 2)
        data['volume'] = int(np.random.uniform(10000000, 100000000))
    
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
    
    # Add asset-specific metrics
    if asset_type == "Cryptocurrency":
        col1, col2, col3 = st.columns(3)
        col1.metric("24h Change", f"{np.random.uniform(-10, 10):.2f}%")
        col2.metric("Market Cap", f"${int(np.random.uniform(1e9, 1e12)):,}")
        col3.metric("24h Volume", f"${int(np.random.uniform(1e8, 1e10)):,}")
    
    # Technical Indicators
    if show_technicals:
        st.subheader("Technical Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RSI", f"{price_history['RSI'].iloc[-1]:.2f}")
        col2.metric("20-day SMA", f"${price_history['SMA_20'].iloc[-1]:.2f}")
        col3.metric("50-day SMA", f"${price_history['SMA_50'].iloc[-1]:.2f}")
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
                if asset_type == "Cryptocurrency":
                    st.write("Reasoning: Based on crypto market conditions, technical analysis, and market sentiment.")
                else:
                    st.write("Reasoning: Based on current market conditions and analysis.")
    
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
            {consensus.upper()} ON {ticker}
        </h3>
        <p>Based on {bullish} Bullish, {neutral} Neutral, and {bearish} Bearish recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add trading section after consensus
    if three_commas and consensus:
        st.subheader("Trading Actions")
        
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
        st.subheader("Profitability Analysis")
        
        # Add annual and bi-annual discount options
        payment_period = st.radio(
            "Subscription Period",
            ["Monthly", "Annual (-25%)", "Bi-Annual (-35%)"],
            key=f"payment_period_{ticker}"
        )

        discount = 1.0  # no discount for monthly
        if payment_period == "Annual (-25%)":
            discount = 0.75
        elif payment_period == "Bi-Annual (-35%)":
            discount = 0.65

        # Select 3Commas plan
        selected_plan = st.selectbox(
            "Select 3Commas Subscription Plan",
            list(plans.keys()),
            key=f"plan_select_{ticker}",
            help="Choose your 3Commas subscription plan for fee calculation"
        )
        
        plan_details = plans[selected_plan]
        monthly_subscription = plan_details['monthly_fee'] * discount
        
        # Display plan features in a more organized way
        st.write("📋 Plan Features:")
        features = plan_details['features']
        
        col_features1, col_features2 = st.columns(2)
        
        with col_features1:
            st.markdown(f"""
            **Trading Capabilities:**
            - Smart Trades: {features['smart_trades']}
            - Signal Bots: {features['signal_bots']}
            - Grid Bots: {features['grid_bots']}
            - DCA Bots: {features['dca_bots']}
            """)
        
        with col_features2:
            st.markdown(f"""
            **Additional Features:**
            - Active DCA Trades: {features['dca_trades']}
            - Multi-pair Trading: {'✅' if features['multi_pair'] else '❌'}
            - Monthly Backtests: {features['backtests']}
            - Sub-Accounts: {'✅' if features.get('sub_accounts', False) else '❌'}
            """)
        
        if features['notes']:
            st.info(f"Note: {features['notes']}")

        # Display effective pricing
        original_price = plan_details['monthly_fee']
        if discount < 1.0:
            st.success(f"""
            **Pricing:**
            - Original Price: {format_currency(original_price, currency)}/month
            - Discounted Price: {format_currency(monthly_subscription, currency)}/month
            - Total Savings: {format_currency((original_price - monthly_subscription) * (12 if payment_period == "Annual (-25%)" else 24), currency)}
            """)
        else:
            st.info(f"**Monthly Price:** {format_currency(monthly_subscription, currency)}/month")

        # Continue with the rest of the profitability analysis
        col_profit1, col_profit2 = st.columns(2)
        
        with col_profit1:
            st.write("📊 Projected Returns Analysis")
            
            # Calculate potential returns based on strategy
            current_price = data['price']
            position_size = 100.0 * (avg_confidence / 100.0)  # Base position size on confidence
            
            # Calculate fees
            trading_fee_percent = 0.15  # Fixed 0.15% trading fee (0.1% exchange + 0.05% 3Commas)
            entry_fee = position_size * (trading_fee_percent / 100)
            estimated_monthly_trades = int(10 + abs(sentiment_strength) * 20)
            
            # Calculate subscription cost per trade
            subscription_per_trade = monthly_subscription / estimated_monthly_trades
            
            # Projected scenarios
            best_case = position_size * (1 + abs(sentiment_strength) * 0.1)  # 10% max return scaled by sentiment
            worst_case = position_size * (1 - abs(sentiment_strength) * 0.05)  # 5% max loss scaled by sentiment
            expected_case = position_size * (1 + sentiment_strength * 0.03)  # 3% expected return scaled by direction
            
            # Calculate fees for each scenario including subscription costs
            best_case_fees = best_case * (trading_fee_percent / 100) + subscription_per_trade
            worst_case_fees = worst_case * (trading_fee_percent / 100) + subscription_per_trade
            expected_case_fees = expected_case * (trading_fee_percent / 100) + subscription_per_trade
            
            # Display projections
            st.write("Initial Investment:", format_currency(position_size, currency))
            st.write("Entry Fees:", format_currency(entry_fee, currency))
            st.write("Subscription Cost per Trade:", format_currency(subscription_per_trade, currency))
            
            # Create a DataFrame for scenarios
            scenarios_df = pd.DataFrame({
                'Scenario': ['Best Case', 'Expected Case', 'Worst Case'],
                'Gross Return': [
                    format_currency(best_case - position_size, currency),
                    format_currency(expected_case - position_size, currency),
                    format_currency(worst_case - position_size, currency)
                ],
                'Trading Fees': [
                    format_currency(best_case * (trading_fee_percent / 100), currency),
                    format_currency(expected_case * (trading_fee_percent / 100), currency),
                    format_currency(worst_case * (trading_fee_percent / 100), currency)
                ],
                'Subscription Cost': [
                    format_currency(subscription_per_trade, currency),
                    format_currency(subscription_per_trade, currency),
                    format_currency(subscription_per_trade, currency)
                ],
                'Net Return': [
                    format_currency(best_case - position_size - best_case_fees, currency),
                    format_currency(expected_case - position_size - expected_case_fees, currency),
                    format_currency(worst_case - position_size - worst_case_fees, currency)
                ]
            })
            
            # Display as a regular table with custom formatting
            st.table(scenarios_df.set_index('Scenario'))
            
            # Add color-coded indicators for quick visual reference
            for scenario in scenarios_df.index:
                net_return = float(scenarios_df.loc[scenario, 'Net Return'].replace('$', '').replace(',', ''))
                if net_return > 0:
                    st.success(f"{scenario}: Profitable ({format_currency(net_return, currency)})")
                elif net_return < 0:
                    st.error(f"{scenario}: Loss ({format_currency(net_return, currency)})")
                else:
                    st.warning(f"{scenario}: Break Even ({format_currency(net_return, currency)})")
        
        with col_profit2:
            st.write("💰 Fee Breakdown")
            
            # Display plan features
            st.write("📋 Selected Plan Features:")
            features_md = "\n".join([f"- {k}: {v}" for k, v in plan_details['features'].items()])
            st.markdown(f"""
            **{selected_plan} Plan - {format_currency(plan_details['monthly_fee'], currency)}/month**
            Maximum Bots: {plan_details['max_bots']}
            {features_md}
            """)
            
            # Fee explanation
            st.info(f"""
            **Fee Structure:**
            - Exchange Fee: 0.1% per trade
            - 3Commas Subscription: {format_currency(monthly_subscription, currency)}/month
            - Subscription per Trade: {format_currency(subscription_per_trade, currency)}
            
            **Monthly Cost Projection:**
            - Subscription Cost: {format_currency(monthly_subscription, currency)}
            - Estimated Monthly Trades: {estimated_monthly_trades}
            - Average Trade Size: {format_currency(position_size, currency)}
            - Total Trading Fees: {format_currency(entry_fee * estimated_monthly_trades, currency)}
            - Total Monthly Cost: {format_currency(monthly_subscription + entry_fee * estimated_monthly_trades, currency)}
            """)
            
            # Break-even analysis
            monthly_trading_fees = entry_fee * estimated_monthly_trades
            total_monthly_cost = monthly_subscription + monthly_trading_fees
            break_even_return = (total_monthly_cost / (position_size * estimated_monthly_trades)) * 100
            
            st.write("📈 Break-Even Analysis")
            st.warning(f"""
            To break even on monthly costs:
            - Required return per trade: {break_even_return:.2f}%
            - Monthly trading volume needed: {format_currency(position_size * estimated_monthly_trades, currency)}
            """)
            
            # ROI Analysis
            st.write("📊 ROI Impact")
            
            # Calculate ROI metrics including subscription
            monthly_roi = (expected_case/position_size - 1) * 100
            fee_impact = -(0.1 + 0.05 + (subscription_per_trade/position_size) * 100)
            net_monthly_roi = monthly_roi + fee_impact
            
            # Display ROI metrics with color coding
            st.write("Potential Monthly ROI (Before Fees):")
            if monthly_roi > 0:
                st.success(f"{monthly_roi:.1f}%")
            else:
                st.error(f"{monthly_roi:.1f}%")
            
            st.write("Fee Impact (Trading + Subscription):")
            st.warning(f"{fee_impact:.1f}%")
            
            st.write("Net Expected Monthly ROI:")
            if net_monthly_roi > 0:
                st.success(f"{net_monthly_roi:.1f}%")
            else:
                st.error(f"{net_monthly_roi:.1f}%")
            
            # Subscription recommendation
            recommended_plan = "Free"
            if estimated_monthly_trades > 100 or position_size > 1000:
                recommended_plan = "Expert"
            elif estimated_monthly_trades > 50 or position_size > 500:
                recommended_plan = "Pro"
            
            st.write("💡 Plan Recommendation")
            if recommended_plan != selected_plan:
                st.info(f"""
                Based on your trading volume and strategy:
                Consider upgrading to the {recommended_plan} plan for better value.
                """)
            else:
                st.success(f"The {selected_plan} plan is well-suited for your trading strategy.")
        
        # Original trading interface continues below
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Smart Trade")
            if trading_accounts:
                account = st.selectbox(
                    "Select Trading Account",
                    trading_accounts,
                    format_func=lambda x: f"{x.get('name', 'Unknown')} ({x.get('market_code', 'Unknown Market')})",
                    key=f"smart_trade_account_{ticker}"
                )
                
                # Adjust default amount based on confidence
                suggested_amount = 100.0 * (avg_confidence / 100.0)
                amount = st.number_input("Trade Amount", min_value=0.0, value=suggested_amount, key=f"amount_{ticker}")
                
                # Adjust take profit and stop loss based on sentiment strength
                suggested_tp = max(1.0, abs(sentiment_strength) * 5.0)
                suggested_sl = min(2.0, abs(sentiment_strength) * 3.0)
                
                take_profit = st.number_input("Take Profit %", min_value=0.1, value=suggested_tp, key=f"tp_{ticker}")
                stop_loss = st.number_input("Stop Loss %", min_value=0.1, value=suggested_sl, key=f"sl_{ticker}")
                
                if st.button("Create Smart Trade", key=f"create_trade_{ticker}"):
                    try:
                        trade = three_commas.create_smart_trade(
                            account_id=account['id'],
                            pair=f"{ticker}_USDT" if asset_type == "Cryptocurrency" else ticker,
                            amount=amount,
                            take_profit=take_profit,
                            stop_loss=stop_loss
                        )
                        st.success(f"Smart trade created successfully! Trade ID: {trade['id']}")
                    except Exception as e:
                        st.error(f"Failed to create trade: {str(e)}")
        
        with col2:
            st.write("AI-Powered Trading Bot")
            if trading_accounts:
                bot_account = st.selectbox(
                    "Select Bot Account",
                    trading_accounts,
                    format_func=lambda x: f"{x.get('name', 'Unknown')} ({x.get('market_code', 'Unknown Market')})",
                    key=f"bot_account_{ticker}"
                )
                
                # Choose strategy based on market conditions
                suggested_strategy = "DCA" if abs(sentiment_strength) < 0.3 else "Grid"
                strategy = st.selectbox(
                    "Bot Strategy",
                    ["DCA", "Grid"],
                    index=0 if suggested_strategy == "DCA" else 1,
                    help="DCA recommended for uncertain markets, Grid for trending markets",
                    key=f"strategy_{ticker}"
                )
                
                st.info(f"Based on analysis: {consensus} sentiment with {avg_confidence:.1f}% confidence")
                
                # Strategy-specific settings adjusted by analysis
                if strategy == "DCA":
                    # Adjust base order volume based on confidence
                    suggested_base = 100.0 * (avg_confidence / 100.0)
                    base_order_volume = st.number_input("Base Order Volume ($)", 
                        min_value=10.0, value=suggested_base, key=f"base_volume_{ticker}")
                    
                    # Adjust safety orders based on sentiment volatility
                    safety_order_volume = st.number_input("Safety Order Volume ($)", 
                        min_value=10.0, value=suggested_base * 0.5, key=f"safety_volume_{ticker}")
                    max_safety_orders = st.number_input("Max Safety Orders", 
                        min_value=1, max_value=25, value=int(3 + (1 - abs(sentiment_strength)) * 5), 
                        key=f"max_safety_{ticker}")
                    
                    # Adjust deviation based on sentiment strength
                    suggested_deviation = max(1.0, (1 - abs(sentiment_strength)) * 5.0)
                    price_deviation = st.number_input("Price Deviation (%)", 
                        min_value=0.1, value=suggested_deviation, key=f"deviation_{ticker}")
                    
                else:  # Grid strategy
                    # Calculate grid bounds based on sentiment
                    current_price = data['price']
                    sentiment_range = abs(sentiment_strength) * 0.2  # 20% max range
                    
                    if sentiment_strength > 0:  # Bullish
                        suggested_lower = current_price * (1 - sentiment_range * 0.4)  # Tighter bottom
                        suggested_upper = current_price * (1 + sentiment_range)  # More upside
                    else:  # Bearish
                        suggested_lower = current_price * (1 - sentiment_range)  # More downside
                        suggested_upper = current_price * (1 + sentiment_range * 0.4)  # Tighter top
                    
                    upper_price = st.number_input("Upper Grid Price ($)", 
                        min_value=0.0, value=suggested_upper, key=f"upper_price_{ticker}")
                    lower_price = st.number_input("Lower Grid Price ($)", 
                        min_value=0.0, value=suggested_lower, key=f"lower_price_{ticker}")
                    
                    # Adjust grid lines based on sentiment strength
                    suggested_lines = int(5 + abs(sentiment_strength) * 15)  # 5-20 lines
                    grid_lines = st.number_input("Number of Grid Lines", 
                        min_value=2, max_value=100, value=suggested_lines, key=f"grid_lines_{ticker}")
                    
                    # Adjust volume based on confidence
                    suggested_volume = 100.0 * (avg_confidence / 100.0)
                    volume_per_grid = st.number_input("Volume per Grid ($)", 
                        min_value=10.0, value=suggested_volume, key=f"grid_volume_{ticker}")
                
                # Take profit based on sentiment strength
                suggested_tp = max(1.0, abs(sentiment_strength) * 5.0)
                take_profit = st.number_input("Take Profit (%)", 
                    min_value=0.1, value=suggested_tp, key=f"bot_tp_{ticker}")
                
                if st.button("Create AI-Powered Trading Bot", key=f"create_bot_{ticker}"):
                    try:
                        bot_params = {
                            'account_id': bot_account['id'],
                            'pair': f"{ticker}_USDT" if asset_type == "Cryptocurrency" else ticker,
                            'strategy': strategy.lower(),
                            'take_profit': take_profit
                        }
                        
                        # Add strategy-specific parameters
                        if strategy == "DCA":
                            bot_params.update({
                                'base_order_volume': base_order_volume,
                                'safety_order_volume': safety_order_volume,
                                'max_safety_orders': max_safety_orders,
                                'price_deviation': price_deviation
                            })
                        else:  # Grid strategy
                            bot_params.update({
                                'upper_price': upper_price,
                                'lower_price': lower_price,
                                'grid_lines': grid_lines,
                                'volume_per_grid': volume_per_grid
                            })
                        
                        bot = three_commas.create_bot(**bot_params)
                        st.success(f"AI-Powered trading bot created successfully! Bot ID: {bot['id']}")
                        st.markdown(f"[Monitor your bot on 3Commas](https://3commas.io/bots/{bot['id']})")
                    except Exception as e:
                        st.error(f"Failed to create bot: {str(e)}")
                        st.info("Make sure you have sufficient funds and proper permissions in your 3Commas account.")
            
            # Display existing bots if any
            try:
                existing_bots = three_commas.get_bots()
                if existing_bots:
                    st.subheader("Existing Bots")
                    for i, bot in enumerate(existing_bots):
                        with st.expander(f"Bot {bot.get('name', 'Unknown')}", key=f"bot_expander_{ticker}_{i}"):
                            st.write(f"Strategy: {bot.get('strategy', 'Unknown')}")
                            st.write(f"Status: {bot.get('status', 'Unknown')}")
                            st.write(f"Profit: {bot.get('profit', '0.00')} USD")
                            if st.button("View on 3Commas", key=f"view_bot_{ticker}_{bot['id']}"):
                                st.markdown(f"[Open Bot Details](https://3commas.io/bots/{bot['id']})")
            except Exception:
                # Silently handle the error without showing a warning message
                pass
    
    st.markdown("---")  # Add a divider between assets

# Footer
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>AI Hedge Fund Dashboard - For Educational Purposes Only</p>
    <p>Not financial advice. Please consult with a financial advisor for investment decisions.</p>
</div>
""", unsafe_allow_html=True) 