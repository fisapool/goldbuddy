# GOLDBUDDY - AI Trading Dashboard 🏆

GOLDBUDDY is an advanced AI-powered trading dashboard specifically designed for gold trading analysis. It combines real-time market data, technical analysis, sentiment analysis, and AI-driven insights to help traders make informed decisions.

## Features

### 📊 Market Analysis
- Real-time gold price tracking
- Historical price charts with customizable timeframes
- Volume analysis and market depth
- Multiple currency display options (USD, EUR, GBP, JPY, MYR)

### 📈 Technical Indicators
- Moving Averages (20-day and 50-day SMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Custom indicator overlays

### 🤖 AI-Powered Analysis
- Multi-agent sentiment analysis
- Market trend predictions
- Risk assessment
- Position sizing recommendations
- Confidence indicators for all data points

### 📰 News & Sentiment
- Real-time news aggregation
- Sentiment analysis of market news
- Social media trend analysis
- Impact assessment on gold prices

### 💹 Risk Management
- Dynamic position sizing calculator
- Risk/reward ratio analysis
- Portfolio exposure calculator
- Maximum drawdown analysis
- Value at Risk (VaR) calculations

### 🎯 Trading Tools
- Position size calculator
- Risk tolerance assessment
- Leverage optimization
- Entry/exit point suggestions
- Stop-loss calculator

### 🌐 Social Sharing
- Quick screenshot capture guides
- Direct sharing to Twitter/X and LinkedIn
- Screenshot best practices
- Security-conscious sharing tips
- Professional presentation guidelines

## Data Confidence System
The dashboard features a comprehensive confidence indicator system:
- High confidence (≥80%): Strong reliability
- Medium confidence (60-79%): Moderate reliability
- Low confidence (<60%): Exercise caution
- Real-time data freshness indicators
- Source transparency for all data points

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fisapool/goldbuddy.git
cd goldbuddy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/gold_dashboard.py
```

2. Access the dashboard at `http://localhost:8501`

## Configuration

### Environment Variables
Create a `.env` file with the following variables:
```
API_KEY=your_api_key_here
ENVIRONMENT=development
DEBUG=True
```

### Streamlit Configuration
The `.streamlit/config.toml` file contains UI customization:
- Theme settings
- Server configurations
- Browser settings

## Dependencies

- Python 3.8+
- streamlit==1.31.0
- plotly==5.18.0
- pandas==2.2.0
- numpy==1.26.0
- python-dotenv==1.0.0

## Development

### Project Structure
```
goldbuddy/
├── src/
│   ├── gold_dashboard.py
│   └── main.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- API keys and sensitive data are stored in `.env`
- `.env` is listed in `.gitignore`
- Use `.env.example` for template configuration
- Never commit real API keys to the repository

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational and informational purposes only. Trading involves risk, and past performance is not indicative of future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

## Support

For support, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

## Acknowledgments

- Built with Streamlit
- Powered by Python
- AI analysis components
- Technical analysis libraries
- Financial data providers

---
Made with ❤️ by GOLDBUDDY Team 