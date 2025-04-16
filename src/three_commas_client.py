import hmac
import hashlib
import requests
import json
from typing import Dict, Optional, List
from datetime import datetime

class ThreeCommasClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.3commas.io"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def _generate_signature(self, path: str, data: str = "") -> str:
        """Generate HMAC SHA256 signature for 3Commas API authentication"""
        encoded_key = self.api_secret.encode('utf-8')
        message = (path + data).encode('utf-8')
        signature = hmac.new(encoded_key, message, hashlib.sha256).hexdigest()
        return signature

    def _make_request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None):
        """Make authenticated request to 3Commas API"""
        url = f"{self.base_url}{path}"
        
        # Prepare request data
        request_data = ""
        if data:
            request_data = json.dumps(data)
        elif params:
            request_data = json.dumps(params)

        # Generate signature
        signature = self._generate_signature(path, request_data)
        
        # Prepare headers
        headers = {
            'APIKEY': self.api_key,
            'Signature': signature,
            'Content-Type': 'application/json'
        }

        # Make request
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data
        )
        
        if response.status_code == 401:
            raise ValueError("Invalid API credentials")
        elif response.status_code == 403:
            raise ValueError("Insufficient permissions. Check your API key permissions.")
        
        response.raise_for_status()
        return response.json()

    def validate_credentials(self) -> bool:
        """Test API credentials validity"""
        try:
            response = self._make_request('GET', '/public/api/ver1/validate')
            return response.get('valid', False)
        except Exception as e:
            print(f"Error validating credentials: {e}")
            return False

    def get_accounts(self) -> List[Dict]:
        """Get all connected trading accounts"""
        return self._make_request('GET', '/public/api/ver1/accounts')

    def create_smart_trade(self, account_id: int, pair: str, amount: float, take_profit: float, stop_loss: float) -> Dict:
        """Create a new smart trade"""
        data = {
            'account_id': account_id,
            'pair': pair,
            'position': {
                'type': 'buy',
                'units': amount,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
        }
        return self._make_request('POST', '/public/api/ver1/smart_trades/create', data=data)

    def get_smart_trades(self, limit: int = 10) -> List[Dict]:
        """Get list of smart trades"""
        params = {'limit': limit}
        return self._make_request('GET', '/public/api/ver1/smart_trades', params=params)

    def create_bot(self, account_id: int, pair: str, strategy: str = 'dca', **kwargs) -> Dict:
        """
        Create a new trading bot with enhanced parameters
        
        Args:
            account_id: Trading account ID
            pair: Trading pair (e.g., 'BTC_USDT')
            strategy: Bot strategy ('dca' or 'grid')
            **kwargs: Additional strategy-specific parameters
                For DCA:
                    - base_order_volume: Initial order volume
                    - safety_order_volume: Safety order volume
                    - max_safety_orders: Maximum number of safety orders
                    - price_deviation: Price deviation percentage
                For Grid:
                    - upper_price: Upper grid price
                    - lower_price: Lower grid price
                    - grid_lines: Number of grid lines
                    - volume_per_grid: Volume per grid line
        """
        data = {
            'account_id': account_id,
            'pair': pair,
            'strategy': strategy,
            'name': f"{strategy.upper()} Bot - {pair}"
        }
        
        # Add strategy-specific parameters
        if strategy == 'dca':
            data.update({
                'base_order_volume': kwargs.get('base_order_volume', 100),
                'safety_order_volume': kwargs.get('safety_order_volume', 50),
                'max_safety_orders': kwargs.get('max_safety_orders', 3),
                'price_deviation': kwargs.get('price_deviation', 2.0),
                'take_profit': kwargs.get('take_profit', 2.0),
                'martingale_volume_coefficient': kwargs.get('martingale_coefficient', 1.5),
                'martingale_step_coefficient': kwargs.get('step_coefficient', 1.1),
                'active_safety_orders_count': kwargs.get('active_safety_orders', 1)
            })
        elif strategy == 'grid':
            data.update({
                'upper_price': kwargs.get('upper_price'),
                'lower_price': kwargs.get('lower_price'),
                'quantity_per_grid': kwargs.get('volume_per_grid', 100),
                'grids': kwargs.get('grid_lines', 10),
                'take_profit': kwargs.get('take_profit', 2.0)
            })
        
        return self._make_request('POST', '/public/api/ver1/bots/create', data=data)

    def get_bots(self, limit: int = 10) -> List[Dict]:
        """Get list of trading bots"""
        params = {'limit': limit}
        return self._make_request('GET', '/public/api/ver1/bots', params=params)

    def get_bot_deals(self, bot_id: int, limit: int = 10) -> List[Dict]:
        """Get deals for a specific bot"""
        params = {'bot_id': bot_id, 'limit': limit}
        return self._make_request('GET', '/public/api/ver1/deals', params=params)

    def start_bot(self, bot_id: int) -> Dict:
        """Start a bot"""
        return self._make_request('POST', f'/public/api/ver1/bots/{bot_id}/start')

    def stop_bot(self, bot_id: int) -> Dict:
        """Stop a bot"""
        return self._make_request('POST', f'/public/api/ver1/bots/{bot_id}/stop')

    def delete_bot(self, bot_id: int) -> Dict:
        """Delete a bot"""
        return self._make_request('POST', f'/public/api/ver1/bots/{bot_id}/delete') 