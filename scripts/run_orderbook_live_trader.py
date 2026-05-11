#!/usr/bin/env python3
"""
Live Order Book Trading Bot for Experiment 7 Model

Runs the order book spike prediction model (Experiment 7) on live websocket data
with paper trading capabilities.

Architecture:
    WebSocket → Order Book Aggregator → Feature Calculator → Sequence Buffer → Model → Paper Trader

Key Differences from Candle-Based Trader:
- 10-second order book windows (not 1-minute candles)
- 24 order book microstructure features (not OHLCV candle features)
- Real-time feature computation from raw order book + trade data
- 60 timesteps = 10 minutes of history (not 60 minutes)

Usage:
    # Live trading with paper account
    python scripts/run_orderbook_live_trader.py \
        --model models/orderbook_hf_120sym_27days_with_dups/best_model.pt \
        --scaler models/orderbook_hf_120sym_27days_with_dups/scaler.pkl \
        --symbols BTC-USD,ETH-USD,SOL-USD

    # Full 120 symbols
    python scripts/run_orderbook_live_trader.py \
        --model models/orderbook_hf_120sym_27days_with_dups/best_model.pt \
        --scaler models/orderbook_hf_120sym_27days_with_dups/scaler.pkl \
        --all-symbols
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
import signal
import pickle
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Websocket client
try:
    from src.inference.public_websocket import PublicWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("WebSocket components not available")


# === ORDER BOOK FEATURE CALCULATOR ===
class OrderBookFeatureCalculator:
    """
    Calculates 24 order book microstructure features from raw order book + trade data.

    Based on HighFreqFeatureExtractor._calculate_features() from extract_orderbook_features_hf.py
    """

    def __init__(self, window_seconds: int = 10):
        self.window_seconds = window_seconds

    def calculate_features(self, ob_data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Calculate 24 features from order book + trade data.

        Args:
            ob_data: DataFrame with columns:
                - bid_depth_1, bid_depth_5, bid_depth_10
                - ask_depth_1, ask_depth_5, ask_depth_10
                - best_bid, best_ask
                - trade_volume (optional)
                - buy_volume, sell_volume (optional)

        Returns:
            Array of 24 features, or None if insufficient data
        """
        if len(ob_data) < 6:  # Need at least 6 timesteps for ROC calculations
            return None

        try:
            features = {}

            # === 1. ORDER BOOK IMBALANCE ===
            features['bid_ask_imbalance'] = (
                (ob_data['bid_depth_10'] - ob_data['ask_depth_10']) /
                (ob_data['bid_depth_10'] + ob_data['ask_depth_10'] + 1e-9)
            ).iloc[-1]

            # === 2. DEPTH RATE OF CHANGE ===
            features['bid_depth_roc'] = ob_data['bid_depth_10'].pct_change(5).iloc[-1]
            features['ask_depth_roc'] = ob_data['ask_depth_10'].pct_change(5).iloc[-1]

            # === 3. DEPTH LEVELS ===
            features['bid_depth'] = ob_data['bid_depth_10'].iloc[-1]
            features['ask_depth'] = ob_data['ask_depth_10'].iloc[-1]

            # === 4. SPREAD DYNAMICS ===
            spread = ob_data['best_ask'] - ob_data['best_bid']
            mid_price = (ob_data['best_bid'] + ob_data['best_ask']) / 2
            features['spread_pct'] = (spread / mid_price).iloc[-1]
            features['spread_roc'] = spread.pct_change(3).iloc[-1]

            # === 5. LARGE ORDER IMBALANCE ===
            # Use depth at level 10 as proxy for "large orders"
            large_bid = ob_data['bid_depth_10'] - ob_data['bid_depth_5']
            large_ask = ob_data['ask_depth_10'] - ob_data['ask_depth_5']
            features['large_order_imbalance'] = (
                (large_bid - large_ask) / (large_bid + large_ask + 1e-9)
            ).iloc[-1]
            features['large_bid_count'] = (large_bid > 0).sum()
            features['large_ask_count'] = (large_ask > 0).sum()

            # === 6. ORDER FLOW IMBALANCE (OFI) ===
            if 'buy_volume' in ob_data.columns and 'sell_volume' in ob_data.columns:
                features['ofi'] = (
                    (ob_data['buy_volume'] - ob_data['sell_volume']) /
                    (ob_data['buy_volume'] + ob_data['sell_volume'] + 1e-9)
                ).iloc[-1]
                features['buy_volume'] = ob_data['buy_volume'].sum()
                features['sell_volume'] = ob_data['sell_volume'].sum()
            else:
                features['ofi'] = 0.0
                features['buy_volume'] = 0.0
                features['sell_volume'] = 0.0

            # === 7. TRADE INTENSITY ===
            if 'trade_volume' in ob_data.columns:
                features['trade_count'] = (ob_data['trade_volume'] > 0).sum()
                features['trade_velocity_roc'] = ob_data['trade_volume'].pct_change(3).iloc[-1]
            else:
                features['trade_count'] = 0
                features['trade_velocity_roc'] = 0.0

            # === 8. PRICE MOMENTUM ===
            features['price'] = mid_price.iloc[-1]
            features['price_change_10sec'] = mid_price.pct_change(1).iloc[-1]
            features['price_change_30sec'] = mid_price.pct_change(3).iloc[-1]
            features['price_change_60sec'] = mid_price.pct_change(6).iloc[-1] if len(ob_data) >= 6 else 0.0

            # === 9. VOLATILITY ===
            features['price_volatility_60sec'] = mid_price.pct_change().tail(6).std() if len(ob_data) >= 6 else 0.0

            # === 10. CUMULATIVE IMBALANCE ===
            imbalance = (ob_data['bid_depth_10'] - ob_data['ask_depth_10']) / (ob_data['bid_depth_10'] + ob_data['ask_depth_10'] + 1e-9)
            features['cumulative_imbalance_30sec'] = imbalance.tail(3).sum()
            features['imbalance_velocity'] = imbalance.diff().iloc[-1]

            # === 11. WEIGHTED MID PRICE ===
            total_depth = ob_data['bid_depth_1'] + ob_data['ask_depth_1']
            wmp = (
                (ob_data['best_bid'] * ob_data['ask_depth_1'] + ob_data['best_ask'] * ob_data['bid_depth_1']) /
                (total_depth + 1e-9)
            )
            features['weighted_mid_price'] = wmp.iloc[-1]
            features['wmp_roc'] = wmp.pct_change(3).iloc[-1]

            # Convert to array in correct order (must match training order)
            feature_array = np.array([
                features['bid_ask_imbalance'],
                features['bid_depth_roc'],
                features['ask_depth_roc'],
                features['bid_depth'],
                features['ask_depth'],
                features['spread_pct'],
                features['spread_roc'],
                features['large_order_imbalance'],
                features['large_bid_count'],
                features['large_ask_count'],
                features['ofi'],
                features['buy_volume'],
                features['sell_volume'],
                features['trade_count'],
                features['trade_velocity_roc'],
                features['price'],
                features['price_change_10sec'],
                features['price_change_30sec'],
                features['price_change_60sec'],
                features['price_volatility_60sec'],
                features['cumulative_imbalance_30sec'],
                features['imbalance_velocity'],
                features['weighted_mid_price'],
                features['wmp_roc']
            ], dtype=np.float32)

            # Replace NaN/inf with 0
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

            return feature_array

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None


# === ORDER BOOK AGGREGATOR ===
class OrderBookAggregator:
    """
    Aggregates raw order book and trade data into 10-second windows.
    """

    def __init__(self, symbols: List[str], window_seconds: int = 10, history_length: int = 60):
        self.symbols = symbols
        self.window_seconds = window_seconds
        self.history_length = history_length

        # Store raw data for each symbol
        self.order_books: Dict[str, deque] = {sym: deque(maxlen=history_length) for sym in symbols}
        self.trades: Dict[str, deque] = {sym: deque(maxlen=1000) for sym in symbols}

        # Track last aggregation time
        self.last_agg_time: Dict[str, datetime] = {}

    async def process_order_book(self, symbol: str, data: dict):
        """Process incoming order book snapshot."""
        try:
            timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))

            # Extract best bid/ask and depths
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            if not bids or not asks:
                return

            # Calculate depth at different levels
            bid_depth_1 = float(bids[0][1]) if len(bids) > 0 else 0
            bid_depth_5 = sum(float(b[1]) for b in bids[:5]) if len(bids) >= 5 else bid_depth_1
            bid_depth_10 = sum(float(b[1]) for b in bids[:10]) if len(bids) >= 10 else bid_depth_5

            ask_depth_1 = float(asks[0][1]) if len(asks) > 0 else 0
            ask_depth_5 = sum(float(a[1]) for a in asks[:5]) if len(asks) >= 5 else ask_depth_1
            ask_depth_10 = sum(float(a[1]) for a in asks[:10]) if len(asks) >= 10 else ask_depth_5

            ob_snapshot = {
                'timestamp': timestamp,
                'best_bid': float(bids[0][0]),
                'best_ask': float(asks[0][0]),
                'bid_depth_1': bid_depth_1,
                'bid_depth_5': bid_depth_5,
                'bid_depth_10': bid_depth_10,
                'ask_depth_1': ask_depth_1,
                'ask_depth_5': ask_depth_5,
                'ask_depth_10': ask_depth_10,
            }

            self.order_books[symbol].append(ob_snapshot)

        except Exception as e:
            logger.error(f"Error processing order book for {symbol}: {e}")

    async def process_trade(self, symbol: str, data: dict):
        """Process incoming trade."""
        try:
            timestamp = datetime.fromisoformat(data['time'].replace('Z', '+00:00'))

            trade = {
                'timestamp': timestamp,
                'price': float(data['price']),
                'size': float(data['size']),
                'side': data['side']  # 'BUY' or 'SELL'
            }

            self.trades[symbol].append(trade)

        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")

    def aggregate_window(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Aggregate data for the last 60 windows (10 minutes).

        Returns:
            DataFrame with 60 rows (one per 10-second window) or None if insufficient data
        """
        if symbol not in self.order_books or len(self.order_books[symbol]) < 6:
            return None

        try:
            # Convert order book snapshots to DataFrame
            ob_list = list(self.order_books[symbol])
            df = pd.DataFrame(ob_list)

            # Add trade volume aggregation
            trades_list = list(self.trades[symbol])
            if trades_list:
                trades_df = pd.DataFrame(trades_list)

                # Aggregate trades into 10-second windows matching order book
                df['trade_volume'] = 0.0
                df['buy_volume'] = 0.0
                df['sell_volume'] = 0.0

                for i, row in df.iterrows():
                    window_start = row['timestamp'] - timedelta(seconds=self.window_seconds)
                    window_trades = trades_df[
                        (trades_df['timestamp'] >= window_start) &
                        (trades_df['timestamp'] < row['timestamp'])
                    ]

                    if len(window_trades) > 0:
                        df.at[i, 'trade_volume'] = window_trades['size'].sum()
                        df.at[i, 'buy_volume'] = window_trades[window_trades['side'] == 'BUY']['size'].sum()
                        df.at[i, 'sell_volume'] = window_trades[window_trades['side'] == 'SELL']['size'].sum()

            return df

        except Exception as e:
            logger.error(f"Error aggregating window for {symbol}: {e}")
            return None


# === SEQUENCE BUFFER ===
class SequenceBuffer:
    """
    Maintains rolling window of 60 timesteps of features for each symbol.
    """

    def __init__(self, sequence_length: int = 60, n_features: int = 24, scaler_path: Optional[str] = None):
        self.sequence_length = sequence_length
        self.n_features = n_features

        # Load scaler
        self.scaler = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_path}")

        # Buffer for each symbol: deque of feature vectors
        self.buffers: Dict[str, deque] = {}

    def add_features(self, symbol: str, features: np.ndarray) -> bool:
        """
        Add feature vector to buffer.

        Returns:
            True if buffer is ready for inference (has sequence_length timesteps)
        """
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.sequence_length)

        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))[0]

        self.buffers[symbol].append(features)

        return len(self.buffers[symbol]) == self.sequence_length

    def get_sequence(self, symbol: str) -> Optional[np.ndarray]:
        """Get sequence for inference (shape: [sequence_length, n_features])."""
        if symbol not in self.buffers or len(self.buffers[symbol]) < self.sequence_length:
            return None

        return np.array(list(self.buffers[symbol]))


# === INFERENCE ENGINE ===
class InferenceEngine:
    """
    Runs CNN-LSTM model inference on sequences.
    """

    def __init__(self, model_path: str, device: str = 'mps'):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'mps' else 'cpu')

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)

        # Build model
        from scripts.train_orderbook_model_hf import CNNLSTMModel
        self.model = CNNLSTMModel(
            input_dim=24,
            sequence_length=60,
            hidden_dim=checkpoint['config']['hidden_dim'],
            num_lstm_layers=checkpoint['config']['num_lstm_layers'],
            dropout=checkpoint['config']['dropout']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")

    def predict(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Run inference on sequence.

        Returns:
            (probability, logit) tuple
        """
        with torch.no_grad():
            x = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)  # [1, 60, 24]
            logit = self.model(x).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()

        return prob, logit


# === PAPER TRADING MANAGER ===
class PaperTrader:
    """
    Simulates trading with $10k virtual account.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 0.20,
        max_positions: int = 5,
        prob_threshold: float = 0.6,
        hold_minutes: int = 60,
        stop_loss: float = 0.03,
        log_file: str = "orderbook_live_trades.json"
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.prob_threshold = prob_threshold
        self.hold_minutes = hold_minutes
        self.stop_loss = stop_loss
        self.log_file = log_file

        # Track positions: {symbol: {'entry_price', 'entry_time', 'size', 'prob'}}
        self.positions: Dict[str, dict] = {}

        # Track all trades
        self.trade_history: List[dict] = []

        logger.info(f"Paper trader initialized: ${initial_capital:,.0f} capital, {max_positions} max positions")

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.max_positions and self.cash > 0

    def open_position(self, symbol: str, price: float, probability: float):
        """Open a new position."""
        if not self.can_open_position():
            return

        # Calculate position size
        position_value = self.cash * self.position_size
        size = position_value / price

        self.positions[symbol] = {
            'entry_price': price,
            'entry_time': datetime.now(),
            'size': size,
            'prob': probability,
            'position_value': position_value
        }

        self.cash -= position_value

        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'size': size,
            'value': position_value,
            'probability': probability,
            'cash_remaining': self.cash
        }

        self.trade_history.append(trade)
        self._save_trades()

        logger.info(f"OPENED {symbol}: ${position_value:,.0f} @ ${price:.4f} (prob={probability:.3f})")

    def close_position(self, symbol: str, price: float, reason: str):
        """Close an existing position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        exit_value = pos['size'] * price
        pnl = exit_value - pos['position_value']
        pnl_pct = (pnl / pos['position_value']) * 100
        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60

        self.cash += exit_value

        trade = {
            'timestamp': datetime.now().isoformat(),
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'size': pos['size'],
            'value': exit_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_minutes': hold_time,
            'reason': reason,
            'cash_after': self.cash
        }

        self.trade_history.append(trade)
        self._save_trades()

        del self.positions[symbol]

        logger.info(f"CLOSED {symbol}: P&L ${pnl:+,.2f} ({pnl_pct:+.2f}%) after {hold_time:.0f}m - {reason}")

    def check_exits(self, symbol: str, current_price: float):
        """Check if position should be closed."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        hold_time = (datetime.now() - pos['entry_time']).total_seconds() / 60
        pnl_pct = ((current_price / pos['entry_price']) - 1)

        # Check stop loss
        if pnl_pct <= -self.stop_loss:
            self.close_position(symbol, current_price, "STOP_LOSS")
            return

        # Check hold time
        if hold_time >= self.hold_minutes:
            self.close_position(symbol, current_price, "TIME_LIMIT")
            return

    def _save_trades(self):
        """Save trade history to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")

    def get_stats(self) -> dict:
        """Get current account statistics."""
        position_value = sum(p['position_value'] for p in self.positions.values())
        total_value = self.cash + position_value

        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'pnl': total_value - self.initial_capital,
            'pnl_pct': ((total_value / self.initial_capital) - 1) * 100,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }


# === MAIN ORCHESTRATOR ===
class OrderBookLiveTrader:
    """
    Main orchestrator for live order book trading.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        symbols: List[str],
        prob_threshold: float = 0.6,
        device: str = 'mps'
    ):
        self.symbols = symbols
        self.prob_threshold = prob_threshold

        # Initialize components
        self.aggregator = OrderBookAggregator(symbols)
        self.feature_calc = OrderBookFeatureCalculator()
        self.buffer = SequenceBuffer(scaler_path=scaler_path)
        self.inference = InferenceEngine(model_path, device=device)
        self.trader = PaperTrader(prob_threshold=prob_threshold)

        # Track last inference time for each symbol
        self.last_inference: Dict[str, datetime] = {}

        logger.info(f"Initialized live trader for {len(symbols)} symbols")

    async def handle_order_book(self, message: dict):
        """Handle incoming order book message."""
        symbol = message.get('product_id')
        if symbol not in self.symbols:
            return

        await self.aggregator.process_order_book(symbol, message)
        await self._try_inference(symbol)

    async def handle_trade(self, message: dict):
        """Handle incoming trade message."""
        symbol = message.get('product_id')
        if symbol not in self.symbols:
            return

        await self.aggregator.process_trade(symbol, message)

    async def _try_inference(self, symbol: str):
        """Try to run inference if we have enough data."""
        # Throttle: only run inference every 10 seconds per symbol
        now = datetime.now()
        if symbol in self.last_inference:
            if (now - self.last_inference[symbol]).total_seconds() < 10:
                return

        self.last_inference[symbol] = now

        # Aggregate data
        df = self.aggregator.aggregate_window(symbol)
        if df is None or len(df) < 60:
            return

        # Calculate features
        features = self.feature_calc.calculate_features(df)
        if features is None:
            return

        # Add to buffer
        ready = self.buffer.add_features(symbol, features)
        if not ready:
            return

        # Get sequence and run inference
        sequence = self.buffer.get_sequence(symbol)
        if sequence is None:
            return

        prob, logit = self.inference.predict(sequence)

        # Get current price
        current_price = df['price'].iloc[-1] if 'price' in df.columns else (df['best_bid'].iloc[-1] + df['best_ask'].iloc[-1]) / 2

        # Trading logic
        if prob >= self.prob_threshold:
            if self.trader.can_open_position() and symbol not in self.trader.positions:
                self.trader.open_position(symbol, current_price, prob)
                logger.warning(f"⚠️  SIGNAL: {symbol} @ ${current_price:.4f} | P={prob:.3f}")

        # Check exits for existing positions
        self.trader.check_exits(symbol, current_price)

        # Log status every 50 inferences
        if sum(1 for _ in self.last_inference.values()) % 50 == 0:
            stats = self.trader.get_stats()
            logger.info(f"Account: ${stats['total_value']:,.0f} | P&L: ${stats['pnl']:+,.0f} ({stats['pnl_pct']:+.2f}%) | Positions: {stats['num_positions']}/{self.trader.max_positions}")

    def _on_ticker_update(self, symbol: str, price: float, volume_24h: float):
        """Handle ticker update from WebSocket."""
        # Ticker updates provide current price - we don't need to do anything special here
        # The aggregator will handle this
        pass

    def _on_trade_update(self, symbol: str, price: float, size: float, side: str):
        """Handle trade update from WebSocket."""
        # Create trade record
        timestamp = datetime.now()
        self.aggregator.add_trade(symbol, {
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'side': side
        })

    def _on_level2_update(self, symbol: str, bids: list, asks: list):
        """Handle level2 order book update from WebSocket."""
        # Create order book snapshot
        timestamp = datetime.now()
        self.aggregator.add_order_book(symbol, {
            'timestamp': timestamp,
            'bids': bids,  # List of [price, size] pairs
            'asks': asks   # List of [price, size] pairs
        })

    async def run(self):
        """Run the live trader."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket client not available")
            return

        logger.info(f"Starting live trader for {len(self.symbols)} symbols...")

        # Create websocket client with callback handlers
        # Note: PublicWebSocketClient only provides ticker and trade updates
        # We'll need to construct order book data from trades
        ws_client = PublicWebSocketClient(
            symbols=self.symbols,
            on_ticker=self._on_ticker_update,
            on_trade=self._on_trade_update,
            channels=['ticker', 'matches']
        )

        # Create periodic tasks
        ws_task = asyncio.create_task(ws_client.start())

        # Periodic aggregation task - process windows every 10 seconds
        async def process_windows_periodically():
            while True:
                await asyncio.sleep(10)
                # Process all symbols
                for symbol in self.symbols:
                    try:
                        await self.process_window(symbol)
                    except Exception as e:
                        logger.error(f"Error processing window for {symbol}: {e}")

        process_task = asyncio.create_task(process_windows_periodically())

        # Statistics printing task
        async def print_stats_loop():
            while True:
                await asyncio.sleep(60)  # Every minute
                stats = self.trader.get_stats()
                logger.info(f"Status: ${stats['total_value']:,.0f} | P&L: ${stats['pnl']:+,.0f} ({stats['pnl_pct']:+.2f}%) | Positions: {stats['num_positions']}")

        stats_task = asyncio.create_task(print_stats_loop())

        # Wait forever (until interrupted)
        try:
            await asyncio.gather(ws_task, process_task, stats_task)

        except asyncio.CancelledError:
            logger.info("Shutting down...")
            await ws_client.stop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await ws_client.stop()
        finally:
            # Print final stats
            stats = self.trader.get_stats()
            logger.info("=" * 80)
            logger.info("FINAL RESULTS")
            logger.info("=" * 80)
            logger.info(f"Initial Capital: ${self.trader.initial_capital:,.0f}")
            logger.info(f"Final Value:     ${stats['total_value']:,.0f}")
            logger.info(f"P&L:             ${stats['pnl']:+,.0f} ({stats['pnl_pct']:+.2f}%)")
            logger.info(f"Total Trades:    {stats['num_trades']}")
            logger.info(f"Trades Log:      {self.trader.log_file}")


# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Live Order Book Trading Bot")
    parser.add_argument('--model', required=True, help="Path to model checkpoint")
    parser.add_argument('--scaler', required=True, help="Path to scaler pickle")
    parser.add_argument('--symbols', help="Comma-separated list of symbols (e.g., BTC-USD,ETH-USD)")
    parser.add_argument('--all-symbols', action='store_true', help="Use all 120 training symbols")
    parser.add_argument('--prob-threshold', type=float, default=0.6, help="Minimum probability threshold")
    parser.add_argument('--device', default='mps', help="Device (mps/cuda/cpu)")

    args = parser.parse_args()

    # Get symbols
    if args.all_symbols:
        # 120 symbols from training
        symbols = [
            'VTHO-USD', 'DOGE-USD', 'LINK-USD', 'XLM-USD', 'SUI-USD', 'AVNT-USD', 'ADA-USD', 'ZEC-USD',
            'HBAR-USD', 'ONDO-USD', 'AAVE-USD', 'LTC-USD', 'ZORA-USD', 'AERO-USD', 'MORPHO-USD', 'MAMO-USD',
            'BNKR-USD', 'PUMP-USD', 'TAO-USD', 'PENGU-USD', 'AVAX-USD', 'SPA-USD', 'FARTCOIN-USD', 'EDGE-USD',
            'DIMO-USD', 'YFI-USD', 'SUPER-USD', 'DOT-USD', 'IP-USD', 'TOSHI-USD', 'BCH-USD', 'PNG-USD',
            'ABT-USD', 'ENA-USD', 'UNI-USD', 'BONK-USD', 'CLANKER-USD', 'PROMPT-USD', 'FIL-USD', 'CRO-USD',
            'GFI-USD', 'LOKA-USD', 'PAXG-USD', 'WELL-USD', 'CRV-USD', 'KEYCAT-USD', 'DASH-USD', 'POLS-USD',
            'ZRX-USD', 'BARD-USD', 'B3-USD', 'SQD-USD', 'NEAR-USD', 'QNT-USD', 'WLFI-USD', 'USELESS-USD',
            'ARB-USD', 'ATH-USD', 'FET-USD', 'SPX-USD', 'INJ-USD', 'SEI-USD', 'COOKIE-USD', 'MANTLE-USD',
            'CBETH-USD', 'PEPE-USD', 'ZKC-USD', 'W-USD', 'ZEN-USD', 'APT-USD', 'KAITO-USD', 'SWFTC-USD',
            'PRO-USD', 'ACH-USD', 'DOGINME-USD', 'WLD-USD', 'TRAC-USD', 'EIGEN-USD', 'MNDE-USD', 'S-USD',
            'PRIME-USD', 'JTO-USD', 'DIA-USD', 'WIF-USD', 'RSC-USD', 'ICP-USD', 'MUSE-USD', 'LRDS-USD',
            'ACS-USD', 'CVX-USD', 'ATOM-USD', 'GST-USD', 'FLOCK-USD', 'BIO-USD', 'PROVE-USD', 'QI-USD',
            'VVV-USD', 'RENDER-USD', 'IMX-USD', 'CAKE-USD', 'SAPIEN-USD', 'ZRO-USD', 'ALGO-USD', 'SEAM-USD',
            'SUSHI-USD', 'LCX-USD', 'AWE-USD', 'POPCAT-USD', 'SHIB-USD', 'ETHFI-USD', 'SNX-USD', 'OMNI-USD',
            'ORCA-USD', 'TIA-USD', 'MOODENG-USD', 'VET-USD', 'RSR-USD', 'SYRUP-USD', 'SHPING-USD', 'PENDLE-USD'
        ]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        # Default: test with 10 symbols
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD',
                   'UNI-USD', 'AAVE-USD', 'DOT-USD', 'MATIC-USD', 'ADA-USD']

    logger.info(f"Trading {len(symbols)} symbols with model {args.model}")

    # Create and run trader
    trader = OrderBookLiveTrader(
        model_path=args.model,
        scaler_path=args.scaler,
        symbols=symbols,
        prob_threshold=args.prob_threshold,
        device=args.device
    )

    asyncio.run(trader.run())


if __name__ == '__main__':
    main()
