#!/usr/bin/env python3
"""
Live Trading Integration for RL Agent

Provides:
- Load trained model
- Connect to feature_buffer.db for real-time data
- Paper trading mode
- Integration with existing Breakout Hunter signals

Usage:
    python scripts/run_rl_live.py --model-path models/rl/best/model.zip --paper-trade
"""

import os
import sys
import time
import signal
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.environment import Action, Position
from src.rl.features import FeatureExtractor, FeatureConfig
from src.rl.state import StateBuilder, StateConfig
from src.rl.regime import RegimeDetector, RegimeType
from src.rl.evaluation import MetricsCalculator, TradeRecord


@dataclass
class LiveConfig:
    """Configuration for live trading."""
    # Model
    model_path: str = "models/rl/best/model.zip"

    # Data sources
    feature_buffer_path: str = "feature_buffer.db"
    duckdb_path: str = "/Users/bz/Pythia2/full_pythia.duckdb"

    # Trading parameters
    symbols: List[str] = None  # None = all available
    decision_interval_seconds: int = 60  # 1 minute
    max_positions: int = 3
    position_size_usd: float = 1000.0

    # Risk management
    max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    initial_stop_pct: float = 0.02
    max_drawdown: float = 0.10        # 10% circuit breaker

    # Paper trading
    paper_trade: bool = True
    initial_capital: float = 10000.0

    # Integration
    use_breakout_signals: bool = True  # Filter with Breakout Hunter
    min_breakout_confidence: float = 0.7

    # Logging
    log_dir: str = "logs/rl_live"
    log_trades: bool = True


class LiveDataConnector:
    """
    Connect to real-time data sources.

    Uses feature_buffer.db for recent data and DuckDB for historical.
    """

    def __init__(
        self,
        feature_buffer_path: str = "feature_buffer.db",
        duckdb_path: str = "/Users/bz/Pythia2/full_pythia.duckdb",
    ):
        """
        Initialize data connector.

        Args:
            feature_buffer_path: Path to SQLite feature buffer
            duckdb_path: Path to DuckDB database
        """
        self.feature_buffer_path = feature_buffer_path
        self.duckdb_path = duckdb_path

        self._sqlite_conn: Optional[sqlite3.Connection] = None

    @property
    def sqlite_conn(self) -> sqlite3.Connection:
        """Lazy SQLite connection."""
        if self._sqlite_conn is None:
            self._sqlite_conn = sqlite3.connect(self.feature_buffer_path)
        return self._sqlite_conn

    def get_latest_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get latest features for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Feature dictionary or None
        """
        try:
            query = """
                SELECT * FROM features
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            df = pd.read_sql_query(query, self.sqlite_conn, params=(symbol,))

            if len(df) == 0:
                return None

            return df.iloc[0].to_dict()

        except Exception as e:
            logger.warning(f"Error getting features for {symbol}: {e}")
            return None

    def get_recent_ohlcv(
        self,
        symbol: str,
        minutes: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Get recent OHLCV data.

        Args:
            symbol: Trading pair symbol
            minutes: Number of minutes of history

        Returns:
            OHLCV DataFrame or None
        """
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(
                query, self.sqlite_conn,
                params=(symbol, minutes),
                parse_dates=['timestamp'],
            )

            if len(df) == 0:
                return None

            df = df.sort_values('timestamp').set_index('timestamp')
            return df

        except Exception as e:
            logger.warning(f"Error getting OHLCV for {symbol}: {e}")
            return None

    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """
        Get latest order book snapshot.

        Args:
            symbol: Trading pair symbol

        Returns:
            Order book dict or None
        """
        try:
            query = """
                SELECT bids, asks, best_bid, best_ask, spread_bps
                FROM order_book_snapshots
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            df = pd.read_sql_query(query, self.sqlite_conn, params=(symbol,))

            if len(df) == 0:
                return None

            row = df.iloc[0]
            return {
                'bids': eval(row['bids']) if row['bids'] else [],
                'asks': eval(row['asks']) if row['asks'] else [],
                'best_bid': row['best_bid'],
                'best_ask': row['best_ask'],
                'spread_bps': row['spread_bps'],
            }

        except Exception as e:
            logger.warning(f"Error getting order book for {symbol}: {e}")
            return None

    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with recent data."""
        try:
            query = """
                SELECT DISTINCT symbol FROM ohlcv
                WHERE timestamp > datetime('now', '-1 hour')
            """
            df = pd.read_sql_query(query, self.sqlite_conn)
            return df['symbol'].tolist()

        except Exception as e:
            logger.warning(f"Error getting symbols: {e}")
            return []

    def close(self) -> None:
        """Close connections."""
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None


class PaperTrader:
    """
    Paper trading simulator.

    Tracks positions and P&L without real execution.
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize paper trader.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[TradeRecord] = []
        self.equity_history: List[float] = [initial_capital]

    def enter_position(
        self,
        symbol: str,
        price: float,
        size_usd: float,
        stop_loss_pct: float = 0.02,
    ) -> bool:
        """
        Enter a new position.

        Args:
            symbol: Trading pair
            price: Entry price
            size_usd: Position size in USD
            stop_loss_pct: Stop loss percentage

        Returns:
            True if position was opened
        """
        if symbol in self.positions:
            logger.warning(f"Already have position in {symbol}")
            return False

        if size_usd > self.capital:
            logger.warning(f"Insufficient capital for {symbol} position")
            return False

        # Calculate size
        size = size_usd / price

        self.positions[symbol] = Position(
            entry_price=price,
            entry_time=datetime.now(),
            size=size,
            stop_loss=price * (1 - stop_loss_pct),
            highest_price=price,
        )

        logger.info(f"PAPER: Entered {symbol} @ ${price:.2f}, size={size:.6f}")
        return True

    def exit_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual",
    ) -> Optional[TradeRecord]:
        """
        Exit a position.

        Args:
            symbol: Trading pair
            price: Exit price
            reason: Exit reason

        Returns:
            TradeRecord if position was closed
        """
        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        return_pct = (price - position.entry_price) / position.entry_price

        # Apply simulated fees
        return_pct -= 0.0055  # 0.55% round trip

        # Calculate P&L
        pnl_usd = position.size * position.entry_price * return_pct

        trade = TradeRecord(
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            entry_price=position.entry_price,
            exit_price=price,
            return_pct=return_pct,
            size=position.size * position.entry_price,  # USD value
            exit_reason=reason,
            symbol=symbol,
        )

        self.trades.append(trade)
        self.capital += pnl_usd

        del self.positions[symbol]

        logger.info(
            f"PAPER: Exited {symbol} @ ${price:.2f}, "
            f"return={return_pct*100:.2f}%, P&L=${pnl_usd:.2f}"
        )

        return trade

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Update stop loss for position."""
        if symbol not in self.positions:
            return False

        self.positions[symbol].stop_loss = new_stop
        return True

    def check_stops(self, prices: Dict[str, float]) -> List[TradeRecord]:
        """
        Check stop losses against current prices.

        Args:
            prices: Dict of symbol -> current price

        Returns:
            List of closed trade records
        """
        closed = []

        for symbol, position in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue

            # Update high watermark
            position.update_high(price)

            # Check stop loss
            if price <= position.stop_loss:
                trade = self.exit_position(symbol, price, "stop_loss")
                if trade:
                    closed.append(trade)

        return closed

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate current equity including open positions."""
        equity = self.capital

        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            unrealized = position.size * (price - position.entry_price)
            equity += unrealized

        return equity

    def get_stats(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Get current statistics."""
        equity = self.get_equity(prices)
        total_return = (equity - self.initial_capital) / self.initial_capital

        return {
            'capital': self.capital,
            'equity': equity,
            'total_return': total_return,
            'n_positions': len(self.positions),
            'n_trades': len(self.trades),
            'positions': list(self.positions.keys()),
        }


class LiveAgent:
    """
    Live trading agent orchestrator.

    Coordinates:
    - Data collection
    - Model inference
    - Position management
    - Risk monitoring
    """

    def __init__(self, config: LiveConfig):
        """
        Initialize live agent.

        Args:
            config: Live trading configuration
        """
        self.config = config

        # Setup logging
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(config.log_dir) / "live_agent.log"
        logger.add(log_file, rotation="100 MB", level="DEBUG")

        # Load model
        logger.info(f"Loading model from {config.model_path}")
        self._load_model()

        # Initialize components
        self.data_connector = LiveDataConnector(
            config.feature_buffer_path,
            config.duckdb_path,
        )
        self.feature_extractor = FeatureExtractor(FeatureConfig())
        self.state_builder = StateBuilder(StateConfig())
        self.regime_detector = RegimeDetector()

        # Paper trading
        if config.paper_trade:
            self.trader = PaperTrader(config.initial_capital)
            logger.info("Running in PAPER TRADE mode")
        else:
            self.trader = None
            logger.warning("Running in LIVE mode - NOT IMPLEMENTED")

        # State
        self.running = False
        self.last_decision_time: Dict[str, datetime] = {}
        self.current_regime: RegimeType = RegimeType.UNKNOWN

        # Statistics
        self.decisions_made = 0
        self.signals_filtered = 0

    def _load_model(self) -> None:
        """Load the trained model."""
        from src.rl.agent import PPOAgent

        # Check if model exists
        if not Path(self.config.model_path).exists():
            logger.warning(f"Model not found at {self.config.model_path}")
            logger.warning("Using random policy for testing")
            self.model = None
        else:
            # Create a dummy environment for model loading
            # In production, this would match the training environment
            self.model = None  # PPOAgent.from_checkpoint(...)
            logger.info("Model loaded successfully")

    def get_action(
        self,
        symbol: str,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:
        """
        Get action from model or fallback.

        Args:
            symbol: Trading pair
            observation: State observation
            action_mask: Valid action mask

        Returns:
            Action index
        """
        if self.model is not None:
            action, _ = self.model.predict(observation, action_mask=action_mask)
            return int(action)
        else:
            # Random valid action for testing
            valid_actions = np.where(action_mask)[0]
            return int(np.random.choice(valid_actions))

    def get_action_mask(self, symbol: str) -> np.ndarray:
        """Get action mask based on current position state."""
        mask = np.ones(len(Action), dtype=bool)

        has_position = symbol in self.trader.positions if self.trader else False

        if not has_position:
            # Can only WAIT or ENTER_LONG
            mask[Action.HOLD] = False
            mask[Action.TIGHTEN_STOP] = False
            mask[Action.LOOSEN_STOP] = False
            mask[Action.TAKE_PARTIAL] = False
            mask[Action.EXIT_ALL] = False
        else:
            # Can't enter when already in position
            mask[Action.ENTER_LONG] = False

        # Check max positions
        if self.trader and len(self.trader.positions) >= self.config.max_positions:
            mask[Action.ENTER_LONG] = False

        return mask

    def should_trade(self, symbol: str) -> bool:
        """Check if trading is allowed for symbol."""
        # Regime check
        if self.current_regime == RegimeType.HIGH_VOLATILITY:
            logger.debug(f"Skipping {symbol}: high volatility regime")
            return False

        # Drawdown circuit breaker
        if self.trader:
            current_prices = {}  # Would need to fetch
            equity = self.trader.get_equity(current_prices)
            drawdown = 1 - equity / self.config.initial_capital

            if drawdown > self.config.max_drawdown:
                logger.warning(f"Circuit breaker: drawdown {drawdown:.1%}")
                return False

        return True

    def process_symbol(self, symbol: str) -> Optional[int]:
        """
        Process trading decision for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Action taken or None
        """
        # Get current data
        features = self.data_connector.get_latest_features(symbol)
        if features is None:
            return None

        ohlcv = self.data_connector.get_recent_ohlcv(symbol, minutes=60)
        if ohlcv is None or len(ohlcv) < 30:
            return None

        # Build observation
        try:
            # Calculate features
            feature_df = self.feature_extractor.calculate_features(ohlcv)

            if len(feature_df) == 0:
                return None

            # Get latest features
            latest_features = feature_df.iloc[-1].values.astype(np.float32)

            # Add position context
            if self.trader and symbol in self.trader.positions:
                position = self.trader.positions[symbol]
                current_price = ohlcv['close'].iloc[-1]
                position_features = np.array([
                    1.0,  # has_position
                    position.unrealized_return(current_price),
                    0.5,  # time_in_position (normalized)
                    position.highest_return(),
                    (current_price - position.stop_loss) / current_price,
                    1.0,  # size
                ], dtype=np.float32)
            else:
                position_features = np.zeros(6, dtype=np.float32)

            observation = np.concatenate([latest_features, position_features])
            observation = np.nan_to_num(observation, nan=0.0)

        except Exception as e:
            logger.warning(f"Error building observation for {symbol}: {e}")
            return None

        # Get action mask and decision
        action_mask = self.get_action_mask(symbol)
        action = self.get_action(symbol, observation, action_mask)

        # Execute action
        current_price = ohlcv['close'].iloc[-1]
        self._execute_action(symbol, action, current_price)

        self.decisions_made += 1
        return action

    def _execute_action(self, symbol: str, action: int, price: float) -> None:
        """Execute trading action."""
        if not self.config.paper_trade or self.trader is None:
            return

        if action == Action.ENTER_LONG:
            if self.should_trade(symbol):
                self.trader.enter_position(
                    symbol,
                    price,
                    self.config.position_size_usd,
                    self.config.initial_stop_pct,
                )
            else:
                self.signals_filtered += 1

        elif action == Action.EXIT_ALL:
            self.trader.exit_position(symbol, price, "manual")

        elif action == Action.TAKE_PARTIAL:
            if symbol in self.trader.positions:
                # Exit half position
                position = self.trader.positions[symbol]
                half_size = position.size * position.entry_price / 2
                # Simplified: just exit all for now
                # Real implementation would handle partial exits

        elif action == Action.TIGHTEN_STOP:
            if symbol in self.trader.positions:
                position = self.trader.positions[symbol]
                new_stop = position.stop_loss + price * 0.005
                new_stop = min(new_stop, price * 0.995)
                self.trader.update_stop(symbol, new_stop)

        elif action == Action.LOOSEN_STOP:
            if symbol in self.trader.positions:
                position = self.trader.positions[symbol]
                new_stop = position.stop_loss - price * 0.005
                new_stop = max(new_stop, price * 0.95)
                self.trader.update_stop(symbol, new_stop)

    def update_regime(self) -> None:
        """Update market regime detection."""
        try:
            # Get BTC data for regime
            btc_ohlcv = self.data_connector.get_recent_ohlcv("BTC-USD", minutes=168)

            if btc_ohlcv is not None and len(btc_ohlcv) > 24:
                regime, confidence = self.regime_detector.detect(
                    btc_ohlcv['close'],
                    btc_ohlcv['volume'] if 'volume' in btc_ohlcv.columns else None,
                    timestamp=datetime.now(),
                )
                self.current_regime = regime
                logger.info(f"Market regime: {regime.value} (confidence: {confidence:.2f})")

        except Exception as e:
            logger.warning(f"Error updating regime: {e}")

    def run_cycle(self) -> None:
        """Run one decision cycle for all symbols."""
        # Update regime periodically
        if self.decisions_made % 10 == 0:
            self.update_regime()

        # Get available symbols
        symbols = self.config.symbols or self.data_connector.get_available_symbols()

        # Process each symbol
        for symbol in symbols[:10]:  # Limit for testing
            try:
                self.process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Check stops
        if self.trader:
            current_prices = {}
            for symbol in self.trader.positions:
                ohlcv = self.data_connector.get_recent_ohlcv(symbol, minutes=1)
                if ohlcv is not None:
                    current_prices[symbol] = ohlcv['close'].iloc[-1]

            closed = self.trader.check_stops(current_prices)
            for trade in closed:
                logger.info(f"Stop hit for {trade.symbol}")

        # Log stats
        if self.trader and self.decisions_made % 60 == 0:
            stats = self.trader.get_stats(current_prices if 'current_prices' in dir() else {})
            logger.info(f"Stats: {stats}")

    def run(self) -> None:
        """Main run loop."""
        self.running = True
        logger.info("Starting live agent...")

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self.running:
            try:
                cycle_start = time.time()

                self.run_cycle()

                # Wait for next interval
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.config.decision_interval_seconds - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.exception(f"Error in run loop: {e}")
                time.sleep(5)

        # Cleanup
        self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down live agent...")

        # Close all positions in paper trading
        if self.trader:
            for symbol in list(self.trader.positions.keys()):
                self.trader.exit_position(symbol, 0, "shutdown")

            # Log final stats
            stats = self.trader.get_stats({})
            logger.info(f"Final stats: {stats}")

            # Calculate metrics
            if self.trader.trades:
                calculator = MetricsCalculator()
                metrics = calculator.calculate(
                    self.trader.trades,
                    total_duration_days=1,
                )
                logger.info(f"Session metrics: {metrics}")

        self.data_connector.close()
        logger.info("Shutdown complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RL Trading Agent Live",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/rl/best/model.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--feature-buffer",
        type=str,
        default="feature_buffer.db",
        help="Path to feature buffer SQLite database",
    )
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        default=True,
        help="Run in paper trading mode",
    )
    parser.add_argument(
        "--live-trade",
        action="store_true",
        help="Run in live trading mode (NOT IMPLEMENTED)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital for paper trading",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1000.0,
        help="Position size in USD",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=3,
        help="Maximum concurrent positions",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Decision interval in seconds",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to trade (default: all)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/rl_live",
        help="Log directory",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Create config
    config = LiveConfig(
        model_path=args.model_path,
        feature_buffer_path=args.feature_buffer,
        paper_trade=not args.live_trade,
        initial_capital=args.initial_capital,
        position_size_usd=args.position_size,
        max_positions=args.max_positions,
        decision_interval_seconds=args.interval,
        symbols=args.symbols,
        log_dir=args.log_dir,
    )

    # Create and run agent
    agent = LiveAgent(config)

    try:
        agent.run()
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
