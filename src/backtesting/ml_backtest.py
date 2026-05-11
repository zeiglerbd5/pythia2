"""
ML-Enhanced Backtest

Uses the spike prediction model to filter trades:
- Only enters when model predicts spike probability above threshold
- Combines catalyst signals with market context
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import duckdb
from loguru import logger


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    position_size: float = 0.0
    pred_probability: float = 0.0


@dataclass
class MLBacktestConfig:
    """ML backtest configuration."""
    initial_capital: float = 10000.0
    position_size_pct: float = 10.0
    max_positions: int = 3

    # ML threshold
    min_pred_probability: float = 0.3

    # Exit parameters
    take_profit_pct: float = 10.0
    stop_loss_pct: float = 5.0
    max_hold_hours: int = 24

    # Costs
    trading_fee_pct: float = 0.1
    slippage_pct: float = 0.1


@dataclass
class MLBacktestResult:
    """ML backtest results."""
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    exits_take_profit: int = 0
    exits_stop_loss: int = 0
    exits_time_limit: int = 0

    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class MLBacktester:
    """Backtester using ML model predictions to filter trades."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)

    def run(
        self,
        predictions_csv: str,
        features_csv: str,
        config: MLBacktestConfig,
    ) -> MLBacktestResult:
        """
        Run backtest using model predictions.

        Args:
            predictions_csv: CSV with model predictions
            features_csv: CSV with features (for symbol info)
            config: Backtest configuration

        Returns:
            MLBacktestResult with performance metrics
        """
        result = MLBacktestResult()

        # Load predictions and features
        preds = pd.read_csv(predictions_csv, parse_dates=['timestamp'])
        features = pd.read_csv(features_csv, parse_dates=['timestamp'])

        # Merge predictions with features
        signals = preds.merge(
            features[['timestamp', 'symbol', 'price_at_signal']],
            on=['timestamp', 'symbol'],
            how='left'
        )

        # Filter by prediction threshold
        signals = signals[signals['y_pred_proba'] >= config.min_pred_probability]
        signals = signals.sort_values('timestamp')

        logger.info(f"Signals after ML filter: {len(signals)} (threshold: {config.min_pred_probability})")

        if len(signals) == 0:
            return result

        # Initialize state
        capital = config.initial_capital
        equity_curve = [capital]
        open_positions: Dict[str, Trade] = {}

        # Process signals
        for _, signal in signals.iterrows():
            signal_time = signal['timestamp']
            symbol = signal['symbol']

            # Check exits
            closed = self._check_exits(open_positions, signal_time, config, result)
            for trade in closed:
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

            # Skip if already have position
            if symbol in open_positions:
                continue

            # Skip if at max positions
            if len(open_positions) >= config.max_positions:
                continue

            # Enter position
            trade = self._try_entry(signal, capital, config)
            if trade:
                open_positions[symbol] = trade
                capital -= trade.position_size

            equity_curve.append(capital + sum(t.position_size for t in open_positions.values()))

        # Close remaining positions
        if len(signals) > 0:
            end_time = signals['timestamp'].max() + timedelta(hours=1)
            for symbol, trade in list(open_positions.items()):
                self._close_position(trade, end_time, "end_of_backtest", config, result)
                capital += trade.pnl_usd + trade.position_size
                result.trades.append(trade)

        equity_curve.append(capital)
        result.equity_curve = equity_curve

        # Calculate metrics
        self._calculate_metrics(result, config.initial_capital)

        return result

    def _try_entry(
        self,
        signal: pd.Series,
        available_capital: float,
        config: MLBacktestConfig,
    ) -> Optional[Trade]:
        """Try to enter a position."""
        symbol = signal['symbol']
        signal_time = signal['timestamp']

        # Get entry price
        entry_price = self._get_price_at_time(symbol, signal_time + timedelta(minutes=5))
        if entry_price is None:
            return None

        # Position size
        position_size = min(
            available_capital * (config.position_size_pct / 100),
            available_capital * 0.95
        )

        if position_size < 10:
            return None

        # Apply slippage
        entry_price *= (1 + config.slippage_pct / 100)

        return Trade(
            symbol=symbol,
            entry_time=signal_time,
            entry_price=entry_price,
            position_size=position_size,
            pred_probability=signal['y_pred_proba'],
        )

    def _check_exits(
        self,
        open_positions: Dict[str, Trade],
        current_time: datetime,
        config: MLBacktestConfig,
        result: MLBacktestResult,
    ) -> List[Trade]:
        """Check and close positions that hit exit conditions."""
        closed = []

        for symbol, trade in list(open_positions.items()):
            current_price = self._get_price_at_time(symbol, current_time)
            if current_price is None:
                continue

            pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100

            # Take profit
            if pnl_pct >= config.take_profit_pct:
                self._close_position(trade, current_time, "take_profit", config, result)
                closed.append(trade)
                del open_positions[symbol]
                continue

            # Stop loss
            if pnl_pct <= -config.stop_loss_pct:
                self._close_position(trade, current_time, "stop_loss", config, result)
                closed.append(trade)
                del open_positions[symbol]
                continue

            # Time limit
            hold_hours = (current_time - trade.entry_time).total_seconds() / 3600
            if hold_hours >= config.max_hold_hours:
                self._close_position(trade, current_time, "time_limit", config, result)
                closed.append(trade)
                del open_positions[symbol]

        return closed

    def _close_position(
        self,
        trade: Trade,
        exit_time: datetime,
        reason: str,
        config: MLBacktestConfig,
        result: MLBacktestResult,
    ):
        """Close a position and calculate PnL."""
        exit_price = self._get_price_at_time(trade.symbol, exit_time)
        if exit_price is None:
            exit_price = trade.entry_price

        exit_price *= (1 - config.slippage_pct / 100)

        gross_pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        net_pnl_pct = gross_pnl_pct - (2 * config.trading_fee_pct)

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.pnl_pct = net_pnl_pct
        trade.pnl_usd = trade.position_size * (net_pnl_pct / 100)

        if reason == "take_profit":
            result.exits_take_profit += 1
        elif reason == "stop_loss":
            result.exits_stop_loss += 1
        elif reason == "time_limit":
            result.exits_time_limit += 1

    def _get_price_at_time(self, symbol: str, time: datetime) -> Optional[float]:
        """Get price at or near given time."""
        try:
            query = f"""
                SELECT close
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{time - timedelta(minutes=30)}'
                  AND timestamp <= '{time + timedelta(minutes=30)}'
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(TIMESTAMP '{time}'))
                LIMIT 1
            """
            result = self.conn.execute(query).fetchone()
            return result[0] if result else None
        except:
            return None

    def _calculate_metrics(self, result: MLBacktestResult, initial_capital: float):
        """Calculate final performance metrics."""
        if not result.trades:
            return

        result.total_trades = len(result.trades)

        wins = [t for t in result.trades if t.pnl_pct > 0]
        losses = [t for t in result.trades if t.pnl_pct <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(result.trades) * 100 if result.trades else 0

        result.avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        result.avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0

        result.total_pnl_usd = sum(t.pnl_usd for t in result.trades)
        result.total_pnl_pct = (result.total_pnl_usd / initial_capital) * 100

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio
        if len(result.equity_curve) > 1:
            returns = pd.Series(result.equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Max drawdown
        equity = pd.Series(result.equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        result.max_drawdown_pct = abs(drawdown.min()) * 100


def main():
    """Run ML-enhanced backtest."""
    print("=" * 70)
    print("ML-ENHANCED BACKTEST")
    print("=" * 70)

    backtester = MLBacktester()

    # Test different prediction thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    for threshold in thresholds:
        config = MLBacktestConfig(
            initial_capital=10000,
            position_size_pct=10,
            max_positions=3,
            min_pred_probability=threshold,
            take_profit_pct=10,
            stop_loss_pct=5,
            max_hold_hours=24,
        )

        result = backtester.run(
            predictions_csv="spike_predictions.csv",
            features_csv="whale_features.csv",
            config=config,
        )

        print(f"\n{'='*70}")
        print(f"Threshold: {threshold}")
        print(f"{'='*70}")
        print(f"  Trades:       {result.total_trades}")
        print(f"  Win Rate:     {result.win_rate:.1f}%")
        print(f"  Total PnL:    ${result.total_pnl_usd:+,.2f} ({result.total_pnl_pct:+.1f}%)")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")

        if result.trades:
            print(f"\n  Exits: TP={result.exits_take_profit}, SL={result.exits_stop_loss}, Time={result.exits_time_limit}")


if __name__ == "__main__":
    main()
