"""Run backtest with filtered symbols and optimized params."""
import sys
sys.path.insert(0, '.')

from src.backtesting.catalyst_backtest import (
    CatalystBacktester, BacktestConfig, print_results
)
import duckdb

# Major liquid symbols with good signal coverage
LIQUID_SYMBOLS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'AAVE-USD', 'XRP-USD',
    'UNI-USD', 'DOGE-USD', 'COW-USD', 'COMP-USD', 'LINK-USD',
    'LTC-USD', 'OP-USD', 'TRUMP-USD', 'BCH-USD'
]

class FilteredBacktester(CatalystBacktester):
    """Backtest with symbol filter."""

    def _load_signals(self):
        symbols_str = ", ".join(f"'{s}'" for s in LIQUID_SYMBOLS)

        query = f"""
            SELECT
                symbol,
                timestamp,
                source,
                event_type,
                title,
                event_priority as priority
            FROM news_signals
            WHERE timestamp >= '2025-12-10'
              AND symbol IN ({symbols_str})
        """

        if self.config.min_signal_priority:
            query += f" AND event_priority >= {self.config.min_signal_priority}"

        if self.config.signal_types:
            types_str = ", ".join(f"'{t}'" for t in self.config.signal_types)
            query += f" AND event_type IN ({types_str})"

        query += " ORDER BY timestamp"

        return self.conn.execute(query).df()


def main():
    configs = [
        # Conservative: small positions, tight stops
        BacktestConfig(
            initial_capital=10000,
            position_size_pct=5,
            take_profit_pct=8,
            stop_loss_pct=4,
            max_hold_hours=12,
            min_signal_priority=0.7,
        ),
        # Moderate: medium positions
        BacktestConfig(
            initial_capital=10000,
            position_size_pct=10,
            take_profit_pct=15,
            stop_loss_pct=7,
            max_hold_hours=24,
            min_signal_priority=0.7,
        ),
        # Aggressive: larger positions, wider stops
        BacktestConfig(
            initial_capital=10000,
            position_size_pct=15,
            take_profit_pct=20,
            stop_loss_pct=10,
            max_hold_hours=48,
            min_signal_priority=0.6,
        ),
    ]

    print("=" * 70)
    print("FILTERED BACKTEST: Major Liquid Coins Only")
    print(f"Symbols: {', '.join(LIQUID_SYMBOLS[:7])}...")
    print("=" * 70)

    backtester = FilteredBacktester(db_path="full_pythia.duckdb")

    for i, config in enumerate(configs):
        print(f"\n{'='*70}")
        print(f"CONFIG {i+1}: TP={config.take_profit_pct}%, SL={config.stop_loss_pct}%, " +
              f"Size={config.position_size_pct}%, Hold={config.max_hold_hours}h")
        print("=" * 70)

        result = backtester.run(config)

        print(f"\nResults:")
        print(f"  Trades:       {result.total_trades}")
        print(f"  Win Rate:     {result.win_rate:.1f}%")
        print(f"  Total PnL:    ${result.total_pnl_usd:+,.2f} ({result.total_pnl_pct:+.1f}%)")
        print(f"  Sharpe:       {result.sharpe_ratio:.2f}")
        print(f"  Max DD:       {result.max_drawdown_pct:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        if result.trades:
            print(f"\n  Exit Reasons: TP={result.exits_take_profit}, " +
                  f"SL={result.exits_stop_loss}, Time={result.exits_time_limit}")

            # Best trades
            best = sorted(result.trades, key=lambda t: t.pnl_pct, reverse=True)[:3]
            print(f"\n  Best Trades:")
            for t in best:
                print(f"    {t.symbol:12} {t.pnl_pct:+.1f}% ({t.signal_type})")

if __name__ == "__main__":
    main()
