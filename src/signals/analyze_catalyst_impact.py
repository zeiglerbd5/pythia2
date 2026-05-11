"""
Catalyst Impact Analysis

Correlates historical catalyst signals with price movements to:
1. Identify which catalyst types preceded 20%+ spikes
2. Measure average price impact by catalyst type
3. Calculate hit rate (% of signals that led to spikes)
4. Build training labels for ML model

Output: CSV with catalyst signals and their price outcomes.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger
import os


class CatalystImpactAnalyzer:
    """Analyzes historical impact of catalyst signals on price."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)

    def analyze_all_signals(
        self,
        lookforward_hours: List[int] = [1, 4, 12, 24],
        min_spike_pct: float = 10.0,
    ) -> pd.DataFrame:
        """
        Analyze price impact for all catalyst signals.

        Args:
            lookforward_hours: Time windows to measure price change
            lookforward_hours: Time windows to check for price moves
            min_spike_pct: Minimum % move to count as a "spike"

        Returns:
            DataFrame with signals and their outcomes
        """
        logger.info("Loading catalyst signals...")

        # Get all signals with symbols that exist in OHLCV
        signals_df = self.conn.execute("""
            SELECT DISTINCT
                ns.symbol,
                ns.timestamp,
                ns.source,
                ns.event_type,
                ns.confidence,
                ns.title,
                ns.event_priority
            FROM news_signals ns
            WHERE ns.symbol != 'UNKNOWN-USD'
              AND ns.timestamp >= '2025-10-01'
              AND EXISTS (
                  SELECT 1 FROM ohlcv o
                  WHERE o.symbol = ns.symbol
                  LIMIT 1
              )
            ORDER BY ns.timestamp
        """).df()

        logger.info(f"Found {len(signals_df)} signals to analyze")

        if len(signals_df) == 0:
            return pd.DataFrame()

        # Calculate price changes for each signal
        results = []
        total = len(signals_df)

        for idx, row in signals_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing {idx}/{total}...")

            symbol = row["symbol"]
            signal_time = row["timestamp"]

            # Get price at signal time and future prices
            price_data = self._get_price_window(symbol, signal_time, max(lookforward_hours))

            if price_data is None:
                continue

            result = {
                "symbol": symbol,
                "signal_time": signal_time,
                "source": row["source"],
                "event_type": row["event_type"],
                "confidence": row["confidence"],
                "title": row["title"][:100] if row["title"] else "",
                "priority": row["event_priority"],
                "price_at_signal": price_data["price_at_signal"],
            }

            # Calculate returns for each lookforward window
            max_return = 0
            for hours in lookforward_hours:
                col_name = f"return_{hours}h"
                ret = price_data.get(f"return_{hours}h", 0)
                result[col_name] = ret
                max_return = max(max_return, ret)

            result["max_return"] = max_return
            result["is_spike"] = max_return >= min_spike_pct

            results.append(result)

        results_df = pd.DataFrame(results)
        logger.info(f"Analyzed {len(results_df)} signals with price data")

        return results_df

    def _get_price_window(
        self, symbol: str, signal_time: datetime, max_hours: int
    ) -> Dict:
        """Get price at signal time and future prices."""
        try:
            # Get price at signal time (closest candle)
            base_price_query = """
                SELECT close as price, timestamp
                FROM ohlcv
                WHERE symbol = ?
                  AND timeframe = '5m'
                  AND timestamp >= ? - INTERVAL 30 MINUTE
                  AND timestamp <= ? + INTERVAL 30 MINUTE
                ORDER BY ABS(EPOCH(timestamp) - EPOCH(?::TIMESTAMP))
                LIMIT 1
            """
            base_result = self.conn.execute(
                base_price_query, [symbol, signal_time, signal_time, signal_time]
            ).fetchone()

            if not base_result:
                return None

            base_price = base_result[0]

            # Get max price in each lookforward window
            result = {"price_at_signal": base_price}

            for hours in [1, 4, 12, 24]:
                if hours > max_hours:
                    continue

                max_price_query = f"""
                    SELECT MAX(high) as max_price
                    FROM ohlcv
                    WHERE symbol = ?
                      AND timeframe = '5m'
                      AND timestamp > ?
                      AND timestamp <= ? + INTERVAL '{hours} hours'
                """
                max_result = self.conn.execute(
                    max_price_query, [symbol, signal_time, signal_time]
                ).fetchone()

                if max_result and max_result[0]:
                    max_price = max_result[0]
                    pct_return = ((max_price - base_price) / base_price) * 100
                    result[f"return_{hours}h"] = round(pct_return, 2)
                else:
                    result[f"return_{hours}h"] = 0

            return result

        except Exception as e:
            logger.debug(f"Price lookup error for {symbol}: {e}")
            return None

    def summarize_by_event_type(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize performance by event type."""
        if len(results_df) == 0:
            return pd.DataFrame()

        summary = results_df.groupby("event_type").agg({
            "symbol": "count",
            "return_1h": "mean",
            "return_4h": "mean",
            "return_12h": "mean",
            "return_24h": "mean",
            "max_return": "mean",
            "is_spike": "mean",
        }).round(2)

        summary.columns = [
            "count", "avg_1h", "avg_4h", "avg_12h", "avg_24h", "avg_max", "spike_rate"
        ]
        summary["spike_rate"] = (summary["spike_rate"] * 100).round(1)

        return summary.sort_values("spike_rate", ascending=False)

    def summarize_by_source(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize performance by source."""
        if len(results_df) == 0:
            return pd.DataFrame()

        summary = results_df.groupby("source").agg({
            "symbol": "count",
            "return_4h": "mean",
            "return_24h": "mean",
            "max_return": "mean",
            "is_spike": "mean",
        }).round(2)

        summary.columns = ["count", "avg_4h", "avg_24h", "avg_max", "spike_rate"]
        summary["spike_rate"] = (summary["spike_rate"] * 100).round(1)

        return summary.sort_values("spike_rate", ascending=False)

    def find_best_signals(
        self, results_df: pd.DataFrame, min_spike_pct: float = 20.0, top_n: int = 20
    ) -> pd.DataFrame:
        """Find signals that preceded the biggest spikes."""
        if len(results_df) == 0:
            return pd.DataFrame()

        spikes = results_df[results_df["max_return"] >= min_spike_pct].copy()
        spikes = spikes.sort_values("max_return", ascending=False)

        return spikes.head(top_n)[
            ["symbol", "signal_time", "event_type", "source", "max_return", "title"]
        ]


def main():
    """Run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze catalyst impact on prices")
    parser.add_argument("--db", default="full_pythia.duckdb", help="Database path")
    parser.add_argument("--output", default="catalyst_analysis.csv", help="Output CSV")
    parser.add_argument("--spike-pct", type=float, default=10.0, help="Min spike percent")
    args = parser.parse_args()

    analyzer = CatalystImpactAnalyzer(db_path=args.db)

    print("=" * 70)
    print("CATALYST IMPACT ANALYSIS")
    print("=" * 70)

    # Run analysis
    results = analyzer.analyze_all_signals(min_spike_pct=args.spike_pct)

    if len(results) == 0:
        print("No signals with matching price data found")
        return

    # Save full results
    results.to_csv(args.output, index=False)
    print(f"\nFull results saved to: {args.output}")

    # Print summaries
    print("\n" + "=" * 70)
    print("PERFORMANCE BY EVENT TYPE")
    print("=" * 70)
    by_type = analyzer.summarize_by_event_type(results)
    print(by_type.to_string())

    print("\n" + "=" * 70)
    print("PERFORMANCE BY SOURCE")
    print("=" * 70)
    by_source = analyzer.summarize_by_source(results)
    print(by_source.to_string())

    print("\n" + "=" * 70)
    print(f"TOP SIGNALS PRECEDING {args.spike_pct}%+ SPIKES")
    print("=" * 70)
    best = analyzer.find_best_signals(results, min_spike_pct=args.spike_pct)
    if len(best) > 0:
        for _, row in best.iterrows():
            print(
                f"{row['signal_time'].strftime('%Y-%m-%d %H:%M')} | "
                f"{row['symbol']:12} | {row['event_type']:15} | "
                f"+{row['max_return']:.1f}% | {row['title'][:40]}..."
            )
    else:
        print("No spikes found")

    # Overall stats
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total signals analyzed: {len(results)}")
    print(f"Signals with {args.spike_pct}%+ spike: {results['is_spike'].sum()} ({results['is_spike'].mean()*100:.1f}%)")
    print(f"Average max return: {results['max_return'].mean():.2f}%")
    print(f"Median max return: {results['max_return'].median():.2f}%")


if __name__ == "__main__":
    main()
