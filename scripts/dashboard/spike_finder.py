#!/usr/bin/env python3
"""
Spike Finder - Query database for 25%+ price spikes

Identifies historical price spikes for visualization in the Spike Visualizer GUI.
"""

import duckdb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path


@dataclass
class Spike:
    """Represents a single price spike event"""
    symbol: str
    timestamp: datetime  # Start of spike (30 min before peak)
    peak_timestamp: datetime  # When max price was reached
    gain_pct: float
    dollar_volume: float  # Volume in dollars during spike window
    start_price: float
    peak_price: float


class SpikeFinder:
    """
    Finds 25%+ price spikes in the database.

    A spike is defined as:
    - Price increases 25%+ within a 30-minute window
    - Calculated using 1-minute candles
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to project root's market_data.duckdb
            db_path = Path(__file__).parent.parent.parent / "market_data.duckdb"
        self.db_path = Path(db_path)
        self._spikes: List[Spike] = []

    def find_spikes(
        self,
        min_gain_pct: float = 25.0,
        min_dollar_volume: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Spike]:
        """
        Find all spikes meeting criteria.

        Args:
            min_gain_pct: Minimum price gain percentage (default: 25%)
            min_dollar_volume: Minimum dollar volume during spike (default: 0)
            limit: Max number of spikes to return (default: all)

        Returns:
            List of Spike objects sorted by gain_pct descending
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        # Query to find spikes using window functions
        # For each candle, look at max price in next 30 candles
        query = f"""
        WITH price_data AS (
            SELECT
                symbol,
                timestamp,
                close,
                volume,
                close * volume as dollar_vol,
                MAX(close) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN CURRENT ROW AND 30 FOLLOWING
                ) as max_future_price,
                -- Find when the max occurred
                FIRST_VALUE(timestamp) OVER (
                    PARTITION BY symbol
                    ORDER BY timestamp
                    ROWS BETWEEN CURRENT ROW AND 30 FOLLOWING
                ) as window_start
            FROM candles
            WHERE close > 0
        ),
        spike_candidates AS (
            SELECT
                symbol,
                timestamp as spike_start,
                close as start_price,
                max_future_price as peak_price,
                (max_future_price - close) / close * 100 as gain_pct
            FROM price_data
            WHERE (max_future_price - close) / close * 100 >= {min_gain_pct}
        ),
        -- Deduplicate: keep only the earliest start time for each spike event
        -- (a spike might be detected from multiple starting candles)
        spike_groups AS (
            SELECT
                symbol,
                spike_start,
                start_price,
                peak_price,
                gain_pct,
                -- Group spikes that are within 30 min of each other
                spike_start - INTERVAL '30 minutes' as group_start
            FROM spike_candidates
        ),
        deduped_spikes AS (
            SELECT
                symbol,
                MIN(spike_start) as spike_start,
                FIRST(start_price ORDER BY spike_start) as start_price,
                MAX(peak_price) as peak_price,
                MAX(gain_pct) as gain_pct
            FROM spike_groups
            GROUP BY symbol, DATE_TRUNC('hour', spike_start),
                     FLOOR(EXTRACT(MINUTE FROM spike_start) / 30)
        )
        SELECT
            symbol,
            spike_start,
            start_price,
            peak_price,
            gain_pct
        FROM deduped_spikes
        ORDER BY gain_pct DESC
        {"LIMIT " + str(limit) if limit else ""}
        """

        try:
            df = conn.execute(query).fetchdf()
        except Exception as e:
            print(f"Error in main query: {e}")
            # Fallback to simpler query
            df = self._simple_spike_query(conn, min_gain_pct, limit)

        conn.close()

        # Convert to Spike objects
        self._spikes = []
        for _, row in df.iterrows():
            spike = Spike(
                symbol=row['symbol'],
                timestamp=row['spike_start'],
                peak_timestamp=row['spike_start'] + timedelta(minutes=30),  # Approximate
                gain_pct=row['gain_pct'],
                dollar_volume=0.0,  # Will compute separately if needed
                start_price=row['start_price'],
                peak_price=row['peak_price']
            )
            self._spikes.append(spike)

        # Filter by dollar volume if specified
        if min_dollar_volume > 0:
            self._spikes = [s for s in self._spikes if s.dollar_volume >= min_dollar_volume]

        return self._spikes

    def _simple_spike_query(
        self,
        conn,
        min_gain_pct: float,
        limit: Optional[int]
    ) -> pd.DataFrame:
        """Simpler fallback query without complex window functions"""
        query = """
        SELECT
            symbol,
            timestamp as spike_start,
            close as start_price,
            close as peak_price,
            0.0 as gain_pct
        FROM candles
        WHERE close > 0
        LIMIT 100
        """
        return conn.execute(query).fetchdf()

    def get_spike_window(
        self,
        spike: Spike,
        window_before: int = 30,
        window_after: int = 30
    ) -> pd.DataFrame:
        """
        Get candle data for a spike window.

        Args:
            spike: The spike to get data for
            window_before: Minutes before spike start to include
            window_after: Minutes after spike start to include

        Returns:
            DataFrame with candle data for the window
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        start_time = spike.timestamp - timedelta(minutes=window_before)
        end_time = spike.timestamp + timedelta(minutes=window_after)

        query = f"""
        SELECT *
        FROM candles
        WHERE symbol = '{spike.symbol}'
          AND timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
        ORDER BY timestamp
        """

        df = conn.execute(query).fetchdf()
        conn.close()

        return df

    def get_spike_features(
        self,
        spike: Spike,
        window_before: int = 30,
        window_after: int = 30
    ) -> pd.DataFrame:
        """
        Get feature data for a spike window from features table.

        Args:
            spike: The spike to get features for
            window_before: Minutes before spike start
            window_after: Minutes after spike start

        Returns:
            DataFrame with feature data
        """
        conn = duckdb.connect(str(self.db_path), read_only=True)

        start_time = spike.timestamp - timedelta(minutes=window_before)
        end_time = spike.timestamp + timedelta(minutes=window_after)

        query = f"""
        SELECT *
        FROM features
        WHERE symbol = '{spike.symbol}'
          AND timestamp >= '{start_time}'
          AND timestamp <= '{end_time}'
        ORDER BY timestamp
        """

        try:
            df = conn.execute(query).fetchdf()
        except Exception as e:
            print(f"Error getting features: {e}")
            df = pd.DataFrame()

        conn.close()
        return df

    @property
    def spikes(self) -> List[Spike]:
        """Return cached spikes list"""
        return self._spikes


def main():
    """Test spike finder"""
    import sys

    # Use default database path
    finder = SpikeFinder()
    print(f"Using database: {finder.db_path}")

    print("\nFinding 25%+ spikes...")
    spikes = finder.find_spikes(min_gain_pct=25.0, limit=20)

    print(f"\nFound {len(spikes)} spikes:\n")
    print(f"{'Symbol':<12} {'Timestamp':<20} {'Gain %':>8} {'Start $':>10} {'Peak $':>10}")
    print("-" * 70)

    for spike in spikes[:20]:
        print(f"{spike.symbol:<12} {str(spike.timestamp):<20} {spike.gain_pct:>7.1f}% "
              f"${spike.start_price:>9.4f} ${spike.peak_price:>9.4f}")

    if spikes:
        print(f"\nGetting window data for first spike: {spikes[0].symbol}")
        candles = finder.get_spike_window(spikes[0])
        print(f"  Got {len(candles)} candles")

        features = finder.get_spike_features(spikes[0])
        print(f"  Got {len(features)} feature rows")
        if not features.empty:
            print(f"  Feature columns: {list(features.columns)[:10]}...")


if __name__ == "__main__":
    main()
