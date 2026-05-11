#!/usr/bin/env python3
"""
Enhanced Batch Feature Calculator - 24 Features for Pre-Spike Detection

Calculates optimized feature set from candles table:
- 9 price features
- 5 volume features
- 6 microstructure features
- 4 multi-timeframe features

Anti-leakage: All features use shift(1) - only past data
Performance: Vectorized operations with timing profiler

Usage:
    python scripts/calculate_features_batch_v2.py --db "/path/to/database.db"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
import time
from loguru import logger
import argparse

# Feature calculation modules (not needed - implementing manually)
# from src.features.microstructure import calculate_roll_measure, calculate_vpin
# from src.features.price_indicators import calculate_price_features
# from src.features.volume_indicators import calculate_volume_features


class FeatureTimer:
    """Simple timing profiler for feature calculation"""

    def __init__(self):
        self.times = {}

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        pass

    def record(self, name: str, start_time: float):
        """Record time for a feature group"""
        elapsed = time.time() - start_time
        self.times[name] = elapsed

    def summary(self):
        """Print timing summary"""
        total = sum(self.times.values())
        logger.info("="*60)
        logger.info("FEATURE CALCULATION TIMING")
        logger.info("="*60)
        for name, elapsed in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = (elapsed / total * 100) if total > 0 else 0
            logger.info(f"  {name:30s}: {elapsed:6.2f}s ({pct:5.1f}%)")
        logger.info("="*60)
        logger.info(f"  {'TOTAL':30s}: {total:6.2f}s")
        logger.info("="*60)


class EnhancedFeatureCalculator:
    """
    Calculate 24 optimized features for pre-spike pattern detection.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.timer = FeatureTimer()
        logger.info(f"EnhancedFeatureCalculator initialized: {db_path}")

    def get_all_symbols(self):
        """Get all symbols from candles table"""
        conn = duckdb.connect(self.db_path)
        try:
            query = "SELECT DISTINCT symbol FROM candles ORDER BY symbol"
            result = conn.execute(query).fetchall()
            symbols = [r[0] for r in result]
            return symbols
        finally:
            conn.close()

    def load_candles(self, symbol: str, timeframe: str = '1m'):
        """Load candles for a symbol"""
        conn = duckdb.connect(self.db_path)
        try:
            query = f"""
                SELECT
                    timestamp, open, high, low, close, volume,
                    buy_volume, sell_volume, num_trades, is_filled
                FROM candles
                WHERE symbol = '{symbol}'
                ORDER BY timestamp ASC
            """
            df = conn.execute(query).df()

            if df.empty:
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            return df
        finally:
            conn.close()

    def calculate_price_features_manual(self, df: pd.DataFrame):
        """
        Calculate 9 price features manually with anti-leakage
        """
        features = pd.DataFrame(index=df.index)

        # 1. Returns (percent change)
        features['returns'] = df['close'].pct_change().shift(1)

        # 2-4. MACD (12,26,9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        features['MACD'] = macd.shift(1)
        features['MACD_signal'] = macd_signal.shift(1)
        features['MACD_hist'] = (macd - macd_signal).shift(1)

        # 5. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['RSI_14'] = rsi.shift(1)

        # 6. NATR (Normalized ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        natr = (atr / df['close']) * 100
        features['NATR'] = natr.shift(1)

        # 7-8. Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        bb_width = (std20 * 4) / sma20  # Width normalized by price
        features['BB_width'] = bb_width.shift(1)

        # BB squeeze: width relative to recent average width
        bb_width_ma = bb_width.rolling(window=20).mean()
        bb_squeeze = bb_width / bb_width_ma
        features['BB_squeeze'] = bb_squeeze.shift(1)

        # 9. VWAP distance
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        vwap_distance = (df['close'] - vwap) / vwap
        features['VWAP_distance'] = vwap_distance.shift(1)

        return features

    def calculate_volume_features_manual(self, df: pd.DataFrame):
        """
        Calculate 5 volume features with anti-leakage
        """
        features = pd.DataFrame(index=df.index)

        # 1. Volume z-score (normalized volume anomaly)
        volume_mean = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        volume_zscore = (df['volume'] - volume_mean) / volume_std
        features['volume_zscore'] = volume_zscore.shift(1)

        # 2. Volume rate of change
        volume_roc = df['volume'].pct_change(periods=5)
        features['volume_roc'] = volume_roc.shift(1)

        # 3. OBV (On-Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['OBV'] = obv.shift(1)

        # 4. Trade count
        features['trade_count'] = df['num_trades'].shift(1)

        # 5. Buy/Sell ratio
        buy_sell_ratio = df['buy_volume'] / (df['sell_volume'] + 1e-10)  # Avoid division by zero
        features['buy_sell_ratio'] = buy_sell_ratio.shift(1)

        return features

    def calculate_microstructure_features_manual(self, df: pd.DataFrame):
        """
        Calculate 6 microstructure features with anti-leakage
        """
        features = pd.DataFrame(index=df.index)

        # 1. Roll measure (bid-ask spread estimator, top predictor)
        # Based on consecutive price changes
        price_changes = df['close'].diff()
        roll_cov = price_changes.rolling(window=50).cov(price_changes.shift(1))
        roll_measure = 2 * np.sqrt(-roll_cov.clip(upper=0))
        features['roll_measure'] = roll_measure.shift(1)

        # 2-3. VPIN and Order Flow Imbalance
        # These require buy/sell volume which we now have!
        buy_volume = df['buy_volume'].fillna(0)
        sell_volume = df['sell_volume'].fillna(0)
        total_volume = buy_volume + sell_volume + 1e-10

        # Order flow imbalance: (buy - sell) / (buy + sell)
        order_flow_imbalance = (buy_volume - sell_volume) / total_volume
        features['order_flow_imbalance'] = order_flow_imbalance.shift(1)

        # VPIN: rolling standard deviation of order flow imbalance
        vpin = order_flow_imbalance.rolling(window=50).std()
        features['vpin'] = vpin.shift(1)

        # 4-6. Order book features (from order_book_features table if available)
        # For now, create placeholder versions from candle data
        # These will be enhanced if order_book_features table is joined

        # Proxy for bid-ask spread: high-low range as % of price
        bid_ask_spread_pct = ((df['high'] - df['low']) / df['close']) * 100
        features['bid_ask_spread_pct'] = bid_ask_spread_pct.shift(1)

        # Proxy for depth ratio: volume traded relative to typical
        volume_ma = df['volume'].rolling(window=20).mean()
        order_book_depth_ratio = df['volume'] / (volume_ma + 1e-10)
        features['order_book_depth_ratio'] = order_book_depth_ratio.shift(1)

        # Large order imbalance: detect abnormally large buy vs sell trades
        large_buy_threshold = buy_volume.rolling(window=50).quantile(0.9)
        large_sell_threshold = sell_volume.rolling(window=50).quantile(0.9)
        large_buy = (buy_volume > large_buy_threshold).astype(int)
        large_sell = (sell_volume > large_sell_threshold).astype(int)
        large_order_imbalance = large_buy - large_sell
        features['large_order_imbalance'] = large_order_imbalance.shift(1)

        return features

    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str):
        """Resample 1-minute candles to higher timeframe"""
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'num_trades': 'sum'
        }

        resampled = df.resample(timeframe).agg(ohlc_dict)
        return resampled.dropna()

    def calculate_multitime_frame_features(self, df: pd.DataFrame):
        """
        Calculate 4 multi-timeframe features (5m, 15m)
        """
        features = pd.DataFrame(index=df.index)

        # Resample to 5-minute and 15-minute
        df_5m = self.resample_to_timeframe(df, '5min')
        df_15m = self.resample_to_timeframe(df, '15min')

        # Calculate features on higher timeframes
        returns_5m = df_5m['close'].pct_change()
        volume_zscore_5m = (df_5m['volume'] - df_5m['volume'].rolling(20).mean()) / df_5m['volume'].rolling(20).std()

        returns_15m = df_15m['close'].pct_change()
        volume_zscore_15m = (df_15m['volume'] - df_15m['volume'].rolling(20).mean()) / df_15m['volume'].rolling(20).std()

        # Align back to 1-minute index with forward-fill
        features['returns_5m'] = returns_5m.reindex(df.index, method='ffill').shift(1)
        features['volume_zscore_5m'] = volume_zscore_5m.reindex(df.index, method='ffill').shift(1)
        features['returns_15m'] = returns_15m.reindex(df.index, method='ffill').shift(1)
        features['volume_zscore_15m'] = volume_zscore_15m.reindex(df.index, method='ffill').shift(1)

        return features

    def calculate_features(self, symbol: str):
        """
        Calculate all 24 features for a symbol
        """
        logger.info(f"Calculating features for {symbol}...")

        # Load candles
        t0 = time.time()
        candles = self.load_candles(symbol)

        if candles.empty:
            logger.warning(f"No candles found for {symbol}")
            return None

        if len(candles) < 100:
            logger.warning(f"Insufficient data for {symbol} ({len(candles)} candles)")
            return None

        self.timer.record('data_loading', t0)

        # Calculate feature groups with timing
        t1 = time.time()
        price_features = self.calculate_price_features_manual(candles)
        self.timer.record('price_features', t1)

        t2 = time.time()
        volume_features = self.calculate_volume_features_manual(candles)
        self.timer.record('volume_features', t2)

        t3 = time.time()
        microstructure_features = self.calculate_microstructure_features_manual(candles)
        self.timer.record('microstructure_features', t3)

        t4 = time.time()
        multitime_features = self.calculate_multitime_frame_features(candles)
        self.timer.record('multitime_features', t4)

        # Combine all features
        features = pd.concat([
            price_features,
            volume_features,
            microstructure_features,
            multitime_features
        ], axis=1)

        # Add metadata
        features['symbol'] = symbol
        features['timeframe'] = '1m'

        # Reset index to have timestamp as column
        features = features.reset_index()

        logger.success(f"✓ {symbol}: {len(features)} rows, {len(features.columns)-2} features")

        return features

    def write_features(self, features: pd.DataFrame):
        """Write features to database"""
        if features.empty:
            return

        conn = duckdb.connect(self.db_path)
        try:
            # Create features table if doesn't exist
            # Get column types dynamically
            columns = []
            for col, dtype in features.dtypes.items():
                if col == 'timestamp':
                    sql_type = 'DATETIME'
                elif col in ['symbol', 'timeframe']:
                    sql_type = 'TEXT'
                elif dtype == 'float64':
                    sql_type = 'REAL'
                elif dtype == 'int64':
                    sql_type = 'INTEGER'
                else:
                    sql_type = 'REAL'
                columns.append(f"{col} {sql_type}")

            create_table = f"""
                CREATE TABLE IF NOT EXISTS features (
                    {', '.join(columns)}
                )
            """
            conn.execute(create_table)

            # Write features using DuckDB
            conn.register('temp_features', features)
            conn.execute("INSERT INTO features SELECT * FROM temp_features")
            conn.unregister('temp_features')
            logger.info(f"Wrote {len(features)} feature rows to database")

        finally:
            conn.close()

    def process_all_symbols(self, symbols=None):
        """Process all symbols"""
        if symbols is None:
            symbols = self.get_all_symbols()

        logger.info(f"Processing {len(symbols)} symbols...")
        print()

        success = 0
        failed = 0

        overall_start = time.time()

        for i, symbol in enumerate(symbols):
            logger.info(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")

            try:
                features = self.calculate_features(symbol)

                if features is not None:
                    self.write_features(features)
                    success += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        overall_time = time.time() - overall_start

        # Final summary
        logger.info("="*80)
        logger.success("FEATURE GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Symbols processed: {success + failed}")
        logger.success(f"✓ Success: {success} symbols")
        if failed > 0:
            logger.warning(f"✗ Failed: {failed} symbols")
        logger.info(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} min)")
        logger.info(f"Avg time per symbol: {overall_time/len(symbols):.1f}s")
        logger.info("="*80)

        # Print timing breakdown
        self.timer.summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Calculate 24 features for ML training')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--symbols', help='Comma-separated symbols (default: all)')

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols:
        if args.symbols.lower() == 'all':
            symbols = None
        else:
            symbols = [s.strip() for s in args.symbols.split(',')]

    # Initialize calculator
    calculator = EnhancedFeatureCalculator(args.db)

    # Process symbols
    calculator.process_all_symbols(symbols)


if __name__ == "__main__":
    main()
