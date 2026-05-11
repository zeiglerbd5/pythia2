"""
Catalyst Feature Engineering

Transforms raw catalyst signals into ML-ready features:
1. Whale direction parsing (to_exchange = bearish, from_exchange = bullish)
2. Stablecoin filtering
3. Price/volume context at signal time
4. Signal aggregation features

Output: Feature matrix for spike prediction models.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import re


# Stablecoins to exclude (not predictive of price moves)
STABLECOINS = {'USDT', 'USDC', 'DAI', 'PYUSD', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'FRAX', 'LUSD'}

# Known exchanges for direction parsing
KNOWN_EXCHANGES = [
    'binance', 'coinbase', 'kraken', 'bybit', 'okx', 'huobi', 'kucoin',
    'bitfinex', 'gemini', 'bitstamp', 'crypto.com', 'gate.io', 'mexc',
    'htx', 'coinone', 'upbit', 'bithumb', 'fixedfloat'
]


class CatalystFeatureEngineer:
    """Engineers features from catalyst signals for ML models."""

    def __init__(self, db_path: str = "full_pythia.duckdb"):
        self.conn = duckdb.connect(db_path)

    def build_whale_features(self, min_usd: float = 500_000) -> pd.DataFrame:
        """
        Build feature matrix from whale signals with direction parsing.

        Args:
            min_usd: Minimum USD value to include

        Returns:
            DataFrame with engineered features
        """
        logger.info("Building whale features...")

        # Load whale signals
        signals = self.conn.execute("""
            SELECT
                symbol,
                timestamp,
                title,
                event_priority,
                sentiment_score
            FROM news_signals
            WHERE event_type = 'whale_move'
              AND timestamp >= '2025-10-01'
            ORDER BY timestamp
        """).df()

        logger.info(f"Loaded {len(signals)} whale signals")

        if len(signals) == 0:
            return pd.DataFrame()

        # Parse features from each signal
        features = []
        for _, row in signals.iterrows():
            feat = self._parse_whale_signal(row)
            if feat is not None:
                features.append(feat)

        df = pd.DataFrame(features)
        logger.info(f"Parsed {len(df)} valid whale features")

        # Filter stablecoins
        df = df[~df['asset'].isin(STABLECOINS)]
        logger.info(f"After stablecoin filter: {len(df)} signals")

        # Add price context
        df = self._add_price_context(df)

        # Add forward returns (labels)
        df = self._add_forward_returns(df)

        return df

    def _parse_whale_signal(self, row: pd.Series) -> Optional[Dict]:
        """Parse a whale signal title into structured features."""
        title = row['title'] or ""

        # Extract amount and asset
        # Patterns: "1,234 BTC ($1,234,567)" or "1,234 BTC"
        amount_match = re.search(r'([\d,\.]+)\s+([A-Z]{2,10})', title)
        if not amount_match:
            return None

        amount = float(amount_match.group(1).replace(',', ''))
        asset = amount_match.group(2)

        # Extract USD value
        usd_match = re.search(r'\$([\d,\.]+)', title)
        usd_value = float(usd_match.group(1).replace(',', '')) if usd_match else 0

        # Parse direction from "from → to" pattern
        direction = 'unknown'
        from_exchange = False
        to_exchange = False

        if '→' in title:
            parts = title.split('|')
            if len(parts) > 1:
                flow_part = parts[1].strip()
                if '→' in flow_part:
                    from_part, to_part = flow_part.split('→')
                    from_part = from_part.strip().lower()
                    to_part = to_part.strip().lower()

                    # Check if from/to is an exchange
                    from_exchange = any(ex in from_part for ex in KNOWN_EXCHANGES)
                    to_exchange = any(ex in to_part for ex in KNOWN_EXCHANGES)

                    if to_exchange and not from_exchange:
                        direction = 'to_exchange'  # BEARISH - selling pressure
                    elif from_exchange and not to_exchange:
                        direction = 'from_exchange'  # BULLISH - accumulation
                    elif from_exchange and to_exchange:
                        direction = 'exchange_to_exchange'
                    else:
                        direction = 'wallet_to_wallet'

        # Determine trading symbol
        trading_symbol = row['symbol']
        if not trading_symbol.endswith('-USD'):
            trading_symbol = f"{asset}-USD"

        return {
            'timestamp': row['timestamp'],
            'symbol': trading_symbol,
            'asset': asset,
            'amount': amount,
            'usd_value': usd_value,
            'direction': direction,
            'from_exchange': from_exchange,
            'to_exchange': to_exchange,
            'event_priority': row['event_priority'],
            'sentiment_score': row['sentiment_score'],
            # Derived features
            'is_bearish_flow': direction == 'to_exchange',
            'is_bullish_flow': direction == 'from_exchange',
            'has_direction': direction not in ('unknown', 'wallet_to_wallet'),
            'log_usd_value': np.log10(max(usd_value, 1)),
        }

    def _add_price_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price and volume context at signal time."""
        logger.info("Adding price context...")

        price_features = []

        for _, row in df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']

            ctx = self._get_price_context(symbol, timestamp)
            price_features.append(ctx)

        # Merge price context
        ctx_df = pd.DataFrame(price_features)
        result = pd.concat([df.reset_index(drop=True), ctx_df], axis=1)

        # Drop rows without price data
        result = result.dropna(subset=['price_at_signal'])
        logger.info(f"After price context: {len(result)} signals")

        return result

    def _get_price_context(self, symbol: str, timestamp: datetime) -> Dict:
        """Get price/volume context around signal time."""
        try:
            # Get recent OHLCV data
            query = f"""
                SELECT
                    timestamp,
                    open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp >= '{timestamp - timedelta(hours=24)}'
                  AND timestamp <= '{timestamp + timedelta(minutes=30)}'
                ORDER BY timestamp
            """
            ohlcv = self.conn.execute(query).df()

            if len(ohlcv) == 0:
                return self._empty_price_context()

            # Split into before/after signal
            before = ohlcv[ohlcv['timestamp'] <= timestamp]

            if len(before) == 0:
                return self._empty_price_context()

            # Price at signal time
            price_at_signal = before.iloc[-1]['close']

            # Recent price stats (4h lookback)
            lookback_4h = before.tail(48)  # 48 x 5min = 4h

            # Volatility (std of returns)
            returns = lookback_4h['close'].pct_change().dropna()
            volatility_4h = returns.std() * 100 if len(returns) > 0 else 0

            # Price momentum
            if len(lookback_4h) >= 2:
                momentum_4h = ((lookback_4h.iloc[-1]['close'] / lookback_4h.iloc[0]['close']) - 1) * 100
            else:
                momentum_4h = 0

            # Volume context
            avg_volume_4h = lookback_4h['volume'].mean()
            recent_volume = before.tail(12)['volume'].mean()  # Last hour
            volume_ratio = recent_volume / avg_volume_4h if avg_volume_4h > 0 else 1

            # RSI approximation (price position in range)
            high_4h = lookback_4h['high'].max()
            low_4h = lookback_4h['low'].min()
            if high_4h > low_4h:
                rsi_proxy = (price_at_signal - low_4h) / (high_4h - low_4h) * 100
            else:
                rsi_proxy = 50

            return {
                'price_at_signal': price_at_signal,
                'volatility_4h': volatility_4h,
                'momentum_4h': momentum_4h,
                'volume_ratio': volume_ratio,
                'rsi_proxy': rsi_proxy,
                'high_4h': high_4h,
                'low_4h': low_4h,
            }

        except Exception as e:
            logger.debug(f"Price context error for {symbol}: {e}")
            return self._empty_price_context()

    def _empty_price_context(self) -> Dict:
        """Return empty price context."""
        return {
            'price_at_signal': None,
            'volatility_4h': None,
            'momentum_4h': None,
            'volume_ratio': None,
            'rsi_proxy': None,
            'high_4h': None,
            'low_4h': None,
        }

    def _add_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward price returns as labels."""
        logger.info("Adding forward returns...")

        returns = []
        for _, row in df.iterrows():
            symbol = row['symbol']
            timestamp = row['timestamp']

            ret = self._get_forward_returns(symbol, timestamp)
            returns.append(ret)

        ret_df = pd.DataFrame(returns)
        result = pd.concat([df.reset_index(drop=True), ret_df], axis=1)

        # Add spike labels
        result['spike_5pct_4h'] = result['max_return_4h'] >= 5
        result['spike_10pct_4h'] = result['max_return_4h'] >= 10
        result['spike_10pct_24h'] = result['max_return_24h'] >= 10
        result['spike_20pct_24h'] = result['max_return_24h'] >= 20

        return result

    def _get_forward_returns(self, symbol: str, timestamp: datetime) -> Dict:
        """Get forward price returns after signal."""
        try:
            query = f"""
                SELECT MAX(high) as max_high
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp > '{timestamp}'
                  AND timestamp <= '{timestamp + timedelta(hours=4)}'
            """
            result_4h = self.conn.execute(query).fetchone()
            max_4h = result_4h[0] if result_4h and result_4h[0] else None

            query = f"""
                SELECT MAX(high) as max_high
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp > '{timestamp}'
                  AND timestamp <= '{timestamp + timedelta(hours=24)}'
            """
            result_24h = self.conn.execute(query).fetchone()
            max_24h = result_24h[0] if result_24h and result_24h[0] else None

            # Get base price
            query = f"""
                SELECT close
                FROM ohlcv
                WHERE symbol = '{symbol}'
                  AND timeframe = '5m'
                  AND timestamp <= '{timestamp}'
                ORDER BY timestamp DESC
                LIMIT 1
            """
            base_result = self.conn.execute(query).fetchone()
            base_price = base_result[0] if base_result else None

            if base_price and base_price > 0:
                return_4h = ((max_4h / base_price) - 1) * 100 if max_4h else 0
                return_24h = ((max_24h / base_price) - 1) * 100 if max_24h else 0
            else:
                return_4h = 0
                return_24h = 0

            return {
                'max_return_4h': return_4h,
                'max_return_24h': return_24h,
            }

        except Exception as e:
            logger.debug(f"Forward returns error for {symbol}: {e}")
            return {'max_return_4h': 0, 'max_return_24h': 0}

    def build_all_catalyst_features(self) -> pd.DataFrame:
        """Build features from all catalyst types."""
        logger.info("Building features from all catalyst types...")

        # Load all signals
        signals = self.conn.execute("""
            SELECT
                symbol,
                timestamp,
                source,
                event_type,
                title,
                event_priority,
                sentiment_score,
                source_credibility
            FROM news_signals
            WHERE timestamp >= '2025-10-01'
              AND symbol != 'UNKNOWN-USD'
            ORDER BY timestamp
        """).df()

        logger.info(f"Loaded {len(signals)} total signals")

        # Build features per signal type
        features = []

        for _, row in signals.iterrows():
            feat = {
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'source': row['source'],
                'event_type': row['event_type'],
                'event_priority': row['event_priority'],
                'sentiment_score': row['sentiment_score'],
                'source_credibility': row['source_credibility'],
            }

            # Type-specific features
            if row['event_type'] == 'whale_move':
                whale_feat = self._parse_whale_signal(row)
                if whale_feat:
                    feat.update({
                        'is_whale': True,
                        'whale_direction': whale_feat['direction'],
                        'whale_usd_value': whale_feat['usd_value'],
                        'whale_is_bearish': whale_feat['is_bearish_flow'],
                        'whale_is_bullish': whale_feat['is_bullish_flow'],
                    })
                else:
                    feat['is_whale'] = False
            else:
                feat['is_whale'] = False

            # Listing-related features
            feat['is_listing'] = row['event_type'] in ('listing', 'futures_listing', 'margin_listing')
            feat['is_binance_listing'] = 'binance' in row['source'].lower() and feat['is_listing']

            features.append(feat)

        df = pd.DataFrame(features)

        # Add price context
        df = self._add_price_context(df)
        df = self._add_forward_returns(df)

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> None:
        """Print feature summary statistics."""
        print("\n" + "=" * 70)
        print("CATALYST FEATURE SUMMARY")
        print("=" * 70)

        print(f"\nTotal signals: {len(df)}")
        print(f"Unique symbols: {df['symbol'].nunique()}")

        if 'direction' in df.columns:
            print("\nWhale Direction Distribution:")
            print(df['direction'].value_counts().to_string())

            print("\nSpike Rate by Direction (10%+ in 24h):")
            by_dir = df.groupby('direction')['spike_10pct_24h'].agg(['mean', 'count'])
            by_dir['mean'] = (by_dir['mean'] * 100).round(1)
            by_dir.columns = ['spike_rate_pct', 'count']
            print(by_dir.to_string())

        if 'event_type' in df.columns:
            print("\nSpike Rate by Event Type:")
            by_type = df.groupby('event_type')['spike_10pct_24h'].agg(['mean', 'count'])
            by_type['mean'] = (by_type['mean'] * 100).round(1)
            by_type.columns = ['spike_rate_pct', 'count']
            print(by_type.sort_values('spike_rate_pct', ascending=False).to_string())

        print("\nFeature Correlations with Spike:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        for col in numeric_cols:
            if col not in ('spike_5pct_4h', 'spike_10pct_4h', 'spike_10pct_24h', 'spike_20pct_24h'):
                corr = df[col].corr(df['spike_10pct_24h'])
                if not np.isnan(corr):
                    correlations[col] = corr

        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for col, corr in sorted_corr[:10]:
            print(f"  {col:30} {corr:+.3f}")


def main():
    """Build and analyze catalyst features."""
    engineer = CatalystFeatureEngineer(db_path="full_pythia.duckdb")

    print("=" * 70)
    print("WHALE SIGNAL FEATURE ENGINEERING")
    print("=" * 70)

    # Build whale features
    whale_df = engineer.build_whale_features()

    if len(whale_df) > 0:
        engineer.get_feature_summary(whale_df)

        # Save features
        whale_df.to_csv("whale_features.csv", index=False)
        print(f"\nSaved whale features to: whale_features.csv")

        # Key insight: direction-based spike rates
        print("\n" + "=" * 70)
        print("KEY INSIGHT: DIRECTION-BASED SPIKE RATES")
        print("=" * 70)

        if 'direction' in whale_df.columns and 'spike_10pct_24h' in whale_df.columns:
            for direction in whale_df['direction'].unique():
                subset = whale_df[whale_df['direction'] == direction]
                spike_rate = subset['spike_10pct_24h'].mean() * 100
                avg_return = subset['max_return_24h'].mean()
                print(f"  {direction:20} | {len(subset):4} signals | {spike_rate:5.1f}% spike rate | {avg_return:+.1f}% avg return")


if __name__ == "__main__":
    main()
