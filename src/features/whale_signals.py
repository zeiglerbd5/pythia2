"""
Whale Signal Buffer

Rolling buffer of whale transactions for feature calculation.
Receives raw whale alert data and provides aggregated features for entry decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from collections import defaultdict
import threading
import numpy as np
from loguru import logger


@dataclass
class WhaleTransaction:
    """Single whale transaction from Whale Alert."""
    timestamp: datetime
    symbol: str              # "BTC-USD"
    amount_usd: float        # USD value
    subtype: str             # exchange_inflow/outflow/transfer/wallet_transfer
    from_name: str
    to_name: str
    blockchain: str


class WhaleSignalBuffer:
    """
    Rolling buffer of whale transactions for feature calculation.

    Maintains a time-based rolling window of whale transactions and provides
    aggregated features for entry signal decisions.

    Features calculated:
    - whale_net_flow_1h: Net USD flow (outflows - inflows), normalized
    - whale_exchange_pressure_1h: Ratio of inflows to total flow
    - whale_activity_zscore: Transaction count vs baseline
    - whale_largest_move_recency: Minutes since last $10M+ move
    - whale_btc_eth_pressure: Aggregate BTC+ETH exchange pressure
    """

    def __init__(self, ttl_minutes: int = 120):
        """
        Initialize whale signal buffer.

        Args:
            ttl_minutes: How long to keep transactions (default: 2 hours)
        """
        self.ttl_minutes = ttl_minutes
        self._transactions: List[WhaleTransaction] = []
        self._by_symbol: Dict[str, List[WhaleTransaction]] = defaultdict(list)
        self._lock = threading.Lock()

        # Baseline stats for z-score calculation (rolling averages)
        # {symbol: {'tx_count_mean': float, 'tx_count_std': float, 'last_update': datetime}}
        self._baseline_stats: Dict[str, Dict] = defaultdict(lambda: {
            'tx_count_mean': 0.0,
            'tx_count_std': 1.0,  # Avoid div by zero
            'tx_counts': [],  # Rolling window of hourly counts
            'last_update': None
        })

        # Statistics
        self.stats = {
            'transactions_added': 0,
            'transactions_pruned': 0,
        }

        logger.info(f"WhaleSignalBuffer initialized (TTL: {ttl_minutes} minutes)")

    def add_transaction(self, tx: WhaleTransaction):
        """
        Add a whale transaction to the buffer.

        Args:
            tx: WhaleTransaction to add
        """
        with self._lock:
            self._transactions.append(tx)
            self._by_symbol[tx.symbol].append(tx)
            self.stats['transactions_added'] += 1

            # Prune old transactions periodically (every 100 additions)
            if self.stats['transactions_added'] % 100 == 0:
                self._prune_old()

            # Update baseline stats
            self._update_baseline(tx.symbol)

        logger.debug(f"[WHALE_BUFFER] Added {tx.symbol} ${tx.amount_usd:,.0f} {tx.subtype}")

    def _prune_old(self):
        """Remove transactions older than TTL (must hold lock)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.ttl_minutes)

        # Prune main list
        old_count = len(self._transactions)
        self._transactions = [tx for tx in self._transactions if tx.timestamp >= cutoff]
        pruned = old_count - len(self._transactions)

        # Prune by-symbol index
        for symbol in list(self._by_symbol.keys()):
            self._by_symbol[symbol] = [
                tx for tx in self._by_symbol[symbol] if tx.timestamp >= cutoff
            ]
            # Remove empty symbol entries
            if not self._by_symbol[symbol]:
                del self._by_symbol[symbol]

        if pruned > 0:
            self.stats['transactions_pruned'] += pruned
            logger.debug(f"[WHALE_BUFFER] Pruned {pruned} old transactions")

    def _update_baseline(self, symbol: str):
        """Update baseline stats for z-score calculation (must hold lock)."""
        stats = self._baseline_stats[symbol]
        now = datetime.now(timezone.utc)

        # Update hourly count tracking
        if stats['last_update'] is None or (now - stats['last_update']).seconds >= 3600:
            # Calculate current hourly count
            hour_ago = now - timedelta(hours=1)
            hourly_count = len([
                tx for tx in self._by_symbol.get(symbol, [])
                if tx.timestamp >= hour_ago
            ])

            # Add to rolling window (keep last 24 hours = 24 data points)
            stats['tx_counts'].append(hourly_count)
            if len(stats['tx_counts']) > 24:
                stats['tx_counts'] = stats['tx_counts'][-24:]

            # Update mean and std
            if len(stats['tx_counts']) >= 2:
                stats['tx_count_mean'] = np.mean(stats['tx_counts'])
                stats['tx_count_std'] = max(np.std(stats['tx_counts']), 0.1)  # Min std to avoid div/0
            else:
                stats['tx_count_mean'] = hourly_count
                stats['tx_count_std'] = 1.0

            stats['last_update'] = now

    def _get_recent(self, symbol: str, window_minutes: int) -> List[WhaleTransaction]:
        """Get transactions for symbol within time window (must hold lock)."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        return [tx for tx in self._by_symbol.get(symbol, []) if tx.timestamp >= cutoff]

    def get_net_flow_usd(self, symbol: str, window_minutes: int = 60) -> float:
        """
        Calculate net USD flow (outflows - inflows) for a symbol.

        Positive = accumulation (bullish), Negative = distribution (bearish)

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            window_minutes: Time window (default: 60)

        Returns:
            Net USD flow (outflows - inflows)
        """
        with self._lock:
            txs = self._get_recent(symbol, window_minutes)

            outflow = sum(
                tx.amount_usd for tx in txs
                if tx.subtype == 'exchange_outflow'
            )
            inflow = sum(
                tx.amount_usd for tx in txs
                if tx.subtype == 'exchange_inflow'
            )

            return outflow - inflow  # Positive = accumulation (bullish)

    def get_exchange_pressure(self, symbol: str, window_minutes: int = 60) -> float:
        """
        Calculate exchange pressure ratio.

        Returns ratio of inflows to total exchange flow.
        >0.5 = selling pressure (more going to exchanges)
        <0.5 = accumulation (more leaving exchanges)

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            window_minutes: Time window (default: 60)

        Returns:
            Exchange pressure ratio (0 to 1, 0.5 = neutral)
        """
        with self._lock:
            txs = self._get_recent(symbol, window_minutes)

            inflow = sum(
                tx.amount_usd for tx in txs
                if tx.subtype == 'exchange_inflow'
            )
            outflow = sum(
                tx.amount_usd for tx in txs
                if tx.subtype == 'exchange_outflow'
            )

            total = inflow + outflow
            if total == 0:
                return 0.5  # Neutral when no data

            return inflow / total  # >0.5 = selling pressure, <0.5 = accumulation

    def get_activity_zscore(self, symbol: str, window_minutes: int = 60) -> float:
        """
        Calculate activity z-score (transaction count vs baseline).

        High values indicate unusual whale activity for this symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            window_minutes: Time window (default: 60)

        Returns:
            Z-score of current activity vs baseline
        """
        with self._lock:
            txs = self._get_recent(symbol, window_minutes)
            current_count = len(txs)

            stats = self._baseline_stats[symbol]
            mean = stats['tx_count_mean']
            std = stats['tx_count_std']

            if std <= 0:
                return 0.0

            return (current_count - mean) / std

    def get_largest_move_recency(self, symbol: str, min_usd: float = 10_000_000) -> float:
        """
        Get minutes since last large whale move.

        Args:
            symbol: Trading pair (e.g., 'BTC-USD')
            min_usd: Minimum USD threshold for "large" move (default: $10M)

        Returns:
            Minutes since last large move, or float('inf') if none
        """
        with self._lock:
            # Look at all transactions for this symbol in buffer
            txs = self._by_symbol.get(symbol, [])

            # Filter to large moves
            large_moves = [tx for tx in txs if tx.amount_usd >= min_usd]

            if not large_moves:
                return float('inf')

            # Find most recent
            most_recent = max(large_moves, key=lambda tx: tx.timestamp)
            elapsed = datetime.now(timezone.utc) - most_recent.timestamp

            return elapsed.total_seconds() / 60  # Convert to minutes

    def get_market_leader_pressure(self, window_minutes: int = 60) -> float:
        """
        Calculate aggregate BTC+ETH exchange pressure.

        This is a market-wide risk indicator - when BTC and ETH are being
        moved to exchanges, it often precedes broader market drops.

        Args:
            window_minutes: Time window (default: 60)

        Returns:
            Weighted average pressure (0 to 1)
        """
        btc_pressure = self.get_exchange_pressure('BTC-USD', window_minutes)
        eth_pressure = self.get_exchange_pressure('ETH-USD', window_minutes)

        # Weight BTC slightly more (market leader)
        return 0.6 * btc_pressure + 0.4 * eth_pressure

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                'total_transactions': len(self._transactions),
                'symbols_tracked': len(self._by_symbol),
                'transactions_added': self.stats['transactions_added'],
                'transactions_pruned': self.stats['transactions_pruned'],
                'ttl_minutes': self.ttl_minutes,
            }


if __name__ == "__main__":
    # Test the whale signal buffer
    from datetime import datetime, timezone

    buffer = WhaleSignalBuffer(ttl_minutes=120)

    # Add some test transactions
    now = datetime.now(timezone.utc)

    # BTC exchange inflow (selling pressure)
    buffer.add_transaction(WhaleTransaction(
        timestamp=now - timedelta(minutes=30),
        symbol='BTC-USD',
        amount_usd=50_000_000,
        subtype='exchange_inflow',
        from_name='unknown wallet',
        to_name='binance',
        blockchain='bitcoin'
    ))

    # BTC exchange outflow (accumulation)
    buffer.add_transaction(WhaleTransaction(
        timestamp=now - timedelta(minutes=15),
        symbol='BTC-USD',
        amount_usd=30_000_000,
        subtype='exchange_outflow',
        from_name='coinbase',
        to_name='unknown wallet',
        blockchain='bitcoin'
    ))

    # ETH large move
    buffer.add_transaction(WhaleTransaction(
        timestamp=now - timedelta(minutes=5),
        symbol='ETH-USD',
        amount_usd=15_000_000,
        subtype='exchange_inflow',
        from_name='unknown wallet',
        to_name='kraken',
        blockchain='ethereum'
    ))

    print("\nWhale Signal Buffer Test:")
    print(f"  Statistics: {buffer.get_statistics()}")
    print(f"\nBTC-USD Features:")
    print(f"  Net Flow (1h): ${buffer.get_net_flow_usd('BTC-USD', 60):,.0f}")
    print(f"  Exchange Pressure: {buffer.get_exchange_pressure('BTC-USD', 60):.2f}")
    print(f"  Activity Z-Score: {buffer.get_activity_zscore('BTC-USD', 60):.2f}")
    print(f"  Largest Move Recency: {buffer.get_largest_move_recency('BTC-USD', 10_000_000):.1f} min")
    print(f"\nMarket Leader Pressure: {buffer.get_market_leader_pressure(60):.2f}")
