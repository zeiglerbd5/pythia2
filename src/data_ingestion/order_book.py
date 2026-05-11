"""
Order book state tracking for Coinbase level2 channel.

Implements snapshot-and-update protocol with validation per implementation guide:
- Snapshot initialization: Full bid/ask sides from first message
- Delta updates: Absolute quantities (not deltas), remove when qty=0
- Sequence number tracking: Detect gaps requiring REST snapshot recovery
- Spread validation: Negative spread indicates desynchronization
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from loguru import logger


@dataclass
class PriceLevel:
    """
    Represents a single price level in the order book.

    Per implementation guide:
    - price: Price level
    - quantity: Absolute quantity at this level (not a delta)
    """
    price: float
    quantity: float
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price:.2f}, qty={self.quantity:.6f})"


@dataclass
class OrderBookSnapshot:
    """
    Snapshot of order book state at a point in time.

    Used for:
    - Periodic snapshots to database
    - Recovery after desynchronization
    - Feature calculation (order book imbalance)
    """
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]
    sequence_num: Optional[int] = None
    spread: Optional[float] = None
    mid_price: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bids": self.bids,
            "asks": self.asks,
            "sequence_num": self.sequence_num,
            "spread": self.spread,
            "mid_price": self.mid_price,
        }


class OrderBook:
    """
    Real-time order book state engine for Coinbase level2 channel.

    Per implementation guide protocol:
    1. Initialize with snapshot message (full bid/ask sides)
    2. Apply delta updates with absolute quantities
    3. Track sequence numbers for gap detection
    4. Monitor spread for negative values (desync indicator)

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC-USD")
        bids: Sorted bid levels (price -> PriceLevel)
        asks: Sorted ask levels (price -> PriceLevel)
        last_sequence: Last processed sequence number
        last_update: Timestamp of last update
        is_synchronized: True if order book state is valid
    """

    def __init__(self, symbol: str):
        """
        Initialize empty order book for a trading pair.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
        """
        self.symbol = symbol

        # Order book state (sorted by price)
        # Bids: highest to lowest (descending)
        # Asks: lowest to highest (ascending)
        self.bids: OrderedDict[float, PriceLevel] = OrderedDict()
        self.asks: OrderedDict[float, PriceLevel] = OrderedDict()

        # State tracking
        self.last_sequence: Optional[int] = None
        self.last_update: Optional[float] = None
        self.is_synchronized = False
        self.snapshot_count = 0
        self.update_count = 0

        logger.debug(f"OrderBook initialized for {symbol}")

    def apply_snapshot(self, snapshot_data: dict) -> None:
        """
        Apply full order book snapshot from level2 channel.

        Per implementation guide:
        - First message is snapshot with full bid/ask sides
        - Contains all price levels and quantities
        - Resets order book state

        Args:
            snapshot_data: Snapshot message from WebSocket
                {
                    'events': [{
                        'type': 'snapshot',
                        'product_id': 'BTC-USD',
                        'updates': [
                            {'side': 'bid', 'price_level': '45000.00', 'new_quantity': '1.5'},
                            ...
                        ]
                    }]
                }
        """
        # Clear existing state
        self.bids.clear()
        self.asks.clear()

        # Extract events
        events = snapshot_data.get('events', [])
        if not events:
            logger.warning(f"{self.symbol}: Snapshot has no events")
            return

        event = events[0]
        updates = event.get('updates', [])
        current_time = time.time()

        # Process all price levels
        for update in updates:
            side = update.get('side')
            price = float(update.get('price_level', 0))
            quantity = float(update.get('new_quantity', 0))

            if price <= 0:
                continue

            level = PriceLevel(price=price, quantity=quantity, timestamp=current_time)

            if side == 'bid':
                self.bids[price] = level
            elif side == 'offer':  # Coinbase uses 'offer' for asks
                self.asks[price] = level

        # Sort order books
        self._sort_books()

        # Update state
        self.is_synchronized = True
        self.last_update = current_time
        self.snapshot_count += 1

        logger.info(
            f"{self.symbol}: Snapshot applied - {len(self.bids)} bids, {len(self.asks)} asks",
            extra={
                "best_bid": self.get_best_bid(),
                "best_ask": self.get_best_ask(),
                "spread": self.get_spread()
            }
        )

    def apply_update(self, update_data: dict) -> bool:
        """
        Apply incremental update to order book.

        Per implementation guide:
        - new_quantity is ABSOLUTE quantity (not delta)
        - When new_quantity = 0, REMOVE that price level
        - Sequence number gaps indicate dropped messages

        Args:
            update_data: Update message from WebSocket

        Returns:
            True if update applied successfully, False if desync detected
        """
        events = update_data.get('events', [])
        if not events:
            return True

        event = events[0]
        updates = event.get('updates', [])
        current_time = time.time()

        # Check for sequence number gaps (if provided)
        if 'sequence_num' in update_data:
            seq_num = update_data['sequence_num']
            if self.last_sequence is not None:
                gap = seq_num - self.last_sequence
                if gap > 1:
                    logger.error(
                        f"{self.symbol}: Sequence gap detected! "
                        f"Last: {self.last_sequence}, Current: {seq_num}, Gap: {gap-1}"
                    )
                    self.is_synchronized = False
                    return False
            self.last_sequence = seq_num

        # Apply updates
        for update in updates:
            side = update.get('side')
            price = float(update.get('price_level', 0))
            new_quantity = float(update.get('new_quantity', 0))

            if price <= 0:
                continue

            # Select correct book
            book = self.bids if side == 'bid' else self.asks

            # Per guide: new_quantity = 0 means REMOVE level
            if new_quantity == 0:
                if price in book:
                    del book[price]
            else:
                # Update with absolute quantity
                book[price] = PriceLevel(
                    price=price,
                    quantity=new_quantity,
                    timestamp=current_time
                )

        # Re-sort after updates
        self._sort_books()

        # Validate spread (negative spread = desync per guide)
        spread = self.get_spread()
        if spread is not None and spread < 0:
            logger.error(
                f"{self.symbol}: Negative spread detected ({spread:.4f}) - "
                "Order book desynchronized!"
            )
            self.is_synchronized = False
            return False

        # Update state
        self.last_update = current_time
        self.update_count += 1

        return True

    def _sort_books(self) -> None:
        """
        Sort order books.

        Bids: Descending (highest price first)
        Asks: Ascending (lowest price first)
        """
        # Sort bids descending
        self.bids = OrderedDict(
            sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        )

        # Sort asks ascending
        self.asks = OrderedDict(
            sorted(self.asks.items(), key=lambda x: x[0], reverse=False)
        )

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price (highest bid)."""
        if not self.bids:
            return None
        return next(iter(self.bids))

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price (lowest ask)."""
        if not self.asks:
            return None
        return next(iter(self.asks))

    def get_spread(self) -> Optional[float]:
        """
        Get bid-ask spread.

        Per implementation guide:
        - Negative spread indicates desynchronization
        - Should be positive in normal conditions

        Returns:
            Spread (ask - bid) or None if incomplete
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        return best_ask - best_bid

    def get_mid_price(self) -> Optional[float]:
        """Get mid-market price (average of best bid/ask)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2.0

    def get_depth(self, levels: int = 5) -> Tuple[List[PriceLevel], List[PriceLevel]]:
        """
        Get top N levels of order book depth.

        Per implementation guide: L=5 depth is optimal for crypto prediction

        Args:
            levels: Number of levels to retrieve (default: 5)

        Returns:
            Tuple of (bid_levels, ask_levels)
        """
        bid_levels = list(self.bids.values())[:levels]
        ask_levels = list(self.asks.values())[:levels]

        return bid_levels, ask_levels

    def calculate_imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate order book imbalance at L levels.

        Per implementation guide:
        ρ = (Σ V_bid - Σ V_ask) / (Σ V_bid + Σ V_ask)

        Threshold: |ρ| > 0.3 signals directional pressure
        Lead time: 10 seconds to 2 minutes before price movement

        Args:
            levels: Depth levels to include (default: 5 per guide)

        Returns:
            Imbalance value [-1, 1] or None if insufficient data
        """
        bids, asks = self.get_depth(levels)

        if len(bids) < levels or len(asks) < levels:
            return None

        # Sum bid and ask volumes
        bid_volume = sum(level.quantity for level in bids)
        ask_volume = sum(level.quantity for level in asks)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        # Calculate imbalance
        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def get_snapshot(self) -> OrderBookSnapshot:
        """
        Get current order book snapshot for storage/analysis.

        Returns:
            OrderBookSnapshot with current state
        """
        bids = [(price, level.quantity) for price, level in self.bids.items()]
        asks = [(price, level.quantity) for price, level in self.asks.items()]

        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=self.last_update or time.time(),
            bids=bids,
            asks=asks,
            sequence_num=self.last_sequence,
            spread=self.get_spread(),
            mid_price=self.get_mid_price(),
        )

    def get_statistics(self) -> dict:
        """
        Get order book statistics.

        Returns:
            Dictionary with stats
        """
        imbalance = self.calculate_imbalance(levels=5)

        return {
            "symbol": self.symbol,
            "is_synchronized": self.is_synchronized,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "spread_bps": self.get_spread() / self.get_mid_price() * 10000 if self.get_mid_price() else None,
            "imbalance_l5": imbalance,
            "snapshot_count": self.snapshot_count,
            "update_count": self.update_count,
            "last_update": datetime.fromtimestamp(self.last_update).isoformat() if self.last_update else None,
        }


class OrderBookManager:
    """
    Manages multiple order books for different trading pairs.
    """

    def __init__(self):
        """Initialize order book manager."""
        self.books: Dict[str, OrderBook] = {}
        logger.info("OrderBookManager initialized")

    def get_book(self, symbol: str) -> OrderBook:
        """
        Get or create order book for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            OrderBook instance
        """
        if symbol not in self.books:
            self.books[symbol] = OrderBook(symbol)
            logger.info(f"Created new order book for {symbol}")

        return self.books[symbol]

    def process_message(self, message: dict) -> bool:
        """
        Process level2 channel message.

        Args:
            message: WebSocket message

        Returns:
            True if processed successfully
        """
        channel = message.get('channel')
        if channel != 'level2':
            return False

        events = message.get('events', [])
        if not events:
            return False

        event = events[0]
        symbol = event.get('product_id')

        if not symbol:
            return False

        book = self.get_book(symbol)

        # Determine message type
        event_type = event.get('type')

        if event_type == 'snapshot':
            book.apply_snapshot(message)
            return True
        elif event_type == 'update':
            return book.apply_update(message)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return False

    def get_all_snapshots(self) -> List[OrderBookSnapshot]:
        """
        Get snapshots from all order books.

        Returns:
            List of OrderBookSnapshot objects
        """
        return [book.get_snapshot() for book in self.books.values()]

    def get_statistics_all(self) -> Dict[str, dict]:
        """
        Get statistics for all order books.

        Returns:
            Dictionary of {symbol: stats}
        """
        return {
            symbol: book.get_statistics()
            for symbol, book in self.books.items()
        }


if __name__ == "__main__":
    # Test order book engine
    book = OrderBook("BTC-USD")

    # Simulate snapshot
    snapshot_msg = {
        'channel': 'level2',
        'events': [{
            'type': 'snapshot',
            'product_id': 'BTC-USD',
            'updates': [
                {'side': 'bid', 'price_level': '45000.00', 'new_quantity': '1.5'},
                {'side': 'bid', 'price_level': '44999.00', 'new_quantity': '2.0'},
                {'side': 'bid', 'price_level': '44998.00', 'new_quantity': '0.5'},
                {'side': 'offer', 'price_level': '45001.00', 'new_quantity': '1.0'},
                {'side': 'offer', 'price_level': '45002.00', 'new_quantity': '1.5'},
            ]
        }]
    }

    book.apply_snapshot(snapshot_msg)

    print("\n=== Order Book Test ===")
    print(f"\nBest Bid: {book.get_best_bid()}")
    print(f"Best Ask: {book.get_best_ask()}")
    print(f"Spread: {book.get_spread()}")
    print(f"Mid Price: {book.get_mid_price()}")
    print(f"Imbalance (L=5): {book.calculate_imbalance():.4f}")

    print(f"\nStatistics: {book.get_statistics()}")

    # Simulate update
    update_msg = {
        'channel': 'level2',
        'sequence_num': 100,
        'events': [{
            'type': 'update',
            'product_id': 'BTC-USD',
            'updates': [
                {'side': 'bid', 'price_level': '45000.00', 'new_quantity': '2.5'},  # Update
                {'side': 'bid', 'price_level': '44997.00', 'new_quantity': '1.0'},  # New
                {'side': 'offer', 'price_level': '45001.00', 'new_quantity': '0'},  # Remove
            ]
        }]
    }

    success = book.apply_update(update_msg)
    print(f"\nUpdate applied: {success}")
    print(f"New Imbalance: {book.calculate_imbalance():.4f}")
