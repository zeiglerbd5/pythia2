#!/usr/bin/env python3
"""
Real-Time Pre-Spike Detection Inference

Runs live inference on WebSocket market data to detect pre-spike patterns.

Architecture:
    WebSocket → Candle Aggregator → Feature Calculator → Sequence Buffer → Model → Alerts

Features:
- Monitors 319 cryptocurrency pairs in real-time
- Calculates 24 features per candle
- Maintains 60-candle rolling windows
- Runs CNN-LSTM inference every minute
- Alerts on high-probability signals

Usage:
    # Basic usage (with Coinbase authentication)
    python scripts/run_live_inference.py \\
        --model models/cnn_lstm_chunked_20251101_192036/best_model.pt \\
        --scaler models/cnn_lstm_chunked_20251101_192036/scaler.pkl

    # Test mode (no WebSocket, load from database)
    python scripts/run_live_inference.py \\
        --model models/cnn_lstm_chunked_20251101_192036/best_model.pt \\
        --scaler models/cnn_lstm_chunked_20251101_192036/scaler.pkl \\
        --test-mode \\
        --db market_data.db

    # Custom settings
    python scripts/run_live_inference.py \\
        --model path/to/model.pt \\
        --threshold 0.7 \\
        --min-probability 0.6 \\
        --alert-log signals.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import argparse
import signal
from datetime import datetime
from loguru import logger
import duckdb

# Inference components
from src.inference import (
    CandleAggregator,
    FeatureCalculator,
    SequenceBuffer,
    InferenceEngine,
    AlertManager
)

# Data ingestion (for live mode)
try:
    from src.inference.public_websocket import PublicWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("WebSocket components not available - test mode only")


class LiveInferenceOrchestrator:
    """
    Orchestrates real-time inference pipeline.

    Coordinates data flow from WebSocket → alerts.
    """

    def __init__(
        self,
        model_path: str,
        symbols: list[str],
        scaler_path: str = None,
        model_type: str = 'cnn_lstm',
        sequence_length: int = 60,
        threshold: float = 0.5,
        min_alert_probability: float = 0.6,
        alert_log_file: str = None,
        device: str = 'mps'
    ):
        """
        Initialize orchestrator.

        Args:
            model_path: Path to model checkpoint
            symbols: List of trading pairs to monitor
            scaler_path: Path to scaler pickle file
            model_type: 'cnn_lstm' or 'tcn'
            sequence_length: Candle sequence length (60)
            threshold: Model prediction threshold
            min_alert_probability: Minimum probability to trigger alert
            alert_log_file: Path to alert log JSON file
            device: PyTorch device
        """
        self.symbols = symbols

        logger.info("="*80)
        logger.info("LIVE INFERENCE ORCHESTRATOR")
        logger.info("="*80)
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Sequence length: {sequence_length}")
        logger.info(f"Device: {device}")
        logger.info("="*80)
        logger.info("")

        # Initialize components
        logger.info("Initializing inference components...")

        # 1. Candle Aggregator
        self.candle_aggregator = CandleAggregator(
            symbols=symbols,
            on_candle_complete=self._on_candle_complete,
            fill_missing=True
        )

        # 2. Feature Calculator
        self.feature_calculator = FeatureCalculator(symbols=symbols)

        # 3. Sequence Buffer
        self.sequence_buffer = SequenceBuffer(
            symbols=symbols,
            sequence_length=sequence_length,
            n_features=24,
            scaler_path=scaler_path,
            device=device
        )

        # 4. Inference Engine
        self.inference_engine = InferenceEngine(
            model_path=model_path,
            model_type=model_type,
            n_features=24,
            sequence_length=sequence_length,
            device=device,
            threshold=threshold
        )

        # 5. Alert Manager
        self.alert_manager = AlertManager(
            min_probability=min_alert_probability,
            alert_log_file=alert_log_file,
            throttle_seconds=60  # Max 1 alert per minute per symbol
        )

        logger.info("All components initialized successfully")
        logger.info("")

        # Statistics
        self.candles_processed = 0
        self.features_calculated = 0
        self.inferences_run = 0
        self.alerts_sent = 0
        self.start_time = datetime.now()

    def _on_candle_complete(self, symbol: str, candle: dict):
        """
        Callback when a 1-minute candle is completed.

        This triggers the inference pipeline:
        1. Calculate features
        2. Add to sequence buffer
        3. Run inference if buffer is full
        4. Check for alerts
        """
        self.candles_processed += 1

        # Calculate features
        features = self.feature_calculator.on_candle_complete(symbol, candle)

        if features is None:
            # Not enough history yet
            return

        self.features_calculated += 1

        # Add to sequence buffer
        timestamp = candle['timestamp']
        self.sequence_buffer.add_features(symbol, features, timestamp)

        # Check if ready for inference
        if not self.sequence_buffer.is_ready(symbol):
            return

        # Get normalized sequence tensor
        sequence_tensor = self.sequence_buffer.get_sequence_tensor(symbol)

        if sequence_tensor is None:
            return

        # Run inference
        result = self.inference_engine.predict(sequence_tensor)

        self.inferences_run += 1

        # Log inference result (debug)
        if self.inferences_run % 100 == 0:
            logger.info(
                f"Inference #{self.inferences_run}: {symbol} "
                f"prob={result['probability']:.3f}"
            )

        # Check for alert
        if result['prediction'] == 1 or result['probability'] >= self.alert_manager.min_probability:
            self.alert_manager.send_alert(
                symbol=symbol,
                probability=result['probability'],
                confidence=result['confidence'],
                timestamp=timestamp,
                additional_data={
                    'inference_time_ms': result['inference_time_ms'],
                }
            )
            self.alerts_sent += 1

    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()

        stats = {
            'uptime_seconds': uptime,
            'candles_processed': self.candles_processed,
            'features_calculated': self.features_calculated,
            'inferences_run': self.inferences_run,
            'alerts_sent': self.alerts_sent,
            'candle_aggregator': self.candle_aggregator.get_statistics(),
            'sequence_buffer': self.sequence_buffer.get_statistics(),
            'inference_engine': self.inference_engine.get_statistics(),
            'alert_manager': self.alert_manager.get_statistics(),
        }

        return stats

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        logger.info("="*80)
        logger.info("LIVE INFERENCE STATISTICS")
        logger.info("="*80)
        logger.info(f"Uptime: {stats['uptime_seconds']:.0f}s")
        logger.info(f"Candles processed: {stats['candles_processed']}")
        logger.info(f"Features calculated: {stats['features_calculated']}")
        logger.info(f"Inferences run: {stats['inferences_run']}")
        logger.info(f"Alerts sent: {stats['alerts_sent']}")
        logger.info("")
        logger.info(f"Avg inference time: {stats['inference_engine']['avg_inference_time_ms']:.2f}ms")
        logger.info(f"Symbols ready: {stats['sequence_buffer']['symbols_ready']}/{len(self.symbols)}")
        logger.info("="*80)

    async def run_with_websocket(self):
        """Run inference with live WebSocket data (public, no auth required)."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket components not available")
            return

        logger.info("Starting live inference with public WebSocket...")
        logger.info("No authentication required - using public market data")
        logger.info("")

        # Create public WebSocket client
        websocket_client = PublicWebSocketClient(
            symbols=self.symbols,
            on_ticker=self._on_ticker_update,
            on_trade=self._on_trade_update,
            channels=['ticker', 'matches']  # 'matches' is the correct channel for trades
        )

        # Create background tasks
        ws_task = asyncio.create_task(websocket_client.start())

        flush_task = asyncio.create_task(
            self.candle_aggregator.periodic_flush(interval_seconds=5)
        )

        # Statistics printing task
        async def print_stats_loop():
            while True:
                await asyncio.sleep(60)  # Every minute
                self.print_statistics()

        stats_task = asyncio.create_task(print_stats_loop())

        # Wait forever (until interrupted)
        try:
            await asyncio.gather(ws_task, flush_task, stats_task)

        except asyncio.CancelledError:
            logger.info("Shutting down...")
            await websocket_client.stop()

    def _on_ticker_update(self, symbol: str, price: float, volume_24h: float):
        """Handle ticker update from WebSocket."""
        self.candle_aggregator.on_ticker_update(symbol, price, volume_24h)

    def _on_trade_update(self, symbol: str, price: float, size: float, side: str):
        """Handle trade update from WebSocket."""
        self.candle_aggregator.on_trade_update(symbol, price, size, side)

    async def run_test_mode(self, db_path: str, start_idx: int = 0, max_candles: int = 1000):
        """
        Run inference on historical data from database (for testing).

        Args:
            db_path: Path to database
            start_idx: Starting candle index
            max_candles: Maximum candles to process
        """
        logger.info(f"Running in test mode: {db_path}")
        logger.info(f"Processing {max_candles} candles starting at index {start_idx}")
        logger.info("")

        # Load historical candles
        conn = duckdb.connect(db_path, read_only=True)

        try:
            # Get sample of candles for all symbols
            query = f"""
                SELECT symbol, timestamp, open, high, low, close, volume,
                       buy_volume, sell_volume, num_trades
                FROM candles
                ORDER BY timestamp
                LIMIT {max_candles} OFFSET {start_idx}
            """

            result = conn.execute(query).fetchall()
            logger.info(f"Loaded {len(result)} candles from database")

            # Process each candle
            for row in result:
                symbol, timestamp, open_, high, low, close, volume, buy_vol, sell_vol, num_trades = row

                candle = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': open_,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'buy_volume': buy_vol or 0.0,
                    'sell_volume': sell_vol or 0.0,
                    'num_trades': num_trades or 0,
                }

                # Trigger pipeline
                self._on_candle_complete(symbol, candle)

                # Periodic stats
                if self.candles_processed % 100 == 0:
                    logger.info(f"Processed {self.candles_processed} candles...")

            # Final stats
            self.print_statistics()

        finally:
            conn.close()


def get_symbols_from_db(db_path: str) -> list[str]:
    """Get list of symbols from database."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        result = conn.execute(
            "SELECT DISTINCT symbol FROM candles ORDER BY symbol"
        ).fetchall()
        symbols = [row[0] for row in result]
        return symbols
    finally:
        conn.close()


async def main():
    parser = argparse.ArgumentParser(description='Real-time pre-spike detection inference')

    parser.add_argument('--model', required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--scaler', help='Path to scaler pickle file')
    parser.add_argument('--model-type', default='cnn_lstm', choices=['cnn_lstm', 'tcn'], help='Model architecture')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length (candles)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--min-probability', type=float, default=0.6, help='Minimum probability for alert')
    parser.add_argument('--alert-log', default='data/alerts.json', help='Path to alert log file')
    parser.add_argument('--device', default='mps', choices=['mps', 'cuda', 'cpu'], help='PyTorch device')

    # Test mode options
    parser.add_argument('--test-mode', action='store_true', help='Run on historical data (no WebSocket)')
    parser.add_argument('--db', help='Database path (required for test mode)')
    parser.add_argument('--max-candles', type=int, default=1000, help='Max candles to process in test mode')
    parser.add_argument('--symbols-file', help='Path to file containing symbols (one per line)')

    args = parser.parse_args()

    # Validate
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return

    if args.scaler and not Path(args.scaler).exists():
        logger.warning(f"Scaler file not found: {args.scaler}")
        args.scaler = None

    # Get symbols
    if args.symbols_file:
        # Load from file
        with open(args.symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    elif args.test_mode:
        if not args.db:
            logger.error("--db required for test mode")
            return

        symbols = get_symbols_from_db(args.db)
        logger.info(f"Loaded {len(symbols)} symbols from database")

    else:
        # For live mode, would load from config or database
        if not args.db:
            logger.error("--db required to get symbol list")
            return

        symbols = get_symbols_from_db(args.db)
        logger.info(f"Will monitor {len(symbols)} symbols")

    # Create orchestrator
    orchestrator = LiveInferenceOrchestrator(
        model_path=args.model,
        symbols=symbols,
        scaler_path=args.scaler,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
        min_alert_probability=args.min_probability,
        alert_log_file=args.alert_log,
        device=args.device
    )

    # Run
    if args.test_mode:
        await orchestrator.run_test_mode(args.db, max_candles=args.max_candles)

    else:
        # Live mode (public WebSocket - no auth required)
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket components not available - install dependencies")
            return

        logger.info("="*80)
        logger.info("LIVE MODE: Connecting to Coinbase Public WebSocket")
        logger.info("="*80)
        logger.info("No authentication required")
        logger.info("Monitoring public market data for all symbols")
        logger.info("="*80)
        logger.info("")

        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            await orchestrator.run_with_websocket()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        finally:
            orchestrator.print_statistics()


if __name__ == "__main__":
    asyncio.run(main())
