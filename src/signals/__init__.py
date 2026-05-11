"""
News and Signal Monitoring System

Detects events (listings, whale moves, partnerships) that cause 20%+ price spikes
BEFORE or AS they happen.
"""

from .symbol_mapper import SymbolMapper
from .news_monitor import NewsMonitor, NewsSignal

__all__ = ['SymbolMapper', 'NewsMonitor', 'NewsSignal']
