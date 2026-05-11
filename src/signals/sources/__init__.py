"""
News Source Adapters

Each adapter implements a consistent interface for fetching news/signals
from different sources with built-in rate limiting.
"""

from .base import BaseSource, NewsItem
from .exchange_listings import ExchangeListingsSource
from .whale_alert import WhaleAlertSource
from .twitter_rss import TwitterRSSSource
from .reddit import RedditSource

__all__ = [
    'BaseSource',
    'NewsItem',
    'ExchangeListingsSource',
    'WhaleAlertSource',
    'TwitterRSSSource',
    'RedditSource',
]
