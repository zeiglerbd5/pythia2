"""
Reddit Source

Monitors cryptocurrency subreddits for trending posts and news.

Priority: Tier 2
Cost: Free (Reddit API)
Rate Limit: 60 req/min
"""

import os
import asyncio
import base64
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Set
import aiohttp
from loguru import logger

from .base import BaseSource, NewsItem


class RedditSource(BaseSource):
    """
    Monitors Reddit for cryptocurrency news and sentiment.

    Subreddits monitored:
    - r/cryptocurrency
    - r/CryptoMarkets
    - r/altcoin

    Signal triggers:
    - Posts with 50+ upvotes in first hour mentioning our coins
    - "News" and "Breaking" flaired posts
    - High upvote velocity = retail attention incoming
    """

    OAUTH_URL = "https://oauth.reddit.com"
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"

    # Subreddits to monitor
    DEFAULT_SUBREDDITS = [
        "cryptocurrency",
        "CryptoMarkets",
        "altcoin",
        "CryptoCurrency",  # Case variation
        "Bitcoin",
        "ethereum",
        "solana",
    ]

    # Flairs that indicate important news
    NEWS_FLAIRS = ["news", "breaking", "announcement", "official", "media"]

    # Minimum upvotes for signal
    MIN_UPVOTES = 50

    # Minimum upvote ratio (to filter controversial/negative posts)
    MIN_UPVOTE_RATIO = 0.7

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        min_upvotes: int = 50,
        request_timeout: int = 30,
    ):
        """
        Initialize Reddit source.

        Args:
            client_id: Reddit API client ID (or REDDIT_CLIENT_ID env var)
            client_secret: Reddit API client secret (or REDDIT_CLIENT_SECRET env var)
            subreddits: List of subreddits to monitor
            min_upvotes: Minimum upvotes for signal
            request_timeout: Timeout for requests
        """
        super().__init__(
            rate_limit_per_minute=60,  # Reddit's free tier limit
            request_timeout=request_timeout,
        )

        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self.subreddits = subreddits or self.DEFAULT_SUBREDDITS
        self.min_upvotes = min_upvotes

        # OAuth token
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Track seen posts
        self._seen_posts: Set[str] = set()

        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not set - using unauthenticated access (limited)")

    @property
    def source_name(self) -> str:
        return "reddit"

    @property
    def source_credibility(self) -> float:
        return 0.4  # Community source, lower credibility

    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Get OAuth access token from Reddit."""
        if self._access_token and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                return self._access_token

        if not self.client_id or not self.client_secret:
            return None

        try:
            # Create basic auth header
            auth_str = f"{self.client_id}:{self.client_secret}"
            auth_bytes = base64.b64encode(auth_str.encode()).decode()

            headers = {
                "Authorization": f"Basic {auth_bytes}",
                "User-Agent": "PythiaNewsBot/1.0",
            }

            data = {
                "grant_type": "client_credentials",
            }

            async with session.post(self.AUTH_URL, headers=headers, data=data, timeout=self.request_timeout) as response:
                if response.status != 200:
                    logger.warning(f"Reddit OAuth failed: {response.status}")
                    return None

                result = await response.json()

                self._access_token = result.get("access_token")
                expires_in = result.get("expires_in", 3600)
                self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

                return self._access_token

        except Exception as e:
            logger.warning(f"Reddit OAuth error: {e}")
            return None

    async def fetch_items(self) -> List[NewsItem]:
        """Fetch trending posts from monitored subreddits."""
        items = []
        posts_checked = 0
        posts_filtered = 0

        async with aiohttp.ClientSession() as session:
            # Get OAuth token if credentials are available
            token = await self._get_access_token(session)

            for subreddit in self.subreddits:
                try:
                    subreddit_items, checked, filtered = await self._fetch_subreddit(session, subreddit, token)
                    items.extend(subreddit_items)
                    posts_checked += checked
                    posts_filtered += filtered
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Error fetching r/{subreddit}: {e}")

        # Log diagnostic info
        if posts_checked > 0:
            logger.debug(f"[reddit] Checked {posts_checked} posts, {posts_filtered} already seen, {len(items)} new items")

        return items

    async def _fetch_subreddit(
        self,
        session: aiohttp.ClientSession,
        subreddit: str,
        token: Optional[str]
    ) -> tuple:
        """Fetch hot posts from a subreddit. Returns (items, posts_checked, posts_filtered)."""
        items = []
        posts_checked = 0
        posts_filtered = 0

        # Use OAuth endpoint if we have a token, otherwise public API
        if token:
            url = f"{self.OAUTH_URL}/r/{subreddit}/hot"
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": "PythiaNewsBot/1.0",
            }
        else:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {
                "User-Agent": "PythiaNewsBot/1.0",
            }

        params = {
            "limit": 25,
            "t": "day",  # Last 24 hours
        }

        try:
            async with session.get(url, headers=headers, params=params, timeout=self.request_timeout) as response:
                if response.status == 429:
                    logger.warning("Reddit rate limited")
                    return [], 0, 0

                if response.status != 200:
                    return [], 0, 0

                data = await response.json()
                posts = data.get("data", {}).get("children", [])
                posts_checked = len(posts)

                for post_wrapper in posts:
                    post = post_wrapper.get("data", {})
                    post_id = post.get("id", "")

                    # Check if already seen before processing
                    if post_id in self._seen_posts:
                        posts_filtered += 1
                        continue

                    item = self._process_post(post, subreddit)
                    if item:
                        items.append(item)

        except aiohttp.ClientError as e:
            logger.debug(f"Reddit network error for r/{subreddit}: {e}")

        return items, posts_checked, posts_filtered

    def _process_post(self, post: Dict[str, Any], subreddit: str) -> Optional[NewsItem]:
        """Process a single Reddit post."""
        post_id = post.get("id", "")

        # Mark as seen (filtering already done in _fetch_subreddit)
        self._seen_posts.add(post_id)

        # Limit cache size
        if len(self._seen_posts) > 10000:
            self._seen_posts = set(list(self._seen_posts)[-5000:])

        # Extract post data
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        url = post.get("url", "")
        permalink = post.get("permalink", "")
        upvotes = post.get("ups", 0)
        upvote_ratio = post.get("upvote_ratio", 0.5)
        created_utc = post.get("created_utc", 0)
        flair = post.get("link_flair_text", "").lower() if post.get("link_flair_text") else ""
        author = post.get("author", "")

        # Check age (skip posts older than 24 hours)
        timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc) if created_utc else datetime.now(timezone.utc)
        age_hours = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600

        if age_hours > 24:
            return None

        # Determine if this is a signal-worthy post
        is_news_flair = any(nf in flair for nf in self.NEWS_FLAIRS)
        has_enough_upvotes = upvotes >= self.min_upvotes
        has_good_ratio = upvote_ratio >= self.MIN_UPVOTE_RATIO

        # Calculate upvote velocity (upvotes per hour)
        if age_hours > 0:
            upvote_velocity = upvotes / age_hours
        else:
            upvote_velocity = upvotes

        # High velocity in first hour is a strong signal
        high_velocity = age_hours <= 1 and upvotes >= 25

        # Must meet either upvote threshold or be a news flair with moderate engagement
        if not (has_enough_upvotes or (is_news_flair and upvotes >= 20) or high_velocity):
            return None

        if not has_good_ratio:
            return None

        # Classify event type
        if is_news_flair:
            event_type = "sentiment_spike"  # News flair = curated news
        elif high_velocity:
            event_type = "sentiment_spike"  # High velocity = viral content
        else:
            event_type = "mention"

        # Build Reddit URL
        if permalink:
            full_url = f"https://reddit.com{permalink}"
        else:
            full_url = url

        return NewsItem(
            source=self.source_name,
            event_type=event_type,
            title=title[:300],
            content=selftext[:500] if selftext else title,
            url=full_url,
            timestamp=timestamp,
            author=f"u/{author}",
            verified=False,
            engagement={
                "upvotes": upvotes,
                "upvote_ratio": upvote_ratio,
                "upvote_velocity": upvote_velocity,
            },
            raw_data={
                "post_id": post_id,
                "subreddit": subreddit,
                "flair": flair,
                "age_hours": age_hours,
            },
        )


if __name__ == "__main__":
    # Test the Reddit source
    import asyncio

    async def test():
        source = RedditSource()

        print(f"Source: {source.source_name}")
        print(f"Credibility: {source.source_credibility}")
        print(f"API credentials: {'set' if source.client_id else 'NOT SET'}")
        print(f"Monitoring {len(source.subreddits)} subreddits")

        print("\nFetching Reddit posts...")
        items = await source.fetch_with_rate_limit()

        print(f"\nFound {len(items)} relevant posts:")
        for item in items[:10]:
            engagement = item.engagement or {}
            print(f"  - [{item.event_type}] r/{item.raw_data.get('subreddit')}: {item.title[:50]}...")
            print(f"    Upvotes: {engagement.get('upvotes', 0)} | Ratio: {engagement.get('upvote_ratio', 0):.2f}")
            print()

        print(f"Health status: {source.get_health_status()}")

    asyncio.run(test())
