"""
Coinbase Advanced Trade API ES256 JWT Authentication

Implements ES256 (ECDSA P-256) JWT token generation for Coinbase WebSocket
and REST API authentication with automatic 90-second refresh logic.

Based on implementation guide specifications:
- ES256 JWT tokens with 2-minute expiration
- ECDSA P-256 curve private key signing
- Automatic refresh every 90 seconds for long-running connections
- Random 16-byte nonce for security
"""

import os
import time
import secrets
from typing import Optional
from datetime import datetime, timedelta

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from loguru import logger


class CoinbaseAuth:
    """
    Handles Coinbase Advanced Trade API authentication using ES256 JWT tokens.

    Per implementation guide:
    - Token expiration: 2 minutes
    - Refresh interval: 90 seconds (for WebSocket connections)
    - Algorithm: ES256 (ECDSA with P-256 curve)
    - Nonce: Random 16-byte value for each token
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        private_key_path: Optional[str] = None
    ):
        """
        Initialize Coinbase authentication.

        Args:
            api_key: Coinbase API key (used as 'sub' claim and 'kid' header)
            api_secret: Coinbase API secret (private key PEM format)
            private_key_path: Optional path to PEM file (alternative to api_secret)
        """
        self.api_key = api_key

        # Load private key from file or use provided secret
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, 'r') as f:
                self.private_key = f.read()
            logger.info(f"Loaded private key from {private_key_path}")
        else:
            self.private_key = api_secret

        # Parse private key for JWT signing
        self._private_key_obj = serialization.load_pem_private_key(
            self.private_key.encode('utf-8'),
            password=None,
            backend=default_backend()
        )

        # Token state
        self._current_token: Optional[str] = None
        self._token_created_at: Optional[float] = None
        self._token_expires_at: Optional[float] = None

        logger.info("CoinbaseAuth initialized successfully")

    def generate_token(
        self,
        service: str = "retail_rest_api_proxy",
        for_websocket: bool = False,
        request_method: str = None,
        request_path: str = None
    ) -> str:
        """
        Generate a new ES256 JWT token.

        Per Coinbase SDK:
        - Algorithm: ES256 (ECDSA P-256)
        - Expiration: 2 minutes
        - Issuer: "cdp" for both WebSocket and REST
        - REST requires 'uri' claim with "METHOD api.coinbase.com/path"
        - Headers: kid (API key), nonce (16 random bytes)

        Args:
            service: Service name for 'aud' claim (default: retail_rest_api_proxy)
            for_websocket: If True, generating for WebSocket (no uri claim needed)
            request_method: HTTP method for REST (e.g., "GET", "POST")
            request_path: Path for REST (e.g., "/api/v3/brokerage/products/BTC-USD/ticker")

        Returns:
            JWT token string
        """
        current_time = int(time.time())

        # Token valid for 2 minutes (120 seconds) per guide
        expiration_time = current_time + 120

        # Generate random 16-byte nonce for security
        nonce = secrets.token_hex(16)

        # Build JWT claims - issuer is always "cdp" per Coinbase SDK
        payload = {
            "sub": self.api_key,  # Subject: API key
            "iss": "cdp",  # Issuer: always "cdp"
            "nbf": current_time,  # Not before
            "exp": expiration_time,  # Expiration (2 min)
        }

        # REST API requires uri claim with method and path
        if not for_websocket and request_method and request_path:
            # Format: "METHOD api.coinbase.com/path"
            uri = f"{request_method} api.coinbase.com{request_path}"
            payload["uri"] = uri

        # Build JWT headers
        headers = {
            "kid": self.api_key,  # Key ID: API key
            "nonce": nonce,  # Random nonce
        }

        # Sign with ES256 (ECDSA P-256)
        token = jwt.encode(
            payload,
            self._private_key_obj,
            algorithm="ES256",
            headers=headers
        )

        # Update state
        self._current_token = token
        self._token_created_at = current_time
        self._token_expires_at = expiration_time

        logger.debug(
            f"Generated new JWT token (expires in 120s)",
            extra={
                "created_at": datetime.fromtimestamp(current_time).isoformat(),
                "expires_at": datetime.fromtimestamp(expiration_time).isoformat(),
                "nonce": nonce
            }
        )

        return token

    def get_token(self, auto_refresh: bool = True, for_websocket: bool = False) -> str:
        """
        Get current JWT token, optionally auto-refreshing if needed.

        Per implementation guide: Refresh every 90 seconds for WebSocket connections

        Args:
            auto_refresh: If True, automatically refresh if token is >90s old
            for_websocket: If True, use "cdp" issuer for WebSocket; else "coinbase-cloud" for REST

        Returns:
            Valid JWT token
        """
        if self._current_token is None:
            # No token yet, generate new one
            return self.generate_token(for_websocket=for_websocket)

        if not auto_refresh:
            return self._current_token

        # Check if token needs refresh (90 second threshold per guide)
        current_time = time.time()
        token_age = current_time - self._token_created_at if self._token_created_at else float('inf')

        if token_age >= 90:
            logger.info(f"Token is {token_age:.1f}s old, refreshing (90s threshold)")
            return self.generate_token(for_websocket=for_websocket)

        return self._current_token

    def is_token_expired(self) -> bool:
        """
        Check if current token is expired.

        Returns:
            True if token is expired or doesn't exist
        """
        if self._token_expires_at is None:
            return True

        return time.time() >= self._token_expires_at

    def needs_refresh(self, threshold_seconds: int = 90) -> bool:
        """
        Check if token needs refresh based on age threshold.

        Per implementation guide: Refresh every 90 seconds for WebSocket

        Args:
            threshold_seconds: Age threshold for refresh (default: 90)

        Returns:
            True if token should be refreshed
        """
        if self._token_created_at is None:
            return True

        token_age = time.time() - self._token_created_at
        return token_age >= threshold_seconds

    def get_auth_headers(self, method: str = "GET", path: str = "") -> dict:
        """
        Get authentication headers for REST API requests.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /api/v3/brokerage/products/BTC-USD/ticker)

        Returns:
            Dictionary with Authorization header
        """
        # Always generate fresh token for REST with URI claim
        token = self.generate_token(
            for_websocket=False,
            request_method=method,
            request_path=path
        )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def get_websocket_auth_message(
        self,
        channel: str,
        product_ids: list[str]
    ) -> dict:
        """
        Build WebSocket subscription message with authentication.

        Per implementation guide:
        - Include JWT token in subscription message
        - Subscribe to specific channels and products

        Args:
            channel: Channel name (e.g., 'level2', 'market_trades', 'ticker')
            product_ids: List of trading pair symbols (e.g., ['BTC-USD'])

        Returns:
            WebSocket subscription message dictionary
        """
        # WebSocket requires "cdp" issuer
        token = self.generate_token(for_websocket=True)

        message = {
            "type": "subscribe",
            "channel": channel,
            "product_ids": product_ids,
            "jwt": token
        }

        logger.debug(
            f"Created WebSocket auth message",
            extra={
                "channel": channel,
                "product_ids": product_ids,
                "token_age": time.time() - self._token_created_at if self._token_created_at else 0
            }
        )

        return message

    def refresh_websocket_subscription(
        self,
        channel: str,
        product_ids: list[str]
    ) -> tuple[dict, dict]:
        """
        Create unsubscribe and resubscribe messages for JWT refresh.

        Per implementation guide:
        - Unsubscribe from channel
        - Wait 100ms
        - Resubscribe with fresh JWT token

        Args:
            channel: Channel name
            product_ids: List of trading pair symbols

        Returns:
            Tuple of (unsubscribe_message, subscribe_message)
        """
        # Unsubscribe message (doesn't need JWT)
        unsubscribe_msg = {
            "type": "unsubscribe",
            "channel": channel,
            "product_ids": product_ids
        }

        # Generate fresh WebSocket token and create subscribe message
        self.generate_token(for_websocket=True)
        subscribe_msg = self.get_websocket_auth_message(channel, product_ids)

        logger.info(
            f"Created WebSocket refresh messages for {channel}",
            extra={"product_ids": product_ids}
        )

        return unsubscribe_msg, subscribe_msg

    @classmethod
    def from_env(cls) -> "CoinbaseAuth":
        """
        Create CoinbaseAuth instance from environment variables.

        Expects:
            COINBASE_API_KEY: API key
            COINBASE_API_SECRET: Private key in PEM format
            COINBASE_PRIVATE_KEY_PATH: Optional path to PEM file

        Returns:
            CoinbaseAuth instance
        """
        api_key = os.getenv("COINBASE_API_KEY")
        api_secret = os.getenv("COINBASE_API_SECRET")
        private_key_path = os.getenv("COINBASE_PRIVATE_KEY_PATH")

        if not api_key:
            raise ValueError("COINBASE_API_KEY environment variable not set")

        if not api_secret and not private_key_path:
            raise ValueError(
                "Either COINBASE_API_SECRET or COINBASE_PRIVATE_KEY_PATH must be set"
            )

        return cls(
            api_key=api_key,
            api_secret=api_secret or "",
            private_key_path=private_key_path
        )


if __name__ == "__main__":
    # Test the authentication module
    from dotenv import load_dotenv

    load_dotenv()

    # Create auth instance from environment
    auth = CoinbaseAuth.from_env()

    # Generate and display token
    token = auth.generate_token()
    print(f"Generated token: {token[:50]}...")
    print(f"Token expires in: {auth._token_expires_at - time.time():.0f} seconds")

    # Test WebSocket message
    ws_msg = auth.get_websocket_auth_message("level2", ["BTC-USD"])
    print(f"\nWebSocket subscription message: {ws_msg}")

    # Test refresh
    print(f"\nNeeds refresh (90s threshold): {auth.needs_refresh()}")
