"""
Configuration management utilities for Pythia.

Loads and validates configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from loguru import logger


class TradingConfig(BaseModel):
    """Trading configuration."""
    context_pairs: list[str]
    target_pairs: list[str]
    pair_criteria: Dict[str, Any]


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""
    endpoint: str
    channels: list[str]
    jwt_refresh_interval_seconds: int
    reconnect_delay_seconds: int
    max_reconnect_delay_seconds: int
    reconnect_backoff_multiplier: float
    message_queue_size: int
    batch_insert_interval_seconds: int


class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str
    retention: Dict[str, int]
    batch_size: int
    batch_timeout_seconds: int


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    timeframes: Dict[str, str]
    lookback: Dict[str, int]
    order_book: Dict[str, Any]
    microstructure: Dict[str, Any]
    volume: Dict[str, Any]
    price: Dict[str, Any]
    whale: Dict[str, int]


class SignalsConfig(BaseModel):
    """Signal generation configuration."""
    ensemble: Dict[str, Any]
    confidence: Dict[str, float]
    filters: Dict[str, Any]
    target_signals_per_day: int
    max_signals_per_day: int


class RiskConfig(BaseModel):
    """Risk management configuration."""
    position_sizing: Dict[str, Any]
    stop_loss: Dict[str, Any]
    portfolio: Dict[str, Any]
    daily: Dict[str, Any]
    volatility: Dict[str, Any]


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    default_order_type: str
    limit_order_offset_pct: float
    expected_slippage_pct: float
    maker_fee_pct: float
    taker_fee_pct: float
    order_timeout_seconds: int
    cancel_unfilled_after_seconds: int
    position_reconciliation_interval_seconds: int


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    log_level: str
    log_file: str
    log_rotation: str
    log_retention: str
    structured_logging: bool
    metrics_enabled: bool
    metrics_port: int
    alerts: Dict[str, Any]
    alert_triggers: Dict[str, Any]


class BacktestingConfig(BaseModel):
    """Backtesting configuration."""
    enabled: bool
    start_date: str
    end_date: str
    initial_capital: float
    commission_pct: float
    slippage_pct: float


class TrainingConfig(BaseModel):
    """Model training configuration."""
    data_split: Dict[str, int]
    target: Dict[str, Any]
    retraining: Dict[str, Any]


class SystemConfig(BaseModel):
    """System configuration."""
    environment: str
    paper_trading: bool
    max_workers: int
    enable_gpu: bool


class Config(BaseModel):
    """Main configuration container."""
    trading: TradingConfig
    websocket: WebSocketConfig
    database: DatabaseConfig
    features: FeaturesConfig
    signals: SignalsConfig
    risk: RiskConfig
    execution: ExecutionConfig
    monitoring: MonitoringConfig
    backtesting: BacktestingConfig
    training: TrainingConfig
    system: SystemConfig


class ModelConfig(BaseModel):
    """ML Model configuration container."""
    model_architecture: Dict[str, Any]
    alternative_architectures: Dict[str, Any]
    loss: Dict[str, Any]
    optimizer: Dict[str, Any]
    training: Dict[str, Any]
    preprocessing: Dict[str, Any]
    imbalance: Dict[str, Any]
    feature_selection: Dict[str, Any]
    ensemble: Dict[str, Any]
    walk_forward: Dict[str, Any]
    target: Dict[str, Any]
    evaluation: Dict[str, Any]
    persistence: Dict[str, Any]
    inference: Dict[str, Any]
    monitoring: Dict[str, Any]
    reproducibility: Dict[str, Any]
    performance_targets: Dict[str, float]


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings from .env file."""

    # Coinbase API
    coinbase_api_key: str = Field(default="", alias="COINBASE_API_KEY")
    coinbase_api_secret: str = Field(default="", alias="COINBASE_API_SECRET")
    coinbase_private_key_path: Optional[str] = Field(default=None, alias="COINBASE_PRIVATE_KEY_PATH")

    # Database
    database_path: str = Field(default="data/pythia.duckdb", alias="DATABASE_PATH")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="logs/pythia.log", alias="LOG_FILE")

    # Telegram
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # Email
    smtp_server: Optional[str] = Field(default=None, alias="SMTP_SERVER")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, alias="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")

    # Trading
    environment: str = Field(default="development", alias="ENVIRONMENT")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    initial_capital: float = Field(default=10000.0, alias="INITIAL_CAPITAL")

    # News monitoring APIs
    whale_alert_api_key: Optional[str] = Field(default=None, alias="WHALE_ALERT_API_KEY")
    cryptopanic_api_key: Optional[str] = Field(default=None, alias="CRYPTOPANIC_API_KEY")
    coinmarketcal_api_key: Optional[str] = Field(default=None, alias="COINMARKETCAL_API_KEY")
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")

    # PyTorch
    torch_device: str = Field(default="mps", alias="TORCH_DEVICE")
    mps_fallback: bool = Field(default=True, alias="MPS_FALLBACK")

    # Performance
    max_workers: int = Field(default=4, alias="MAX_WORKERS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ConfigManager:
    """Manages configuration loading and access."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory (default: ./config)
        """
        if config_dir is None:
            # Find config directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)

        # Load configurations
        self.config = self._load_config()
        self.model_config = self._load_model_config()
        self.env = self._load_env()

        logger.info(f"Configuration loaded from {self.config_dir}")

    def _load_config(self) -> Config:
        """Load main configuration from config.yaml."""
        config_file = self.config_dir / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        return Config(**data)

    def _load_model_config(self) -> ModelConfig:
        """Load model configuration from model_config.yaml."""
        config_file = self.config_dir / "model_config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Model config file not found: {config_file}")

        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)

        return ModelConfig(**data)

    def _load_env(self) -> EnvironmentSettings:
        """Load environment settings from .env file."""
        # Look for .env in project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            return EnvironmentSettings(_env_file=str(env_file))
        else:
            logger.warning(".env file not found, using defaults")
            return EnvironmentSettings()

    def get_trading_pairs(self) -> tuple[list[str], list[str]]:
        """
        Get trading pairs separated into context and targets.

        Returns:
            Tuple of (context_pairs, target_pairs)
        """
        return (
            self.config.trading.context_pairs,
            self.config.trading.target_pairs
        )

    def get_all_pairs(self) -> list[str]:
        """
        Get all trading pairs (context + target).

        Returns:
            List of all trading pair symbols
        """
        return self.config.trading.context_pairs + self.config.trading.target_pairs

    def is_paper_trading(self) -> bool:
        """Check if paper trading mode is enabled."""
        return self.env.paper_trading

    def get_database_path(self) -> Path:
        """Get database file path."""
        db_path = Path(self.env.database_path)
        if not db_path.is_absolute():
            # Make relative to project root
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / db_path
        return db_path

    def get_log_path(self) -> Path:
        """Get log file path."""
        log_path = Path(self.env.log_file)
        if not log_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            log_path = project_root / log_path
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get global configuration manager instance.

    Returns:
        ConfigManager singleton
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager


def reload_config() -> ConfigManager:
    """
    Reload configuration from files.

    Returns:
        New ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager


if __name__ == "__main__":
    # Test configuration loading
    config_mgr = get_config()

    print("=== Configuration Test ===\n")

    print(f"Environment: {config_mgr.config.system.environment}")
    print(f"Paper Trading: {config_mgr.is_paper_trading()}")
    print(f"\nContext Pairs: {config_mgr.config.trading.context_pairs}")
    print(f"Target Pairs: {config_mgr.config.trading.target_pairs[:5]}...")
    print(f"\nWebSocket Endpoint: {config_mgr.config.websocket.endpoint}")
    print(f"Channels: {config_mgr.config.websocket.channels}")
    print(f"\nDatabase Path: {config_mgr.get_database_path()}")
    print(f"Log Path: {config_mgr.get_log_path()}")

    print(f"\n=== Model Configuration ===")
    print(f"Model Type: {config_mgr.model_config.model_architecture['type']}")
    print(f"CNN Filters: {config_mgr.model_config.model_architecture['cnn']['filters']}")
    print(f"LSTM Units: {config_mgr.model_config.model_architecture['lstm']['layer1_units']}, "
          f"{config_mgr.model_config.model_architecture['lstm']['layer2_units']}")
    print(f"Focal Loss Alpha: {config_mgr.model_config.loss['focal_loss']['alpha']}")
    print(f"Focal Loss Gamma: {config_mgr.model_config.loss['focal_loss']['gamma']}")
