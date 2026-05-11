"""
Evaluation Utilities for RL Trading Agent (Phase 3)

Implements:
- Walk-forward backtesting
- Performance metrics (Sharpe, win rate, drawdown, profit factor)
- Comparison to baselines
- A/B testing framework

Proper evaluation is critical to ensure the RL agent performs
well on unseen data and doesn't overfit to historical patterns.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
from scipy import stats
from loguru import logger


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    return_pct: float
    size: float
    exit_reason: str
    regime: Optional[str] = None
    symbol: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0              # Value at Risk (95%)
    cvar_95: float = 0.0             # Conditional VaR

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trading stats
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0  # Minutes

    # Activity
    trades_per_day: float = 0.0
    exposure_pct: float = 0.0        # Time in position

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'n_trades': self.n_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'trades_per_day': self.trades_per_day,
            'exposure_pct': self.exposure_pct,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Total Return: {self.total_return*100:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Win Rate: {self.win_rate*100:.1f}% | "
            f"Max DD: {self.max_drawdown*100:.1f}% | "
            f"Trades: {self.n_trades}"
        )


class MetricsCalculator:
    """Calculate evaluation metrics from trades."""

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252 * 24,  # Hourly data
    ):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Risk-free rate (annual)
            annualization_factor: Factor for annualizing returns
        """
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

    def calculate(
        self,
        trades: List[TradeRecord],
        equity_curve: Optional[pd.Series] = None,
        total_duration_days: Optional[float] = None,
    ) -> EvaluationMetrics:
        """
        Calculate comprehensive metrics from trades.

        Args:
            trades: List of trade records
            equity_curve: Equity curve series (optional)
            total_duration_days: Total evaluation period in days

        Returns:
            EvaluationMetrics object
        """
        metrics = EvaluationMetrics()

        if not trades:
            return metrics

        # Extract returns
        returns = np.array([t.return_pct * t.size for t in trades])

        # Basic stats
        metrics.n_trades = len(trades)
        metrics.total_return = np.sum(returns)

        wins = returns > 0
        losses = returns <= 0

        metrics.win_rate = wins.mean() if len(returns) > 0 else 0

        if wins.any():
            metrics.avg_win = returns[wins].mean()
        if losses.any():
            metrics.avg_loss = returns[losses].mean()

        # Profit factor
        gross_profit = returns[wins].sum() if wins.any() else 0
        gross_loss = abs(returns[losses].sum()) if losses.any() else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Trade duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in trades
        ]
        metrics.avg_trade_duration = np.mean(durations) if durations else 0

        # Risk metrics from equity curve
        if equity_curve is not None and len(equity_curve) > 1:
            metrics = self._calculate_risk_metrics(metrics, equity_curve)

        # Activity metrics
        if total_duration_days and total_duration_days > 0:
            metrics.trades_per_day = len(trades) / total_duration_days

            # Exposure
            total_minutes = total_duration_days * 24 * 60
            in_position_minutes = sum(durations)
            metrics.exposure_pct = in_position_minutes / total_minutes if total_minutes > 0 else 0

        return metrics

    def _calculate_risk_metrics(
        self,
        metrics: EvaluationMetrics,
        equity_curve: pd.Series,
    ) -> EvaluationMetrics:
        """Calculate risk metrics from equity curve."""
        # Returns from equity curve
        returns = equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return metrics

        # Volatility (annualized)
        metrics.volatility = returns.std() * np.sqrt(self.annualization_factor)

        # Annualized return
        total_periods = len(returns)
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        metrics.annual_return = (1 + total_return) ** (self.annualization_factor / total_periods) - 1

        # Sharpe ratio
        excess_return = returns.mean() - self.risk_free_rate / self.annualization_factor
        if returns.std() > 0:
            metrics.sharpe_ratio = excess_return / returns.std() * np.sqrt(self.annualization_factor)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(self.annualization_factor)
            if downside_std > 0:
                metrics.sortino_ratio = (metrics.annual_return - self.risk_free_rate) / downside_std

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = abs(drawdown.min())

        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown

        # VaR and CVaR
        metrics.var_95 = np.percentile(returns, 5)
        metrics.cvar_95 = returns[returns <= metrics.var_95].mean() if len(returns[returns <= metrics.var_95]) > 0 else metrics.var_95

        return metrics


@dataclass
class WalkForwardFold:
    """Single fold for walk-forward validation."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    fold_index: int


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Properly respects temporal ordering to prevent lookahead bias.

    +-------+-------+-------+-------+-------+
    |Train  |Train  |Train  | Test  |       |
    +-------+-------+-------+-------+-------+
    |       |Train  |Train  |Train  | Test  |
    +-------+-------+-------+-------+-------+
    """

    def __init__(
        self,
        train_period_days: int = 90,
        test_period_days: int = 30,
        step_days: int = 30,
        gap_days: int = 1,  # Gap between train and test to avoid leakage
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_period_days: Training period length
            test_period_days: Test period length
            step_days: Step size between folds
            gap_days: Gap between train and test
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.step_days = step_days
        self.gap_days = gap_days

    def get_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardFold]:
        """
        Generate walk-forward folds.

        Args:
            start_date: Data start date
            end_date: Data end date

        Returns:
            List of WalkForwardFold objects
        """
        folds = []
        fold_index = 0

        current_train_start = start_date

        while True:
            train_end = current_train_start + timedelta(days=self.train_period_days)
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_period_days)

            if test_end > end_date:
                break

            folds.append(WalkForwardFold(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_index=fold_index,
            ))

            current_train_start += timedelta(days=self.step_days)
            fold_index += 1

        logger.info(f"Generated {len(folds)} walk-forward folds")
        return folds


class Evaluator:
    """
    Complete evaluation framework for RL trading agent.

    Provides:
    - Walk-forward backtesting
    - Comparison to baselines
    - Statistical significance testing
    """

    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None,
        validator: Optional[WalkForwardValidator] = None,
    ):
        """
        Initialize evaluator.

        Args:
            metrics_calculator: Metrics calculator
            validator: Walk-forward validator
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.validator = validator or WalkForwardValidator()

        # Results storage
        self.results: Dict[str, List[EvaluationMetrics]] = defaultdict(list)

    def evaluate_agent(
        self,
        agent: Any,
        env: Any,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Tuple[EvaluationMetrics, List[TradeRecord], pd.Series]:
        """
        Evaluate agent on environment.

        Args:
            agent: Trained agent
            env: Trading environment
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            (metrics, trades, equity_curve)
        """
        all_trades = []
        equity = [1.0]  # Start with 1.0
        total_steps = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_trades = []

            while not done:
                # Get action (with mask if available)
                action_mask = env.get_action_mask() if hasattr(env, 'get_action_mask') else None
                action, _ = agent.predict(obs, action_mask=action_mask, deterministic=deterministic)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_steps += 1

                # Record trade if completed
                if info.get('trade_closed'):
                    # Extract trade info from environment
                    if hasattr(env, 'episode_trades') and env.episode_trades:
                        trade = env.episode_trades[-1]
                        episode_trades.append(TradeRecord(
                            entry_time=trade.entry_time,
                            exit_time=trade.exit_time,
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            return_pct=trade.return_pct,
                            size=trade.size,
                            exit_reason=trade.exit_reason,
                            symbol=info.get('symbol'),
                        ))

            # Update equity
            episode_return = info.get('episode_return', 0)
            equity.append(equity[-1] * (1 + episode_return))

            all_trades.extend(episode_trades)

        # Calculate metrics
        equity_curve = pd.Series(equity)
        total_days = n_episodes  # Assuming ~1 day per episode

        metrics = self.metrics_calculator.calculate(
            all_trades,
            equity_curve,
            total_days,
        )

        return metrics, all_trades, equity_curve

    def walk_forward_evaluate(
        self,
        agent_factory: Callable,
        env_factory: Callable,
        start_date: datetime,
        end_date: datetime,
        train_timesteps: int = 100_000,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward evaluation.

        Args:
            agent_factory: Function to create new agent
            env_factory: Function to create environment
            start_date: Evaluation start date
            end_date: Evaluation end date
            train_timesteps: Training timesteps per fold

        Returns:
            Walk-forward results
        """
        folds = self.validator.get_folds(start_date, end_date)
        fold_results = []

        for fold in folds:
            logger.info(f"Processing fold {fold.fold_index}: "
                       f"Train {fold.train_start} to {fold.train_end}, "
                       f"Test {fold.test_start} to {fold.test_end}")

            # Create training environment
            train_env = env_factory(
                start_time=fold.train_start,
                end_time=fold.train_end,
            )

            # Create and train agent
            agent = agent_factory(train_env)
            agent.train(total_timesteps=train_timesteps)

            # Create test environment
            test_env = env_factory(
                start_time=fold.test_start,
                end_time=fold.test_end,
            )

            # Evaluate
            metrics, trades, equity = self.evaluate_agent(
                agent, test_env, n_episodes=30
            )

            fold_results.append({
                'fold': fold.fold_index,
                'train_period': (fold.train_start.isoformat(), fold.train_end.isoformat()),
                'test_period': (fold.test_start.isoformat(), fold.test_end.isoformat()),
                'metrics': metrics.to_dict(),
                'n_trades': len(trades),
            })

            # Cleanup
            train_env.close()
            test_env.close()

        # Aggregate results
        all_sharpes = [r['metrics']['sharpe_ratio'] for r in fold_results]
        all_returns = [r['metrics']['total_return'] for r in fold_results]

        return {
            'folds': fold_results,
            'summary': {
                'mean_sharpe': np.mean(all_sharpes),
                'std_sharpe': np.std(all_sharpes),
                'mean_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'n_folds': len(folds),
            }
        }

    def compare_to_baseline(
        self,
        agent_metrics: EvaluationMetrics,
        baseline_metrics: EvaluationMetrics,
        baseline_name: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Compare agent performance to baseline.

        Args:
            agent_metrics: Agent evaluation metrics
            baseline_metrics: Baseline metrics
            baseline_name: Name of baseline

        Returns:
            Comparison results
        """
        comparison = {
            'agent': agent_metrics.to_dict(),
            'baseline': baseline_metrics.to_dict(),
            'baseline_name': baseline_name,
            'improvements': {},
        }

        # Calculate improvements
        for key in ['sharpe_ratio', 'total_return', 'win_rate']:
            agent_val = getattr(agent_metrics, key)
            base_val = getattr(baseline_metrics, key)

            if base_val != 0:
                improvement = (agent_val - base_val) / abs(base_val) * 100
            else:
                improvement = float('inf') if agent_val > 0 else 0

            comparison['improvements'][key] = improvement

        return comparison


class ABTester:
    """
    A/B testing framework for comparing model versions.

    Allows safe testing of new models against production model.
    """

    def __init__(
        self,
        current_model: Any,
        candidate_model: Any,
        test_allocation: float = 0.1,
    ):
        """
        Initialize A/B tester.

        Args:
            current_model: Production model
            candidate_model: Candidate model to test
            test_allocation: Fraction of decisions for candidate
        """
        self.current = current_model
        self.candidate = candidate_model
        self.allocation = test_allocation

        # Track results
        self.current_trades: List[TradeRecord] = []
        self.candidate_trades: List[TradeRecord] = []

    def select_model(self) -> Tuple[Any, str]:
        """
        Select which model to use.

        Returns:
            (model, model_type)
        """
        if np.random.random() < self.allocation:
            return self.candidate, 'candidate'
        return self.current, 'current'

    def record_trade(self, model_type: str, trade: TradeRecord) -> None:
        """Record trade result."""
        if model_type == 'current':
            self.current_trades.append(trade)
        else:
            self.candidate_trades.append(trade)

    def evaluate(
        self,
        min_trades: int = 50,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate vs current model.

        Args:
            min_trades: Minimum trades for significance
            significance_level: P-value threshold

        Returns:
            Evaluation results
        """
        if len(self.candidate_trades) < min_trades:
            return {
                'status': 'insufficient_data',
                'candidate_trades': len(self.candidate_trades),
                'required_trades': min_trades,
            }

        # Calculate metrics
        calculator = MetricsCalculator()

        current_metrics = calculator.calculate(self.current_trades, total_duration_days=30)
        candidate_metrics = calculator.calculate(self.candidate_trades, total_duration_days=30)

        # Statistical test
        current_returns = [t.return_pct * t.size for t in self.current_trades]
        candidate_returns = [t.return_pct * t.size for t in self.candidate_trades]

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(candidate_returns, current_returns)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(current_returns) + np.var(candidate_returns)) / 2
        )
        effect_size = (np.mean(candidate_returns) - np.mean(current_returns)) / (pooled_std + 1e-8)

        # Decision
        is_significant = p_value < significance_level
        is_better = candidate_metrics.sharpe_ratio > current_metrics.sharpe_ratio

        recommendation = 'keep_current'
        if is_significant and is_better:
            recommendation = 'promote_candidate'
        elif not is_significant:
            recommendation = 'continue_testing'

        return {
            'status': 'evaluated',
            'current_metrics': current_metrics.to_dict(),
            'candidate_metrics': candidate_metrics.to_dict(),
            'statistical_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'is_significant': is_significant,
            },
            'recommendation': recommendation,
        }

    def reset(self) -> None:
        """Reset test results."""
        self.current_trades = []
        self.candidate_trades = []


if __name__ == "__main__":
    # Test evaluation utilities
    print("Testing Evaluation Utilities\n" + "=" * 50)

    # Generate synthetic trades
    np.random.seed(42)
    n_trades = 100

    trades = []
    current_time = datetime(2024, 1, 1)

    for i in range(n_trades):
        duration = np.random.randint(30, 480)  # 30 min to 8 hours
        return_pct = np.random.randn() * 0.03 + 0.005  # Slight positive bias

        trades.append(TradeRecord(
            entry_time=current_time,
            exit_time=current_time + timedelta(minutes=duration),
            entry_price=100,
            exit_price=100 * (1 + return_pct),
            return_pct=return_pct,
            size=1.0,
            exit_reason='manual' if return_pct > 0 else 'stop_loss',
        ))

        current_time += timedelta(hours=np.random.randint(1, 24))

    # Generate equity curve
    returns = [t.return_pct for t in trades]
    equity = pd.Series([1.0] + list(np.cumprod([1 + r for r in returns])))

    # Test MetricsCalculator
    print("\n1. MetricsCalculator")
    calculator = MetricsCalculator()
    metrics = calculator.calculate(trades, equity, total_duration_days=100)
    print(f"   {metrics}")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Sortino: {metrics.sortino_ratio:.2f}")

    # Test WalkForwardValidator
    print("\n2. WalkForwardValidator")
    validator = WalkForwardValidator(
        train_period_days=60,
        test_period_days=20,
        step_days=20,
    )

    folds = validator.get_folds(
        datetime(2024, 1, 1),
        datetime(2024, 6, 1),
    )

    for fold in folds[:3]:
        print(f"   Fold {fold.fold_index}: Train {fold.train_start.date()} to {fold.train_end.date()}, "
              f"Test {fold.test_start.date()} to {fold.test_end.date()}")

    # Test comparison
    print("\n3. Baseline Comparison")
    evaluator = Evaluator()

    # Fake baseline metrics
    baseline_metrics = EvaluationMetrics(
        total_return=0.15,
        sharpe_ratio=1.0,
        win_rate=0.45,
        max_drawdown=0.12,
    )

    comparison = evaluator.compare_to_baseline(metrics, baseline_metrics, "Buy and Hold")
    print(f"   Sharpe improvement: {comparison['improvements']['sharpe_ratio']:.1f}%")
    print(f"   Return improvement: {comparison['improvements']['total_return']:.1f}%")

    # Test A/B Tester
    print("\n4. A/B Tester")

    class MockModel:
        pass

    ab_tester = ABTester(MockModel(), MockModel(), test_allocation=0.2)

    # Simulate trades
    for _ in range(60):
        model, model_type = ab_tester.select_model()

        # Candidate slightly better
        bias = 0.001 if model_type == 'candidate' else 0
        return_pct = np.random.randn() * 0.03 + bias

        ab_tester.record_trade(
            model_type,
            TradeRecord(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                entry_price=100,
                exit_price=100 * (1 + return_pct),
                return_pct=return_pct,
                size=1.0,
                exit_reason='manual',
            )
        )

    result = ab_tester.evaluate(min_trades=10)
    print(f"   Status: {result['status']}")
    print(f"   Recommendation: {result['recommendation']}")
    if 'statistical_test' in result:
        print(f"   P-value: {result['statistical_test']['p_value']:.4f}")

    print("\nEvaluation tests passed!")
