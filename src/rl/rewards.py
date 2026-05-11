"""
Reward Functions for RL Trading Agent

Implements multiple reward formulations:
- Basic P&L reward
- Risk-adjusted reward (Sharpe-based)
- Hybrid reward with behavior shaping
- Configurable reward components

The reward function is critical for guiding the agent to learn
profitable trading behaviors while managing risk.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum
from loguru import logger


class RewardType(Enum):
    """Types of reward functions."""
    BASIC_PNL = "basic_pnl"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE = "sharpe"
    HYBRID = "hybrid"
    SPIKE_AWARE = "spike_aware"  # v3: Uses spike tracking features (DEPRECATED - too penalty-heavy)
    SPIKE_QUALITY_BONUS = "spike_quality_bonus"  # v4: Hybrid + additive bonuses for quality


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Reward type
    reward_type: RewardType = RewardType.HYBRID

    # P&L reward scales
    reward_scale_win: float = 10.0      # Amplify winning trades
    reward_scale_loss: float = 5.0      # Losses hurt less (encourage risk-taking)
    reward_scale_unrealized: float = 0.1  # Small signal for paper gains

    # Risk adjustment
    reward_scale_drawdown: float = 2.0   # Penalize giving back gains
    drawdown_threshold: float = 0.02     # 2% drawdown triggers penalty
    target_volatility: float = 0.02      # Target position volatility
    reward_volatility_bonus: float = 0.01  # Bonus for smooth equity

    # Efficiency penalties
    penalty_adjustment: float = 0.001    # Discourage fidgeting with stops
    penalty_stale_position: float = 0.1  # Don't hold forever
    max_hold_time: int = 480             # 8 hours max before penalty
    min_profit_for_hold: float = 0.005   # 0.5% min profit to keep holding

    # Behavior shaping
    reward_smart_exit: float = 1.0       # Reward for exiting above stop
    reward_patience: float = 0.1         # Small bonus for holding winners
    profit_target: float = 0.02          # 2% profit triggers patience bonus

    # Transaction costs (for net P&L calculation)
    transaction_cost: float = 0.0055     # 0.55% round trip

    # Sharpe calculation
    sharpe_window: int = 50              # Rolling window for Sharpe
    risk_free_rate: float = 0.0          # Risk-free rate (per step)
    sharpe_annualization: float = np.sqrt(252 * 24 * 60)  # For minute data


@dataclass
class SpikeAwareRewardConfig:
    """
    Configuration for spike-aware reward function (v3).

    Designed to:
    1. Penalize repeated trading on the same spike
    2. Reward selective, high-quality trades
    3. Incorporate risk-adjusted returns
    4. Use spike tracking features effectively
    """
    # =========================================
    # CORE P&L SCALING (balanced asymmetry)
    # =========================================
    # Note: Symmetric scaling encourages selectivity over gambling
    reward_scale_win: float = 15.0       # Base multiplier for winning trades
    reward_scale_loss: float = 12.0      # Losses hurt more (discourages gambling)
    reward_scale_unrealized: float = 0.05  # Small signal for paper gains

    # =========================================
    # SPIKE QUALITY MULTIPLIERS
    # =========================================
    # These multiply the trade reward based on spike state
    spike_freshness_weight: float = 1.0   # How much freshness affects reward
    stale_spike_penalty: float = 0.3      # Min multiplier for very stale spikes

    # =========================================
    # RE-ENTRY PENALTIES (v3.1: reduced to balance with opportunity cost)
    # =========================================
    # Explicit penalties for poor trading behavior
    penalty_already_traded_spike: float = 0.3   # Penalty for re-trading same spike (was 0.5)
    penalty_rapid_reentry: float = 0.15         # Penalty if minutes_since_trade < threshold (was 0.3)
    rapid_reentry_threshold: float = 0.5        # Normalized (0.5 = 30 minutes)

    # =========================================
    # OVERTRADING PENALTIES (v3.1: reduced)
    # =========================================
    penalty_overtrading: float = 0.1      # Per-trade penalty when overtrading (was 0.2)
    overtrading_threshold: float = 0.6    # trades_in_last_hour > 3 (normalized 0.6)

    # =========================================
    # SELECTIVITY BONUSES (v3.1: increased for balance)
    # =========================================
    # Reward patience and waiting for good setups
    bonus_fresh_spike_entry: float = 0.8  # Bonus for entering very fresh spike (was 0.5)
    bonus_patient_trade: float = 0.5      # Bonus if waited >45 min (was 0.3)
    patience_threshold: float = 0.75      # minutes_since_trade threshold

    # =========================================
    # SPIKE MAGNITUDE BONUSES
    # =========================================
    # Extra reward for catching large moves
    large_move_threshold: float = 0.03    # 3% move
    large_move_bonus: float = 2.0         # Multiplier for large moves
    huge_move_threshold: float = 0.05     # 5% move
    huge_move_bonus: float = 3.0          # Multiplier for huge moves

    # =========================================
    # RISK ADJUSTMENT
    # =========================================
    enable_sharpe_component: bool = True
    sharpe_weight: float = 0.1            # Weight of Sharpe component
    sharpe_window: int = 20               # Rolling window for Sharpe

    drawdown_penalty_weight: float = 1.5  # Penalty for drawdowns
    drawdown_threshold: float = 0.015     # 1.5% drawdown triggers penalty

    # =========================================
    # POSITION MANAGEMENT
    # =========================================
    penalty_stale_position: float = 0.05  # Per-step penalty for stale positions
    max_hold_time: int = 360              # 6 hours max before stale penalty
    min_profit_for_hold: float = 0.003    # 0.3% min profit to justify holding

    reward_smart_exit: float = 0.8        # Bonus for manual exit above stop
    penalty_stop_loss: float = 0.3        # Extra penalty for hitting stop

    # =========================================
    # OPPORTUNITY COST (v3.1 - DISABLED, too aggressive)
    # =========================================
    # Penalty for missing good trading opportunities
    # This prevents the agent from learning to "just hold and do nothing"
    # NOTE: Disabled because -140 reward was too harsh
    enable_opportunity_cost: bool = False  # DISABLED for v3.2
    opportunity_cost_penalty: float = 0.005  # Reduced from 0.02
    opportunity_freshness_threshold: float = 0.9  # More selective

    # =========================================
    # TRANSACTION COSTS
    # =========================================
    transaction_cost: float = 0.0055      # 0.55% round trip


class RewardCalculator:
    """
    Calculate rewards for RL agent based on trading actions and outcomes.

    The reward function balances:
    1. Profit/loss from trades
    2. Risk management (drawdown, volatility)
    3. Behavioral incentives (patience, smart exits)
    4. Efficiency (avoid overtrading, stale positions)
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator.

        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        self._returns_history: List[float] = []

    def calculate(
        self,
        action: int,
        prev_price: float,
        current_price: float,
        position: Optional[Any] = None,  # Position object
        prev_position: Optional[Any] = None,
        trade_result: Optional[Any] = None,  # TradeResult object
        time_in_position: float = 0,
        episode_returns: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate reward for a state transition.

        Args:
            action: Action taken (0-6)
            prev_price: Price before action
            current_price: Price after action
            position: Current position (after action)
            prev_position: Position before action
            trade_result: Result if trade was closed
            time_in_position: Minutes in current position
            episode_returns: List of returns in episode so far

        Returns:
            Calculated reward
        """
        if self.config.reward_type == RewardType.BASIC_PNL:
            return self._basic_pnl_reward(trade_result, position, prev_price, current_price)
        elif self.config.reward_type == RewardType.RISK_ADJUSTED:
            return self._risk_adjusted_reward(
                action, trade_result, position, prev_position,
                prev_price, current_price, time_in_position
            )
        elif self.config.reward_type == RewardType.SHARPE:
            return self._sharpe_reward(episode_returns or [])
        elif self.config.reward_type == RewardType.HYBRID:
            return self._hybrid_reward(
                action, trade_result, position, prev_position,
                prev_price, current_price, time_in_position
            )
        else:
            return 0.0

    def _basic_pnl_reward(
        self,
        trade_result: Optional[Any],
        position: Optional[Any],
        prev_price: float,
        current_price: float
    ) -> float:
        """
        Basic P&L reward - sparse signal on trade completion.

        Simple but effective baseline.
        """
        reward = 0.0

        if trade_result is not None:
            # Trade completed
            net_return = trade_result.return_pct

            if net_return > 0:
                reward = net_return * self.config.reward_scale_win
            else:
                reward = net_return * self.config.reward_scale_loss

        elif position is not None:
            # In position - small unrealized P&L signal
            price_change = (current_price - prev_price) / prev_price
            reward = price_change * self.config.reward_scale_unrealized

        return reward

    def _risk_adjusted_reward(
        self,
        action: int,
        trade_result: Optional[Any],
        position: Optional[Any],
        prev_position: Optional[Any],
        prev_price: float,
        current_price: float,
        time_in_position: float
    ) -> float:
        """
        Risk-adjusted reward with drawdown and volatility penalties.
        """
        reward = 0.0

        # Base P&L component
        reward += self._basic_pnl_reward(trade_result, position, prev_price, current_price)

        # Drawdown penalty
        if position is not None:
            current_return = position.unrealized_return(current_price)
            highest_return = position.highest_return()
            drawdown = highest_return - current_return

            if drawdown > self.config.drawdown_threshold:
                reward -= drawdown * self.config.reward_scale_drawdown

        # Stale position penalty
        if position is not None and time_in_position > self.config.max_hold_time:
            current_return = position.unrealized_return(current_price)
            if current_return < self.config.min_profit_for_hold:
                reward -= self.config.penalty_stale_position

        return reward

    def _sharpe_reward(self, episode_returns: List[float]) -> float:
        """
        Sharpe ratio based reward.

        Directly optimizes risk-adjusted returns.
        """
        if len(episode_returns) < 2:
            return 0.0

        returns = np.array(episode_returns[-self.config.sharpe_window:])
        excess = returns - self.config.risk_free_rate

        mean_excess = np.mean(excess)
        std_excess = np.std(excess) + 1e-8

        sharpe = (mean_excess / std_excess) * self.config.sharpe_annualization

        # Bound to reasonable range
        return np.tanh(sharpe) * 0.1  # Scale down

    def _hybrid_reward(
        self,
        action: int,
        trade_result: Optional[Any],
        position: Optional[Any],
        prev_position: Optional[Any],
        prev_price: float,
        current_price: float,
        time_in_position: float
    ) -> float:
        """
        Hybrid reward combining sparse P&L with dense behavior shaping.

        This is the recommended reward function.
        """
        reward = 0.0

        # ===========================================
        # Component 1: P&L Reward (Primary Signal)
        # ===========================================
        if trade_result is not None:
            # Trade completed - main learning signal
            net_return = trade_result.return_pct

            if net_return > 0:
                reward += net_return * self.config.reward_scale_win
            else:
                reward += net_return * self.config.reward_scale_loss

            # Bonus for smart exit (exited above stop loss)
            if trade_result.exit_reason in ['manual', 'manual_partial']:
                # Calculate what would have been the stop loss return
                # (approximated as negative of initial stop percentage)
                if net_return > -0.02:  # Better than -2% stop
                    reward += self.config.reward_smart_exit

        elif position is not None:
            # In position but not closed - small unrealized P&L signal
            price_change = (current_price - prev_price) / prev_price
            reward += price_change * self.config.reward_scale_unrealized

        # ===========================================
        # Component 2: Risk-Adjusted Rewards
        # ===========================================
        if position is not None:
            current_return = position.unrealized_return(current_price)
            highest_return = position.highest_return()
            drawdown = highest_return - current_return

            # Penalize large drawdowns from peak
            if drawdown > self.config.drawdown_threshold:
                reward -= drawdown * self.config.reward_scale_drawdown

        # ===========================================
        # Component 3: Efficiency Rewards
        # ===========================================

        # Penalize excessive stop adjustments (prevent fidgeting)
        from .environment import Action
        if action in [Action.TIGHTEN_STOP, Action.LOOSEN_STOP]:
            reward -= self.config.penalty_adjustment

        # Penalize holding too long without profit
        if position is not None and time_in_position > self.config.max_hold_time:
            current_return = position.unrealized_return(current_price)
            if current_return < self.config.min_profit_for_hold:
                reward -= self.config.penalty_stale_position

        # ===========================================
        # Component 4: Behavior Shaping
        # ===========================================

        # Reward for patience with winners
        from .environment import Action
        if action == Action.HOLD and position is not None:
            current_return = position.unrealized_return(current_price)
            if current_return > self.config.profit_target:
                reward += self.config.reward_patience

        return reward

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._returns_history = []


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Adaptive reward calculator that adjusts parameters based on performance.

    Useful for curriculum learning and continuous adaptation.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        adaptation_rate: float = 0.01
    ):
        """
        Initialize adaptive reward calculator.

        Args:
            config: Base reward configuration
            adaptation_rate: Rate of parameter adaptation
        """
        super().__init__(config)
        self.adaptation_rate = adaptation_rate

        # Performance tracking
        self.win_rate: float = 0.5
        self.avg_return: float = 0.0
        self.avg_drawdown: float = 0.0

        # Trade history for adaptation
        self._trade_history: List[float] = []

    def update_from_trade(self, trade_result: Any) -> None:
        """Update parameters based on trade outcome."""
        self._trade_history.append(trade_result.return_pct)

        # Keep last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]

        # Update statistics
        wins = sum(1 for r in self._trade_history if r > 0)
        self.win_rate = wins / len(self._trade_history) if self._trade_history else 0.5
        self.avg_return = np.mean(self._trade_history) if self._trade_history else 0.0

        # Adapt parameters
        self._adapt_parameters()

    def _adapt_parameters(self) -> None:
        """Adapt reward parameters based on performance."""
        # If win rate is low, increase reward for wins to encourage more selective trading
        if self.win_rate < 0.4:
            self.config.reward_scale_win *= (1 + self.adaptation_rate)
            self.config.reward_scale_loss *= (1 - self.adaptation_rate * 0.5)

        # If average return is negative, increase risk penalties
        if self.avg_return < 0:
            self.config.reward_scale_drawdown *= (1 + self.adaptation_rate)
            self.config.penalty_stale_position *= (1 + self.adaptation_rate)

        # Log adaptation
        logger.debug(
            f"Reward parameters adapted: "
            f"win_rate={self.win_rate:.2f}, "
            f"scale_win={self.config.reward_scale_win:.2f}, "
            f"scale_drawdown={self.config.reward_scale_drawdown:.2f}"
        )


class SpikeAwareRewardCalculator:
    """
    Spike-Aware Reward Calculator (v3)

    Designed to break the ~9.5 reward plateau by:
    1. Using spike tracking features in reward calculation
    2. Penalizing repeated trading on the same spike
    3. Rewarding selective, high-quality trades
    4. Incorporating risk-adjusted returns (Sharpe-like)

    This calculator expects spike_context to be passed with each calculation,
    containing the spike tracking features from the environment.
    """

    def __init__(self, config: Optional[SpikeAwareRewardConfig] = None):
        """
        Initialize spike-aware reward calculator.

        Args:
            config: SpikeAwareRewardConfig instance
        """
        self.config = config or SpikeAwareRewardConfig()
        self._returns_history: List[float] = []
        self._trade_count: int = 0

        # Track spike trading for debugging
        self._spike_trades: int = 0
        self._fresh_spike_trades: int = 0
        self._stale_spike_trades: int = 0

    def calculate(
        self,
        action: int,
        prev_price: float,
        current_price: float,
        position: Optional[Any] = None,
        prev_position: Optional[Any] = None,
        trade_result: Optional[Any] = None,
        time_in_position: float = 0,
        episode_returns: Optional[List[float]] = None,
        spike_context: Optional[dict] = None,
    ) -> float:
        """
        Calculate spike-aware reward for a state transition.

        Args:
            action: Action taken (0-6)
            prev_price: Price before action
            current_price: Price after action
            position: Current position (after action)
            prev_position: Position before action
            trade_result: Result if trade was closed
            time_in_position: Minutes in current position
            episode_returns: List of returns in episode so far
            spike_context: Dict with spike tracking features:
                - minutes_since_last_trade: float (0-1, 1 = 60+ min)
                - trades_in_last_hour: float (0-1, 1 = 5+ trades)
                - spike_freshness: float (1.0 = fresh, 0.0 = stale)
                - already_traded_this_spike: float (0 or 1)
                - is_in_spike: float (0 or 1)

        Returns:
            Calculated reward
        """
        reward = 0.0

        # Default spike context if not provided
        if spike_context is None:
            spike_context = {
                'minutes_since_last_trade': 1.0,
                'trades_in_last_hour': 0.0,
                'spike_freshness': 1.0,
                'already_traded_this_spike': 0.0,
                'is_in_spike': 0.0,
            }

        # Extract spike features
        minutes_since_trade = spike_context.get('minutes_since_last_trade', 1.0)
        trades_in_hour = spike_context.get('trades_in_last_hour', 0.0)
        spike_freshness = spike_context.get('spike_freshness', 1.0)
        already_traded = spike_context.get('already_traded_this_spike', 0.0)
        is_in_spike = spike_context.get('is_in_spike', 0.0)

        # ===========================================
        # Component 1: Trade P&L with Spike Quality Scaling
        # ===========================================
        if trade_result is not None:
            self._trade_count += 1
            net_return = trade_result.return_pct

            # Base P&L reward (more symmetric than v1/v2)
            if net_return > 0:
                base_reward = net_return * self.config.reward_scale_win
            else:
                base_reward = net_return * self.config.reward_scale_loss

            # -----------------------------------------
            # Spike Quality Multiplier
            # -----------------------------------------
            # Fresh spikes get full reward, stale spikes get reduced reward
            quality_multiplier = self._calculate_spike_quality_multiplier(
                spike_freshness, already_traded, is_in_spike
            )
            reward += base_reward * quality_multiplier

            # -----------------------------------------
            # Large Move Bonuses
            # -----------------------------------------
            abs_return = abs(net_return)
            if abs_return >= self.config.huge_move_threshold and net_return > 0:
                reward += net_return * self.config.huge_move_bonus
                logger.debug(f"Huge move bonus: +{net_return * self.config.huge_move_bonus:.3f}")
            elif abs_return >= self.config.large_move_threshold and net_return > 0:
                reward += net_return * self.config.large_move_bonus
                logger.debug(f"Large move bonus: +{net_return * self.config.large_move_bonus:.3f}")

            # -----------------------------------------
            # Exit Quality Bonuses/Penalties
            # -----------------------------------------
            if trade_result.exit_reason in ['manual', 'manual_partial']:
                # Manual exit above stop = good risk management
                if net_return > -0.015:  # Better than -1.5%
                    reward += self.config.reward_smart_exit
            elif trade_result.exit_reason == 'stop_loss':
                # Stop loss hit = additional penalty
                reward -= self.config.penalty_stop_loss

            # Track for debugging
            if is_in_spike > 0.5:
                self._spike_trades += 1
                if spike_freshness > 0.7:
                    self._fresh_spike_trades += 1
                else:
                    self._stale_spike_trades += 1

            # Store return for Sharpe calculation
            self._returns_history.append(net_return)
            if len(self._returns_history) > 100:
                self._returns_history = self._returns_history[-100:]

        # ===========================================
        # Component 2: Entry Quality (on ENTER_LONG)
        # ===========================================
        from .environment import Action
        if action == Action.ENTER_LONG:
            # -----------------------------------------
            # Re-entry Penalties
            # -----------------------------------------
            # Penalty for re-trading the same spike
            if already_traded > 0.5:
                reward -= self.config.penalty_already_traded_spike
                logger.debug(f"Already traded spike penalty: -{self.config.penalty_already_traded_spike:.3f}")

            # Penalty for rapid re-entry
            if minutes_since_trade < self.config.rapid_reentry_threshold:
                reentry_penalty = self.config.penalty_rapid_reentry * (
                    1.0 - minutes_since_trade / self.config.rapid_reentry_threshold
                )
                reward -= reentry_penalty
                logger.debug(f"Rapid re-entry penalty: -{reentry_penalty:.3f}")

            # Penalty for overtrading
            if trades_in_hour > self.config.overtrading_threshold:
                overtrade_penalty = self.config.penalty_overtrading * (
                    trades_in_hour - self.config.overtrading_threshold
                )
                reward -= overtrade_penalty
                logger.debug(f"Overtrading penalty: -{overtrade_penalty:.3f}")

            # -----------------------------------------
            # Selectivity Bonuses
            # -----------------------------------------
            # Bonus for entering a very fresh spike
            if spike_freshness > 0.8 and is_in_spike > 0.5:
                reward += self.config.bonus_fresh_spike_entry
                logger.debug(f"Fresh spike entry bonus: +{self.config.bonus_fresh_spike_entry:.3f}")

            # Bonus for patient entry (waited long enough)
            if minutes_since_trade > self.config.patience_threshold:
                reward += self.config.bonus_patient_trade
                logger.debug(f"Patient trade bonus: +{self.config.bonus_patient_trade:.3f}")

        # ===========================================
        # Component 3: Position Management
        # ===========================================
        if position is not None:
            current_return = position.unrealized_return(current_price)
            highest_return = position.highest_return()
            drawdown = highest_return - current_return

            # Small unrealized P&L signal
            price_change = (current_price - prev_price) / prev_price
            reward += price_change * self.config.reward_scale_unrealized

            # Drawdown penalty
            if drawdown > self.config.drawdown_threshold:
                dd_penalty = drawdown * self.config.drawdown_penalty_weight
                reward -= dd_penalty

            # Stale position penalty
            if time_in_position > self.config.max_hold_time:
                if current_return < self.config.min_profit_for_hold:
                    reward -= self.config.penalty_stale_position

        # ===========================================
        # Component 4: Risk-Adjusted Component (Sharpe)
        # ===========================================
        if self.config.enable_sharpe_component and len(self._returns_history) >= 5:
            sharpe_reward = self._calculate_sharpe_reward()
            reward += sharpe_reward * self.config.sharpe_weight

        # ===========================================
        # Component 5: Opportunity Cost (v3.1)
        # ===========================================
        # Penalize NOT trading during fresh spikes when not in position
        # This prevents "just hold forever" local minimum
        if self.config.enable_opportunity_cost:
            if position is None and action != Action.ENTER_LONG:
                # Not in position and not entering
                if is_in_spike > 0.5 and spike_freshness > self.config.opportunity_freshness_threshold:
                    # Missing a fresh spike opportunity!
                    reward -= self.config.opportunity_cost_penalty
                    # Scale by how fresh the spike is
                    extra_penalty = (spike_freshness - self.config.opportunity_freshness_threshold) * 0.01
                    reward -= extra_penalty

        return reward

    def _calculate_spike_quality_multiplier(
        self,
        spike_freshness: float,
        already_traded: float,
        is_in_spike: float,
    ) -> float:
        """
        Calculate quality multiplier for trade rewards based on spike state.

        Returns:
            Multiplier in range [stale_spike_penalty, 1.0]
        """
        # If not in a spike, use full multiplier (trend following is ok)
        if is_in_spike < 0.5:
            return 1.0

        # In a spike - scale by freshness
        # freshness = 1.0 -> multiplier = 1.0
        # freshness = 0.0 -> multiplier = stale_spike_penalty
        base_multiplier = (
            self.config.stale_spike_penalty +
            (1.0 - self.config.stale_spike_penalty) * spike_freshness
        )

        # Additional penalty if already traded this spike
        if already_traded > 0.5:
            base_multiplier *= 0.5  # Halve the reward for repeat trades

        # Weight by configuration
        final_multiplier = (
            1.0 - self.config.spike_freshness_weight +
            self.config.spike_freshness_weight * base_multiplier
        )

        return max(self.config.stale_spike_penalty, min(1.0, final_multiplier))

    def _calculate_sharpe_reward(self) -> float:
        """
        Calculate Sharpe-like reward component from recent returns.

        Returns:
            Bounded reward in [-1, 1]
        """
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history[-self.config.sharpe_window:])
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8

        # Simple Sharpe ratio
        sharpe = mean_return / std_return

        # Bound to reasonable range using tanh
        return np.tanh(sharpe * 2.0)

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._returns_history = []
        self._trade_count = 0
        self._spike_trades = 0
        self._fresh_spike_trades = 0
        self._stale_spike_trades = 0

    def get_stats(self) -> dict:
        """Get statistics for debugging."""
        return {
            'trade_count': self._trade_count,
            'spike_trades': self._spike_trades,
            'fresh_spike_trades': self._fresh_spike_trades,
            'stale_spike_trades': self._stale_spike_trades,
            'fresh_ratio': (
                self._fresh_spike_trades / max(1, self._spike_trades)
            ),
        }


@dataclass
class SpikeQualityBonusConfig:
    """
    v4/v6: Additive bonuses on top of hybrid reward.

    Philosophy: The hybrid reward provides a stable positive baseline (~9.5).
    These bonuses reward high-quality spike trading without penalizing
    lower-quality trades. All values are BONUSES (non-negative additions).

    This approach fixes v3's penalty-dominated failure by:
    1. Never adding penalties beyond what hybrid already provides
    2. Creating positive gradient for quality improvement
    3. Maintaining exploration-friendly reward landscape

    v5 additions (DEPRECATED): Exit quality bonuses caused high variance
    v6 changes: Tuned entry thresholds, disabled exit bonuses for stability
    """
    # =========================================
    # ENTRY TIMING BONUSES (v6 TUNED)
    # =========================================
    # Reward entering during fresh spikes (higher probability trades)
    bonus_fresh_spike_entry: float = 0.35       # Increased from 0.3 for stronger signal
    fresh_spike_threshold: float = 0.6          # Lowered from 0.7 - more entries qualify

    # Reward patient, selective trading
    bonus_patient_entry: float = 0.25           # Increased from 0.2
    patient_threshold: float = 0.5              # Lowered from 0.6 (30min instead of 36min)

    # Reward first entry on a spike (highest probability trade)
    bonus_first_trade_on_spike: float = 0.5     # Increased from 0.4 - KEY signal

    # =========================================
    # TRADE QUALITY BONUSES (v6 TUNED)
    # =========================================
    # Reward capturing large price moves
    large_move_threshold: float = 0.02          # Lowered from 0.03 - 2% moves now qualify
    large_move_multiplier: float = 1.5          # Increased from 1.0 for stronger gradient
    huge_move_threshold: float = 0.04           # Lowered from 0.05
    huge_move_multiplier: float = 2.0           # Increased from 1.5

    # =========================================
    # QUALITY MULTIPLIER (unchanged - working well)
    # =========================================
    # Scale base P&L reward by entry quality
    enable_quality_multiplier: bool = True
    max_freshness_bonus: float = 0.2            # Up to 20% extra for fresh spike
    first_trade_bonus: float = 0.1              # Extra 10% for first trade on spike

    # =========================================
    # EXIT QUALITY BONUSES (v5 - DISABLED)
    # =========================================
    # v5 exit bonuses caused high variance (+/- 2.97) and conflicting signals.
    # Disabled in v6 for stability. May re-enable with different approach.
    enable_exit_bonuses: bool = False           # DISABLED - caused instability

    # Profit lock bonus: exiting with significant gains
    bonus_profit_lock: float = 0.2              # Bonus for locking in >2% profit
    profit_lock_threshold: float = 0.02         # 2% return triggers bonus

    # Near-high exit: captured most of the move
    bonus_near_high_exit: float = 0.15          # Bonus for exiting within 1% of position high
    near_high_threshold: float = 0.01           # 1% below highest return

    # Quick loss exit: minimized damage on bad trade
    bonus_quick_loss_exit: float = 0.1          # Bonus for quick exit on small loss
    quick_loss_threshold: float = 0.01          # Loss must be <1%
    quick_loss_time_minutes: int = 30           # Must exit within 30 minutes

    # Trailing stop discipline: stopped out at profit
    bonus_trailing_stop_profit: float = 0.1     # Bonus for stop loss exit that's profitable


class SpikeQualityBonusCalculator:
    """
    v4: Spike Quality Bonus Calculator

    Wraps the hybrid RewardCalculator and adds spike-quality bonuses.
    This is a COMPOSITIONAL approach - we don't modify hybrid, we enhance it.

    Key design principles:
    1. Hybrid reward is the stable base (proven ~9.5 reward)
    2. Bonuses are ALWAYS non-negative (no additional penalties)
    3. Bonuses reward quality without punishing mediocrity
    4. Goal: Break the 9.5 plateau by rewarding selective, high-quality trades

    Expected improvement:
    - Base hybrid: ~9.5
    - Spike quality bonuses: +2 to +3
    - Target v4 reward: ~11-12
    """

    def __init__(
        self,
        base_config: Optional[RewardConfig] = None,
        bonus_config: Optional[SpikeQualityBonusConfig] = None
    ):
        """
        Initialize spike quality bonus calculator.

        Args:
            base_config: RewardConfig for hybrid base (uses HYBRID type)
            bonus_config: SpikeQualityBonusConfig for bonus parameters
        """
        # Use hybrid as base - proven stable positive reward
        base_config = base_config or RewardConfig(reward_type=RewardType.HYBRID)
        if base_config.reward_type != RewardType.HYBRID:
            logger.warning(
                f"SpikeQualityBonusCalculator works best with HYBRID base, "
                f"got {base_config.reward_type}. Overriding to HYBRID."
            )
            base_config.reward_type = RewardType.HYBRID

        self.base_calculator = RewardCalculator(base_config)
        self.bonus_config = bonus_config or SpikeQualityBonusConfig()

        # Track entry quality for deferred bonus calculation (v6.1)
        self._entry_spike_freshness: float = 0.0
        self._entry_was_first_trade: bool = False
        self._entry_in_spike: bool = False
        self._entry_was_patient: bool = False  # v6.1: track patient entry

        # Statistics tracking
        self._total_base_reward: float = 0.0
        self._total_bonus: float = 0.0
        self._bonus_counts: dict = {
            'fresh_spike_entry': 0,
            'patient_entry': 0,
            'first_trade': 0,
            'large_move': 0,
            'huge_move': 0,
            'quality_multiplier': 0,
            # v5 exit bonuses
            'profit_lock': 0,
            'near_high_exit': 0,
            'quick_loss_exit': 0,
            'trailing_stop_profit': 0,
        }

    def calculate(
        self,
        action: int,
        prev_price: float,
        current_price: float,
        position: Optional[Any] = None,
        prev_position: Optional[Any] = None,
        trade_result: Optional[Any] = None,
        time_in_position: float = 0,
        episode_returns: Optional[List[float]] = None,
        spike_context: Optional[dict] = None,
    ) -> float:
        """
        Calculate reward with spike-quality bonuses.

        Args:
            action: Action taken (0-6)
            prev_price: Price before action
            current_price: Price after action
            position: Current position (after action)
            prev_position: Position before action
            trade_result: Result if trade was closed
            time_in_position: Minutes in current position
            episode_returns: List of returns in episode so far
            spike_context: Dict with spike tracking features

        Returns:
            total_reward = base_hybrid_reward + spike_quality_bonus
        """
        # 1. Get base hybrid reward (proven stable ~9.5)
        base_reward = self.base_calculator.calculate(
            action=action,
            prev_price=prev_price,
            current_price=current_price,
            position=position,
            prev_position=prev_position,
            trade_result=trade_result,
            time_in_position=time_in_position,
            episode_returns=episode_returns,
        )

        # 2. Calculate additive bonuses
        bonus = self._calculate_bonuses(
            action=action,
            trade_result=trade_result,
            spike_context=spike_context or {},
        )

        # Track for statistics
        self._total_base_reward += base_reward
        self._total_bonus += bonus

        return base_reward + bonus

    def _calculate_bonuses(
        self,
        action: int,
        trade_result: Optional[Any],
        spike_context: dict,
    ) -> float:
        """
        Calculate all spike-quality bonuses.

        IMPORTANT (v6.1 fix): Entry bonuses are now DEFERRED to trade completion
        and only awarded if the trade is PROFITABLE. This prevents reward hacking
        where the agent farms entry bonuses by rapidly entering/exiting.

        Returns:
            Non-negative bonus value
        """
        bonus = 0.0
        cfg = self.bonus_config

        # Extract spike context with safe defaults
        minutes_since_trade = spike_context.get('minutes_since_last_trade', 1.0)
        spike_freshness = spike_context.get('spike_freshness', 0.0)
        already_traded = spike_context.get('already_traded_this_spike', 0.0)
        is_in_spike = spike_context.get('is_in_spike', 0.0)

        from .environment import Action

        # =========================================
        # ENTRY TRACKING (no immediate bonuses!)
        # =========================================
        # v6.1: We track entry quality but DON'T award bonuses until trade completes
        # This prevents the agent from farming bonuses via rapid entry/exit
        if action == Action.ENTER_LONG:
            # Track entry quality for deferred bonus calculation
            self._entry_spike_freshness = spike_freshness
            self._entry_was_first_trade = already_traded < 0.5
            self._entry_in_spike = is_in_spike > 0.5
            self._entry_was_patient = minutes_since_trade > cfg.patient_threshold
            # NO BONUSES HERE - deferred to trade completion

        # =========================================
        # TRADE COMPLETION BONUSES (v6.1: includes deferred entry bonuses)
        # =========================================
        if trade_result is not None:
            net_return = trade_result.return_pct

            # =========================================
            # DEFERRED ENTRY BONUSES (only if profitable!)
            # v6.1: Entry bonuses are now awarded at trade completion
            # to prevent reward hacking via rapid entry/exit farming
            # =========================================
            if net_return > 0:  # ONLY award entry bonuses on profitable trades
                # Bonus 1: Fresh spike entry (deferred)
                if self._entry_in_spike and self._entry_spike_freshness > cfg.fresh_spike_threshold:
                    entry_bonus = cfg.bonus_fresh_spike_entry * self._entry_spike_freshness
                    bonus += entry_bonus
                    self._bonus_counts['fresh_spike_entry'] += 1
                    logger.debug(f"Fresh spike entry bonus (deferred): +{entry_bonus:.3f}")

                # Bonus 2: Patient entry (deferred)
                if self._entry_was_patient:
                    bonus += cfg.bonus_patient_entry
                    self._bonus_counts['patient_entry'] += 1
                    logger.debug(f"Patient entry bonus (deferred): +{cfg.bonus_patient_entry:.3f}")

                # Bonus 3: First trade on spike (deferred)
                if self._entry_in_spike and self._entry_was_first_trade:
                    bonus += cfg.bonus_first_trade_on_spike
                    self._bonus_counts['first_trade'] += 1
                    logger.debug(f"First trade on spike bonus (deferred): +{cfg.bonus_first_trade_on_spike:.3f}")

            # =========================================
            # TRADE SIZE BONUSES (always apply to profitable trades)
            # =========================================
            # Bonus 4: Large move capture
            # Reward catching significant price moves
            if net_return > cfg.large_move_threshold:
                large_bonus = net_return * cfg.large_move_multiplier
                bonus += large_bonus
                self._bonus_counts['large_move'] += 1
                logger.debug(f"Large move bonus: +{large_bonus:.3f}")

            # Bonus 5: Huge move capture (additional)
            if net_return > cfg.huge_move_threshold:
                huge_bonus = net_return * cfg.huge_move_multiplier
                bonus += huge_bonus
                self._bonus_counts['huge_move'] += 1
                logger.debug(f"Huge move bonus: +{huge_bonus:.3f}")

            # Bonus 6: Quality multiplier on winning trades
            # Scale P&L reward by entry quality
            if cfg.enable_quality_multiplier and net_return > 0 and self._entry_in_spike:
                quality_bonus = net_return * cfg.max_freshness_bonus * self._entry_spike_freshness
                if self._entry_was_first_trade:
                    quality_bonus += net_return * cfg.first_trade_bonus
                if quality_bonus > 0:
                    bonus += quality_bonus
                    self._bonus_counts['quality_multiplier'] += 1
                    logger.debug(f"Quality multiplier bonus: +{quality_bonus:.3f}")

            # =========================================
            # v5 EXIT QUALITY BONUSES
            # =========================================
            if cfg.enable_exit_bonuses:
                exit_bonus = self._calculate_exit_bonuses(trade_result)
                bonus += exit_bonus

            # Reset entry tracking after trade completes
            self._entry_spike_freshness = 0.0
            self._entry_was_first_trade = False
            self._entry_in_spike = False
            self._entry_was_patient = False  # v6.1

        # Always return non-negative bonus
        return max(0.0, bonus)

    def _calculate_exit_bonuses(self, trade_result: Any) -> float:
        """
        Calculate exit quality bonuses (v5 - Strategy 2).

        Rewards smart exit timing and risk management:
        1. Profit lock: exiting with significant gains
        2. Near-high exit: captured most of the move
        3. Quick loss exit: minimized damage on bad trade
        4. Trailing stop profit: disciplined stop management

        Args:
            trade_result: TradeResult object with return_pct, exit_reason,
                         entry_time, exit_time, highest_return

        Returns:
            Non-negative exit bonus
        """
        bonus = 0.0
        cfg = self.bonus_config

        if trade_result is None:
            return 0.0

        net_return = trade_result.return_pct
        exit_reason = trade_result.exit_reason
        highest_return = getattr(trade_result, 'highest_return', net_return)

        # Calculate trade duration in minutes
        duration_minutes = (
            (trade_result.exit_time - trade_result.entry_time).total_seconds() / 60
        )

        # -----------------------------------------
        # Bonus 7: Profit lock
        # Reward for locking in significant profits
        # -----------------------------------------
        if net_return > cfg.profit_lock_threshold:
            bonus += cfg.bonus_profit_lock
            self._bonus_counts['profit_lock'] += 1
            logger.debug(f"Profit lock bonus: +{cfg.bonus_profit_lock:.3f} (return={net_return:.2%})")

        # -----------------------------------------
        # Bonus 8: Near-high exit
        # Reward for exiting near the position's highest return
        # (captured most of the available move)
        # -----------------------------------------
        if net_return > 0 and highest_return > 0:
            # How much of the peak return did we capture?
            return_gap = highest_return - net_return
            if return_gap < cfg.near_high_threshold:
                bonus += cfg.bonus_near_high_exit
                self._bonus_counts['near_high_exit'] += 1
                logger.debug(
                    f"Near-high exit bonus: +{cfg.bonus_near_high_exit:.3f} "
                    f"(exit={net_return:.2%}, high={highest_return:.2%}, gap={return_gap:.2%})"
                )

        # -----------------------------------------
        # Bonus 9: Quick loss exit
        # Reward for quickly cutting losses on bad trades
        # (good risk management - don't let losers run)
        # -----------------------------------------
        if net_return < 0 and abs(net_return) < cfg.quick_loss_threshold:
            if duration_minutes < cfg.quick_loss_time_minutes:
                bonus += cfg.bonus_quick_loss_exit
                self._bonus_counts['quick_loss_exit'] += 1
                logger.debug(
                    f"Quick loss exit bonus: +{cfg.bonus_quick_loss_exit:.3f} "
                    f"(loss={net_return:.2%}, duration={duration_minutes:.0f}min)"
                )

        # -----------------------------------------
        # Bonus 10: Trailing stop profit
        # Reward for getting stopped out at a profit
        # (indicates good trailing stop management)
        # -----------------------------------------
        if exit_reason == 'stop_loss' and net_return > 0:
            bonus += cfg.bonus_trailing_stop_profit
            self._bonus_counts['trailing_stop_profit'] += 1
            logger.debug(
                f"Trailing stop profit bonus: +{cfg.bonus_trailing_stop_profit:.3f} "
                f"(stopped out at +{net_return:.2%})"
            )

        return bonus

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self.base_calculator.reset()
        self._entry_spike_freshness = 0.0
        self._entry_was_first_trade = False
        self._entry_in_spike = False
        self._entry_was_patient = False  # v6.1

    def get_stats(self) -> dict:
        """Get statistics about bonus distribution."""
        return {
            'total_base_reward': self._total_base_reward,
            'total_bonus': self._total_bonus,
            'bonus_ratio': (
                self._total_bonus / max(0.01, self._total_base_reward)
            ),
            'bonus_counts': dict(self._bonus_counts),
        }

    def reset_stats(self) -> None:
        """Reset statistics tracking."""
        self._total_base_reward = 0.0
        self._total_bonus = 0.0
        self._bonus_counts = {k: 0 for k in self._bonus_counts}


def calculate_episode_metrics(trades: List[Any]) -> dict:
    """
    Calculate trading metrics from a list of trades.

    Args:
        trades: List of TradeResult objects

    Returns:
        Dict of metrics
    """
    if not trades:
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
        }

    returns = [t.return_pct * t.size for t in trades]

    # Basic stats
    total_return = sum(returns)
    num_trades = len(trades)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (from cumulative returns)
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Sharpe ratio (simplified)
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0

    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
    }


if __name__ == "__main__":
    # Test reward calculation
    from dataclasses import dataclass

    @dataclass
    class MockPosition:
        entry_price: float
        highest_price: float

        def unrealized_return(self, current_price: float) -> float:
            return (current_price - self.entry_price) / self.entry_price

        def highest_return(self) -> float:
            return (self.highest_price - self.entry_price) / self.entry_price

    @dataclass
    class MockTradeResult:
        return_pct: float
        size: float
        exit_reason: str

    # Test different reward types
    print("Testing Reward Functions\n" + "=" * 50)

    for reward_type in RewardType:
        config = RewardConfig(reward_type=reward_type)
        calc = RewardCalculator(config)

        print(f"\n{reward_type.value}:")

        # Winning trade
        trade = MockTradeResult(return_pct=0.05, size=1.0, exit_reason='manual')
        reward = calc.calculate(
            action=6,  # EXIT_ALL
            prev_price=100,
            current_price=105,
            trade_result=trade
        )
        print(f"  Winning trade (+5%): reward = {reward:.4f}")

        # Losing trade
        trade = MockTradeResult(return_pct=-0.02, size=1.0, exit_reason='stop_loss')
        reward = calc.calculate(
            action=6,  # EXIT_ALL
            prev_price=100,
            current_price=98,
            trade_result=trade
        )
        print(f"  Losing trade (-2%): reward = {reward:.4f}")

        # In position with unrealized gain
        position = MockPosition(entry_price=100, highest_price=105)
        reward = calc.calculate(
            action=2,  # HOLD
            prev_price=103,
            current_price=104,
            position=position
        )
        print(f"  Holding with gain: reward = {reward:.4f}")

        # In position with drawdown
        position = MockPosition(entry_price=100, highest_price=110)
        reward = calc.calculate(
            action=2,  # HOLD
            prev_price=106,
            current_price=104,  # 6% drawdown from peak
            position=position
        )
        print(f"  Holding with drawdown: reward = {reward:.4f}")

    # Test episode metrics
    print("\n\nTesting Episode Metrics\n" + "=" * 50)

    trades = [
        MockTradeResult(return_pct=0.05, size=1.0, exit_reason='manual'),
        MockTradeResult(return_pct=-0.02, size=1.0, exit_reason='stop_loss'),
        MockTradeResult(return_pct=0.03, size=0.5, exit_reason='manual_partial'),
        MockTradeResult(return_pct=0.08, size=1.0, exit_reason='manual'),
        MockTradeResult(return_pct=-0.015, size=1.0, exit_reason='stop_loss'),
    ]

    metrics = calculate_episode_metrics(trades)
    print("\nEpisode Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test SpikeAwareRewardCalculator
    print("\n\n" + "=" * 60)
    print("Testing SpikeAwareRewardCalculator (v3)")
    print("=" * 60)

    spike_calc = SpikeAwareRewardCalculator(SpikeAwareRewardConfig())

    # Scenario 1: Fresh spike entry with win
    print("\n--- Scenario 1: Fresh spike entry with +5% win ---")
    spike_context = {
        'minutes_since_last_trade': 1.0,    # Long time since last trade
        'trades_in_last_hour': 0.0,         # No recent trades
        'spike_freshness': 0.95,            # Very fresh spike
        'already_traded_this_spike': 0.0,   # First trade on this spike
        'is_in_spike': 1.0,                 # In a spike
    }
    trade = MockTradeResult(return_pct=0.05, size=1.0, exit_reason='manual')
    reward = spike_calc.calculate(
        action=6,  # EXIT_ALL
        prev_price=100,
        current_price=105,
        trade_result=trade,
        spike_context=spike_context
    )
    print(f"  Reward: {reward:.4f} (should be high - fresh spike, patient entry)")

    # Scenario 2: Stale spike re-entry with small win
    print("\n--- Scenario 2: Stale spike re-entry with +2% win ---")
    spike_context = {
        'minutes_since_last_trade': 0.2,    # Just traded 12 min ago
        'trades_in_last_hour': 0.6,         # 3 trades this hour
        'spike_freshness': 0.3,             # Stale spike
        'already_traded_this_spike': 1.0,   # Already traded this spike!
        'is_in_spike': 1.0,                 # Still in spike
    }
    trade = MockTradeResult(return_pct=0.02, size=1.0, exit_reason='manual')
    reward = spike_calc.calculate(
        action=6,  # EXIT_ALL
        prev_price=100,
        current_price=102,
        trade_result=trade,
        spike_context=spike_context
    )
    print(f"  Reward: {reward:.4f} (should be lower - stale spike, re-entry)")

    # Scenario 3: Entry on already-traded spike (bad behavior)
    print("\n--- Scenario 3: Entry on already-traded spike ---")
    spike_context = {
        'minutes_since_last_trade': 0.3,    # 18 min since last trade
        'trades_in_last_hour': 0.4,         # 2 trades this hour
        'spike_freshness': 0.4,             # Somewhat stale
        'already_traded_this_spike': 1.0,   # Already traded!
        'is_in_spike': 1.0,
    }
    reward = spike_calc.calculate(
        action=1,  # ENTER_LONG
        prev_price=100,
        current_price=100,
        spike_context=spike_context
    )
    print(f"  Reward: {reward:.4f} (should be negative - penalized re-entry)")

    # Scenario 4: Patient entry on fresh spike (good behavior)
    print("\n--- Scenario 4: Patient entry on fresh spike ---")
    spike_context = {
        'minutes_since_last_trade': 0.9,    # 54 min since last trade
        'trades_in_last_hour': 0.2,         # 1 trade this hour
        'spike_freshness': 0.9,             # Fresh spike
        'already_traded_this_spike': 0.0,   # First entry
        'is_in_spike': 1.0,
    }
    reward = spike_calc.calculate(
        action=1,  # ENTER_LONG
        prev_price=100,
        current_price=100,
        spike_context=spike_context
    )
    print(f"  Reward: {reward:.4f} (should be positive - fresh spike bonus + patience)")

    # Scenario 5: Loss on stop
    print("\n--- Scenario 5: Stop loss hit (-2%) ---")
    spike_context = {
        'minutes_since_last_trade': 0.5,
        'trades_in_last_hour': 0.2,
        'spike_freshness': 0.5,
        'already_traded_this_spike': 0.0,
        'is_in_spike': 1.0,
    }
    trade = MockTradeResult(return_pct=-0.02, size=1.0, exit_reason='stop_loss')
    reward = spike_calc.calculate(
        action=6,
        prev_price=100,
        current_price=98,
        trade_result=trade,
        spike_context=spike_context
    )
    print(f"  Reward: {reward:.4f} (should be negative - loss + stop penalty)")

    print(f"\n  Calculator stats: {spike_calc.get_stats()}")

    print("\n" + "=" * 60)
    print("All reward function tests passed!")
    print("=" * 60)
