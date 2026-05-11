"""
Entry Timing Reward Calculator

Reward function designed specifically for entry-only timing:
- SPARSE rewards: Only reward when trade completes via rule-based exit
- Focus on entry quality: Did entering at this time lead to profit?
- Spike capture bonuses: Extra reward for catching large moves
- Patience rewards: Bonus for waiting for good setups

Key insight: With rule-based exits, the agent can only control WHEN to enter.
A good entry is one that leads to a take-profit exit.
A bad entry is one that leads to a stop-loss exit.

The reward function should be simple and clear:
- Winning trade (TP hit) = positive reward
- Losing trade (SL hit) = negative reward
- Bonuses for capturing large moves
- Small penalty for missing obvious opportunities
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Any
from loguru import logger


@dataclass
class EntryTimingRewardConfig:
    """Configuration for entry timing reward calculation."""

    # =========================================
    # TRADE OUTCOME REWARDS (Primary Signal)
    # =========================================
    # These are the main learning signals - SPARSE, only on trade completion

    # Take profit hit - the ideal outcome (increased for selectivity)
    reward_take_profit: float = 14.520000000000003      # Strong positive signal
    reward_take_profit_bonus_per_pct: float = 0.5  # Extra per 1% above min TP

    # Stop loss hit - the bad outcome (increased for selectivity)
    penalty_stop_loss: float = -6.050000000000001       # Harsher penalty

    # Max hold time exit - neutral-ish (depends on P&L at exit)
    reward_max_hold_positive: float = 1.0  # If profitable at max hold
    penalty_max_hold_negative: float = -1.0  # If unprofitable at max hold (increased)

    # End of episode exit - neutral
    reward_end_episode_positive: float = 0.5
    penalty_end_episode_negative: float = -0.5  # Increased penalty

    # =========================================
    # QUICK WIN #1: ENTRY COST (tuned for selectivity)
    # =========================================
    # Immediate penalty for entering - discourages excessive entries
    entry_cost: float = -1.44              # Strong cost per entry action
    entry_cost_bad_setup: float = -4.2250000000000005    # Heavy extra cost for bad setup (3x asymmetry)
    bad_setup_threshold: float = 0.25     # Setup quality below this = bad

    # =========================================
    # QUICK WIN #2: WAIT REWARD (tuned for selectivity)
    # =========================================
    # Reward for correctly waiting during bad conditions
    wait_reward_bad_conditions: float = 0.14520000000000002   # Meaningful reward for patience
    wait_penalty_missed_spike: float = -0.02   # Small penalty for missing (avoid over-entering)

    # =========================================
    # SPIKE CAPTURE BONUSES
    # =========================================
    # Bonus for catching large moves (encourages timing entries before spikes)

    # Entry quality bonus: entering on fresh spike that leads to profit
    bonus_fresh_spike_profitable: float = 1.5  # Entered fresh spike, made money
    fresh_spike_threshold: float = 0.7         # spike_freshness > this to qualify

    # Large move capture bonus
    large_move_threshold: float = 0.05    # 5% move
    large_move_bonus: float = 2.0         # Extra reward for catching 5%+ moves
    huge_move_threshold: float = 0.10     # 10% move
    huge_move_bonus: float = 4.0          # Extra reward for catching 10%+ moves

    # =========================================
    # TRADE QUALITY METRICS
    # =========================================
    # Reward good risk/reward and quick wins

    # Quick win bonus: TP hit in < N minutes
    quick_win_threshold_minutes: int = 120  # 2 hours
    quick_win_bonus: float = 0.5

    # High watermark capture: exited near the high of the position
    # (highest_return - exit_return < threshold)
    high_capture_threshold: float = 0.02  # Within 2% of highest
    high_capture_bonus: float = 0.3

    # =========================================
    # OPPORTUNITY COST (Optional)
    # =========================================
    # Small penalty for NOT entering during obvious opportunities
    # This prevents the agent from learning to never trade

    enable_opportunity_penalty: bool = True
    opportunity_penalty: float = -0.01    # Very small per-step penalty
    opportunity_threshold: float = 0.8    # Only penalize for very fresh spikes

    # =========================================
    # BEHAVIOR SHAPING (very small signals)
    # =========================================
    # These provide gradient but shouldn't dominate

    # Patience bonus: waited enough time between trades
    patience_bonus: float = 0.1
    min_wait_minutes: int = 60  # Bonus for waiting 60+ min

    # Trade frequency penalty: too many trades
    overtrade_penalty: float = -0.2       # Increased penalty
    max_trades_before_penalty: int = 5    # Reduced from 10

    # =========================================
    # QUICK WIN #3: ENTRY BUDGET (tuned for selectivity)
    # =========================================
    # Hard limit on entries per episode to force selectivity
    entry_budget: int = 1                 # Max 3 entries per episode (very selective)
    entry_budget_exhausted_penalty: float = -2.0  # Strong penalty for exceeding budget

    # =========================================
    # SETUP QUALITY THRESHOLDS
    # =========================================
    # Used to determine if current conditions are good for entry
    good_setup_vol_zscore: float = 1.0    # Volatility z-score for good setup
    good_setup_volume_zscore: float = 1.5  # Volume z-score for good setup
    good_setup_rsi_extreme: float = 30.0  # RSI distance from 50 for good setup


class EntryTimingRewardCalculator:
    """
    Reward calculator for entry-only timing environment.

    Philosophy:
    - SPARSE rewards: Only give meaningful rewards when trades complete
    - CLEAR signal: TP = good, SL = bad
    - Bonuses for high-quality entries (spike capture, quick wins)
    - Minimal shaping to avoid reward hacking

    This is simpler than the full trading reward because:
    - Agent only controls entry timing
    - Exits are deterministic (rule-based)
    - Success/failure is clearly defined
    """

    def __init__(self, config: Optional[EntryTimingRewardConfig] = None):
        """Initialize reward calculator."""
        self.config = config or EntryTimingRewardConfig()

        # Statistics tracking
        self._tp_count: int = 0
        self._sl_count: int = 0
        self._max_hold_count: int = 0
        self._total_reward: float = 0.0
        self._spike_captures: int = 0

        # Entry budget tracking
        self._entries_this_episode: int = 0
        self._waits_rewarded: int = 0
        self._entry_costs_applied: float = 0.0

    def calculate(
        self,
        action: int,
        trade_result: Optional[Any] = None,  # EntryTradeResult
        position: Optional[Any] = None,       # EntryPosition
        spike_context: Optional[dict] = None,
        episode_trades: Optional[List[Any]] = None,
        state_features: Optional[dict] = None,  # NEW: For setup quality estimation
    ) -> float:
        """
        Calculate reward for entry timing.

        Args:
            action: Action taken (0=WAIT, 1=ENTER)
            trade_result: Result if trade completed this step
            position: Current position (None if flat)
            spike_context: Dict with spike features
            episode_trades: List of completed trades
            state_features: Dict with current state features for setup quality

        Returns:
            Calculated reward
        """
        reward = 0.0
        cfg = self.config

        # Default spike context
        if spike_context is None:
            spike_context = {
                'is_in_spike': False,
                'spike_freshness': 0.0,
                'spike_traded': False,
            }

        # Default state features
        if state_features is None:
            state_features = {}

        episode_trades = episode_trades or []

        # Estimate current setup quality
        setup_quality = self._estimate_setup_quality(state_features, spike_context)

        # ===========================================
        # QUICK WIN #1: ENTRY COST
        # ===========================================
        if action == 1 and position is None:  # Just entered
            self._entries_this_episode += 1

            # Base entry cost
            reward += cfg.entry_cost
            self._entry_costs_applied += abs(cfg.entry_cost)

            # Extra penalty for bad setup
            if setup_quality < cfg.bad_setup_threshold:
                reward += cfg.entry_cost_bad_setup
                self._entry_costs_applied += abs(cfg.entry_cost_bad_setup)
                logger.debug(f"Bad setup entry penalty: {cfg.entry_cost_bad_setup}")

        # ===========================================
        # QUICK WIN #2: WAIT REWARD
        # ===========================================
        if action == 0 and position is None:  # Waiting while flat
            spike_freshness = spike_context.get('spike_freshness', 0.0)

            if setup_quality < 0.3:
                # Good job waiting during poor conditions
                reward += cfg.wait_reward_bad_conditions
                self._waits_rewarded += 1
            elif spike_freshness > cfg.opportunity_threshold:
                if not spike_context.get('spike_traded', False):
                    # Missed obvious opportunity
                    reward += cfg.wait_penalty_missed_spike

        # ===========================================
        # QUICK WIN #3: ENTRY BUDGET CHECK
        # ===========================================
        if action == 1 and position is None:
            if self._entries_this_episode > cfg.entry_budget:
                # Exceeded budget - strong penalty
                reward += cfg.entry_budget_exhausted_penalty
                logger.debug(f"Entry budget exceeded: {cfg.entry_budget_exhausted_penalty}")

        # ===========================================
        # TRADE COMPLETION REWARD (Primary Signal)
        # ===========================================
        if trade_result is not None:
            reward += self._calculate_trade_reward(trade_result, spike_context)

        # ===========================================
        # OVERTRADE PENALTY (in addition to budget)
        # ===========================================
        if len(episode_trades) > cfg.max_trades_before_penalty:
            reward += cfg.overtrade_penalty

        self._total_reward += reward
        return reward

    def _estimate_setup_quality(
        self,
        state_features: dict,
        spike_context: dict,
    ) -> float:
        """
        Estimate setup quality from state features (0 = terrible, 1 = excellent).

        This encodes domain knowledge about what makes good entry conditions.
        """
        cfg = self.config
        score = 0.0

        # Volatility conditions (prefer elevated volatility)
        atr_zscore = state_features.get('atr_zscore', 0)
        if atr_zscore > cfg.good_setup_vol_zscore:
            score += 0.25
        elif atr_zscore > 0.5:
            score += 0.15
        elif atr_zscore < -0.5:
            score -= 0.1

        # Volume surge (prefer high volume)
        vol_zscore = state_features.get('volume_zscore', 0)
        if vol_zscore > cfg.good_setup_volume_zscore:
            score += 0.25
        elif vol_zscore > 0.5:
            score += 0.15

        # RSI extremes (oversold/overbought)
        rsi = state_features.get('rsi', 50)
        rsi_distance = abs(rsi - 50)
        if rsi_distance > cfg.good_setup_rsi_extreme:
            score += 0.2
        elif rsi_distance > 15:
            score += 0.1

        # Spike context
        if spike_context.get('is_in_spike', False):
            score += 0.15
        if spike_context.get('spike_freshness', 0) > 0.5:
            score += 0.15

        return np.clip(score, 0.0, 1.0)

    def _calculate_trade_reward(
        self,
        trade_result: Any,
        spike_context: dict,
    ) -> float:
        """Calculate reward for a completed trade."""
        reward = 0.0
        cfg = self.config

        # Import here to avoid circular dependency
        from .entry_timing_env import ExitReason

        exit_reason = trade_result.exit_reason
        net_return = trade_result.return_pct
        highest_return = trade_result.highest_return
        hold_duration = trade_result.hold_duration_minutes

        # -----------------------------------------
        # Base reward by exit reason
        # -----------------------------------------
        if exit_reason == ExitReason.TAKE_PROFIT:
            self._tp_count += 1
            reward += cfg.reward_take_profit

            # Bonus for returns above minimum TP
            extra_pct = (net_return - 0.12) * 100  # Extra above 12%
            if extra_pct > 0:
                reward += extra_pct * cfg.reward_take_profit_bonus_per_pct

        elif exit_reason == ExitReason.STOP_LOSS:
            self._sl_count += 1
            reward += cfg.penalty_stop_loss

        elif exit_reason == ExitReason.MAX_HOLD_TIME:
            self._max_hold_count += 1
            if net_return > 0:
                reward += cfg.reward_max_hold_positive
            else:
                reward += cfg.penalty_max_hold_negative

        elif exit_reason == ExitReason.END_OF_EPISODE:
            if net_return > 0:
                reward += cfg.reward_end_episode_positive
            else:
                reward += cfg.penalty_end_episode_negative

        # -----------------------------------------
        # Spike capture bonuses
        # -----------------------------------------
        spike_freshness = spike_context.get('spike_freshness', 0.0)

        # Bonus for profitable entry on fresh spike
        if net_return > 0 and spike_freshness > cfg.fresh_spike_threshold:
            reward += cfg.bonus_fresh_spike_profitable
            self._spike_captures += 1
            logger.debug(f"Fresh spike capture bonus: +{cfg.bonus_fresh_spike_profitable}")

        # Large move capture
        if net_return > cfg.huge_move_threshold:
            reward += cfg.huge_move_bonus
            logger.debug(f"Huge move bonus: +{cfg.huge_move_bonus}")
        elif net_return > cfg.large_move_threshold:
            reward += cfg.large_move_bonus
            logger.debug(f"Large move bonus: +{cfg.large_move_bonus}")

        # -----------------------------------------
        # Trade quality bonuses
        # -----------------------------------------
        # Quick win bonus
        if exit_reason == ExitReason.TAKE_PROFIT and hold_duration < cfg.quick_win_threshold_minutes:
            reward += cfg.quick_win_bonus
            logger.debug(f"Quick win bonus: +{cfg.quick_win_bonus} (hold={hold_duration}min)")

        # High watermark capture (exited near the position high)
        if net_return > 0:
            return_gap = highest_return - net_return
            if return_gap < cfg.high_capture_threshold:
                reward += cfg.high_capture_bonus
                logger.debug(f"High capture bonus: +{cfg.high_capture_bonus}")

        # -----------------------------------------
        # Patience bonus (waited long enough before entering)
        # -----------------------------------------
        # This is implicit in the trade timing - if the entry led to TP,
        # patience is already rewarded through the trade outcome

        return reward

    def reset(self) -> None:
        """Reset internal state for new episode."""
        self._tp_count = 0
        self._sl_count = 0
        self._max_hold_count = 0
        self._total_reward = 0.0
        self._spike_captures = 0

        # Reset quick win tracking
        self._entries_this_episode = 0
        self._waits_rewarded = 0
        self._entry_costs_applied = 0.0

    def get_stats(self) -> dict:
        """Get statistics for debugging."""
        total_trades = self._tp_count + self._sl_count + self._max_hold_count
        return {
            'tp_count': self._tp_count,
            'sl_count': self._sl_count,
            'max_hold_count': self._max_hold_count,
            'total_trades': total_trades,
            'win_rate': self._tp_count / max(1, total_trades),
            'spike_captures': self._spike_captures,
            'total_reward': self._total_reward,
            # Quick win metrics
            'entries_this_episode': self._entries_this_episode,
            'entry_budget': self.config.entry_budget,
            'waits_rewarded': self._waits_rewarded,
            'entry_costs_applied': self._entry_costs_applied,
        }


class AggressiveEntryRewardCalculator:
    """
    Alternative reward calculator that encourages more entries.

    Use this if the agent learns to be too conservative (never enters).
    Provides stronger positive signals for any profitable trade.
    """

    def __init__(self, config: Optional[EntryTimingRewardConfig] = None):
        """Initialize reward calculator."""
        self.config = config or EntryTimingRewardConfig()
        self._base_calculator = EntryTimingRewardCalculator(self.config)

    def calculate(
        self,
        action: int,
        trade_result: Optional[Any] = None,
        position: Optional[Any] = None,
        spike_context: Optional[dict] = None,
        episode_trades: Optional[List[Any]] = None,
    ) -> float:
        """Calculate reward with aggressive entry encouragement."""
        reward = self._base_calculator.calculate(
            action, trade_result, position, spike_context, episode_trades
        )

        # Extra encouragement for entering
        if action == 1 and position is None:  # Just entered
            reward += 0.1  # Small bonus just for entering

        # Extra penalty for long periods without trading
        episode_trades = episode_trades or []
        if len(episode_trades) == 0 and position is None:
            # No trades yet and not in position - small penalty to encourage action
            reward -= 0.001

        return reward

    def reset(self) -> None:
        """Reset internal state."""
        self._base_calculator.reset()

    def get_stats(self) -> dict:
        """Get statistics."""
        return self._base_calculator.get_stats()


def calculate_entry_metrics(trades: List[Any]) -> dict:
    """
    Calculate trading metrics from entry timing trades.

    Args:
        trades: List of EntryTradeResult objects

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
            'tp_rate': 0.0,
            'sl_rate': 0.0,
            'avg_hold_minutes': 0,
        }

    from .entry_timing_env import ExitReason

    returns = [t.return_pct for t in trades]
    total_return = sum(returns)
    num_trades = len(trades)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Exit reason breakdown
    tp_count = sum(1 for t in trades if t.exit_reason == ExitReason.TAKE_PROFIT)
    sl_count = sum(1 for t in trades if t.exit_reason == ExitReason.STOP_LOSS)

    tp_rate = tp_count / num_trades if num_trades > 0 else 0
    sl_rate = sl_count / num_trades if num_trades > 0 else 0

    avg_hold = np.mean([t.hold_duration_minutes for t in trades])

    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'tp_rate': tp_rate,
        'sl_rate': sl_rate,
        'avg_hold_minutes': avg_hold,
    }


if __name__ == "__main__":
    # Test reward calculation
    from dataclasses import dataclass
    from datetime import datetime, timedelta

    @dataclass
    class MockExitReason:
        TAKE_PROFIT = 1
        STOP_LOSS = 2
        MAX_HOLD_TIME = 3
        END_OF_EPISODE = 4

    @dataclass
    class MockTradeResult:
        return_pct: float
        exit_reason: int
        highest_return: float
        hold_duration_minutes: int
        entry_price: float = 100.0
        exit_price: float = 112.0
        entry_time: datetime = None
        exit_time: datetime = None
        fees_paid: float = 0.55

        def __post_init__(self):
            self.entry_time = self.entry_time or datetime.now()
            self.exit_time = self.exit_time or self.entry_time + timedelta(minutes=self.hold_duration_minutes)

    print("=" * 60)
    print("Testing EntryTimingRewardCalculator")
    print("=" * 60)

    calc = EntryTimingRewardCalculator()

    # Test 1: Take profit hit
    print("\n--- Test 1: Take Profit Hit (12% gain) ---")

    # Mock the exit reason properly
    class MockER:
        TAKE_PROFIT = 1
        STOP_LOSS = 2
        MAX_HOLD_TIME = 3
        END_OF_EPISODE = 4

    trade = MockTradeResult(
        return_pct=0.12,
        exit_reason=MockER.TAKE_PROFIT,
        highest_return=0.13,
        hold_duration_minutes=60,
    )

    # Patch the import in the calculator
    import sys
    from types import ModuleType
    mock_module = ModuleType('entry_timing_env')
    mock_module.ExitReason = MockER
    sys.modules['src.rl.entry_timing_env'] = mock_module

    spike_context = {'spike_freshness': 0.9, 'is_in_spike': True, 'spike_traded': False}
    reward = calc.calculate(action=0, trade_result=trade, spike_context=spike_context)
    print(f"  Reward: {reward:.4f}")
    print(f"  Stats: {calc.get_stats()}")

    # Test 2: Stop loss hit
    print("\n--- Test 2: Stop Loss Hit (-2% loss) ---")
    trade = MockTradeResult(
        return_pct=-0.02,
        exit_reason=MockER.STOP_LOSS,
        highest_return=0.01,
        hold_duration_minutes=30,
    )
    reward = calc.calculate(action=0, trade_result=trade, spike_context=spike_context)
    print(f"  Reward: {reward:.4f}")
    print(f"  Stats: {calc.get_stats()}")

    # Test 3: Quick win
    print("\n--- Test 3: Quick Win (TP in 30 min) ---")
    trade = MockTradeResult(
        return_pct=0.12,
        exit_reason=MockER.TAKE_PROFIT,
        highest_return=0.12,
        hold_duration_minutes=30,
    )
    reward = calc.calculate(action=0, trade_result=trade, spike_context=spike_context)
    print(f"  Reward: {reward:.4f} (should include quick win bonus)")
    print(f"  Stats: {calc.get_stats()}")

    # Test 4: Huge move capture
    print("\n--- Test 4: Huge Move Capture (15% gain) ---")
    trade = MockTradeResult(
        return_pct=0.15,
        exit_reason=MockER.TAKE_PROFIT,
        highest_return=0.16,
        hold_duration_minutes=240,
    )
    reward = calc.calculate(action=0, trade_result=trade, spike_context=spike_context)
    print(f"  Reward: {reward:.4f} (should include huge move bonus)")
    print(f"  Stats: {calc.get_stats()}")

    print("\n" + "=" * 60)
    print("EntryTimingRewardCalculator tests completed!")
    print("=" * 60)
