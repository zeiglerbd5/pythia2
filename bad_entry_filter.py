"""
Bad Entry Filter - Identifies and blocks trades likely to hit stop loss

Based on analysis of 56 historical trades where 43% hit stop loss.
This filter reduces stop loss rate from 44% to 38-40% depending on tier.

Author: ML Analysis
Date: 2026-03-19
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional


class FilterTier(Enum):
    """Filter aggressiveness levels"""
    CONSERVATIVE = "conservative"  # Highest precision, minimal false positives
    BALANCED = "balanced"          # Best PnL improvement
    AGGRESSIVE = "aggressive"      # Maximum stop loss avoidance


@dataclass
class FilterResult:
    """Result of bad entry filter check"""
    should_block: bool
    reason: str
    tier_triggered: Optional[FilterTier]
    risk_score: float  # 0-1, higher = more risky


class BadEntryFilter:
    """
    Filters out entries with high probability of hitting stop loss.

    Key patterns identified from historical analysis:
    1. FOMO Trap: High volatility + High momentum (chasing pumps)
    2. Extreme Momentum: Very high momentum alone indicates overextension
    3. Extreme Volatility: Very high volatility alone is dangerous

    Statistics by tier:
    - CONSERVATIVE: 71% precision, catches 21% of SL, loses 7% of winners
    - BALANCED: 67% precision, catches 25% of SL, loses 10% of winners
    - AGGRESSIVE: 67% precision, catches 33% of SL, loses 13% of winners
    """

    def __init__(self, tier: FilterTier = FilterTier.BALANCED):
        """
        Initialize filter with specified aggressiveness tier.

        Args:
            tier: FilterTier enum - CONSERVATIVE, BALANCED, or AGGRESSIVE
        """
        self.tier = tier

        # Thresholds tuned from historical backtest analysis
        self.thresholds = {
            FilterTier.CONSERVATIVE: {
                'volatility_and_momentum': (0.50, 3.0),  # Both must exceed
            },
            FilterTier.BALANCED: {
                'momentum_or_volatility': (5.0, 0.60),  # Either exceeds
            },
            FilterTier.AGGRESSIVE: {
                'momentum_or_volatility': (5.0, 0.50),  # Either exceeds (looser vol)
            }
        }

    def check(self, volatility_4h: float, momentum_4h: float) -> FilterResult:
        """
        Check if entry should be blocked.

        Args:
            volatility_4h: 4-hour volatility metric (typically 0.1 - 0.8)
            momentum_4h: 4-hour momentum/return (typically -5 to +10)

        Returns:
            FilterResult with decision and reasoning
        """
        risk_score = self._calculate_risk_score(volatility_4h, momentum_4h)

        if self.tier == FilterTier.CONSERVATIVE:
            return self._check_conservative(volatility_4h, momentum_4h, risk_score)
        elif self.tier == FilterTier.BALANCED:
            return self._check_balanced(volatility_4h, momentum_4h, risk_score)
        else:
            return self._check_aggressive(volatility_4h, momentum_4h, risk_score)

    def _calculate_risk_score(self, volatility_4h: float, momentum_4h: float) -> float:
        """Calculate composite risk score 0-1"""
        # Normalize inputs to 0-1 range based on observed distributions
        vol_score = min(volatility_4h / 0.80, 1.0)  # 0.80 is ~max observed
        mom_score = min(abs(momentum_4h) / 10.0, 1.0)  # 10 is ~max observed

        # Combined score weighted towards momentum (more predictive)
        return 0.4 * vol_score + 0.6 * mom_score

    def _check_conservative(self, vol: float, mom: float, risk: float) -> FilterResult:
        """
        Conservative filter: Only block obvious FOMO traps.
        Requires BOTH high volatility AND high momentum.
        """
        vol_thresh, mom_thresh = self.thresholds[FilterTier.CONSERVATIVE]['volatility_and_momentum']

        if vol > vol_thresh and mom > mom_thresh:
            return FilterResult(
                should_block=True,
                reason=f"FOMO trap detected: volatility={vol:.2f}>{vol_thresh} AND momentum={mom:.1f}>{mom_thresh}",
                tier_triggered=FilterTier.CONSERVATIVE,
                risk_score=risk
            )

        return FilterResult(
            should_block=False,
            reason="Entry allowed - no danger signals",
            tier_triggered=None,
            risk_score=risk
        )

    def _check_balanced(self, vol: float, mom: float, risk: float) -> FilterResult:
        """
        Balanced filter: Best PnL improvement.
        Blocks if EITHER extreme momentum OR extreme volatility.
        """
        mom_thresh, vol_thresh = self.thresholds[FilterTier.BALANCED]['momentum_or_volatility']

        if mom > mom_thresh:
            return FilterResult(
                should_block=True,
                reason=f"Extreme momentum: {mom:.1f}>{mom_thresh}",
                tier_triggered=FilterTier.BALANCED,
                risk_score=risk
            )

        if vol > vol_thresh:
            return FilterResult(
                should_block=True,
                reason=f"Extreme volatility: {vol:.2f}>{vol_thresh}",
                tier_triggered=FilterTier.BALANCED,
                risk_score=risk
            )

        return FilterResult(
            should_block=False,
            reason="Entry allowed - within acceptable range",
            tier_triggered=None,
            risk_score=risk
        )

    def _check_aggressive(self, vol: float, mom: float, risk: float) -> FilterResult:
        """
        Aggressive filter: Maximum capital preservation.
        Lower thresholds to catch more potential stop losses.
        """
        mom_thresh, vol_thresh = self.thresholds[FilterTier.AGGRESSIVE]['momentum_or_volatility']

        if mom > mom_thresh:
            return FilterResult(
                should_block=True,
                reason=f"High momentum risk: {mom:.1f}>{mom_thresh}",
                tier_triggered=FilterTier.AGGRESSIVE,
                risk_score=risk
            )

        if vol > vol_thresh:
            return FilterResult(
                should_block=True,
                reason=f"High volatility risk: {vol:.2f}>{vol_thresh}",
                tier_triggered=FilterTier.AGGRESSIVE,
                risk_score=risk
            )

        return FilterResult(
            should_block=False,
            reason="Entry allowed",
            tier_triggered=None,
            risk_score=risk
        )


def should_block_entry(volatility_4h: float, momentum_4h: float,
                       tier: str = "balanced") -> Tuple[bool, str]:
    """
    Simple function interface for bad entry detection.

    Args:
        volatility_4h: 4-hour volatility metric
        momentum_4h: 4-hour momentum/return percentage
        tier: "conservative", "balanced", or "aggressive"

    Returns:
        Tuple of (should_block: bool, reason: str)

    Example:
        >>> should_block, reason = should_block_entry(0.55, 4.5, "balanced")
        >>> if should_block:
        ...     print(f"BLOCKED: {reason}")
    """
    tier_map = {
        "conservative": FilterTier.CONSERVATIVE,
        "balanced": FilterTier.BALANCED,
        "aggressive": FilterTier.AGGRESSIVE
    }

    filter_obj = BadEntryFilter(tier=tier_map.get(tier, FilterTier.BALANCED))
    result = filter_obj.check(volatility_4h, momentum_4h)

    return result.should_block, result.reason


# Quick check functions for integration
def is_fomo_trap(volatility_4h: float, momentum_4h: float) -> bool:
    """Check if entry is a FOMO trap (high vol + high mom)"""
    return volatility_4h > 0.50 and momentum_4h > 3.0


def is_extreme_momentum(momentum_4h: float) -> bool:
    """Check if momentum is dangerously high"""
    return momentum_4h > 5.0


def is_extreme_volatility(volatility_4h: float) -> bool:
    """Check if volatility is dangerously high"""
    return volatility_4h > 0.60


if __name__ == "__main__":
    # Demo usage
    print("Bad Entry Filter - Demo")
    print("=" * 60)

    test_cases = [
        (0.30, 2.0, "Normal conditions"),
        (0.55, 4.0, "High vol + high momentum (FOMO)"),
        (0.40, 6.5, "Extreme momentum only"),
        (0.65, 1.0, "Extreme volatility only"),
        (0.70, 8.0, "Both extreme"),
    ]

    for vol, mom, desc in test_cases:
        print(f"\n{desc}:")
        print(f"  volatility_4h={vol:.2f}, momentum_4h={mom:.1f}")

        for tier in ["conservative", "balanced", "aggressive"]:
            blocked, reason = should_block_entry(vol, mom, tier)
            status = "BLOCKED" if blocked else "ALLOWED"
            print(f"  {tier:12}: {status}")
