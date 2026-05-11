#!/usr/bin/env python3
"""
Compare profitability of Fast & Steep vs Slow & Large spike types
under realistic trading scenarios.
"""

import pandas as pd
import numpy as np

# Load the categorized spikes
df = pd.read_csv('all_spikes_categorized.csv')

# Filter to just the two main types
fast_steep = df[df['category'] == 'Fast & Steep'].copy()
slow_large = df[df['category'] == 'Slow & Large'].copy()

print("=" * 80)
print("PROFITABILITY ANALYSIS: Fast & Steep vs Slow & Large")
print("=" * 80)
print()

# Basic statistics
print("Fast & Steep:")
print(f"  Count: {len(fast_steep)}")
print(f"  Mean gain: {fast_steep['peak_gain'].mean():.1f}%")
print(f"  Median gain: {fast_steep['peak_gain'].median():.1f}%")
print(f"  Mean time to peak: {fast_steep['time_to_peak_min'].mean():.1f} min")
print(f"  Median time to peak: {fast_steep['time_to_peak_min'].median():.1f} min")
print()

print("Slow & Large:")
print(f"  Count: {len(slow_large)}")
print(f"  Mean gain: {slow_large['peak_gain'].mean():.1f}%")
print(f"  Median gain: {slow_large['peak_gain'].median():.1f}%")
print(f"  Mean time to peak: {slow_large['time_to_peak_min'].mean():.1f} min")
print(f"  Median time to peak: {slow_large['time_to_peak_min'].median():.1f} min")
print()

# Entry timing analysis
print("=" * 80)
print("ENTRY TIMING SCENARIOS")
print("=" * 80)
print()

# Scenario 1: Enter immediately (within 1 min of signal)
print("Scenario 1: INSTANT ENTRY (1 min delay)")
print("-" * 80)
print("Fast & Steep:")
# Fast peaks at median 11 min, so entering at 1 min catches most of it
# Assume you catch 90% of the move (conservative)
fast_instant = fast_steep['peak_gain'] * 0.90
print(f"  Estimated capture: 90% of move")
print(f"  Mean profit: {fast_instant.mean():.1f}%")
print(f"  Median profit: {fast_instant.median():.1f}%")
print()

print("Slow & Large:")
# Slow peaks at median 122 min, entering at 1 min catches nearly all of it
slow_instant = slow_large['peak_gain'] * 0.95
print(f"  Estimated capture: 95% of move")
print(f"  Mean profit: {slow_instant.mean():.1f}%")
print(f"  Median profit: {slow_instant.median():.1f}%")
print()

# Scenario 2: Enter after 5 minutes (more realistic)
print("Scenario 2: DELAYED ENTRY (5 min delay)")
print("-" * 80)
print("Fast & Steep:")
# Peak at 11 min, entering at 5 min means you've missed 45% of the time window
# Conservatively, you might catch only 50-60% of the move
fast_delayed = fast_steep['peak_gain'] * 0.55
print(f"  Estimated capture: 55% of move (already halfway to peak)")
print(f"  Mean profit: {fast_delayed.mean():.1f}%")
print(f"  Median profit: {fast_delayed.median():.1f}%")
print()

print("Slow & Large:")
# Peak at 122 min, entering at 5 min is still very early
slow_delayed = slow_large['peak_gain'] * 0.90
print(f"  Estimated capture: 90% of move (still very early)")
print(f"  Mean profit: {slow_delayed.mean():.1f}%")
print(f"  Median profit: {slow_delayed.median():.1f}%")
print()

# Scenario 3: Enter after 15 minutes (realistic with classification delay)
print("Scenario 3: LATE ENTRY (15 min delay - time to classify spike type)")
print("-" * 80)
print("Fast & Steep:")
# Peak at 11 min, entering at 15 min means you've MISSED the peak
# You might only catch 10-20% or even lose money on the way down
fast_late = fast_steep['peak_gain'] * 0.10  # Very conservative
print(f"  Estimated capture: 10% of move (MISSED - already past peak!)")
print(f"  Mean profit: {fast_late.mean():.1f}%")
print(f"  Median profit: {fast_late.median():.1f}%")
print(f"  ** NOT PROFITABLE - Too late to enter **")
print()

print("Slow & Large:")
# Peak at 122 min, entering at 15 min is still quite early
slow_late = slow_large['peak_gain'] * 0.85
print(f"  Estimated capture: 85% of move (still early in the move)")
print(f"  Mean profit: {slow_late.mean():.1f}%")
print(f"  Median profit: {slow_late.median():.1f}%")
print()

# Summary
print("=" * 80)
print("SUMMARY: WHICH IS EASIER TO TRADE PROFITABLY?")
print("=" * 80)
print()

print("WINNER: Slow & Large")
print()
print("Reasons:")
print("1. MORE FORGIVING ENTRY WINDOW")
print("   - Fast & Steep: Must enter within 2-5 min to be profitable")
print("   - Slow & Large: Can enter anytime in first 15-30 min")
print()
print("2. LARGER GAINS")
print(f"   - Fast & Steep median: {fast_steep['peak_gain'].median():.1f}%")
print(f"   - Slow & Large median: {slow_large['peak_gain'].median():.1f}%")
print(f"   - Slow & Large has {slow_large['peak_gain'].median() / fast_steep['peak_gain'].median():.1f}x larger gains")
print()
print("3. EASIER TO CLASSIFY")
print("   - Fast & Steep: Need to classify in <5 min (very difficult)")
print("   - Slow & Large: Can classify in first 15-30 min (much easier)")
print()
print("4. PROFITABILITY COMPARISON (5 min entry delay scenario):")
print(f"   - Fast & Steep: ~{fast_delayed.median():.1f}% median profit")
print(f"   - Slow & Large: ~{slow_delayed.median():.1f}% median profit")
print(f"   - Slow & Large is {slow_delayed.median() / fast_delayed.median():.1f}x more profitable")
print()

# Strategy recommendations
print("=" * 80)
print("STRATEGY RECOMMENDATIONS")
print("=" * 80)
print()

print("OPTION 1: SEPARATE STRATEGIES (Recommended)")
print("-" * 80)
print()
print("Fast & Steep Strategy:")
print("  - Entry: IMMEDIATE (within 1-2 min of signal)")
print("  - Exit: 30 min OR 2% drawdown from peak, whichever comes first")
print("  - Target: 10-20% gain")
print("  - Position size: Smaller (higher risk of late entry)")
print()
print("Slow & Large Strategy:")
print("  - Entry: Within 10-15 min of signal (more time to validate)")
print("  - Exit: 6-24 hours OR 5% drawdown from peak")
print("  - Target: 30-100%+ gain")
print("  - Position size: Larger (more predictable, lower entry risk)")
print()

print("OPTION 2: UNIFIED STRATEGY")
print("-" * 80)
print()
print("Universal Strategy:")
print("  - Entry: IMMEDIATE for ALL signals (within 1-2 min)")
print("  - Initial exit: 10 min mark - check gain")
print("    - If gain >15% already: Likely Fast & Steep, exit within 30 min")
print("    - If gain 8-12%: Likely Slow & Large, hold for 6-24 hours")
print("  - This allows ONE entry strategy with adaptive exit")
print()

print("RECOMMENDATION:")
print("Use UNIFIED STRATEGY with adaptive exit at 10-minute mark.")
print("This captures both types effectively:")
print("  - Catches Fast & Steep early (before peak at 11 min)")
print("  - Catches Slow & Large early (long before peak at 122 min)")
print("  - Lets the spike TYPE reveal itself by 10-minute performance")
print()

# Calculate expected value
print("=" * 80)
print("EXPECTED VALUE ANALYSIS")
print("=" * 80)
print()

# With unified immediate entry strategy
fast_ev = fast_instant.mean()
slow_ev = slow_instant.mean()
overall_ev = (fast_ev * len(fast_steep) + slow_ev * len(slow_large)) / (len(fast_steep) + len(slow_large))

print("With UNIFIED immediate entry strategy:")
print(f"  Fast & Steep EV: {fast_ev:.1f}% per trade")
print(f"  Slow & Large EV: {slow_ev:.1f}% per trade")
print(f"  Overall EV: {overall_ev:.1f}% per trade")
print(f"  Frequency: {len(fast_steep) + len(slow_large)}/602 = {(len(fast_steep) + len(slow_large))/602*100:.1f}% of signals")
print()
print("This assumes:")
print("  - 90% capture on Fast & Steep (immediate entry)")
print("  - 95% capture on Slow & Large (immediate entry)")
print("  - Small Movers filtered out by RSI > 80 check")
