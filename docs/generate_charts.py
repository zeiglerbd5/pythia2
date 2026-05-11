"""
Generate visualization charts for Breakout Detection Strategies document.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_breakout_pattern_chart():
    """Create visualization of T+0, T+1, T+2 triple confirmation pattern."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Simulate price data
    hours = np.arange(-6, 12, 0.25)  # -6h to +12h from breakout

    # Pre-breakout: flat with slight noise
    pre_breakout = np.ones(24) * 100 + np.random.randn(24) * 0.5

    # T+0: Sharp breakout (+8%)
    t0_end = 108

    # T+1: Continuation (+3% from T+0 close)
    t1_end = 111.24

    # T+2: Strong resumption (+6% from T+1 close)
    t2_end = 117.91

    # Post-entry: continues upward then consolidates

    # Build price array
    prices = []
    for i, h in enumerate(hours):
        if h < 0:
            prices.append(100 + np.random.randn() * 0.3)
        elif h < 1:  # T+0 hour
            progress = h
            prices.append(100 + progress * 8 + np.random.randn() * 0.2)
        elif h < 2:  # T+1 hour
            progress = h - 1
            prices.append(108 + progress * 3.24 + np.random.randn() * 0.2)
        elif h < 3:  # T+2 hour (entry point)
            progress = h - 2
            prices.append(111.24 + progress * 6.67 + np.random.randn() * 0.2)
        else:  # Post-entry
            base = 117.91
            growth = (h - 3) * 2.5
            noise = np.sin(h * 2) * 2 + np.random.randn() * 0.5
            prices.append(base + growth + noise)

    prices = np.array(prices)

    # Plot price line
    ax.plot(hours, prices, 'b-', linewidth=2, label='Price')

    # Highlight T+0, T+1, T+2 zones
    ax.axvspan(0, 1, alpha=0.3, color='yellow', label='T+0 (Breakout Detection)')
    ax.axvspan(1, 2, alpha=0.3, color='orange', label='T+1 (Filter - Must be positive)')
    ax.axvspan(2, 3, alpha=0.3, color='green', label='T+2 (Entry Point)')

    # Mark key prices
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Pre-breakout price')
    ax.axhline(y=108, color='yellow', linestyle=':', alpha=0.7)
    ax.axhline(y=111.24, color='orange', linestyle=':', alpha=0.7)

    # Entry point arrow
    entry_price = 111.24 * 1.0  # T+2 open
    ax.annotate('ENTRY\n(T+2 Open)', xy=(2, entry_price), xytext=(2.5, 105),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # Stop loss line
    stop_loss = entry_price * 0.92  # 8% below entry
    ax.axhline(y=stop_loss, color='red', linestyle='--', alpha=0.7, label=f'Stop Loss (-8%): ${stop_loss:.2f}')

    # Annotations for each phase
    ax.text(0.5, 109, 'T+0: +8%\nVol 3x+', ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    ax.text(1.5, 112.5, 'T+1: +3%\n(positive=pass)', ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    ax.text(2.5, 119, 'T+2: +6%\nVol cont.', ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))

    ax.set_xlabel('Hours from Breakout (T+0)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Breakout Hunter v5.3: Triple Confirmation Pattern\n(T+0 Detect, T+1 Filter, T+2 Entry)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-6, 12)
    ax.set_ylim(95, 140)

    plt.tight_layout()
    plt.savefig('/Users/bz/Pythia2/docs/breakout_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: breakout_pattern.png")


def create_profit_lock_chart():
    """Create visualization of stepped profit lock levels."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Entry point
    entry_price = 100

    # Simulate a winning trade that hits multiple levels
    hours = np.arange(0, 24, 0.25)

    # Price trajectory: rises to +45%, then pulls back to +30%
    prices = []
    for h in hours:
        if h < 6:
            # Initial rise
            prices.append(entry_price * (1 + h * 0.03))
        elif h < 12:
            # Accelerating rise
            base = entry_price * 1.18
            prices.append(base * (1 + (h - 6) * 0.045))
        elif h < 18:
            # Peak and pullback
            peak = entry_price * 1.45
            pullback = (h - 12) * 0.025
            prices.append(peak * (1 - pullback))
        else:
            # Stabilize around +30%
            prices.append(entry_price * 1.30 + np.random.randn() * 0.5)

    prices = np.array(prices)

    # Plot price
    ax.plot(hours, prices, 'b-', linewidth=2, label='Price')

    # Entry line
    ax.axhline(y=entry_price, color='gray', linestyle='--', alpha=0.5, label=f'Entry: ${entry_price}')

    # Profit lock levels
    lock_levels = [
        (0.05, -0.02, 'Lock -2%'),   # +5% gain -> lock -2%
        (0.10, 0.00, 'Breakeven'),   # +10% gain -> breakeven
        (0.15, 0.05, 'Lock +5%'),    # +15% gain -> lock +5%
        (0.25, 0.12, 'Lock +12%'),   # +25% gain -> lock +12%
        (0.40, 0.25, 'Lock +25%'),   # +40% gain -> lock +25%
    ]

    colors = ['#ffcccc', '#ffffcc', '#ccffcc', '#ccffff', '#ccccff']

    for i, (trigger, lock, label) in enumerate(lock_levels):
        trigger_price = entry_price * (1 + trigger)
        lock_price = entry_price * (1 + lock)

        # Trigger line (dashed)
        ax.axhline(y=trigger_price, color=colors[i], linestyle=':', alpha=0.8)
        ax.text(24.5, trigger_price, f'+{trigger*100:.0f}%', va='center', fontsize=9)

        # Lock line (solid)
        ax.axhline(y=lock_price, color='green' if lock >= 0 else 'orange',
                   linestyle='-', linewidth=2, alpha=0.6)
        ax.text(-0.5, lock_price, label, va='center', ha='right', fontsize=9,
                color='green' if lock >= 0 else 'orange')

    # Initial stop loss
    stop_loss = entry_price * 0.92
    ax.axhline(y=stop_loss, color='red', linestyle='--', linewidth=2,
               label=f'Initial Stop: ${stop_loss:.0f} (-8%)')

    # Mark where locks activate
    for i, (trigger, lock, label) in enumerate(lock_levels):
        trigger_price = entry_price * (1 + trigger)
        # Find first hour where price exceeds trigger
        for h, p in zip(hours, prices):
            if p >= trigger_price:
                ax.scatter([h], [trigger_price], color='green', s=100, zorder=5)
                ax.annotate(f'{label}\nactivated', xy=(h, trigger_price),
                           xytext=(h+1.5, trigger_price-3), fontsize=8,
                           arrowprops=dict(arrowstyle='->', color='green', lw=1))
                break

    ax.set_xlabel('Hours Since Entry')
    ax.set_ylabel('Price ($)')
    ax.set_title('Breakout Hunter v5.3: Stepped Profit Lock System\n"Never let big winners become losers"', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-1, 26)
    ax.set_ylim(88, 150)

    # Add annotation box explaining the system
    textstr = 'Profit Lock Logic:\n' \
              '1. Peak gain triggers lock level\n' \
              '2. Stop moves up to locked price\n' \
              '3. Multiple locks can stack\n' \
              '4. Exit on pullback to lock level'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('/Users/bz/Pythia2/docs/profit_locks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: profit_locks.png")


def create_trailing_stop_chart():
    """Create visualization of tiered trailing stop system."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Entry point
    entry_price = 100

    # Simulate a big winner: rises to +80%, then pulls back
    hours = np.arange(0, 36, 0.25)

    # Price trajectory
    prices = []
    highest = entry_price
    for h in hours:
        if h < 8:
            # Initial rise to +25%
            p = entry_price * (1 + h * 0.03125)
        elif h < 16:
            # Accelerate to +50%
            p = entry_price * 1.25 * (1 + (h - 8) * 0.03)
        elif h < 24:
            # Push to +80%
            p = entry_price * 1.50 * (1 + (h - 16) * 0.025)
        else:
            # Pullback from +80% to trigger trail
            peak = entry_price * 1.80
            pullback = (h - 24) * 0.006
            p = peak * (1 - pullback)

        # Add noise
        p += np.random.randn() * 0.3
        prices.append(p)
        highest = max(highest, p)

    prices = np.array(prices)

    # Calculate trailing stop levels dynamically
    trail_levels = []
    highest_so_far = entry_price
    for i, p in enumerate(prices):
        highest_so_far = max(highest_so_far, p)
        gain = (highest_so_far - entry_price) / entry_price

        # Determine trail percentage
        if gain >= 0.70:
            trail_pct = 0.05  # 5% trail
        elif gain >= 0.50:
            trail_pct = 0.06  # 6% trail
        elif gain >= 0.30:
            trail_pct = 0.08  # 8% trail
        elif gain >= 0.20:
            trail_pct = 0.10  # 10% trail (default)
        else:
            trail_pct = None  # Not active

        if trail_pct:
            trail_levels.append(highest_so_far * (1 - trail_pct))
        else:
            trail_levels.append(None)

    # Plot price
    ax.plot(hours, prices, 'b-', linewidth=2, label='Price')

    # Plot trailing stop (where active)
    trail_hours = []
    trail_prices = []
    for h, t in zip(hours, trail_levels):
        if t is not None:
            trail_hours.append(h)
            trail_prices.append(t)

    if trail_hours:
        ax.plot(trail_hours, trail_prices, 'r-', linewidth=2, label='Trailing Stop')

    # Entry and initial stop
    ax.axhline(y=entry_price, color='gray', linestyle='--', alpha=0.5, label=f'Entry: ${entry_price}')
    ax.axhline(y=entry_price * 0.92, color='red', linestyle=':', alpha=0.5, label='Initial Stop (-8%)')

    # Mark trail tier transitions
    tier_changes = [
        (0.20, 0.10, '+20%: Trail activates at 10%'),
        (0.30, 0.08, '+30%: Tightens to 8%'),
        (0.50, 0.06, '+50%: Tightens to 6%'),
        (0.70, 0.05, '+70%: Tightens to 5%'),
    ]

    y_offset = 0
    for trigger, trail_pct, label in tier_changes:
        trigger_price = entry_price * (1 + trigger)
        ax.axhline(y=trigger_price, color='purple', linestyle=':', alpha=0.3)
        ax.text(36.5, trigger_price + y_offset, label, va='center', fontsize=9, color='purple')
        y_offset += 2  # Prevent overlap

    # Find and mark exit point
    for i in range(len(prices)):
        if trail_levels[i] is not None and prices[i] <= trail_levels[i]:
            ax.scatter([hours[i]], [prices[i]], color='red', s=200, zorder=5, marker='X')
            ax.annotate('EXIT\n(Trail Stop)', xy=(hours[i], prices[i]),
                       xytext=(hours[i]+2, prices[i]+5), fontsize=11, fontweight='bold',
                       color='red', arrowprops=dict(arrowstyle='->', color='red', lw=2))
            break

    ax.set_xlabel('Hours Since Entry')
    ax.set_ylabel('Price ($)')
    ax.set_title('Breakout Hunter v5.3: Tiered Trailing Stop System\n"Tighten the trail as gains increase"', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-1, 40)
    ax.set_ylim(88, 200)

    # Add tier explanation
    textstr = 'Trailing Stop Tiers:\n' \
              '+20%: 10% trail (default)\n' \
              '+30%: 8% trail\n' \
              '+50%: 6% trail\n' \
              '+70%: 5% trail'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig('/Users/bz/Pythia2/docs/trailing_stop.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: trailing_stop.png")


def create_accumulation_pattern_chart():
    """Create visualization of stealth accumulation pattern."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    hours = np.arange(-48, 24, 1)  # 48h before to 24h after breakout

    # === Price Panel ===
    ax1 = axes[0]

    # Price: flat during accumulation, then breakout
    prices = []
    for h in hours:
        if h < 0:
            # Accumulation phase: flat with slight compression
            prices.append(100 + np.random.randn() * (1 - abs(h)/100))
        elif h < 3:
            # Breakout phase
            prices.append(100 + h * 15 + np.random.randn())
        else:
            # Post-breakout: continues up
            prices.append(145 + (h - 3) * 3 + np.sin(h) * 2 + np.random.randn())

    prices = np.array(prices)

    ax1.plot(hours, prices, 'b-', linewidth=2)
    ax1.axvspan(-48, 0, alpha=0.2, color='blue', label='Accumulation Phase')
    ax1.axvspan(0, 3, alpha=0.3, color='green', label='Breakout')
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Accumulation Hunter v6.0: Stealth Accumulation Pattern Detection', fontsize=14)
    ax1.legend(loc='upper left')

    # Mark entry
    ax1.annotate('ENTRY\n(Breakout confirmed)', xy=(1, 115), xytext=(-10, 130),
                fontsize=11, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # === BAR (Buy/Ask Ratio) Panel ===
    ax2 = axes[1]

    # BAR: builds up during accumulation
    bar_values = []
    for h in hours:
        if h < -36:
            bar_values.append(1.0 + np.random.randn() * 0.2)
        elif h < 0:
            # BAR builds up (stealth accumulation)
            progress = (h + 36) / 36
            bar_values.append(1.0 + progress * 7 + np.random.randn() * 0.5)
        else:
            # After breakout: normalizes
            bar_values.append(4.0 + np.random.randn() * 1.5)

    bar_values = np.array(bar_values)

    ax2.fill_between(hours, bar_values, alpha=0.5, color='purple')
    ax2.plot(hours, bar_values, 'purple', linewidth=2)
    ax2.axhline(y=3.0, color='orange', linestyle='--', label='Watch threshold (3x)')
    ax2.axhline(y=5.0, color='red', linestyle='--', label='Strong signal (5x)')
    ax2.axvspan(-48, 0, alpha=0.1, color='blue')
    ax2.set_ylabel('BAR (Bid/Ask Ratio)\nvs Baseline')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 10)

    # === Volume Panel ===
    ax3 = axes[2]

    # Volume: elevated during accumulation
    volumes = []
    for h in hours:
        if h < -36:
            volumes.append(1.0 + np.random.rand() * 0.5)
        elif h < 0:
            # Volume elevated
            volumes.append(2.5 + np.random.rand() * 1.5)
        elif h < 3:
            # Breakout volume surge
            volumes.append(8 + np.random.rand() * 4)
        else:
            volumes.append(3 + np.random.rand() * 2)

    volumes = np.array(volumes)

    colors = ['blue' if h < 0 else ('green' if h < 3 else 'gray') for h in hours]
    ax3.bar(hours, volumes, color=colors, alpha=0.6, width=0.8)
    ax3.axhline(y=2.0, color='orange', linestyle='--', label='Volume anomaly (2x)')
    ax3.axvspan(-48, 0, alpha=0.1, color='blue')
    ax3.set_ylabel('Volume\nvs Baseline')
    ax3.set_xlabel('Hours from Breakout')
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 15)

    # Add annotation explaining the pattern
    ax1.text(0.02, 0.95,
             'Detection Criteria:\n'
             '1. Volume >2x baseline, price flat\n'
             '2. BAR (Bid/Ask) >3x baseline\n'
             '3. Pattern persists 2+ hours\n'
             '4. Enter when breakout confirms (>3% move)',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/Users/bz/Pythia2/docs/accumulation_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: accumulation_pattern.png")


if __name__ == '__main__':
    print("Generating strategy visualization charts...")
    create_breakout_pattern_chart()
    create_profit_lock_chart()
    create_trailing_stop_chart()
    create_accumulation_pattern_chart()
    print("\nAll charts generated successfully!")
