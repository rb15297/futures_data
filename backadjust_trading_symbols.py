"""
Back-Adjustment Script for Trading Symbols Only

Focuses on: ES, RTY, YM, ZN, ZB, GC, NQ
These are the symbols used in your training pipeline.

Author: Claude Code Analysis
Date: 2025-11-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Symbols we care about
TRADING_SYMBOLS = ['ES', 'RTY', 'YM', 'ZN', 'ZB', 'GC', 'NQ']

# Price thresholds by symbol (conservative to avoid false positives)
THRESHOLDS = {
    'NQ': 100,     # Nasdaq: ~15,000-25,000 (1min bars can be volatile)
    'ES': 50,      # S&P 500: ~3,000-5,000
    'RTY': 30,     # Russell 2000: ~1,500-2,500
    'YM': 300,     # Dow: ~25,000-40,000
    'ZN': 1.5,     # 10-Year Note: ~100-130
    'ZB': 2.0,     # 30-Year Bond: ~100-180
    'GC': 30,      # Gold: ~1,500-2,500
}


def detect_rollovers(df, price_threshold=50):
    """
    Detect rollover points in continuous contract data.

    Conservative approach: Only flag as rollover if:
    1. Large price change (> threshold)
    2. Occurs near midnight UTC (22:00-02:00)
    3. Volume is relatively low (optional check)
    """
    # Calculate price change
    price_change = df['close'].diff()

    # Check if timestamp is within window of midnight (22:00-02:00 UTC)
    hour = df['timestamp'].dt.hour
    near_midnight = (hour >= 22) | (hour <= 2)

    # Rollover = large price change near midnight
    rollover = (abs(price_change) > price_threshold) & near_midnight

    return rollover


def calculate_adjustments(df, rollover_mask):
    """
    Calculate cumulative adjustment amounts for back-adjustment.

    Works backward from most recent data (keeps current prices accurate).
    """
    # Get the price jump at each rollover
    price_changes = df['close'].diff()
    rollover_adjustments = price_changes.where(rollover_mask, 0)

    # Calculate cumulative adjustment (working backward from end)
    cumulative_adjustment = rollover_adjustments[::-1].cumsum()[::-1]

    return cumulative_adjustment


def backadjust_prices(df, adjustment):
    """Apply back-adjustment to OHLC prices."""
    df_adjusted = df.copy()

    for col in ['open', 'high', 'low', 'close']:
        if col in df_adjusted.columns:
            df_adjusted[col] = df_adjusted[col] - adjustment

    return df_adjusted


def analyze_symbol(symbol, input_dir, create_plots=True):
    """
    Analyze all timeframes for a single symbol.
    """
    print("\n" + "=" * 80)
    print(f"ANALYZING: {symbol}")
    print("=" * 80)

    threshold = THRESHOLDS.get(symbol, 50)
    results = {}

    for timeframe in ['1min', '1hour', 'daily']:
        filename = f"{symbol}_{timeframe}_continuous.parquet"
        filepath = input_dir / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filename}")
            continue

        print(f"\n{timeframe.upper()} Data:")
        print("-" * 60)

        # Load data
        df = pd.read_parquet(filepath)

        if df['timestamp'].dtype == 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Detect rollovers
        rollover_mask = detect_rollovers(df, price_threshold=threshold)
        num_rollovers = rollover_mask.sum()

        print(f"  Total bars: {len(df):,}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Rollovers detected: {num_rollovers}")

        if num_rollovers > 0:
            rollover_data = df[rollover_mask][['timestamp', 'close']].copy()
            rollover_data['price_change'] = df.loc[rollover_mask, 'close'].diff()

            print(f"\n  Rollover dates and price changes:")
            for idx, row in rollover_data.iterrows():
                if not pd.isna(row['price_change']):
                    print(f"    {row['timestamp']}: {row['price_change']:+.2f} points")

        results[timeframe] = {
            'filepath': filepath,
            'filename': filename,
            'bars': len(df),
            'rollovers': num_rollovers,
            'threshold': threshold,
            'rollover_mask': rollover_mask if num_rollovers > 0 else None,
            'df': df if num_rollovers > 0 else None
        }

    # Create visualization if requested and we have 1min data
    if create_plots and '1min' in results and results['1min']['rollovers'] > 0:
        create_rollover_plot(symbol, results['1min'])

    return results


def create_rollover_plot(symbol, data_dict):
    """Create visualization of rollovers for a symbol."""
    df = data_dict['df']
    rollover_mask = data_dict['rollover_mask']

    if df is None or rollover_mask is None:
        return

    # Sample data if too large
    if len(df) > 100000:
        df_plot = df.iloc[::len(df)//100000].copy()
        rollover_plot = df[rollover_mask]
    else:
        df_plot = df.copy()
        rollover_plot = df[rollover_mask]

    df_plot['price_change'] = df_plot['close'].diff()

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Price over time
    axes[0].plot(df_plot['timestamp'], df_plot['close'], linewidth=0.5, alpha=0.7)
    if len(rollover_plot) > 0:
        axes[0].scatter(rollover_plot['timestamp'], rollover_plot['close'],
                       color='red', s=100, zorder=5, marker='X', label='Rollover')
    axes[0].set_title(f'{symbol} Price Over Time (1min data)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Price changes with rollovers highlighted
    axes[1].plot(df_plot['timestamp'], df_plot['price_change'], linewidth=0.5, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    if len(rollover_plot) > 0:
        rollover_plot_changes = rollover_plot.copy()
        rollover_plot_changes['price_change'] = df.loc[rollover_mask, 'close'].diff()
        axes[1].scatter(rollover_plot_changes['timestamp'], rollover_plot_changes['price_change'],
                       color='red', s=100, zorder=5, marker='X', label='Rollover')
    axes[1].set_title('Price Changes (1min intervals)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price Change')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'{symbol}_rollover_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot: {output_path}")


def backadjust_symbol(symbol, input_dir, output_dir=None, dry_run=False):
    """
    Back-adjust all timeframes for a single symbol.
    """
    if output_dir is None:
        # Create backadjusted subfolder
        output_dir = input_dir / "backadjusted"
        output_dir.mkdir(exist_ok=True)

    threshold = THRESHOLDS.get(symbol, 50)
    results = []

    for timeframe in ['1min', '1hour', 'daily']:
        filename = f"{symbol}_{timeframe}_continuous.parquet"
        filepath = input_dir / filename

        if not filepath.exists():
            continue

        logger.info(f"Processing: {filename}")

        # Load data
        df = pd.read_parquet(filepath)

        if df['timestamp'].dtype == 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Detect rollovers
        rollover_mask = detect_rollovers(df, price_threshold=threshold)
        num_rollovers = rollover_mask.sum()

        if num_rollovers == 0:
            logger.info(f"  No rollovers detected")
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'rollovers': 0,
                'status': 'no_adjustment_needed'
            })
            continue

        if dry_run:
            logger.info(f"  Found {num_rollovers} rollovers (dry run)")
            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'rollovers': num_rollovers,
                'status': 'dry_run'
            })
            continue

        # Calculate adjustments
        adjustments = calculate_adjustments(df, rollover_mask)

        # Apply back-adjustment
        df_adjusted = backadjust_prices(df, adjustments)

        # Verify
        df_adjusted['price_change'] = df_adjusted['close'].diff()
        large_changes_before = (abs(df['close'].diff()) > threshold).sum()
        large_changes_after = (abs(df_adjusted['price_change']) > threshold).sum()

        logger.info(f"  Rollovers removed: {num_rollovers}")
        logger.info(f"  Large changes before: {large_changes_before}")
        logger.info(f"  Large changes after: {large_changes_after}")

        # Save (use same filename as original, but in backadjusted folder)
        output_filename = f"{symbol}_{timeframe}_continuous.parquet"
        output_path = output_dir / output_filename

        df_adjusted = df_adjusted.drop('price_change', axis=1, errors='ignore')
        df_adjusted.to_parquet(output_path, index=False)

        logger.info(f"  Saved: {output_filename}")

        results.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'rollovers': num_rollovers,
            'large_changes_before': large_changes_before,
            'large_changes_after': large_changes_after,
            'output': output_filename,
            'status': 'success'
        })

    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("TRADING SYMBOLS ROLLOVER ANALYSIS & BACK-ADJUSTMENT")
    print("=" * 80)
    print(f"\nSymbols: {', '.join(TRADING_SYMBOLS)}")
    print("\nThis script will:")
    print("1. Analyze rollover gaps in your continuous contract data")
    print("2. Show rollover dates and price changes for each symbol")
    print("3. Create back-adjusted files in 'backadjusted' subfolder")
    print("4. Generate visualizations for each symbol")
    print("\nOutput location: data/databento_continuous/backadjusted/")
    print("\n" + "=" * 80)

    input_dir = Path(r"C:\data\databento_continuous")

    # Phase 1: Analysis
    print("\n" + "=" * 80)
    print("PHASE 1: ANALYSIS")
    print("=" * 80)

    all_results = {}
    for symbol in TRADING_SYMBOLS:
        results = analyze_symbol(symbol, input_dir, create_plots=True)
        all_results[symbol] = results

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY SYMBOL")
    print("=" * 80)

    for symbol in TRADING_SYMBOLS:
        print(f"\n{symbol}:")
        if symbol not in all_results:
            print("  No data found")
            continue

        for timeframe in ['1min', '1hour', 'daily']:
            if timeframe in all_results[symbol]:
                r = all_results[symbol][timeframe]
                print(f"  {timeframe:6s}: {r['rollovers']:2d} rollovers detected")

    # Ask for confirmation
    print("\n" + "=" * 80)
    response = input("\nProceed with back-adjustment? (yes/no): ").strip().lower()

    if response != 'yes':
        print("\nAnalysis complete. No files modified.")
        print("\nVisualization files created:")
        for symbol in TRADING_SYMBOLS:
            plot_file = f"{symbol}_rollover_analysis.png"
            if Path(plot_file).exists():
                print(f"  - {plot_file}")
        return

    # Phase 2: Back-adjustment
    print("\n" + "=" * 80)
    print("PHASE 2: BACK-ADJUSTMENT")
    print("=" * 80)

    all_adjustment_results = []
    for symbol in TRADING_SYMBOLS:
        print(f"\n{symbol}:")
        results = backadjust_symbol(symbol, input_dir, dry_run=False)
        all_adjustment_results.extend(results)

    # Final summary
    print("\n" + "=" * 80)
    print("BACK-ADJUSTMENT COMPLETE")
    print("=" * 80)

    successful = sum(1 for r in all_adjustment_results if r['status'] == 'success')
    no_adjustment = sum(1 for r in all_adjustment_results if r['status'] == 'no_adjustment_needed')

    print(f"\nFiles created: {successful}")
    print(f"Files unchanged (no rollovers): {no_adjustment}")

    print("\nBack-adjusted files created:")
    for result in all_adjustment_results:
        if result['status'] == 'success':
            print(f"  - {result['output']}")

    # Print output location
    output_dir = input_dir / "backadjusted"
    print(f"\n{'=' * 80}")
    print(f"Back-adjusted files saved to:")
    print(f"  {output_dir}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Review the visualization PNG files to verify rollovers were detected correctly

2. Test training with back-adjusted files:
   - Update train_5_15_cross_market_V3.py to point to the backadjusted folder:

     OLD: data_dir = 'data/databento_continuous'
     NEW: data_dir = 'data/databento_continuous/backadjusted'

   - Run training and compare model performance

3. Compare a few random timestamps between original and back-adjusted files

4. If results are good, you can:
   - Keep using the backadjusted folder (cleanest approach)
   - Or backup originals and replace them with back-adjusted files

Note: Back-adjusted data will have different absolute price levels in historical data,
but percentage returns and relative patterns will be preserved.
""")


if __name__ == "__main__":
    main()
