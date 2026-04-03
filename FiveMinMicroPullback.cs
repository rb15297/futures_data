#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    /// <summary>
    /// 5-Min Micro Pullback with Trailing Stop
    /// -----------------------------------------
    /// 1. Resample to 5-min bars (RTH 9:30-16:00 ET)
    /// 2. Find 3 consecutive bullish (or bearish) 5-min bars
    /// 3. Wait for a pullback bar (close opposite to trend)
    /// 4. Enter at pullback close, stop at pullback low/high
    /// 5. Trail stop once price moves 1x stop distance in favor
    /// 6. Trail tightens to 0.3x stop distance from best price
    /// 7. Force flat at 15:55 ET (EOD safety)
    /// </summary>
    public class FiveMinMicroPullback : Strategy
    {
        #region Private Variables

        // 5-min bar tracking
        private int bullCount;
        private int bearCount;
        private bool hadBullTrend;
        private bool hadBearTrend;

        // Trade management
        private double entryPrice;
        private double stopPrice;
        private double bestPrice;
        private double stopDist;
        private bool trailActive;

        // Series for 5-min data
        private PriceBars fiveMinBars;

        #endregion

        #region Properties

        [NinjaScriptProperty]
        [Range(2, 5)]
        [Display(Name = "Consecutive Trend Bars", Description = "Number of consecutive bullish/bearish 5-min bars to define trend", Order = 1, GroupName = "Strategy Parameters")]
        public int ConsecutiveBars { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 5.0)]
        [Display(Name = "Trail Activation Mult", Description = "Activate trail after price moves this many x stop distance in favor", Order = 2, GroupName = "Strategy Parameters")]
        public double TrailActivationMult { get; set; }

        [NinjaScriptProperty]
        [Range(0.05, 2.0)]
        [Display(Name = "Trail Tightness Mult", Description = "Trail distance = this x stop distance from best price", Order = 3, GroupName = "Strategy Parameters")]
        public double TrailTightnessMult { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "RTH Only", Description = "Only trade during Regular Trading Hours (9:30-16:00 ET)", Order = 4, GroupName = "Strategy Parameters")]
        public bool RthOnly { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Quantity", Description = "Number of contracts to trade", Order = 5, GroupName = "Strategy Parameters")]
        public int Quantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Flatten Time", Description = "Force flat at this time (HHmmss ET)", Order = 6, GroupName = "Strategy Parameters")]
        public int EodFlattenTime { get; set; }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = "5-Min Micro Pullback with Trailing Stop - Enters on pullback after 3 consecutive trend bars, manages with custom trailing stop.";
                Name                        = "FiveMinMicroPullback";
                Calculate                   = Calculate.OnEachTick;
                EntriesPerDirection          = 1;
                EntryHandling               = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = false;  // We handle EOD ourselves
                ExitOnSessionCloseSeconds    = 30;
                IsFillLimitOnTouch           = false;
                MaximumBarsLookBack          = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution          = OrderFillResolution.Standard;
                Slippage                     = 1;
                StartBehavior                = StartBehavior.WaitUntilFlat;
                TimeInForce                  = TimeInForce.Gtc;
                TraceOrders                  = true;
                RealtimeErrorHandling        = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling           = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade          = 5;
                IsInstantiatedOnEachOptimizationIteration = true;

                // Default parameter values (Balanced config from backtest)
                ConsecutiveBars     = 3;
                TrailActivationMult = 1.0;
                TrailTightnessMult  = 0.3;
                RthOnly             = true;
                Quantity            = 1;
                EodFlattenTime      = 155500;  // 3:55 PM ET
            }
            else if (State == State.Configure)
            {
                // Add 5-minute data series for trend detection
                AddDataSeries(BarsPeriodType.Minute, 5);
            }
            else if (State == State.DataLoaded)
            {
                ResetState();
            }
        }

        private void ResetState()
        {
            bullCount     = 0;
            bearCount     = 0;
            hadBullTrend  = false;
            hadBearTrend  = false;
            entryPrice    = 0;
            stopPrice     = 0;
            bestPrice     = 0;
            stopDist      = 0;
            trailActive   = false;
        }

        protected override void OnBarUpdate()
        {
            // ================================================================
            // BarsInProgress == 1 => 5-minute bar update (trend detection)
            // BarsInProgress == 0 => Primary bar update (trade management)
            // ================================================================

            if (BarsInProgress == 1)
            {
                On5MinBarUpdate();
                return;
            }

            if (BarsInProgress != 0)
                return;

            // Need enough bars on both series
            if (CurrentBars[0] < BarsRequiredToTrade || CurrentBars[1] < BarsRequiredToTrade)
                return;

            // === RTH FILTER ===
            bool isRTH = IsInRegularTradingHours();

            // === EOD FLATTEN ===
            if (RthOnly && Position.MarketPosition != MarketPosition.Flat)
            {
                TimeSpan currentTime = Times[0][0].TimeOfDay;
                TimeSpan flattenTime = TimeSpan.ParseExact(
                    EodFlattenTime.ToString("D6"), "hhmmss", null);

                if (currentTime >= flattenTime)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                    {
                        ExitLong("EOD Flat", "Long Entry");
                        Print(Times[0][0] + " | EOD FLATTEN - Closing Long");
                    }
                    else if (Position.MarketPosition == MarketPosition.Short)
                    {
                        ExitShort("EOD Flat", "Short Entry");
                        Print(Times[0][0] + " | EOD FLATTEN - Closing Short");
                    }
                    return;
                }
            }

            // === TRAILING STOP MANAGEMENT (runs every tick) ===
            ManageTrailingStop();

            // === ENTRY SIGNALS (only when flat and in session) ===
            if (Position.MarketPosition == MarketPosition.Flat && isRTH)
            {
                CheckEntrySignals();
            }
        }

        /// <summary>
        /// Called on each 5-minute bar close. Tracks consecutive bullish/bearish bars
        /// and detects pullback conditions.
        /// </summary>
        private void On5MinBarUpdate()
        {
            if (CurrentBars[1] < 1)
                return;

            double open5  = Opens[1][0];
            double close5 = Closes[1][0];
            double high5  = Highs[1][0];
            double low5   = Lows[1][0];

            bool isBull = close5 > open5;
            bool isBear = close5 < open5;

            // Track consecutive bars
            bullCount = isBull ? bullCount + 1 : 0;
            bearCount = isBear ? bearCount + 1 : 0;

            // Detect trend phase
            if (bullCount >= ConsecutiveBars)
            {
                hadBullTrend = true;
                hadBearTrend = false;
            }
            if (bearCount >= ConsecutiveBars)
            {
                hadBearTrend = true;
                hadBullTrend = false;
            }

            // Detect pullback (close opposite to trend, means streak just broke)
            bool bullPullback = hadBullTrend && isBear && bullCount == 0;
            bool bearPullback = hadBearTrend && isBull && bearCount == 0;

            bool isRTH = IsInRegularTradingHours();

            // === ENTRY ON PULLBACK ===
            if (bullPullback && (RthOnly ? isRTH : true) && Position.MarketPosition == MarketPosition.Flat)
            {
                entryPrice  = close5;
                stopPrice   = low5;
                stopDist    = Math.Abs(close5 - low5);
                bestPrice   = close5;
                trailActive = false;
                hadBullTrend = false;

                EnterLong(0, Quantity, "Long Entry");

                Print(Times[1][0] + " | LONG ENTRY @ " + close5.ToString("F2")
                    + " | Stop: " + low5.ToString("F2")
                    + " | StopDist: " + stopDist.ToString("F2"));

                Draw.ArrowUp(this, "LongEntry" + CurrentBars[1], false, 0, low5 - 4 * TickSize, Brushes.Lime);
            }

            if (bearPullback && (RthOnly ? isRTH : true) && Position.MarketPosition == MarketPosition.Flat)
            {
                entryPrice  = close5;
                stopPrice   = high5;
                stopDist    = Math.Abs(high5 - close5);
                bestPrice   = close5;
                trailActive = false;
                hadBearTrend = false;

                EnterShort(0, Quantity, "Short Entry");

                Print(Times[1][0] + " | SHORT ENTRY @ " + close5.ToString("F2")
                    + " | Stop: " + high5.ToString("F2")
                    + " | StopDist: " + stopDist.ToString("F2"));

                Draw.ArrowDown(this, "ShortEntry" + CurrentBars[1], false, 0, high5 + 4 * TickSize, Brushes.Red);
            }
        }

        /// <summary>
        /// Check for entry signals on the primary (tick) data series.
        /// This is a backup — primary entries happen in On5MinBarUpdate.
        /// </summary>
        private void CheckEntrySignals()
        {
            // Entries are handled in On5MinBarUpdate to align with 5-min bar closes.
            // This method is reserved for any additional primary-series entry logic.
        }

        /// <summary>
        /// Manages the trailing stop on every tick of the primary data series.
        /// </summary>
        private void ManageTrailingStop()
        {
            if (stopDist <= 0)
                return;

            // === LONG POSITION ===
            if (Position.MarketPosition == MarketPosition.Long)
            {
                // Update best price seen
                if (Highs[0][0] > bestPrice)
                    bestPrice = Highs[0][0];

                // Activate trail once price moves 1x stop distance in favor
                if (!trailActive && (bestPrice - entryPrice) >= TrailActivationMult * stopDist)
                {
                    trailActive = true;
                    Print(Times[0][0] + " | TRAIL ACTIVATED (Long) | BestPrice: " + bestPrice.ToString("F2"));
                }

                // Tighten trail
                if (trailActive)
                {
                    double newStop = bestPrice - TrailTightnessMult * stopDist;
                    if (newStop > stopPrice)
                    {
                        stopPrice = newStop;
                    }
                }

                // Check if stop hit
                if (Lows[0][0] <= stopPrice)
                {
                    ExitLong("Trail Stop", "Long Entry");
                    Print(Times[0][0] + " | LONG EXIT (Trail Stop) @ " + stopPrice.ToString("F2")
                        + " | P&L: " + (stopPrice - entryPrice).ToString("F2") + " pts");
                }
            }

            // === SHORT POSITION ===
            if (Position.MarketPosition == MarketPosition.Short)
            {
                // Update best price seen
                if (Lows[0][0] < bestPrice || bestPrice == 0)
                    bestPrice = Lows[0][0];

                // Activate trail
                if (!trailActive && (entryPrice - bestPrice) >= TrailActivationMult * stopDist)
                {
                    trailActive = true;
                    Print(Times[0][0] + " | TRAIL ACTIVATED (Short) | BestPrice: " + bestPrice.ToString("F2"));
                }

                // Tighten trail
                if (trailActive)
                {
                    double newStop = bestPrice + TrailTightnessMult * stopDist;
                    if (newStop < stopPrice)
                    {
                        stopPrice = newStop;
                    }
                }

                // Check if stop hit
                if (Highs[0][0] >= stopPrice)
                {
                    ExitShort("Trail Stop", "Short Entry");
                    Print(Times[0][0] + " | SHORT EXIT (Trail Stop) @ " + stopPrice.ToString("F2")
                        + " | P&L: " + (entryPrice - stopPrice).ToString("F2") + " pts");
                }
            }
        }

        /// <summary>
        /// Check if current bar time falls within RTH (9:30-16:00 ET)
        /// </summary>
        private bool IsInRegularTradingHours()
        {
            // Use primary series time
            DateTime barTime = Times[0][0];
            TimeSpan time = barTime.TimeOfDay;
            TimeSpan rthOpen  = new TimeSpan(9, 30, 0);
            TimeSpan rthClose = new TimeSpan(16, 0, 0);
            return time >= rthOpen && time < rthClose;
        }

        /// <summary>
        /// Reset trade state when a position is closed
        /// </summary>
        protected override void OnExecutionUpdate(Execution execution, string executionId,
            double price, int quantity, MarketPosition marketPosition,
            string orderId, DateTime time)
        {
            // When we go flat, reset trail state for next trade
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                trailActive = false;
                bestPrice   = 0;
                stopDist    = 0;
            }
        }

        #region Plot

        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            // Draw stop level on chart when in a position
            if (Position.MarketPosition != MarketPosition.Flat && stopPrice > 0)
            {
                int x1 = ChartBars.GetBarIdxByTime(chartControl, Times[0][0].AddMinutes(-30));
                int x2 = ChartBars.GetBarIdxByTime(chartControl, Times[0][0]);

                double y = chartScale.GetYByValue(stopPrice);

                SharpDX.Direct2D1.Brush stopBrush = Position.MarketPosition == MarketPosition.Long
                    ? Brushes.Lime.ToDxBrush(RenderTarget)
                    : Brushes.Red.ToDxBrush(RenderTarget);

                RenderTarget.DrawLine(
                    new SharpDX.Vector2(chartControl.GetXByBarIndex(ChartBars, x1), (float)y),
                    new SharpDX.Vector2(chartControl.GetXByBarIndex(ChartBars, x2), (float)y),
                    stopBrush,
                    2);

                stopBrush.Dispose();
            }
        }

        #endregion
    }
}
