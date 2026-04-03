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
    /// 5-Min Micro Pullback — Ultra-Selective Config
    /// -----------------------------------------------
    /// Matches the Python backtest logic exactly:
    /// 
    /// 1. Add a secondary 5-minute data series for pattern detection
    /// 2. Track consecutive bullish/bearish 5-min bars (default 3)
    /// 3. On pullback bar, verify:
    ///    a) Pullback close remains above run start (healthy pullback)
    ///    b) Stop distance is within MinStop..MaxStop range
    ///    c) Haven't exceeded MaxTradesPerDay
    /// 4. Enter at pullback bar close with stop at pullback low/high ± 1 pt buffer
    /// 5. Trail stop activates after 0.5x stop distance move in favor
    /// 6. Trail tightens to 0.5x stop distance from best price
    /// 7. Hard target at RRTarget × stop distance (3R default)
    /// 8. Time exit after MaxHoldBars primary bars
    /// 9. Force flat at EOD flatten time
    ///
    /// Ultra-Selective defaults: trail_activation=0.5, trail_distance=0.5
    /// </summary>
    public class FiveMinMicroPullbackUltra : Strategy
    {
        #region Private Variables

        // 5-min bar tracking — store close values to check run and pullback health
        private List<double> fiveMinCloses;
        private List<double> fiveMinOpens;
        private List<double> fiveMinHighs;
        private List<double> fiveMinLows;
        private int bullCount;
        private int bearCount;

        // Trade management
        private double entryPrice;
        private double stopPrice;
        private double targetPrice;
        private double bestPrice;
        private double stopDist;
        private bool trailActive;
        private int barsInTrade;
        private int tradesToday;
        private DateTime lastTradeDate;

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
        [Range(0.0, 100.0)]
        [Display(Name = "Min Stop (pts)", Description = "Minimum stop distance in points. Rejects setups with tighter stops.", Order = 4, GroupName = "Strategy Parameters")]
        public double MinStop { get; set; }

        [NinjaScriptProperty]
        [Range(1.0, 200.0)]
        [Display(Name = "Max Stop (pts)", Description = "Maximum stop distance in points. Rejects setups with wider stops.", Order = 5, GroupName = "Strategy Parameters")]
        public double MaxStop { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 10.0)]
        [Display(Name = "R:R Target", Description = "Take profit at this multiple of stop distance", Order = 6, GroupName = "Strategy Parameters")]
        public double RRTarget { get; set; }

        [NinjaScriptProperty]
        [Range(1, 500)]
        [Display(Name = "Max Hold Bars", Description = "Force exit after this many primary-series bars in trade", Order = 7, GroupName = "Strategy Parameters")]
        public int MaxHoldBars { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Max Trades Per Day", Description = "Maximum number of entries per trading day", Order = 8, GroupName = "Strategy Parameters")]
        public int MaxTradesPerDay { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 10.0)]
        [Display(Name = "Stop Buffer (pts)", Description = "Extra points added beyond pullback low/high for stop", Order = 9, GroupName = "Strategy Parameters")]
        public double StopBuffer { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "RTH Only", Description = "Only trade during Regular Trading Hours (9:30-16:00 ET)", Order = 10, GroupName = "Strategy Parameters")]
        public bool RthOnly { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Quantity", Description = "Number of contracts to trade", Order = 11, GroupName = "Strategy Parameters")]
        public int Quantity { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "EOD Flatten Time", Description = "Force flat at this time (HHmmss ET)", Order = 12, GroupName = "Strategy Parameters")]
        public int EodFlattenTime { get; set; }

        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description                 = "5-Min Micro Pullback Ultra-Selective — Matches Python backtest logic with pullback health check, stop size filter, daily trade cap, R:R target, and max hold bars.";
                Name                        = "FiveMinMicroPullbackUltra";
                Calculate                   = Calculate.OnBarClose;
                EntriesPerDirection          = 1;
                EntryHandling               = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds    = 300;  // 5 min before session close
                IsFillLimitOnTouch           = false;
                MaximumBarsLookBack          = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution          = OrderFillResolution.Standard;
                Slippage                     = 1;
                StartBehavior                = StartBehavior.WaitUntilFlat;
                TimeInForce                  = TimeInForce.Gtc;
                TraceOrders                  = true;
                RealtimeErrorHandling        = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling           = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade          = 20;
                IsInstantiatedOnEachOptimizationIteration = true;

                // Ultra-Selective defaults (from Python backtest)
                ConsecutiveBars     = 3;
                TrailActivationMult = 0.5;
                TrailTightnessMult  = 0.5;
                MinStop             = 3.0;
                MaxStop             = 25.0;
                RRTarget            = 3.0;
                MaxHoldBars         = 120;
                MaxTradesPerDay     = 2;
                StopBuffer          = 1.0;
                RthOnly             = true;
                Quantity            = 1;
                EodFlattenTime      = 155500;  // 3:55 PM ET
            }
            else if (State == State.Configure)
            {
                // Add 5-minute data series for pattern detection
                AddDataSeries(BarsPeriodType.Minute, 5);
            }
            else if (State == State.DataLoaded)
            {
                fiveMinCloses = new List<double>();
                fiveMinOpens  = new List<double>();
                fiveMinHighs  = new List<double>();
                fiveMinLows   = new List<double>();
                ResetTradeState();
                tradesToday   = 0;
                lastTradeDate = DateTime.MinValue;
            }
        }

        private void ResetTradeState()
        {
            entryPrice  = 0;
            stopPrice   = 0;
            targetPrice = 0;
            bestPrice   = 0;
            stopDist    = 0;
            trailActive = false;
            barsInTrade = 0;
        }

        protected override void OnBarUpdate()
        {
            // ================================================================
            // BarsInProgress == 1 => 5-minute bar close (pattern detection + entry)
            // BarsInProgress == 0 => Primary 1-min bar close (trade management)
            // ================================================================

            if (BarsInProgress == 1)
            {
                On5MinBarUpdate();
                return;
            }

            if (BarsInProgress != 0)
                return;

            // Need enough bars on both series
            if (CurrentBars[0] < BarsRequiredToTrade || CurrentBars[1] < ConsecutiveBars + 2)
                return;

            // === RESET DAILY TRADE COUNTER ===
            DateTime today = Times[0][0].Date;
            if (today != lastTradeDate)
            {
                tradesToday   = 0;
                lastTradeDate = today;
            }

            // === EOD FLATTEN ===
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                TimeSpan currentTime = Times[0][0].TimeOfDay;
                int hh = EodFlattenTime / 10000;
                int mm = (EodFlattenTime / 100) % 100;
                int ss = EodFlattenTime % 100;
                TimeSpan flattenTime = new TimeSpan(hh, mm, ss);

                if (currentTime >= flattenTime)
                {
                    FlattenPosition("EOD Flat");
                    return;
                }
            }

            // === TRADE MANAGEMENT (every primary bar) ===
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                barsInTrade++;
                ManagePosition();
            }
        }

        /// <summary>
        /// Called on each 5-minute bar close.
        /// Tracks consecutive bullish/bearish bars, detects pullback,
        /// verifies health check, and enters.
        /// </summary>
        private void On5MinBarUpdate()
        {
            if (CurrentBars[1] < 1)
                return;

            double open5  = Opens[1][0];
            double close5 = Closes[1][0];
            double high5  = Highs[1][0];
            double low5   = Lows[1][0];

            // Store bar data for health check lookback
            fiveMinCloses.Add(close5);
            fiveMinOpens.Add(open5);
            fiveMinHighs.Add(high5);
            fiveMinLows.Add(low5);

            // Keep buffer manageable
            if (fiveMinCloses.Count > 100)
            {
                fiveMinCloses.RemoveAt(0);
                fiveMinOpens.RemoveAt(0);
                fiveMinHighs.RemoveAt(0);
                fiveMinLows.RemoveAt(0);
            }

            int idx = fiveMinCloses.Count - 1;

            bool isBull = close5 > open5;
            bool isBear = close5 < open5;

            // Track consecutive trend bars
            bullCount = isBull ? bullCount + 1 : 0;
            bearCount = isBear ? bearCount + 1 : 0;

            // Not enough history for lookback
            if (idx < ConsecutiveBars + 1)
                return;

            // Only enter when flat, within daily limit, and in session
            if (Position.MarketPosition != MarketPosition.Flat)
                return;
            if (tradesToday >= MaxTradesPerDay)
                return;
            if (RthOnly && !IsInRegularTradingHours())
                return;

            // ============================================================
            // BULLISH PULLBACK DETECTION
            // ============================================================
            // Previous N bars were all bullish, current bar is bearish (pullback)
            if (isBear && idx >= ConsecutiveBars)
            {
                // Check that the previous ConsecutiveBars bars were all bullish
                bool allBull = true;
                for (int k = idx - ConsecutiveBars; k < idx; k++)
                {
                    if (fiveMinCloses[k] <= fiveMinOpens[k])
                    {
                        allBull = false;
                        break;
                    }
                }

                if (allBull)
                {
                    // HEALTH CHECK: pullback close must be above the run start close
                    double runStartClose = fiveMinCloses[idx - ConsecutiveBars];
                    if (close5 > runStartClose)
                    {
                        double stop = low5 - StopBuffer;
                        double dist = close5 - stop;

                        // STOP SIZE FILTER
                        if (dist > MinStop && dist < MaxStop)
                        {
                            entryPrice  = close5;
                            stopPrice   = stop;
                            stopDist    = dist;
                            targetPrice = close5 + dist * RRTarget;
                            bestPrice   = close5;
                            trailActive = false;
                            barsInTrade = 0;

                            EnterLong(0, Quantity, "Long Entry");
                            tradesToday++;

                            Print(Times[1][0] + " | LONG ENTRY @ " + close5.ToString("F2")
                                + " | Stop: " + stop.ToString("F2")
                                + " | Target: " + targetPrice.ToString("F2")
                                + " | StopDist: " + dist.ToString("F2")
                                + " | RunStart: " + runStartClose.ToString("F2"));
                        }
                    }
                }
            }

            // ============================================================
            // BEARISH PULLBACK DETECTION
            // ============================================================
            // Previous N bars were all bearish, current bar is bullish (pullback)
            if (isBull && idx >= ConsecutiveBars)
            {
                bool allBear = true;
                for (int k = idx - ConsecutiveBars; k < idx; k++)
                {
                    if (fiveMinCloses[k] >= fiveMinOpens[k])
                    {
                        allBear = false;
                        break;
                    }
                }

                if (allBear)
                {
                    // HEALTH CHECK: pullback close must be below the run start close
                    double runStartClose = fiveMinCloses[idx - ConsecutiveBars];
                    if (close5 < runStartClose)
                    {
                        double stop = high5 + StopBuffer;
                        double dist = stop - close5;

                        // STOP SIZE FILTER
                        if (dist > MinStop && dist < MaxStop)
                        {
                            entryPrice  = close5;
                            stopPrice   = stop;
                            stopDist    = dist;
                            targetPrice = close5 - dist * RRTarget;
                            bestPrice   = close5;
                            trailActive = false;
                            barsInTrade = 0;

                            EnterShort(0, Quantity, "Short Entry");
                            tradesToday++;

                            Print(Times[1][0] + " | SHORT ENTRY @ " + close5.ToString("F2")
                                + " | Stop: " + stop.ToString("F2")
                                + " | Target: " + targetPrice.ToString("F2")
                                + " | StopDist: " + dist.ToString("F2")
                                + " | RunStart: " + runStartClose.ToString("F2"));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Manages open position: trailing stop, target exit, max hold exit.
        /// Runs on every primary (1-min) bar close.
        /// </summary>
        private void ManagePosition()
        {
            if (stopDist <= 0)
                return;

            // === MAX HOLD BARS EXIT ===
            if (barsInTrade >= MaxHoldBars)
            {
                FlattenPosition("Max Hold");
                return;
            }

            // === LONG POSITION ===
            if (Position.MarketPosition == MarketPosition.Long)
            {
                double barHigh = Highs[0][0];
                double barLow  = Lows[0][0];

                // Check initial stop (conservative: check low first)
                if (barLow <= stopPrice)
                {
                    ExitLong("Stop Hit", "Long Entry");
                    Print(Times[0][0] + " | LONG EXIT (Stop) @ " + stopPrice.ToString("F2")
                        + " | P&L: " + (stopPrice - entryPrice).ToString("F2") + " pts");
                    return;
                }

                // Check target
                if (barHigh >= targetPrice)
                {
                    ExitLong("Target Hit", "Long Entry");
                    Print(Times[0][0] + " | LONG EXIT (Target) @ " + targetPrice.ToString("F2")
                        + " | P&L: " + (targetPrice - entryPrice).ToString("F2") + " pts");
                    return;
                }

                // Update best price
                if (barHigh > bestPrice)
                    bestPrice = barHigh;

                // Trail activation
                if (!trailActive && (bestPrice - entryPrice) >= TrailActivationMult * stopDist)
                {
                    trailActive = true;
                    Print(Times[0][0] + " | TRAIL ACTIVATED (Long) | Best: " + bestPrice.ToString("F2"));
                }

                // Tighten trailing stop
                if (trailActive)
                {
                    double newStop = bestPrice - TrailTightnessMult * stopDist;
                    if (newStop > stopPrice)
                    {
                        stopPrice = newStop;

                        // Intra-bar check: bar low may have hit the new tighter stop
                        if (barLow <= stopPrice)
                        {
                            ExitLong("Trail Stop", "Long Entry");
                            Print(Times[0][0] + " | LONG EXIT (Trail Stop) @ " + stopPrice.ToString("F2")
                                + " | P&L: " + (stopPrice - entryPrice).ToString("F2") + " pts");
                            return;
                        }
                    }
                }
            }

            // === SHORT POSITION ===
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                double barHigh = Highs[0][0];
                double barLow  = Lows[0][0];

                // Check initial stop (conservative: check high first)
                if (barHigh >= stopPrice)
                {
                    ExitShort("Stop Hit", "Short Entry");
                    Print(Times[0][0] + " | SHORT EXIT (Stop) @ " + stopPrice.ToString("F2")
                        + " | P&L: " + (entryPrice - stopPrice).ToString("F2") + " pts");
                    return;
                }

                // Check target
                if (barLow <= targetPrice)
                {
                    ExitShort("Target Hit", "Short Entry");
                    Print(Times[0][0] + " | SHORT EXIT (Target) @ " + targetPrice.ToString("F2")
                        + " | P&L: " + (entryPrice - targetPrice).ToString("F2") + " pts");
                    return;
                }

                // Update best price
                if (barLow < bestPrice || bestPrice <= 0)
                    bestPrice = barLow;

                // Trail activation
                if (!trailActive && (entryPrice - bestPrice) >= TrailActivationMult * stopDist)
                {
                    trailActive = true;
                    Print(Times[0][0] + " | TRAIL ACTIVATED (Short) | Best: " + bestPrice.ToString("F2"));
                }

                // Tighten trailing stop
                if (trailActive)
                {
                    double newStop = bestPrice + TrailTightnessMult * stopDist;
                    if (newStop < stopPrice)
                    {
                        stopPrice = newStop;

                        // Intra-bar check: bar high may have hit the new tighter stop
                        if (barHigh >= stopPrice)
                        {
                            ExitShort("Trail Stop", "Short Entry");
                            Print(Times[0][0] + " | SHORT EXIT (Trail Stop) @ " + stopPrice.ToString("F2")
                                + " | P&L: " + (entryPrice - stopPrice).ToString("F2") + " pts");
                            return;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Flatten any open position with a given signal name.
        /// </summary>
        private void FlattenPosition(string reason)
        {
            if (Position.MarketPosition == MarketPosition.Long)
            {
                ExitLong(reason, "Long Entry");
                Print(Times[0][0] + " | LONG EXIT (" + reason + ") @ Market");
            }
            else if (Position.MarketPosition == MarketPosition.Short)
            {
                ExitShort(reason, "Short Entry");
                Print(Times[0][0] + " | SHORT EXIT (" + reason + ") @ Market");
            }
        }

        /// <summary>
        /// Check if current bar time falls within RTH (9:30-16:00 ET)
        /// </summary>
        private bool IsInRegularTradingHours()
        {
            DateTime barTime = Times[0].Count > 0 ? Times[0][0] : Times[1][0];
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
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                ResetTradeState();
            }
        }
    }
}
