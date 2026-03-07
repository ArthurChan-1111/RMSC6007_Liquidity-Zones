# ============================================================
# Liquidity Zones (Zone-based) + Liquidity Sweeps (Event-based)
# Works with Yahoo Finance daily data (e.g., TSLA)
# ============================================================

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import datetime as dt

# Optional: data download (Yahoo Finance)
try:
    import yfinance as yf
except ImportError:
    yf = None


# -----------------------------
# 0) Data fetch helper (Yahoo)
# -----------------------------
def fetch_yahoo_daily(
    symbol: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches daily OHLCV data from Yahoo Finance using yfinance.
    Returns a DataFrame with columns: [open, high, low, close, volume]

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. 'TSLA').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    """
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    if end is None:
        end = dt.date.today().isoformat()

    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol or date range.")

    # Flatten MultiIndex columns returned by newer yfinance versions (e.g. ('Close', 'TSLA') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]].copy()

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


# -----------------------------
# 1) Technical utilities
# -----------------------------
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range (simple moving average of True Range).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(length, min_periods=length).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


# -----------------------------
# 2) Pivot detection (symmetric)
# -----------------------------
def detect_pivots(
    df: pd.DataFrame,
    left: int = 10,
    right: int = 10,
    unique_in_window: bool = False,
) -> pd.DataFrame:
    """
    Detects pivot highs/lows:
      - Pivot High: high[i] is maximum within [i-left, i+right]
      - Pivot Low : low[i]  is minimum within [i-left, i+right]

    If unique_in_window=True:
      - require that the max (or min) is unique inside the window to reduce duplicates.
    """
    out = df.copy()
    highs = out["high"].values
    lows = out["low"].values
    n = len(out)

    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        wh = highs[i - left: i + right + 1]
        wl = lows[i - left: i + right + 1]

        is_ph = highs[i] == np.max(wh)
        is_pl = lows[i] == np.min(wl)

        if unique_in_window and is_ph:
            is_ph = (np.sum(wh == highs[i]) == 1)
        if unique_in_window and is_pl:
            is_pl = (np.sum(wl == lows[i]) == 1)

        pivot_high[i] = is_ph
        pivot_low[i] = is_pl

    out["pivot_high"] = pivot_high
    out["pivot_low"] = pivot_low
    out["pivot_high_level"] = np.where(pivot_high, out["high"].values, np.nan)
    out["pivot_low_level"] = np.where(pivot_low, out["low"].values, np.nan)
    return out


# -----------------------------
# 3) Zone structure
# -----------------------------
@dataclass
class Zone:
    top: float
    bottom: float
    touches: int
    first_idx: int     # index position in df
    last_idx: int      # updated whenever merged
    first_time: pd.Timestamp
    last_time: pd.Timestamp

    @property
    def height(self) -> float:
        return float(self.top - self.bottom)


# -----------------------------
# 4) Option A: Liquidity Zone Detector (zone-based)
# -----------------------------
def build_liquidity_zones(
    df: pd.DataFrame,
    pivot_left_right: int = 10,
    atr_len: int = 14,
    atr_merge_mult: float = 1.0,
    require_volume: bool = True,
    vol_len: int = 20,
    vol_mult: float = 1.2,
    min_touches: int = 2,
    unique_pivots: bool = False,
    use_both_high_low_pivots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs liquidity zones by clustering volume-qualified pivot highs/lows.
    - A candidate zone is expanded by merging new pivots if the resulting height <= ATR * atr_merge_mult.
    - A zone becomes "confirmed" when touches >= min_touches.

    Parameters
    ----------
    pivot_left_right : int
        Symmetric pivot strength (left=right).
    atr_len : int
        ATR length for merge threshold.
    atr_merge_mult : float
        ATR multiplier controlling max zone height during merging.
    require_volume : bool
        If True, a pivot is only valid if volume at pivot bar > SMA(volume, vol_len) * vol_mult.
    vol_len : int
        SMA length for volume baseline.
    vol_mult : float
        Volume sensitivity multiplier.
    min_touches : int
        Minimum touch count required to confirm and report a zone.
    unique_pivots : bool
        Require unique max/min in the pivot window to reduce duplicates.
    use_both_high_low_pivots : bool
        If True, consume both pivot highs and pivot lows as touches. If False, only pivot highs.

    Returns
    -------
    zones_df : pd.DataFrame
        Confirmed zones with columns:
        [zone_id (str: 'YYYY-MM-DD_to_YYYY-MM-DD'), top, bottom, height, touches, first_idx, last_idx, first_time, last_time]
    df_out : pd.DataFrame
        Original df with added columns: [atr, vol_sma, pivot_high, pivot_low, pivot_high_level, pivot_low_level, zone_id]
    """
    out = detect_pivots(df, left=pivot_left_right, right=pivot_left_right, unique_in_window=unique_pivots).copy()

    # Indicators
    out["atr"] = atr(out, length=atr_len)
    if require_volume:
        out["vol_sma"] = sma(out["volume"], vol_len)
    else:
        out["vol_sma"] = np.nan

    # Pivot price series & index positions
    pivots = []
    for i, row in enumerate(out.itertuples()):
        # Note: pivot confirmation appears at index i (already delayed by right bars in detect_pivots)
        # Volume filter (apply at the pivot bar i)
        vol_ok = True
        if require_volume:
            if np.isnan(row.vol_sma):
                vol_ok = False
            else:
                vol_ok = (row.volume > row.vol_sma * vol_mult)

        if not vol_ok:
            continue

        if row.pivot_high:
            pivots.append((i, float(row.pivot_high_level)))
        if use_both_high_low_pivots and row.pivot_low:
            pivots.append((i, float(row.pivot_low_level)))

    zones: List[Zone] = []

    def best_merge_index(price: float, i: int) -> Optional[int]:
        """
        Find zone index that yields the smallest resulting height after merging,
        while ensuring (new_height <= atr[i] * atr_merge_mult) and time gap <= max_merge_gap_days.
        """
        if np.isnan(out.at[out.index[i], "atr"]):
            return None
        thr = float(out.at[out.index[i], "atr"]) * float(atr_merge_mult)
        max_merge_gap_days = 90   # Max days since last touch to allow merging (3 months)
        best_idx = None
        best_height = np.inf
        for j, z in enumerate(zones):
            # Check time gap
            time_gap = (out.index[i] - z.last_time).days
            if time_gap > max_merge_gap_days:
                continue
            potential_top = max(z.top, price)
            potential_bot = min(z.bottom, price)
            new_h = potential_top - potential_bot
            if new_h <= thr and new_h < best_height:
                best_height = new_h
                best_idx = j
        return best_idx

    # Iterate valid pivots and build/merge zones
    for i, p in pivots:
        j = best_merge_index(p, i)
        if j is None:
            # Create new candidate zone
            zones.append(
                Zone(
                    top=p, bottom=p, touches=1,
                    first_idx=i, last_idx=i,
                    first_time=out.index[i], last_time=out.index[i]
                )
            )
        else:
            # Merge into existing zone and update
            z = zones[j]
            z.top = max(z.top, p)
            z.bottom = min(z.bottom, p)
            z.touches += 1
            z.last_idx = i
            z.last_time = out.index[i]

    # Convert to DataFrame (confirmed zones only)
    records = []
    for idx, z in enumerate(zones):
        if z.touches >= min_touches:
            # Create a unique zone_id based on the zone's start and end dates
            start_date = z.first_time.date()
            end_date = z.last_time.date()
            zone_id_str = f"{start_date}_to_{end_date}"
            records.append({
                "zone_id": zone_id_str,
                "top": z.top,
                "bottom": z.bottom,
                "height": z.height,
                "touches": z.touches,
                "first_idx": z.first_idx,
                "last_idx": z.last_idx,
                "first_time": z.first_time,
                "last_time": z.last_time
            })

    _ZONE_COLS = ["zone_id", "top", "bottom", "height", "touches", "first_idx", "last_idx", "first_time", "last_time"]
    if not records:
        zones_df = pd.DataFrame(columns=_ZONE_COLS)
    else:
        zones_df = pd.DataFrame(records)
        # Sort by last_time, but since zone_id now includes dates, sorting by last_time is still useful
        zones_df = zones_df.sort_values(["last_time", "touches", "height"], ascending=[True, False, True]).reset_index(drop=True)

    # Add zone_id column to out (df_z): for each date, list active zone_ids
    out['zone_id'] = ''
    for ts in out.index:
        active_zones = []
        for _, z in zones_df.iterrows():
            if z['first_time'] <= ts <= z['last_time']:
                active_zones.append(z['zone_id'])
        if active_zones:
            out.at[ts, 'zone_id'] = ', '.join(active_zones)

    return zones_df, out


# -----------------------------
# 5) Liquidity Sweeps (event-based, separate role)
# -----------------------------
def detect_liquidity_sweeps(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
    atr_len: int = 14,
    buffer_atr: float = 0.10,
    unique_pivots: bool = True,
) -> pd.DataFrame:
    """
    Detects liquidity sweeps relative to the *latest pivot levels*:

      - buy_side_sweep (bearish):
          high > last_pivot_high + buffer  AND  close < last_pivot_high
      - sell_side_sweep (bullish):
          low  < last_pivot_low  - buffer  AND  close > last_pivot_low

    buffer = ATR * buffer_atr

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV DataFrame.
    left : int
        Left bars for pivot detection.
    right : int
        Right bars for pivot detection.
    atr_len : int
        ATR length.
    buffer_atr : float
        Buffer multiplier for ATR.
    unique_pivots : bool
        Require unique pivots.

    Returns
    -------
    pd.DataFrame
        DataFrame with sweep detections and pivot levels.
    """
    out = detect_pivots(df, left=left, right=right, unique_in_window=unique_pivots).copy()
    out["atr"] = atr(out, length=atr_len)
    out["buffer"] = out["atr"] * float(buffer_atr)

    # Shift by `right` bars before forward-fill so that a pivot at bar i (confirmed using
    # right future bars) is only made available from bar i+right onward, preventing look-ahead bias.
    out["last_pivot_high"] = out["pivot_high_level"].shift(right).ffill()
    out["last_pivot_low"] = out["pivot_low_level"].shift(right).ffill()

    # Conditions (guard NaNs)
    cond_b = (
        (out["high"] > (out["last_pivot_high"] + out["buffer"])) &
        (out["close"] < out["last_pivot_high"])
    )
    cond_s = (
        (out["low"] < (out["last_pivot_low"] - out["buffer"])) &
        (out["close"] > out["last_pivot_low"])
    )

    out["buy_side_sweep"] = cond_b.fillna(False)
    out["sell_side_sweep"] = cond_s.fillna(False)
    out["swept_level"] = np.where(
        out["buy_side_sweep"], out["last_pivot_high"],
        np.where(out["sell_side_sweep"], out["last_pivot_low"], np.nan)
    )

    return out


# -----------------------------
# 6) Optional: tag sweeps occurring at/near zones
# -----------------------------
def tag_sweeps_at_zones(
    sweeps_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    proximity_atr_mult: float = 0.25,
    max_days_after_zone_end: int = 365,  # Allow tagging up to 1 year after zone's last_time
) -> pd.DataFrame:
    """
    Tags sweep events that occur 'inside or near' a zone.
    Near = expand zone [bottom, top] by ATR * proximity_atr_mult on both sides.
    Only tags zones where the sweep time is within the zone's active period (first_time to last_time + max_days_after_zone_end).

    Returns a copy with boolean columns:
      - sweep_near_zone
      - sweep_zone_id (zone id if matched, else NaN)

    Assumes sweeps_df contains 'atr' column and index aligned by time.
    """
    out = sweeps_df.copy()
    out["sweep_near_zone"] = False
    out["sweep_zone_id"] = pd.Series(dtype=object)

    if zones_df is None or zones_df.empty:
        return out

    # Prebuild tuples for speed: (zone_id, bottom, top, first_time, last_time)
    zones_tuples = list(zones_df[["zone_id", "bottom", "top", "first_time", "last_time"]].itertuples(index=False, name=None))

    for i, (ts, row) in enumerate(out.iterrows()):
        # Only check rows that are sweeps
        if not (row.get("buy_side_sweep", False) or row.get("sell_side_sweep", False)):
            continue
        bar_low = row["low"]
        bar_high = row["high"]
        a = row.get("atr", np.nan)
        if np.isnan(a):
            continue

        pad = a * proximity_atr_mult
        matched_zone_id = None

        for (zid, zb, zt, z_first, z_last) in zones_tuples:
            # Time check: only tag if sweep is within zone's period + buffer
            if ts < z_first or ts > z_last + pd.Timedelta(days=max_days_after_zone_end):
                continue
            # Expand zone by pad
            lo = zb - pad
            hi = zt + pad

            # If bar overlaps with expanded zone
            if (bar_high >= lo) and (bar_low <= hi):
                matched_zone_id = zid
                break

        if matched_zone_id is not None:
            out.at[ts, "sweep_near_zone"] = True
            out.at[ts, "sweep_zone_id"] = matched_zone_id

    return out


# -----------------------------
# 7) Optional plotting (quicklook)
# -----------------------------
def plot_zones_matplotlib(
    df: pd.DataFrame,
    zones_df: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    title: str = "Zones",
    save_dir: str = ".",
):
    """
    Lightweight visualization: high/low range + close price + horizontal bands for zones.
    Zones are built from intraday high/low pivots, so the high-low range is plotted
    alongside close so that zone bands align visually with actual price extremes.
    Requires matplotlib.

    Parameters
    ----------
    save_dir : str
        Directory to auto-save the chart image (PNG). Defaults to current directory.
        The file is named '<title>.png' with spaces replaced by underscores.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        return

    # Filter data by date range if provided, else use last 250 days
    if start_date and end_date:
        tail = df.loc[start_date:end_date]
    else:
        tail = df.tail(250)

    if tail.empty:
        print("No data in the specified date range.")
        return

    x = np.arange(len(tail))
    dates = tail.index

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the high-low range as a light band so zones align with actual price extremes
    ax.fill_between(x, tail["low"].values, tail["high"].values,
                    color="lightblue", alpha=0.25, label="High-Low range")

    # Plot close price on top
    ax.plot(x, tail["close"].values, label="Close", color="black", linewidth=1)

    # Plot pivot markers if available in df
    if "pivot_high_level" in tail.columns:
        ph_mask = ~tail["pivot_high_level"].isna()
        ax.scatter(x[ph_mask], tail["pivot_high_level"].values[ph_mask],
                   marker="v", color="red", s=30, zorder=5, label="Pivot High")
    if "pivot_low_level" in tail.columns:
        pl_mask = ~tail["pivot_low_level"].isna()
        ax.scatter(x[pl_mask], tail["pivot_low_level"].values[pl_mask],
                   marker="^", color="green", s=30, zorder=5, label="Pivot Low")

    if zones_df is not None and not zones_df.empty:
        for _, z in zones_df.iterrows():
            # Only shade x positions where the date falls within the zone's active period
            mask = (dates >= z["first_time"]) & (dates <= z["last_time"])
            x_zone = x[mask]
            if len(x_zone) == 0:
                continue
            ax.fill_between(
                x_zone,
                z["bottom"],
                z["top"],
                color="orange",
                alpha=0.35,
                label=f"Zone {z['zone_id']}"
            )

    ax.set_title(title)
    ax.set_xticks(x[::max(1, len(x) // 10)])
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates[::max(1, len(x) // 10)]], rotation=45)
    ax.grid(True, alpha=0.3)
    # Deduplicate legend entries and place outside the plot to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(
        seen.values(), seen.keys(),
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )
    plt.tight_layout()
    # Auto-save image
    os.makedirs(save_dir, exist_ok=True)
    filename = title.replace(" ", "_").replace("/", "-") + ".png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {save_path}")
    plt.show()


# -----------------------------
# 8) Example usage
# -----------------------------
if __name__ == "__main__":
    SYMBOL = "TSLA"
    START = "2021-01-04"
    END = "2024-09-30"

    # Fetch data (no Excel save yet — all sheets written together at the end)
    df = fetch_yahoo_daily(SYMBOL, start=START, end=END)

    # Build liquidity zones (df_z has a 'zone_id' column for each date row)
    zones_df, df_z = build_liquidity_zones(
        df,
        pivot_left_right=5,    # 5 bars on each side (~2 weeks) to confirm a pivot
        atr_len=14,            # ATR (Average True Range) length for merging pivots into a single zone.
        atr_merge_mult=1.0,    # how tight the merged zone can be (relative to ATR)
        require_volume=True,   # volume qualification ON
        vol_len=20,            # SMA length for volume qualification - 20SMA
        vol_mult=1.0,          # pivot's volume must exceed 20-day SMA (any above-average day)
        min_touches=2,         # Minimum number of pivot interactions
        unique_pivots=False,
        use_both_high_low_pivots=True,
    )

    print("\n=== Liquidity Zones (confirmed) ===")
    print(zones_df.head(10))

    # Optional quick visualization (comment out if running headless)
    plot_zones_matplotlib(df_z, zones_df, start_date="2021-01-01", end_date="2024-09-30", title=f"{SYMBOL} Liquidity Zones (2021-2024)", save_dir="charts")

    # Detect liquidity sweeps
    sweeps_df = detect_liquidity_sweeps(
        df,
        left=3,
        right=3,
        atr_len=14,
        buffer_atr=0.10,       # 10% ATR buffer past the pivot
        unique_pivots=True,
    )

    # Tag sweeps that occur near/inside zones
    sweeps_tagged = tag_sweeps_at_zones(sweeps_df, zones_df, proximity_atr_mult=0.25, max_days_after_zone_end=0)

    print("\n=== Recent Sweeps (last 5 rows) ===")
    cols = ["buy_side_sweep", "sell_side_sweep", "swept_level", "sweep_near_zone", "sweep_zone_id"]
    print(sweeps_tagged[cols].tail(5))
    # buy_side_sweep = True means a bearish sweep (price spiked above last pivot high but closed below it)
    # sell_side_sweep = True means a bullish sweep (price dipped below last pivot low)
    # sweep_near_zone = True means the sweep bar overlaps with a zone (including buffer)
    # sweep_zone_id indicates which zone it is near (NaN if not near any zone)

    # Save all results to a single Excel workbook using ExcelWriter so all sheets are preserved
    excel_path = f"{SYMBOL}_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # OHLCV sheet: Date · open · high · low · close · volume · buy_side_sweep · sell_side_sweep · swept_level · zone_id
        # zone_id shows the active zone name for every date that falls within a zone's period
        sweep_cols = sweeps_df[["buy_side_sweep", "sell_side_sweep", "swept_level"]]
        ohlcv_zone = df_z[["open", "high", "low", "close", "volume"]].join(sweep_cols).assign(zone_id=df_z["zone_id"])
        ohlcv_zone.to_excel(writer, index=True, sheet_name="OHLCV")

        # Liquidity zones summary table
        zones_df.to_excel(writer, index=False, sheet_name="Liquidity_Zones")

        # Full detect_liquidity_sweeps output
        sweeps_df.to_excel(writer, index=True, sheet_name="Sweeps")

        # Tagged sweeps
        sweeps_tagged[cols + ["open", "high", "low", "close", "volume"]].to_excel(
            writer, index=True, sheet_name="Tagged_Sweeps"
        )

    print(f"\nAll results saved to {excel_path}")
    print("  Sheet 'OHLCV'           : Date · open · high · low · close · volume · buy_side_sweep · sell_side_sweep · swept_level · zone_id")
    print("  Sheet 'Liquidity_Zones' : confirmed zone summary")
    print("  Sheet 'Sweeps'          : full detect_liquidity_sweeps output")
    print("  Sheet 'Tagged_Sweeps'   : sweep events with zone tags")