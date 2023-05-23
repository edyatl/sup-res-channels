#!/usr/bin/env python3
"""
    Python porting of Support Resistance Channels TradingView Indicator by LonesomeTheBlue
    https://www.tradingview.com/script/Ej53t8Wv-Support-Resistance-Channels/
    Developed by @edyatl <edyatl@yandex.ru> May 2023
    https://github.com/edyatl

"""
# Standard imports
import os
from os import environ as env
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import talib as tl
from binance import Client

# Load API keys from env
project_dotenv = os.path.join(os.path.abspath(""), ".env")
if os.path.exists(project_dotenv):
    load_dotenv(project_dotenv)

api_key, api_secret = env.get("ENV_API_KEY"), env.get("ENV_SECRET_KEY")

# Make API Client instance
client = Client(api_key, api_secret)

short_col_names = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "num_trades",
    "taker_base_vol",
    "taker_quote_vol",
    "ignore",
]

# Load Dataset
# Get last 500 records of ATOMUSDT 15m Timeframe
klines = client.get_klines(symbol="ATOMUSDT", interval=Client.KLINE_INTERVAL_15MINUTE)
data = pd.DataFrame(klines, columns=short_col_names)

# Convert Open and Close time fields to DateTime
data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")

# --------------------------INPUTS--------------------------------

# "Pivot Period", min = 4, max = 30, "Used while calculating Pivot Points, checks left&right bars"
prd: int = 10

# "Source", ['High/Low', 'Close/Open'], "Source for Pivot Points"
ppsrc: str = "High/Low"

# "Maximum Channel Width %", min = 1, max = 8, "Calculated using Highest/Lowest levels in 300 bars"
ChannelW: int = 5

# "Minimum Strength", minval = 1, "Channel must contain at least 2 Pivot Points"
minstrength: int = 1

# "Maximum Number of S/R", min = 1, max = 10, "Max num of Support/Resistance Channels to Show" - 1
maxnumsr: int = 6 - 1

# "Loopback Period", min = 100, max = 400, "While calc S/R lvls checks Pivots in Loopback Period"
loopback: int = 290

# res_col = input(color.new(color.red, 75), title = "Resistance Color", group = "Colors")
# sup_col = input(color.new(color.lime, 75), title = "Support Color", group = "Colors")
# inch_col = input(color.new(color.gray, 75), title = "Color When P in Chnl", group = "Colors")

showpp: bool = False  # "Show Pivot Points"
showsrbroken: bool = False  # "Show Broken Support/Resistance"

showthema1en: bool = True  # "MA 1"
showthema1len: int = 50  # "ma1 length"
showthema1type: str = "SMA"  # ["SMA", "EMA"]

showthema2en: bool = True  # "MA 2"
showthema2len: int = 200  # "ma2 length"
showthema2type: str = "SMA"  # ["SMA", "EMA"]


# --------------------------FUNCIONS------------------------------

# find/create SR channel of a pivot point
def get_sr_vals(
    ind: int, _pivotvals: np.ndarray, _cwidth: np.ndarray, _bar_index: int
) -> tuple:
    """
    Find/create a support/resistance (SR) channel for a given pivot point.

    Parameters:
        ind (int): Index of the pivot point in the _pivotvals array.
        _pivotvals (np.ndarray): Array of pivot values.
        _cwidth (np.ndarray): Array of channel widths.
        _bar_index (int): Index of the current bar.

    Returns:
        tuple: A tuple containing the high value of the SR channel, the low value
               of the SR channel, and the number of pivot points in the channel.

    """
    lo: np.double = _pivotvals[ind]
    hi: np.double = lo
    numpp: int = 0

    for y in range(len(_pivotvals)):
        cpp: np.double = _pivotvals[y]
        wdth: np.double = hi - cpp if cpp <= hi else cpp - lo
        if wdth <= _cwidth[_bar_index]:  # fits the max channel width?
            if cpp <= hi:
                lo = min(lo, cpp)
            else:
                hi = max(hi, cpp)
            numpp += 20  # each pivot point added as 20
    return hi, lo, numpp


# keep old SR channels and calculate/sort new channels if we met new pivot point
def changeit(x: int, y: int, _suportresistance: np.ndarray):
    """
    Swap the values of SR channels at indices x and y, in the _suportresistance array.

    Parameters:
        x (int): Index of the first SR channel.
        y (int): Index of the second SR channel.
        _suportresistance (np.ndarray): Array containing the SR channels.

    Returns:
        None

    """
    tmp: np.double = _suportresistance[y * 2]
    _suportresistance[y * 2] = _suportresistance[x * 2]
    _suportresistance[x * 2] = tmp
    tmp = _suportresistance[(y * 2) + 1]
    _suportresistance[(y * 2) + 1] = _suportresistance[(x * 2) + 1]
    _suportresistance[(x * 2) + 1] = tmp


def get_level(ind: int, _suportresistance: np.ndarray) -> np.double:
    """
    Retrieve the level at the specified index from the _suportresistance array.

    Parameters:
        ind (int): Index of the level to retrieve.
        _suportresistance (np.ndarray): Array containing the support/resistance levels.

    Returns:
        np.double: The level at the specified index, or None if the index is out of range
                   or the level is 0.

    """
    ret: np.double = None
    if ind < len(_suportresistance):
        if _suportresistance[ind] != 0:
            return _suportresistance[ind]
    return ret


def get_color(ind: int, _close: np.double, _suportresistance: np.ndarray) -> str:
    """
    Determine color for the specified level index based on its relationship with the close price.

    Parameters:
        ind (int): Index of the level to determine the color for.
        _close (np.double): The close price value.
        _suportresistance (np.ndarray): Array containing the support/resistance levels.

    Returns:
        str: The color string corresponding to the level's relationship with the close price,
             or None if the index is out of range or the level is 0.

    """
    ret: str = None
    if ind < len(_suportresistance):
        if _suportresistance[ind] != 0:
            ret = (
                "res_col"
                if _suportresistance[ind] > _close
                and _suportresistance[ind + 1] > _close
                else "sup_col"
                if _suportresistance[ind] < _close
                and _suportresistance[ind + 1] < _close
                else "inch_col"
            )
    return ret


def exclude_repeats(pv: np.ndarray, smpl: np.ndarray, sp: int) -> np.ndarray:
    """
    Exclude repeated values in the pivot array by setting them to NaN.

    Parameters:
        pv (np.ndarray): Array of pivot values.
        smpl (np.ndarray): Array used for comparison to identify repeated values.
        sp (int): Span length for comparison.

    Returns:
        np.ndarray: Array with repeated values set to NaN.

    """
    for i in range(sp, len(pv) - sp):
        for j in range(sp):
            if pv[i] == smpl[i + 1 : i + 1 + sp][j]:
                pv[i] = np.NaN
            if pv[i] == smpl[i - sp : i][j]:
                pv[i] = np.NaN
    return pv


def pivothigh(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Find pivot highs in the given array of high values.

    Parameters:
        high (np.ndarray): Array of high values.
        left (int): Number of bars to the left for comparison.
        right (int): Number of bars to the right for comparison.

    Returns:
        np.ndarray: Array containing pivot high values, with non-pivot values set to NaN.

    """
    pivots = np.roll(tl.MAX(high, left + 1 + right), -right)
    pivots[pivots != high] = np.NaN

    # Exclude repeating pivot highs
    exclude_repeats(pivots, high, right)
    return pivots


def pivotlow(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """
    Find pivot lows in the given array of low values.

    Parameters:
        low (np.ndarray): Array of low values.
        left (int): Number of bars to the left for comparison.
        right (int): Number of bars to the right for comparison.

    Returns:
        np.ndarray: Array containing pivot low values, with non-pivot values set to NaN.

    """
    pivots = np.roll(tl.MIN(low, left + 1 + right), -right)
    pivots[pivots != low] = np.NaN

    # Exclude repeating pivot lows
    exclude_repeats(pivots, low, right)
    return pivots


def main():

    close: np.ndarray = data["close"].to_numpy(dtype=np.double)
    _open: np.ndarray = data["open"].to_numpy(dtype=np.double)
    high: np.ndarray = data["high"].to_numpy(dtype=np.double)
    low: np.ndarray = data["low"].to_numpy(dtype=np.double)

    # min/max levels
    suportresistance: np.ndarray = np.zeros(20, dtype=np.double)

    ma1 = (
        (
            tl.SMA(close, showthema1len)
            if showthema1type == "SMA"
            else tl.EMA(close, showthema1len)
        )
        if showthema1en
        else None
    )

    ma2 = (
        (
            tl.SMA(close, showthema2len)
            if showthema2type == "SMA"
            else tl.EMA(close, showthema2len)
        )
        if showthema2en
        else None
    )

    # get Pivot High/low
    src1: np.ndarray = high if ppsrc == "High/Low" else np.maximum(close, _open)
    src2: np.ndarray = low if ppsrc == "High/Low" else np.minimum(close, _open)

    ph: np.ndarray = pivothigh(src1, prd, prd)
    pl: np.ndarray = pivotlow(src2, prd, prd)

    # calculate maximum S/R channel width
    prdhighest: np.ndarray = tl.MAX(high, 300)
    prdlowest: np.ndarray = tl.MIN(low, 300)
    cwidth: np.ndarray = (prdhighest - prdlowest) * ChannelW / 100

    # get/keep Pivot levels
    pivotvals: np.ndarray = np.zeros(0, dtype=np.double)
    pivotlocs: np.ndarray = np.zeros(0, dtype=np.int)

    for bar_index, (_ph, _pl, ph_nan, pl_nan) in enumerate(
        zip(ph, pl, np.isnan(ph), np.isnan(pl))
    ):
        if not ph_nan or not pl_nan:
            pivotvals = np.insert(pivotvals, 0, [_ph if not ph_nan else _pl])
            pivotlocs = np.insert(pivotlocs, 0, [bar_index])

    for x in reversed(range(len(pivotvals))):
        bar_index = len(high) - 1
        # remove old pivot points
        if bar_index - pivotlocs[x] > loopback:
            pivotvals = np.delete(pivotvals, x, 0)
            pivotlocs = np.delete(pivotlocs, x, 0)

    for bar_index, (_ph, _pl, ph_nan, pl_nan) in enumerate(
        zip(ph, pl, np.isnan(ph), np.isnan(pl))
    ):
        if not ph_nan or not pl_nan:
            # number of pivot, strength, min/max levels
            supres: np.ndarray = np.zeros(0, dtype=np.double)
            stren: np.ndarray = np.zeros(10, dtype=np.double)

            # get levels and strengs
            for x1 in range(len(pivotvals)):
                hi, lo, strength = get_sr_vals(x1, pivotvals, cwidth, bar_index)
                supres = np.append(supres, strength)
                supres = np.append(supres, hi)
                supres = np.append(supres, lo)

            # add each HL to strengh
            for x2 in range(len(pivotvals)):
                h: np.double = supres[x2 * 3 + 1]
                l: np.double = supres[x2 * 3 + 2]
                s: int = 0

                for y2 in range(loopback + 1):
                    if len(high[:bar_index]) - 1 < y2:
                        continue
                    if (
                        high[:bar_index][-y2 - 1] <= h
                        and high[:bar_index][-y2 - 1] >= l
                    ) or (
                        low[:bar_index][-y2 - 1] <= h and low[:bar_index][-y2 - 1] >= l
                    ):
                        s += 1
                supres[x2 * 3] = supres[x2 * 3] + s

            # reset SR levels
            suportresistance.fill(0)

            # get strongest SRs
            src: int = 0
            for _ in range(len(pivotvals)):
                stv: np.double = -1.0  # value
                stl: int = -1  # location
                for y3 in range(len(pivotvals)):
                    if supres[y3 * 3] > stv and supres[y3 * 3] >= minstrength * 20:
                        stv = supres[y3 * 3]
                        stl = y3
                if stl >= 0:
                    # get sr level
                    hh = supres[stl * 3 + 1]
                    ll = supres[stl * 3 + 2]
                    suportresistance[src * 2] = hh
                    suportresistance[src * 2 + 1] = ll
                    stren[src] = supres[stl * 3]

                    # make included pivot points' strength zero
                    for y32 in range(len(pivotvals)):
                        if (
                            supres[y32 * 3 + 1] <= hh and supres[y32 * 3 + 1] >= ll
                        ) or (supres[y32 * 3 + 2] <= hh and supres[y32 * 3 + 2] >= ll):
                            supres[y32 * 3] = -1

                    src += 1
                    if src >= 10:
                        break

            for x4 in range(9):
                for y4 in range(x4 + 1, 10):
                    if stren[y4] > stren[x4]:
                        tmp = stren[y4]
                        stren[y4] = stren[x4]
                        changeit(x4, y4, suportresistance)

    top_level: np.ndarray = np.zeros(10, dtype=np.double)
    bot_level: np.ndarray = np.zeros(10, dtype=np.double)

    for bar_index, _close in enumerate(close):
        for x in range(min(10, (maxnumsr + 1))):
            top_level[x] = np.nan
            bot_level[x] = np.nan

            srcol = get_color(x * 2, _close, suportresistance)
            if srcol:
                top_level[x] = get_level(x * 2, suportresistance)
                bot_level[x] = get_level(x * 2 + 1, suportresistance)

    top_level = np.where(top_level != 0, top_level, np.nan)
    bot_level = np.where(bot_level != 0, bot_level, np.nan)

    res = pd.DataFrame(
        {
            "top_level": top_level,
            "bot_level": bot_level,
        }
    )
    res.dropna().to_csv("sup_res_channels-ATOMUSDT-15m.csv", index=None, header=True)


if __name__ == "__main__":
    main()
