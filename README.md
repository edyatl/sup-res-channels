## Python porting of Support Resistance Channels TradingView Indicator
by LonesomeTheBlue

<https://www.tradingview.com/script/Ej53t8Wv-Support-Resistance-Channels/>

>Developed by [@edyatl](https://github.com/edyatl) May 2023 <edyatl@yandex.ru>

### Using [Python wrapper](https://github.com/TA-Lib/ta-lib-python) for [TA-LIB](http://ta-lib.org/) based on Cython instead of SWIG.

### Original Indicator code

```python
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © LonesomeTheBlue
 
//@version=4
study("Support Resistance Channels", "SRchannel", overlay = true, max_bars_back = 501)
prd = input(defval = 10, title="Pivot Period", minval = 4, maxval = 30, group = "Settings 🔨", tooltip="Used while calculating Pivot Points, checks left&right bars")
ppsrc = input(defval = 'High/Low', title="Source", options = ['High/Low', 'Close/Open'], group = "Settings 🔨", tooltip="Source for Pivot Points")
ChannelW = input(defval = 5, title = "Maximum Channel Width %", minval = 1, maxval = 8, group = "Settings 🔨", tooltip="Calculated using Highest/Lowest levels in 300 bars")
minstrength = input(defval = 1, title = "Minimum Strength", minval = 1, group = "Settings 🔨", tooltip = "Channel must contain at least 2 Pivot Points")
maxnumsr = input(defval = 6, title = "Maximum Number of S/R", minval = 1, maxval = 10, group = "Settings 🔨", tooltip = "Maximum number of Support/Resistance Channels to Show") - 1
loopback = input(defval = 290, title = "Loopback Period", minval = 100, maxval = 400, group = "Settings 🔨", tooltip="While calculating S/R levels it checks Pivots in Loopback Period")
res_col = input(defval = color.new(color.red, 75), title = "Resistance Color", group = "Colors 🟡🟢🟣")
sup_col = input(defval = color.new(color.lime, 75), title = "Support Color", group = "Colors 🟡🟢🟣")
inch_col = input(defval = color.new(color.gray, 75), title = "Color When Price in Channel", group = "Colors 🟡🟢🟣")
showpp = input(defval = false, title = "Show Pivot Points", group = "Extras ⏶⏷")
showsrbroken = input(defval = false, title = "Show Broken Support/Resistance", group = "Extras ⏶⏷")
showthema1en = input(defval = false, title = "MA 1", inline = "ma1")
showthema1len = input(defval = 50, title = "", inline = "ma1")
showthema1type = input(defval = "SMA", title = "", options = ["SMA", "EMA"], inline = "ma1")
showthema2en = input(defval = false, title = "MA 2", inline = "ma2")
showthema2len = input(defval = 200, title = "", inline = "ma2")
showthema2type = input(defval = "SMA", title = "", options = ["SMA", "EMA"], inline = "ma2")

ma1 = showthema1en ? (showthema1type == "SMA" ? sma(close, showthema1len) : ema(close, showthema1len)) : na
ma2 = showthema2en ? (showthema2type == "SMA" ? sma(close, showthema2len) : ema(close, showthema2len)) : na

plot(ma1, color = not na(ma1) ? color.blue : na)
plot(ma2, color = not na(ma2) ? color.red : na)

// get Pivot High/low
float src1 =  ppsrc == 'High/Low' ? high : max(close, open)
float src2 =  ppsrc == 'High/Low' ? low: min(close, open)
float ph = pivothigh(src1, prd, prd)
float pl = pivotlow(src2, prd, prd) 

// draw Pivot points
plotshape(ph and showpp, text = "H",  style = shape.labeldown, color = na, textcolor = color.red, location = location.abovebar, offset = -prd)
plotshape(pl and showpp, text = "L",  style = shape.labelup, color = na, textcolor = color.lime, location = location.belowbar, offset = -prd)

//calculate maximum S/R channel width
prdhighest =  highest(300)
prdlowest = lowest(300)
cwidth = (prdhighest - prdlowest) * ChannelW / 100

// get/keep Pivot levels
var pivotvals= array.new_float(0)
var pivotlocs= array.new_float(0)
if ph or pl
    array.unshift(pivotvals, ph ? ph : pl)
    array.unshift(pivotlocs, bar_index)
    for x = array.size(pivotvals) - 1 to 0
        if bar_index - array.get(pivotlocs, x) > loopback // remove old pivot points
            array.pop(pivotvals)
            array.pop(pivotlocs)
            continue
        break

//find/create SR channel of a pivot point
get_sr_vals(ind)=>
    float lo = array.get(pivotvals, ind)
    float hi = lo
    int numpp = 0
    for y = 0 to array.size(pivotvals) - 1
        float cpp = array.get(pivotvals, y)
        float wdth = cpp <= hi ? hi - cpp : cpp - lo
        if wdth <= cwidth // fits the max channel width?
            if cpp <= hi
                lo := min(lo, cpp)
            else
                hi := max(hi, cpp)
                
            numpp := numpp + 20 // each pivot point added as 20
    [hi, lo, numpp] 

// keep old SR channels and calculate/sort new channels if we met new pivot point
var suportresistance = array.new_float(20, 0) // min/max levels
changeit(x, y)=>
    tmp = array.get(suportresistance, y * 2)
    array.set(suportresistance, y * 2, array.get(suportresistance, x * 2))
    array.set(suportresistance, x * 2, tmp)
    tmp := array.get(suportresistance, y * 2 + 1)
    array.set(suportresistance, y * 2 + 1, array.get(suportresistance, x * 2 + 1))
    array.set(suportresistance, x * 2 + 1, tmp)
    
if ph or pl
    supres = array.new_float(0)  // number of pivot, strength, min/max levels
    stren = array.new_float(10, 0)
    // get levels and strengs
    for x = 0 to array.size(pivotvals) - 1
        [hi, lo, strength] = get_sr_vals(x)
        array.push(supres, strength)
        array.push(supres, hi)
        array.push(supres, lo)
    
    // add each HL to strengh
    for x = 0 to array.size(pivotvals) - 1
        h = array.get(supres, x * 3 + 1)
        l = array.get(supres, x * 3 + 2)
        s = 0
        for y = 0 to loopback
            if (high[y] <= h and high[y] >= l) or
               (low[y] <= h and low[y] >= l)
                s := s + 1
        array.set(supres, x * 3, array.get(supres, x * 3) + s)
    
    //reset SR levels
    array.fill(suportresistance, 0)
    // get strongest SRs
    src = 0
    for x = 0 to array.size(pivotvals) - 1
        stv = -1. // value
        stl = -1 // location
        for y = 0 to array.size(pivotvals) - 1
            if array.get(supres, y * 3) > stv and array.get(supres, y * 3) >= minstrength * 20
                stv := array.get(supres, y * 3)
                stl := y
        if stl >= 0
            //get sr level
            hh = array.get(supres, stl * 3 + 1)
            ll = array.get(supres, stl * 3 + 2)
            array.set(suportresistance, src * 2, hh)
            array.set(suportresistance, src * 2 + 1, ll)
            array.set(stren, src, array.get(supres, stl * 3))
            
            // make included pivot points' strength zero 
            for y = 0 to array.size(pivotvals) - 1
                if (array.get(supres, y * 3 + 1) <= hh and array.get(supres, y * 3 + 1) >= ll) or
                   (array.get(supres, y * 3 + 2) <= hh and array.get(supres, y * 3 + 2) >= ll)
                    array.set(supres, y * 3, -1)

            src += 1
            if src >= 10
                break
    
    for x = 0 to 8
        for y = x + 1 to 9
            if array.get(stren, y) > array.get(stren, x)
                tmp = array.get(stren, y) 
                array.set(stren, y, array.get(stren, x))
                changeit(x, y)
                
    
get_level(ind)=>
    float ret = na
    if ind < array.size(suportresistance)
        if array.get(suportresistance, ind) != 0
            ret := array.get(suportresistance, ind)
    ret
    
get_color(ind)=>
    color ret = na
    if ind < array.size(suportresistance)
        if array.get(suportresistance, ind) != 0
            ret := array.get(suportresistance, ind) > close and array.get(suportresistance, ind + 1) > close ? res_col :
                   array.get(suportresistance, ind) < close and array.get(suportresistance, ind + 1) < close ? sup_col :
                   inch_col
    ret

var srchannels = array.new_box(10)
for x = 0 to min(9, maxnumsr)
    box.delete(array.get(srchannels, x))
    srcol = get_color(x * 2)
    if not na(srcol)
        array.set(srchannels, x, 
                  box.new(left = bar_index, top = get_level(x * 2), right = bar_index + 1, bottom = get_level(x * 2 + 1), 
                          border_color = srcol, 
                          border_width = 1,
                          extend = extend.both, 
                          bgcolor = srcol))

resistancebroken = false
supportbroken = false

// check if it's not in a channel
not_in_a_channel = true
for x = 0 to min(9, maxnumsr)
    if close <= array.get(suportresistance, x * 2) and close >= array.get(suportresistance, x * 2 + 1) 
        not_in_a_channel := false

// if price is not in a channel then check broken ones
if not_in_a_channel
    for x = 0 to min(9, maxnumsr)
        if close[1] <= array.get(suportresistance, x * 2) and close > array.get(suportresistance, x * 2)
            resistancebroken := true
        if close[1] >= array.get(suportresistance, x * 2 + 1) and close < array.get(suportresistance, x * 2 + 1)
            supportbroken := true

alertcondition(resistancebroken, title = "Resistance Broken", message = "Resistance Broken")
alertcondition(supportbroken, title = "Support Broken", message = "Support Broken")
plotshape(showsrbroken and resistancebroken, style = shape.triangleup, location = location.belowbar, color = color.new(color.lime, 0), size = size.tiny)
plotshape(showsrbroken and supportbroken, style = shape.triangledown, location = location.abovebar, color = color.new(color.red, 0), size = size.tiny)
```

### Original Indicator Overview

In the quest to create an effective Support/Resistance Channels script, the author, LonesomeTheBlue, has developed a comprehensive indicator that precisely identifies pivot points and establishes robust support and resistance levels. This powerful tool not only calculates the strength of these levels but also arranges them in order of significance, providing traders with valuable insights into potential price movements.

The indicator employs a meticulous process to identify and analyze pivot points. When a new pivot point is detected, the script clears older support/resistance channels and proceeds to evaluate all pivot points within its own channel. Utilizing a dynamic width calculation based on the highest and lowest levels in the previous 300 bars, the script determines the maximum channel width percentage. This flexible approach ensures that the channels adapt to changing market conditions.

As the script constructs each support/resistance channel, it calculates the channel's strength, considering the number of pivot points it encompasses. These strength values serve as a crucial factor in determining the significance of the channels. By quantifying the number of pivot points, the script provides traders with an objective measure of the channel's reliability.

To optimize visibility, the script sorts the support/resistance channels based on their strength and displays the strongest ones. By adjusting the positions of the channels in the list, the script ensures that the most influential levels are readily visible to traders. This thoughtful design enhances the indicator's usability and helps traders make well-informed decisions.

The Support Resistance Channels indicator also incorporates alerts to notify traders when a support or resistance level has been broken during the latest price movement. These alerts, accompanied by small shapes placed below or above the candle, provide timely and actionable information, allowing traders to react promptly to potential trend reversals or breakout opportunities.

The indicator offers several adjustable settings, allowing traders to customize their analysis based on specific requirements. Traders can modify the pivot period, choose between different data sources (high/low or close/open), and specify the maximum channel width percentage. Additionally, the number of support/resistance levels to display, the loopback period for analyzing pivot points, and the option to show levels only on the last N bars can all be tailored to suit individual preferences.

Furthermore, traders can personalize the indicator's appearance by selecting colors and adjusting transparency. The script's versatility enables users to adapt the visual representation of support and resistance levels to their preferred charting style.

In summary, the Support Resistance Channels indicator offers traders a comprehensive solution for identifying and analyzing pivotal support and resistance levels. With its ability to evaluate pivot points, calculate channel strength, and rank the most significant levels, this indicator equips traders with valuable insights into potential price movements. By combining functionality with customization options, the indicator provides a versatile tool that can be adapted to individual trading strategies and preferences.



