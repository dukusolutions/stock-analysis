# Stock Screening Script Analysis

## Pattern Recognition Metrics (Pattern Score: 0-100)

The script evaluates several key indicators, each contributing to the pattern score:

### Deep Value (30 points)
* Triggers when price is >25% below recent high
* Identifies potentially oversold stocks with recovery potential

### Near Support (20 points)
* Triggers when current price is within 5% of recent low
* Indicates possible price floor/support level

### High Volume (15 points)
* Triggers when current volume is >120% of average volume
* Suggests increased market interest/attention

### RSI Oversold (15 points)
* Triggers when RSI drops below 35
* Classical oversold technical indicator

### Below MA200 (10 points)
* Price below 200-day moving average
* Traditional bearish indicator, but can signal value opportunity

### Stabilizing (10 points)
* Volatility below 2%
* Suggests price consolidation

## Value Score Components (0-100)

### Traditional Value Metrics (40 points max)
* P/E Ratio < 15 (10 points) or < 20 (5 points)
* Dividend Yield > 4% (10 points) or > 2% (5 points)
* Price-to-Book < 1.5 (10 points) or < 3 (5 points)
* P/E vs Industry < -20% (10 points) or < -10% (5 points)

### Market Position/Stability (30 points max)
* Market Cap > $10B (10 points)
* Beta < 1.2 (10 points)
* Current Ratio > 1.5 (10 points)

### Pattern Recognition (30% weight)
* Incorporates the pattern score discussed above

## What to Look For

### High Combined Scores
* Pattern Score > 60 AND Value Score > 70: Added to Watchlist
* Pattern Score > 70 AND Value Score > 80: Priority Watch

### Key Technical Signals
* Price significantly below 52-week high with stabilizing volatility
* RSI showing oversold conditions with increasing volume

### Fundamental Strength
* Strong current ratio (>1.5) indicating good liquidity
* Reasonable debt-to-equity levels
* Positive profit margins and ROE
* Attractive dividend yield if applicable

### Industry Context
* PE ratio relative to industry average
* Sector/industry trends and positioning

## Excel Output Highlighting
* Yellow highlight: Watchlist candidates
* Orange highlight: Priority watch candidates

This screening tool combines technical and fundamental factors to identify oversold but fundamentally sound companies, making it particularly useful for value investors seeking mean reversion opportunities in quality companies.