import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

def identify_value_pattern(stock_data):
    """
    Identify potential value opportunities similar to CVS pattern
    Returns a dictionary of pattern indicators and their values
    """
    df = stock_data.copy()
    
    # Calculate key metrics
    df['52W_High'] = df['Close'].rolling(window=252).max()
    df['52W_Low'] = df['Close'].rolling(window=252).min()
    df['Pct_From_52W_High'] = ((df['52W_High'] - df['Close']) / df['52W_High']) * 100
    df['30D_Vol_Avg'] = df['Volume'].rolling(window=30).mean()
    df['Volume_Ratio'] = df['Volume'] / df['30D_Vol_Avg']
    
    # Get latest values
    latest = df.iloc[-1]
    
    # Define pattern criteria
    pattern = {
        'Deep_Value': latest['Pct_From_52W_High'] > 25,  # More than 25% below 52-week high
        'Near_Support': latest['Close'] <= df['Low'].rolling(window=50).quantile(0.1).iloc[-1],
        'High_Volume': latest['Volume_Ratio'] > 1.2,  # Above average volume
        'RSI_Oversold': latest.get('RSI', 50) < 35,  # Oversold condition
        'Below_MA200': latest['Close'] < latest['MA200'],
        'Stabilizing': df['Close'].pct_change().rolling(window=5).std().iloc[-1] < 0.02  # Low recent volatility
    }
    
    # Calculate pattern score (0-100)
    pattern_score = sum([
        30 if pattern['Deep_Value'] else 0,
        20 if pattern['Near_Support'] else 0,
        15 if pattern['High_Volume'] else 0,
        15 if pattern['RSI_Oversold'] else 0,
        10 if pattern['Below_MA200'] else 0,
        10 if pattern['Stabilizing'] else 0
    ])
    
    pattern['Pattern_Score'] = pattern_score
    pattern['Pattern_Detected'] = pattern_score >= 60  # Threshold for pattern detection
    
    return pattern

def calculate_value_metrics(stock_data, fundamental_data):
    """
    Calculate comprehensive value metrics combining technical and fundamental data
    """
    metrics = {}
    
    # Technical metrics
    latest = stock_data.iloc[-1]
    metrics['Price'] = latest['Close']
    metrics['Pct_From_52W_High'] = ((stock_data['Close'].rolling(window=252).max().iloc[-1] - latest['Close']) /
                                   stock_data['Close'].rolling(window=252).max().iloc[-1]) * 100
    
    # Fundamental metrics
    metrics.update(fundamental_data)
    
    # Calculate industry relative valuations
    if 'PE_Ratio' in fundamental_data and 'Industry' in fundamental_data:
        try:
            industry_pe = yf.Ticker('^SPX').info.get('forwardPE', 20)  # Use S&P 500 as benchmark
            metrics['PE_vs_Industry'] = (metrics['PE_Ratio'] / industry_pe - 1) * 100 if metrics['PE_Ratio'] else None
        except:
            metrics['PE_vs_Industry'] = None
    
    return metrics

def calculate_enhanced_value_score(metrics, pattern_data):
    """
    Calculate an enhanced value score incorporating pattern recognition
    """
    score = 50  # Base score
    
    # Pattern recognition score (0-30 points)
    pattern_score = pattern_data.get('Pattern_Score', 0)
    score += pattern_score * 0.3  # Weight pattern score as 30% of total
    
    # Traditional value metrics (0-40 points)
    if metrics.get('PE_Ratio'):
        if metrics['PE_Ratio'] < 15:
            score += 10
        elif metrics['PE_Ratio'] < 20:
            score += 5
            
    if metrics.get('Dividend_Yield'):
        div_yield = float(metrics['Dividend_Yield']) if metrics['Dividend_Yield'] else 0
        if div_yield > 0.04:  # 4% yield
            score += 10
        elif div_yield > 0.02:  # 2% yield
            score += 5
            
    if metrics.get('Price_to_Book'):
        if metrics['Price_to_Book'] < 1.5:
            score += 10
        elif metrics['Price_to_Book'] < 3:
            score += 5
            
    if metrics.get('PE_vs_Industry'):
        if metrics['PE_vs_Industry'] < -20:  # 20% below industry
            score += 10
        elif metrics['PE_vs_Industry'] < -10:
            score += 5
            
    # Market position and stability (0-30 points)
    if metrics.get('Market_Cap'):
        if metrics['Market_Cap'] > 10e9:  # 10B+ market cap
            score += 10
            
    if metrics.get('Beta'):
        if metrics['Beta'] < 1.2:  # Lower volatility
            score += 10
            
    if metrics.get('Current_Ratio'):
        if metrics['Current_Ratio'] > 1.5:
            score += 10
            
    return min(max(score, 0), 100)  # Ensure score is between 0 and 100

def analyze_stocks(tickers=None):
    """
    Enhanced stock analysis with pattern recognition
    """
    if tickers is None:
        print("Fetching S&P 500 tickers...")
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data for better pattern recognition
    
    results = []
    
    for ticker in tickers:
        try:
            print(f"\nAnalyzing {ticker}...")
            
            # Download historical data
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if stock.empty:
                continue
                
            # Get fundamental data
            fund_data = get_fundamental_metrics(ticker)
            
            # Calculate technical indicators
            stock = calculate_technical_indicators(stock)
            
            # Identify value pattern
            pattern_data = identify_value_pattern(stock)
            
            # Calculate comprehensive metrics
            value_metrics = calculate_value_metrics(stock, fund_data)
            
            # Calculate enhanced value score
            value_score = calculate_enhanced_value_score(value_metrics, pattern_data)
            
            # Combine all metrics
            result = {
                'Ticker': ticker,
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
                'Current_Price': stock['Close'].iloc[-1],
                'Value_Score': value_score,
                'Pattern_Score': pattern_data['Pattern_Score'],
                'Pattern_Detected': pattern_data['Pattern_Detected'],
                'Pct_From_52W_High': value_metrics['Pct_From_52W_High'],
                **fund_data
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add watchlist flags
    df['Watchlist'] = (df['Pattern_Detected'] & (df['Value_Score'] > 70))
    df['Priority_Watch'] = (df['Pattern_Detected'] & (df['Value_Score'] > 80))
    
    # Save to Excel with enhanced formatting
    save_to_enhanced_excel(df, 'value_stock_analysis.xlsx')
    
    return df

def save_to_enhanced_excel(df, filename):
    """
    Enhanced Excel output with better formatting and highlighting
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Stock Analysis"
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    watchlist_fill = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")
    priority_fill = PatternFill(start_color="F4B084", end_color="F4B084", fill_type="solid")
    
    # Write headers
    headers = list(df.columns)
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = Font(bold=True, color="FFFFFF")
    
    # Write data
    for row_idx, row in enumerate(df.values, 2):
        for col_idx, value in enumerate(row, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            
            # Apply watchlist highlighting
            if headers[col_idx-1] == 'Watchlist' and value:
                for c in range(1, len(headers) + 1):
                    ws.cell(row=row_idx, column=c).fill = watchlist_fill
            
            # Apply priority watch highlighting
            if headers[col_idx-1] == 'Priority_Watch' and value:
                for c in range(1, len(headers) + 1):
                    ws.cell(row=row_idx, column=c).fill = priority_fill
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column = list(column)
        for cell in column:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        ws.column_dimensions[get_column_letter(column[0].column)].width = max_length + 2
    
    # Add filters
    ws.auto_filter.ref = ws.dimensions
    
    # Save workbook
    wb.save(filename)

if __name__ == "__main__":
    # Test with some example tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'CVS', 'WBA']
    results = analyze_stocks(test_tickers)
    
    # Display watchlist stocks
    watchlist = results[results['Watchlist']]
    if not watchlist.empty:
        print("\nWatchlist Stocks:")
        print(watchlist[['Ticker', 'Current_Price', 'Value_Score', 'Pattern_Score', 'Pct_From_52W_High']])