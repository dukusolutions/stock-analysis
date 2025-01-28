import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

def calculate_technical_indicators(stock_data):
    """
    Calculate technical indicators for trend analysis
    """
    df = stock_data.copy()
    
    # Calculate 50-day and 200-day moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def detect_macd_crossover(macd_values, signal_values):
    """
    Detect if a MACD crossover occurred in the recent values
    Returns True if MACD crossed above signal line
    """
    if len(macd_values) < 2 or len(signal_values) < 2:
        return False
        
    # Check if MACD was below signal and is now above
    was_below = macd_values.iloc[0] < signal_values.iloc[0]
    is_above = macd_values.iloc[-1] > signal_values.iloc[-1]
    
    return bool(was_below and is_above)

def get_fundamental_metrics(ticker):
    """
    Fetch fundamental metrics from yfinance
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'Market_Cap': info.get('marketCap'),
            'PE_Ratio': info.get('forwardPE'),
            'PEG_Ratio': info.get('pegRatio'),
            'Price_to_Book': info.get('priceToBook'),
            'Debt_to_Equity': info.get('debtToEquity'),
            'Current_Ratio': info.get('currentRatio'),
            'Profit_Margins': info.get('profitMargins'),
            'ROE': info.get('returnOnEquity'),
            'Dividend_Rate': info.get('dividendRate'),
            'Dividend_Yield': info.get('dividendYield'),
            'Beta': info.get('beta'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry')
        }
    except Exception as e:
        print(f"Error fetching fundamental data for {ticker}: {str(e)}")
        return {}

def calculate_value_score(metrics):
    """
    Calculate a composite value score (0-100) based on fundamental and technical metrics
    """
    score = 50  # Start with neutral score
    
    try:
        # Helper function to safely get numeric values
        def safe_float(value):
            try:
                if isinstance(value, pd.Series):
                    return float(value.iloc[0])
                return float(value) if value is not None and not pd.isna(value) else None
            except:
                return None

        # Fundamental metrics scoring
        pe_ratio = safe_float(metrics.get('PE_Ratio'))
        if pe_ratio is not None:
            if pe_ratio < 15:
                score += 10
            elif pe_ratio > 30:
                score -= 5
                
        peg_ratio = safe_float(metrics.get('PEG_Ratio'))
        if peg_ratio is not None:
            if peg_ratio < 1:
                score += 10
            elif peg_ratio > 2:
                score -= 5
                
        ptb = safe_float(metrics.get('Price_to_Book'))
        if ptb is not None:
            if ptb < 3:
                score += 5
                
        div_yield = safe_float(metrics.get('Dividend_Yield'))
        if div_yield is not None:
            if div_yield > 0.02:  # 2% yield
                score += 5
                
        roe = safe_float(metrics.get('ROE'))
        if roe is not None:
            if roe > 0.15:  # 15% ROE
                score += 5
                
        # Technical metrics scoring
        rsi = safe_float(metrics.get('RSI'))
        if rsi is not None:
            if rsi < 30:
                score += 10  # Oversold condition
            elif rsi > 70:
                score -= 5   # Overbought condition
                
        price_vs_ma200 = safe_float(metrics.get('Price_vs_MA200'))
        if price_vs_ma200 is not None:
            if price_vs_ma200 < -0.1:  # 10% below 200-day MA
                score += 10
                
        # Trend reversal scoring
        if metrics.get('MACD_Crossover') is True:  # Explicit boolean check
            score += 5
            
    except Exception as e:
        print(f"Error calculating value score: {str(e)}")
        return 50  # Return neutral score on error
        
    return min(max(score, 0), 100)  # Ensure score is between 0 and 100

def analyze_stocks(tickers=None):
    """
    Enhanced stock analysis with value investing metrics
    """
    if tickers is None:
        print("Fetching S&P 500 tickers...")
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
        company_names = dict(zip(sp500['Symbol'], sp500['Security']))
    else:
        company_names = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                company_names[ticker] = stock.info.get('longName', ticker)
            except:
                company_names[ticker] = ticker

    end_date = datetime.now()
    start_date = end_date - timedelta(days=400)
    
    results = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"\nProcessing {ticker} ({i}/{len(tickers)})...")
            
            # Download historical data
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if stock.empty:
                print(f"No data found for {ticker}")
                continue
                
            # Calculate technical indicators
            tech_data = calculate_technical_indicators(stock)
            
            # Get fundamental metrics
            fund_metrics = get_fundamental_metrics(ticker)
            
            # Current metrics - ensure scalar values
            current_price = float(tech_data['Close'].iloc[-1])
            current_ma50 = float(tech_data['MA50'].iloc[-1]) if not pd.isna(tech_data['MA50'].iloc[-1]) else None
            current_ma200 = float(tech_data['MA200'].iloc[-1]) if not pd.isna(tech_data['MA200'].iloc[-1]) else None
            current_rsi = float(tech_data['RSI'].iloc[-1]) if not pd.isna(tech_data['RSI'].iloc[-1]) else None
            
            # Check for MACD crossover using separate function
            macd_crossover = detect_macd_crossover(
                tech_data['MACD'].tail(5),
                tech_data['Signal_Line'].tail(5)
            )
            
            # Calculate price vs moving averages - ensure scalar values
            price_vs_ma50 = float((current_price / current_ma50 - 1) * 100) if current_ma50 is not None else None
            price_vs_ma200 = float((current_price / current_ma200 - 1) * 100) if current_ma200 is not None else None
            
            # Combine metrics
            metrics = {
                'Ticker': ticker,
                'Company_Name': company_names.get(ticker, ticker),
                'Current_Price': current_price,
                'MA50': current_ma50,
                'MA200': current_ma200,
                'RSI': current_rsi,
                'Price_vs_MA50': price_vs_ma50,
                'Price_vs_MA200': price_vs_ma200,
                'MACD_Crossover': macd_crossover,
                **fund_metrics
            }
            
            # Calculate value score
            metrics['Value_Score'] = calculate_value_score(metrics)
            
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not results:
        print("No data was collected. Please check the errors above.")
        return None
        
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add analysis timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Reorder and select columns for final output
    columns = [
        'Analysis_Date',
        'Ticker',
        'Company_Name',
        'Sector',
        'Industry',
        'Current_Price',
        'Value_Score',
        'PE_Ratio',
        'PEG_Ratio',
        'Price_to_Book',
        'ROE',
        'Profit_Margins',
        'Dividend_Yield',
        'RSI',
        'Price_vs_MA50',
        'Price_vs_MA200',
        'MACD_Crossover',
        'Beta',
        'Market_Cap',
        'Debt_to_Equity',
        'Current_Ratio'
    ]
    
    # Only include columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    
    # Save to Excel
    excel_filename = f'value_stock_analysis_{timestamp}.xlsx'
    save_to_excel(df, excel_filename)
    
    # Display value opportunities
    print("\nTop Value Opportunities:")
    value_columns = ['Ticker', 'Company_Name', 'Value_Score', 'Current_Price', 'PE_Ratio', 'RSI', 'Price_vs_MA200']
    available_columns = [col for col in value_columns if col in df.columns]
    value_stocks = df.nlargest(10, 'Value_Score')[available_columns]
    print(value_stocks)
    
    return df

def save_to_excel(df, filename):
    """
    Save DataFrame to Excel with proper formatting
    """
    # Ensure all numeric columns are float or int
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Stock Analysis"
    
    # Convert DataFrame to rows and write to worksheet
    rows = dataframe_to_rows(df, index=False)
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=value)
            
            # Format header row
            if r_idx == 1:
                cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                cell.font = Font(bold=True, color="FFFFFF", size=11)
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                # Format data cells
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        if "%" in df.columns[c_idx-1]:
                            cell.number_format = '0.00"%"'
                        else:
                            cell.number_format = '#,##0.00'
                    else:
                        cell.number_format = '#,##0'
                
                cell.alignment = Alignment(horizontal="right" if isinstance(value, (int, float)) else "left")
    
    # Add borders
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
        for cell in row:
            cell.border = thin_border
    
    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        
        for cell in column:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        
        adjusted_width = max_length + 2
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Freeze top row
    ws.freeze_panes = "A2"
    
    # Add filters
    ws.auto_filter.ref = ws.dimensions
    
    # Save workbook
    wb.save(filename)

if __name__ == "__main__":
    # Test with a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ']
    stock_data = analyze_stocks()
    
    if stock_data is not None:
        print("\nAnalysis complete!")
    else:
        print("\nAnalysis failed. Please check the errors above.")