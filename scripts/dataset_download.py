import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os

OUTPUTDIR = "asset_data"
os.makedirs(OUTPUTDIR, exist_ok=True)
warnings.filterwarnings('ignore')

def fetch_data(tickers, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    print(f"\nRequesting data from Yahoo Finance:")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    
    data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker")
    
    print("\nActual data received:")
    print(f"Start date: {data.index.min()}")
    print(f"End date: {data.index.max()}")
    print(f"Total days: {len(data)}")
    
    return data

def compute_features_and_scale(df):
    """
    Compute log returns and squared returns, then scale features.
    Squared returns are kept unscaled since they should be positive.
    """
    calc_df = df.copy()
    
    # Compute log returns
    calc_df['log_returns'] = np.log(calc_df['Close'] / calc_df['Close'].shift(1))
    calc_df['squared_returns'] = calc_df['log_returns'] ** 2
    
    # Remove NaN values
    calc_df = calc_df.dropna()
    
    # Define columns to scale (excluding squared_returns)
    cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_returns', 'squared_returns']
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(calc_df[cols_to_scale]),
        columns=cols_to_scale,
        index=calc_df.index
    )
    
    # Add unscaled squared returns
    scaled_features['squared_returns'] = calc_df['squared_returns']
    
    print(f"Final dataset shape: {scaled_features.shape}")
    return scaled_features

if __name__ == "__main__":
    # Define parameters
    ASSETS = {
        '^GSPC': 'S&P500',
        '^DJI': 'DowJones',
        '^FTSE': 'FTSE100',
        'CL=F': 'OilSpot',
        'GC=F': 'GoldSpot',
        'USDJPY=X': 'USDJPY'
    }

    tickers = list(ASSETS.keys())
    asset_names = list(ASSETS.values())
    start_date = "2000-01-01"
    end_date = "2025-01-01"

    # Fetch financial data
    financial_data = fetch_data(tickers, start_date, end_date)

    # Process each ticker
    for ticker, asset_name in zip(tickers, asset_names):
        if ticker in financial_data.columns.levels[0]:
            print(f"\nProcessing {ticker}...")
            
            # Get ticker data
            ticker_data = financial_data[ticker]
            
            print(f"Data period: {ticker_data.index.min()} to {ticker_data.index.max()}")
            
            # Process and scale the data
            processed_data = compute_features_and_scale(ticker_data)
            
            # Add asset name prefix to columns
            prefix = asset_name.lower()
            processed_data.columns = [f"{prefix}_{col}" for col in processed_data.columns]
            
            # Save to CSV
            output_path = os.path.join(OUTPUTDIR, f"{prefix}_data.csv")
            processed_data.to_csv(output_path)
            
            print(f"✓ Saved {asset_name} data to {output_path}")
            print(f"  Shape: {processed_data.shape}")
            print(f"  Date range: {processed_data.index.min()} to {processed_data.index.max()}") 