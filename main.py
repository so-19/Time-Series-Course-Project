import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import (download_stock_data, plot_time_series, check_stationarity, 
                  plot_acf_pacf, decompose_time_series, calculate_spread,
                  plot_stocks_together, plot_normalized_stocks, plot_spread_histogram,
                  plot_rolling_stats, plot_volatility, plot_rolling_correlation)

def main():
    ticker1 = "AAPL"
    ticker2 = "MSFT"
    start_date = "2019-01-01"
    end_date = "2023-12-31"
    
    print(f"Downloading data for {ticker1} and {ticker2} from {start_date} to {end_date}...")
    data = download_stock_data(ticker1, ticker2, start_date, end_date)
    
    print("Data head:")
    print(data.head())
    
    print("\nData info:")
    print(data.info())
    
    print("\nCalculating spread...")
    spread_column = calculate_spread(data, f"{ticker1}_Close", f"{ticker2}_Close")
    
    print("\nPlotting individual time series...")
    plot_time_series(data, f"{ticker1}_Close")
    plot_time_series(data, f"{ticker2}_Close")
    plot_time_series(data, spread_column)
    
    print("\nPlotting stocks together...")
    plot_stocks_together(data, f"{ticker1}_Close", f"{ticker2}_Close")
    
    print("\nPlotting normalized stocks comparison...")
    plot_normalized_stocks(data, f"{ticker1}_Close", f"{ticker2}_Close")
    
    print("\nPlotting spread distribution...")
    plot_spread_histogram(data, spread_column)
    
    print("\nPlotting rolling statistics of spread...")
    plot_rolling_stats(data, spread_column, window=20)
    
    print("\nAnalyzing volatility...")
    plot_volatility(data, f"{ticker1}_Close", f"{ticker2}_Close", spread_column, window=20)
    
    print("\nPlotting rolling correlation...")
    plot_rolling_correlation(data, f"{ticker1}_Close", f"{ticker2}_Close", window=30)
    
    print("\nChecking stationarity of spread...")
    spread_stationary = check_stationarity(data, spread_column)
    
    print("\nPlotting ACF and PACF of spread...")
    plot_acf_pacf(data, spread_column, lags=40)
    
    print("\nDecomposing spread time series (using 21 trading days as period)...")
    decompose_time_series(data, spread_column, period=21)
    
    print("\nTime series analysis complete. Check the generated plots for visualization.")

if __name__ == "__main__":
    main()
