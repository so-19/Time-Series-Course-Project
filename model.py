import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf
import seaborn as sns
from scipy import stats

def download_stock_data(ticker1, ticker2, start_date, end_date):
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date)
    data = data['Close']
    data.columns = [f"{ticker1}_Close", f"{ticker2}_Close"]
    return data

def calculate_spread(data, stock1_col, stock2_col):
    spread = data[stock1_col] - data[stock2_col]
    spread_name = f"{stock1_col.split('_')[0]}_{stock2_col.split('_')[0]}_Spread"
    data[spread_name] = spread
    return spread_name

def plot_time_series(data, column):
    plt.figure(figsize=(12, 6))
    plt.plot(data[column])
    plt.title(f"{column} Time Series")
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.grid(True)
    plt.savefig(f"{column}_time_series.png")
    plt.close()

def plot_stocks_together(data, stock1_col, stock2_col):
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(data.index, data[stock1_col], 'b-', label=stock1_col)
    ax2.plot(data.index, data[stock2_col], 'r-', label=stock2_col)
    ax1.set_ylabel(stock1_col, color='b')
    ax2.set_ylabel(stock2_col, color='r')
    
    plt.title(f"{stock1_col} vs {stock2_col}")
    plt.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.savefig(f"{stock1_col}_vs_{stock2_col}.png")
    plt.close()

def plot_normalized_stocks(data, stock1_col, stock2_col):
    normalized = pd.DataFrame()
    normalized[stock1_col] = data[stock1_col] / data[stock1_col].iloc[0] * 100
    normalized[stock2_col] = data[stock2_col] / data[stock2_col].iloc[0] * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(normalized[stock1_col], 'b-', label=stock1_col)
    plt.plot(normalized[stock2_col], 'r-', label=stock2_col)
    plt.title(f"Normalized {stock1_col} vs {stock2_col} (Base=100)")
    plt.ylabel('Normalized Price (Base=100)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"normalized_{stock1_col}_vs_{stock2_col}.png")
    plt.close()

def plot_spread_histogram(data, spread_column):
    plt.figure(figsize=(10, 6))
    plt.hist(data[spread_column].dropna(), bins=50, alpha=0.7, color='skyblue')
    sns.kdeplot(data[spread_column].dropna(), color='darkblue')
    plt.axvline(data[spread_column].mean(), color='red', linestyle='dashed', 
                linewidth=2, label=f'Mean: {data[spread_column].mean():.2f}')
    plt.axvline(data[spread_column].median(), color='green', linestyle='dashed', 
                linewidth=2, label=f'Median: {data[spread_column].median():.2f}')
    
    plt.title(f'Distribution of {spread_column}')
    plt.xlabel('Spread Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{spread_column}_histogram.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    stats.probplot(data[spread_column].dropna(), plot=plt)
    plt.title(f'Q-Q Plot of {spread_column}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{spread_column}_qqplot.png")
    plt.close()

def plot_rolling_stats(data, spread_column, window=20):
    rolling_mean = data[spread_column].rolling(window=window).mean()
    rolling_std = data[spread_column].rolling(window=window).std()
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(data[spread_column], label='Spread', color='blue', alpha=0.5)
    plt.plot(rolling_mean, label=f'{window}-day Rolling Mean', color='red')
    plt.legend()
    plt.title(f'Spread and {window}-day Rolling Mean')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(rolling_std, label=f'{window}-day Rolling Std', color='green')
    plt.legend()
    plt.title(f'{window}-day Rolling Standard Deviation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{spread_column}_rolling_stats.png")
    plt.close()

def plot_volatility(data, stock1_col, stock2_col, spread_column, window=20):
    returns1 = data[stock1_col].pct_change()
    returns2 = data[stock2_col].pct_change()
    spread_returns = data[spread_column].pct_change()
    
    vol1 = returns1.rolling(window=window).std() * np.sqrt(252) 
    vol2 = returns2.rolling(window=window).std() * np.sqrt(252) 
    spread_vol = spread_returns.rolling(window=window).std() * np.sqrt(252) 
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(vol1, label=f'{stock1_col} Volatility', color='blue')
    plt.plot(vol2, label=f'{stock2_col} Volatility', color='red')
    plt.legend()
    plt.title(f'{window}-day Rolling Volatility (Annualized)')
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(spread_vol, label=f'{spread_column} Volatility', color='green')
    plt.legend()
    plt.title(f'{window}-day Rolling Spread Volatility (Annualized)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"volatility_comparison.png")
    plt.close()

def plot_rolling_correlation(data, stock1_col, stock2_col, window=30):
    returns1 = data[stock1_col].pct_change()
    returns2 = data[stock2_col].pct_change()
    rolling_corr = returns1.rolling(window=window).corr(returns2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr)
    plt.axhline(y=0, linestyle='dashed', color='red', alpha=0.5)
    plt.axhline(y=rolling_corr.mean(), linestyle='dashed', color='green', 
               label=f'Mean: {rolling_corr.mean():.2f}')
    plt.title(f'{window}-day Rolling Correlation between {stock1_col} and {stock2_col}')
    plt.ylim(-1.1, 1.1) 
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"rolling_correlation_{window}day.png")
    plt.close()

def check_stationarity(data, column):
    rolling_mean = data[column].rolling(window=20).mean()
    rolling_std = data[column].rolling(window=20).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(data[column], label='Original', alpha=0.5)
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.title(f'Rolling Statistics for {column}')
    plt.xlabel('Date')
    plt.grid(True)
    plt.savefig(f"{column}_stationarity_check.png")
    plt.close()
    
    result = sm.tsa.stattools.adfuller(data[column].dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    if result[1] <= 0.05:
        print("Stationary: The time series is stationary (p-value <= 0.05)")
    else:
        print("Non-Stationary: The time series is not stationary (p-value > 0.05)")
    
    return result[1] <= 0.05

def plot_acf_pacf(data, column, lags=40):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(211)
    acf_values = acf(data[column].dropna(), nlags=lags)
    plt.stem(range(len(acf_values)), acf_values)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
    plt.title(f'Autocorrelation Function for {column}')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    
    plt.subplot(212)
    pacf_values = pacf(data[column].dropna(), nlags=lags)
    plt.stem(range(len(pacf_values)), pacf_values)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
    plt.title(f'Partial Autocorrelation Function for {column}')
    plt.xlabel('Lags')
    plt.ylabel('PACF')
    
    plt.tight_layout()
    plt.savefig(f"{column}_acf_pacf.png")
    plt.close()

def decompose_time_series(data, column, period):
    decomposition = seasonal_decompose(data[column], model='additive', period=period)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(data[column], label='Original')
    plt.legend(loc='upper left')
    plt.title(f'Decomposition of {column} Time Series')
    
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='upper left')
    
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonality')
    plt.legend(loc='upper left')
    
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{column}_decomposition.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(decomposition.resid.dropna(), bins=30, alpha=0.7)
    plt.title('Histogram of Residuals')
    
    plt.subplot(122)
    stats.probplot(decomposition.resid.dropna(), plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.savefig(f"{column}_decomposition_residuals.png")
    plt.close()
    
    return decomposition
