import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf
import requests
import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Set API keys
fred_api_key = '28e06e7d302abe40f5c313ffe9e16beb'  # Your FRED API key
coingecko_api_url = 'https://api.coingecko.com/api/v3'

# Step 2: Initialize Fred
fred = Fred(api_key=fred_api_key)

# Step 3: Define date range (increase time range to ensure enough data points)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # Use 2 years of data instead of 1
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Debug information
print(f"Fetching data from {start_date_str} to {end_date_str}")

# Step 4: Fetch macroeconomic data from FRED
try:
    cpi = fred.get_series('CPIAUCSL', observation_start=start_date_str, observation_end=end_date_str)  # CPI
    unemployment = fred.get_series('UNRATE', observation_start=start_date_str, observation_end=end_date_str)  # Unemployment Rate
    print(f"FRED data fetched. CPI length: {len(cpi)}, Unemployment length: {len(unemployment)}")
except Exception as e:
    print(f"Error fetching FRED data: {e}")
    print("Please ensure your FRED API key is valid and you have a stable internet connection")
    exit()

# Step 5: Fetch S&P 500 data from Yahoo Finance
try:
    sp500_df = yf.download('^GSPC', start=start_date_str, end=end_date_str, interval='1mo', auto_adjust=True)
    sp500 = sp500_df['Close'].squeeze()  # Ensure it's a Series
    print(f"S&P 500 data fetched. Length: {len(sp500)}")
except Exception as e:
    print(f"Error fetching S&P 500 data from Yahoo Finance: {e}")
    print("Please ensure you have a stable internet connection")
    exit()

# Step 6: Fetch Bitcoin data from CoinGecko and resample to monthly
# Use the maximum days available in free tier and handle response properly
try:
    response = requests.get(
        f'{coingecko_api_url}/coins/bitcoin/market_chart', 
        params={'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    response_data = response.json()
    
    if 'prices' not in response_data:
        print("Error: 'prices' key not found in CoinGecko response. Full response:")
        print(response_data)
        raise KeyError("'prices' key missing in CoinGecko API response")
    
    prices = response_data['prices']
    bitcoin_df = pd.DataFrame(prices, columns=['date', 'price'])
    bitcoin_df['date'] = pd.to_datetime(bitcoin_df['date'], unit='ms')
    bitcoin_df.set_index('date', inplace=True)
    bitcoin_monthly = bitcoin_df['price'].resample('ME').mean()
    print(f"Bitcoin data fetched. Length: {len(bitcoin_monthly)}")
except Exception as e:
    print(f"Error fetching Bitcoin data from CoinGecko: {e}")
    print("Please ensure you have a stable internet connection")
    exit()

# Step 7: Align data to monthly frequency
# Create a monthly date range and align all data series to it
print(f"CPI index: {cpi.index}")
print(f"Unemployment index: {unemployment.index}")
print(f"S&P 500 index: {sp500.index}")
print(f"Bitcoin index: {bitcoin_monthly.index}")

# Convert all data to monthly frequency if not already
if not isinstance(cpi.index, pd.DatetimeIndex):
    cpi.index = pd.to_datetime(cpi.index)
if not isinstance(unemployment.index, pd.DatetimeIndex):
    unemployment.index = pd.to_datetime(unemployment.index)

# Resample to ensure all are at month-end frequency
cpi_monthly = cpi.resample('ME').last()
unemployment_monthly = unemployment.resample('ME').last()
sp500_monthly = sp500.resample('ME').last()

# Find the common date range for all datasets
common_start = max(
    cpi_monthly.index.min(),
    unemployment_monthly.index.min(),
    sp500_monthly.index.min(),
    bitcoin_monthly.index.min()
)
common_end = min(
    cpi_monthly.index.max(),
    unemployment_monthly.index.max(),
    sp500_monthly.index.max(),
    bitcoin_monthly.index.max()
)

print(f"Common date range: {common_start} to {common_end}")

# Create a common date range
date_range = pd.date_range(start=common_start, end=common_end, freq='ME')

# Align all data to this common range
cpi_monthly = cpi_monthly.reindex(date_range)
unemployment_monthly = unemployment_monthly.reindex(date_range)
sp500_monthly = sp500_monthly.reindex(date_range)
bitcoin_monthly = bitcoin_monthly.reindex(date_range)

# Forward fill any missing values (maximum 2 consecutive missing values)
cpi_monthly = cpi_monthly.fillna(method='ffill', limit=2)
unemployment_monthly = unemployment_monthly.fillna(method='ffill', limit=2)
sp500_monthly = sp500_monthly.fillna(method='ffill', limit=2)
bitcoin_monthly = bitcoin_monthly.fillna(method='ffill', limit=2)

# Step 8: Merge into a DataFrame
data = pd.DataFrame({
    'CPI': cpi_monthly,
    'Unemployment': unemployment_monthly,
    'SP500': sp500_monthly,
    'Bitcoin': bitcoin_monthly
})

# Print data info before calculations
print("\nData before calculating changes:")
print(data.describe())
print(f"NaN count: {data.isna().sum()}")

# Step 9: Calculate changes with explicit fill_method=None to avoid warnings
data['CPI_change'] = data['CPI'].pct_change(fill_method=None)
data['Unemployment_change'] = data['Unemployment'].diff()
data['SP500_return'] = data['SP500'].pct_change(fill_method=None)
data['Bitcoin_return'] = data['Bitcoin'].pct_change(fill_method=None)

# Step 10: Drop missing values and confirm we have data to work with
data = data.dropna()
print(f"\nShape after dropping NaN values: {data.shape}")

if len(data) < 3:  # Not enough data points for meaningful analysis
    print("ERROR: Not enough valid data points after preprocessing.")
    print("Try increasing the date range or check the data sources.")
    exit()

# Step 11: Calculate correlation
correlation = data[['CPI_change', 'Unemployment_change', 'SP500_return', 'Bitcoin_return']].corr()
print("\nCorrelation Matrix:")
print(correlation)

# Add visualization for correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=.5)
plt.title('Correlation Matrix of Economic Indicators and Asset Returns', fontsize=16)
plt.tight_layout()
plt.savefig('c:/Users/neves/OneDrive/Ambiente de Trabalho/Projects/Post Fiat Task 11/correlation_heatmap.png', dpi=300)
print("Correlation heatmap saved to correlation_heatmap.png")

# Step 12: Run regression analysis
X = data[['CPI_change', 'Unemployment_change']]
X = sm.add_constant(X)

# Check if we have valid data for regression
if X.shape[0] > 0 and not X.isna().any().any():
    # Regression for S&P 500
    y_sp500 = data['SP500_return']
    model_sp500 = sm.OLS(y_sp500, X).fit()
    print("\nRegression Results for S&P 500:")
    print(model_sp500.summary())

    # Regression for Bitcoin
    y_bitcoin = data['Bitcoin_return']
    model_bitcoin = sm.OLS(y_bitcoin, X).fit()
    print("\nRegression Results for Bitcoin:")
    print(model_bitcoin.summary())
    
    # Visualization for regression results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot actual vs fitted values for S&P 500
    axes[0, 0].scatter(y_sp500, model_sp500.fittedvalues)
    axes[0, 0].plot([y_sp500.min(), y_sp500.max()], [y_sp500.min(), y_sp500.max()], 'k--')
    axes[0, 0].set_xlabel('Actual S&P 500 Returns')
    axes[0, 0].set_ylabel('Predicted S&P 500 Returns')
    axes[0, 0].set_title('S&P 500: Actual vs Predicted Returns')
    
    # Plot residuals for S&P 500
    axes[0, 1].scatter(model_sp500.fittedvalues, model_sp500.resid)
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_xlabel('Predicted S&P 500 Returns')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('S&P 500 Regression Residuals')
    
    # Plot actual vs fitted values for Bitcoin
    axes[1, 0].scatter(y_bitcoin, model_bitcoin.fittedvalues)
    axes[1, 0].plot([y_bitcoin.min(), y_bitcoin.max()], [y_bitcoin.min(), y_bitcoin.max()], 'k--')
    axes[1, 0].set_xlabel('Actual Bitcoin Returns')
    axes[1, 0].set_ylabel('Predicted Bitcoin Returns')
    axes[1, 0].set_title('Bitcoin: Actual vs Predicted Returns')
    
    # Plot residuals for Bitcoin
    axes[1, 1].scatter(model_bitcoin.fittedvalues, model_bitcoin.resid)
    axes[1, 1].axhline(y=0, color='k', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Bitcoin Returns')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Bitcoin Regression Residuals')
    
    plt.tight_layout()
    plt.savefig('c:/Users/neves/OneDrive/Ambiente de Trabalho/Projects/Post Fiat Task 11/regression_analysis.png', dpi=300)
    print("Regression analysis plots saved to regression_analysis.png")
    
    # Time series plot of asset returns and economic indicators
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Asset Returns
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['SP500_return'], label='S&P 500 Returns', color='blue')
    plt.plot(data.index, data['Bitcoin_return'], label='Bitcoin Returns', color='orange')
    plt.title('Asset Returns Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Economic Indicators
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['CPI_change'], label='CPI Change (Inflation)', color='green')
    plt.plot(data.index, data['Unemployment_change'], label='Unemployment Rate Change', color='red')
    plt.title('Economic Indicators Over Time', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:/Users/neves/OneDrive/Ambiente de Trabalho/Projects/Post Fiat Task 11/time_series_analysis.png', dpi=300)
    print("Time series analysis saved to time_series_analysis.png")

    # Step 13: Summarize findings
    print("\nSummary of Findings:")
    print(f"Correlation between CPI change and S&P 500 returns: {correlation.loc['CPI_change', 'SP500_return']:.4f}")
    print(f"Correlation between Unemployment change and S&P 500 returns: {correlation.loc['Unemployment_change', 'SP500_return']:.4f}")
    print(f"Correlation between CPI change and Bitcoin returns: {correlation.loc['CPI_change', 'Bitcoin_return']:.4f}")
    print(f"Correlation between Unemployment change and Bitcoin returns: {correlation.loc['Unemployment_change', 'Bitcoin_return']:.4f}")

    # Interpret regression significance
    if model_sp500.pvalues['CPI_change'] < 0.05:
        print(f"CPI change significantly predicts S&P 500 returns (coefficient: {model_sp500.params['CPI_change']:.4f}).")
    else:
        print("CPI change does not significantly predict S&P 500 returns.")

    if model_sp500.pvalues['Unemployment_change'] < 0.05:
        print(f"Unemployment change significantly predicts S&P 500 returns (coefficient: {model_sp500.params['Unemployment_change']:.4f}).")
    else:
        print("Unemployment change does not significantly predict S&P 500 returns.")

    if model_bitcoin.pvalues['CPI_change'] < 0.05:
        print(f"CPI change significantly predicts Bitcoin returns (coefficient: {model_bitcoin.params['CPI_change']:.4f}).")
    else:
        print("CPI change does not significantly predict Bitcoin returns.")

    if model_bitcoin.pvalues['Unemployment_change'] < 0.05:
        print(f"Unemployment change significantly predicts Bitcoin returns (coefficient: {model_bitcoin.params['Unemployment_change']:.4f}).")
    else:
        print("Unemployment change does not significantly predict Bitcoin returns.")

    # Step 14: Highlight potential trading signals
    if correlation.loc['Unemployment_change', 'SP500_return'] < -0.3:
        print("Potential Trading Signal: Rising unemployment may signal a decline in S&P 500.")
    if correlation.loc['CPI_change', 'Bitcoin_return'] > 0.3:
        print("Potential Trading Signal: Rising inflation may drive Bitcoin returns higher.")

    # Create a text file with all findings
    results_file_path = 'c:/Users/neves/OneDrive/Ambiente de Trabalho/Projects/Post Fiat Task 11/analysis_results.txt'
    with open(results_file_path, 'w') as f:
        f.write("ECONOMIC INDICATORS AND ASSET RETURNS ANALYSIS\n")
        f.write("==========================================\n\n")
        f.write(f"Analysis Period: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"Number of data points after preprocessing: {len(data)}\n\n")
        
        f.write("CORRELATION ANALYSIS\n")
        f.write("-------------------\n")
        f.write(f"Correlation between CPI change and S&P 500 returns: {correlation.loc['CPI_change', 'SP500_return']:.4f}\n")
        f.write(f"Correlation between Unemployment change and S&P 500 returns: {correlation.loc['Unemployment_change', 'SP500_return']:.4f}\n")
        f.write(f"Correlation between CPI change and Bitcoin returns: {correlation.loc['CPI_change', 'Bitcoin_return']:.4f}\n")
        f.write(f"Correlation between Unemployment change and Bitcoin returns: {correlation.loc['Unemployment_change', 'Bitcoin_return']:.4f}\n\n")
        
        f.write("REGRESSION ANALYSIS - S&P 500\n")
        f.write("---------------------------\n")
        f.write(f"R-squared: {model_sp500.rsquared:.4f}\n")
        f.write(f"Adjusted R-squared: {model_sp500.rsquared_adj:.4f}\n")
        f.write(f"F-statistic: {model_sp500.fvalue:.4f} (p-value: {model_sp500.f_pvalue:.4f})\n\n")
        f.write("Coefficients:\n")
        f.write(f"Constant: {model_sp500.params['const']:.6f} (p-value: {model_sp500.pvalues['const']:.4f})\n")
        f.write(f"CPI change: {model_sp500.params['CPI_change']:.6f} (p-value: {model_sp500.pvalues['CPI_change']:.4f})\n")
        f.write(f"Unemployment change: {model_sp500.params['Unemployment_change']:.6f} (p-value: {model_sp500.pvalues['Unemployment_change']:.4f})\n\n")
        
        f.write("REGRESSION ANALYSIS - Bitcoin\n")
        f.write("---------------------------\n")
        f.write(f"R-squared: {model_bitcoin.rsquared:.4f}\n")
        f.write(f"Adjusted R-squared: {model_bitcoin.rsquared_adj:.4f}\n")
        f.write(f"F-statistic: {model_bitcoin.fvalue:.4f} (p-value: {model_bitcoin.f_pvalue:.4f})\n\n")
        f.write("Coefficients:\n")
        f.write(f"Constant: {model_bitcoin.params['const']:.6f} (p-value: {model_bitcoin.pvalues['const']:.4f})\n")
        f.write(f"CPI change: {model_bitcoin.params['CPI_change']:.6f} (p-value: {model_bitcoin.pvalues['CPI_change']:.4f})\n")
        f.write(f"Unemployment change: {model_bitcoin.params['Unemployment_change']:.6f} (p-value: {model_bitcoin.pvalues['Unemployment_change']:.4f})\n\n")
        
        f.write("INTERPRETATIONS & TRADING SIGNALS\n")
        f.write("-------------------------------\n")
        
        # Interpretation of S&P 500 regression
        if model_sp500.pvalues['CPI_change'] < 0.05:
            f.write(f"CPI change significantly predicts S&P 500 returns (coefficient: {model_sp500.params['CPI_change']:.4f}).\n")
            if model_sp500.params['CPI_change'] > 0:
                f.write("This suggests higher inflation is associated with higher S&P 500 returns.\n")
            else:
                f.write("This suggests higher inflation is associated with lower S&P 500 returns.\n")
        else:
            f.write("CPI change does not significantly predict S&P 500 returns.\n")

        if model_sp500.pvalues['Unemployment_change'] < 0.05:
            f.write(f"Unemployment change significantly predicts S&P 500 returns (coefficient: {model_sp500.params['Unemployment_change']:.4f}).\n")
            if model_sp500.params['Unemployment_change'] > 0:
                f.write("This suggests rising unemployment is associated with higher S&P 500 returns.\n")
            else:
                f.write("This suggests rising unemployment is associated with lower S&P 500 returns.\n")
        else:
            f.write("Unemployment change does not significantly predict S&P 500 returns.\n")
        
        # Interpretation of Bitcoin regression
        if model_bitcoin.pvalues['CPI_change'] < 0.05:
            f.write(f"CPI change significantly predicts Bitcoin returns (coefficient: {model_bitcoin.params['CPI_change']:.4f}).\n")
            if model_bitcoin.params['CPI_change'] > 0:
                f.write("This suggests higher inflation is associated with higher Bitcoin returns.\n")
            else:
                f.write("This suggests higher inflation is associated with lower Bitcoin returns.\n")
        else:
            f.write("CPI change does not significantly predict Bitcoin returns.\n")

        if model_bitcoin.pvalues['Unemployment_change'] < 0.05:
            f.write(f"Unemployment change significantly predicts Bitcoin returns (coefficient: {model_bitcoin.params['Unemployment_change']:.4f}).\n")
            if model_bitcoin.params['Unemployment_change'] > 0:
                f.write("This suggests rising unemployment is associated with higher Bitcoin returns.\n")
            else:
                f.write("This suggests rising unemployment is associated with lower Bitcoin returns.\n")
        else:
            f.write("Unemployment change does not significantly predict Bitcoin returns.\n")
        
        # Trading signals
        f.write("\nPOTENTIAL TRADING SIGNALS\n")
        f.write("-----------------------\n")
        if correlation.loc['Unemployment_change', 'SP500_return'] < -0.3:
            f.write("- Rising unemployment may signal a decline in S&P 500.\n")
        if correlation.loc['CPI_change', 'Bitcoin_return'] > 0.3:
            f.write("- Rising inflation may drive Bitcoin returns higher.\n")
        
        f.write("\nAnalysis completed on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"\nResults saved to text file: {results_file_path}")
    
    # Additional pie charts for portfolio allocation suggestion based on findings
    # Only if there are significant findings
    if (model_sp500.pvalues['CPI_change'] < 0.05 or model_sp500.pvalues['Unemployment_change'] < 0.05 or
        model_bitcoin.pvalues['CPI_change'] < 0.05 or model_bitcoin.pvalues['Unemployment_change'] < 0.05):
        
        plt.figure(figsize=(15, 6))
        
        # Create allocation suggestions based on current economic conditions
        current_cpi = data['CPI_change'].iloc[-1]
        current_unemp = data['Unemployment_change'].iloc[-1]
        
        # Basic allocation logic based on current conditions and regression results
        btc_weight = 0.3  # Base allocation
        sp500_weight = 0.5  # Base allocation
        cash_weight = 0.2   # Base allocation
        
        # Adjust weights based on economic conditions and regression results
        if current_cpi > 0 and model_bitcoin.params['CPI_change'] > 0:
            btc_weight += 0.1
            sp500_weight -= 0.1
        elif current_cpi > 0 and model_sp500.params['CPI_change'] > 0:
            sp500_weight += 0.1
            btc_weight -= 0.1
            
        if current_unemp > 0 and model_bitcoin.params['Unemployment_change'] < 0:
            btc_weight -= 0.1
            cash_weight += 0.1
        elif current_unemp > 0 and model_sp500.params['Unemployment_change'] < 0:
            sp500_weight -= 0.1
            cash_weight += 0.1
            
        # Make sure weights sum to 1
        total = btc_weight + sp500_weight + cash_weight
        btc_weight /= total
        sp500_weight /= total
        cash_weight /= total
        
        # Create pie chart
        plt.subplot(1, 2, 1)
        labels = ['Bitcoin', 'S&P 500', 'Cash']
        sizes = [btc_weight, sp500_weight, cash_weight]
        colors = ['#f7931a', '#4169e1', '#4caf50']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Suggested Portfolio Allocation\nBased on Current Economic Conditions')
        
        # Add a summary of current conditions
        plt.subplot(1, 2, 2)
        plt.axis('off')
        conditions_text = f"""
        CURRENT ECONOMIC CONDITIONS:
        
        • Inflation (CPI change): {current_cpi:.4f}
          {'Increasing' if current_cpi > 0 else 'Decreasing'}
        
        • Unemployment change: {current_unemp:.4f}
          {'Increasing' if current_unemp > 0 else 'Decreasing'}
        
        • S&P 500 recent return: {data['SP500_return'].iloc[-1]:.4f}
        
        • Bitcoin recent return: {data['Bitcoin_return'].iloc[-1]:.4f}
        
        ALLOCATION REASONING:
        
        Based on historical relationships between
        macroeconomic indicators and asset returns,
        this allocation aims to optimize performance
        given current economic conditions.
        """
        plt.text(0, 0.5, conditions_text, fontsize=10, va='center')
        
        plt.tight_layout()
        plt.savefig('c:/Users/neves/OneDrive/Ambiente de Trabalho/Projects/Post Fiat Task 11/portfolio_allocation.png', dpi=300)
        print("Portfolio allocation suggestion saved to portfolio_allocation.png")
        
else:
    print("ERROR: Not enough valid data for regression analysis.")
    print("Data preview:")
    print(X.head())
    print("NaN count in regression variables:")
    print(X.isna().sum())