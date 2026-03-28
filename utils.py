import numpy as np
import yfinance as yf
from scipy.stats import norm

def fetch_and_process_data(tickers, start_date, end_date):
    """获取数据并计算收益率"""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    tnx = yf.Ticker("^TNX")
    risk_free_rate = tnx.history(period="1d")['Close'].iloc[-1] / 100
    return returns, risk_free_rate

def bs_put_price(S, K, T, r, sigma):
    """B-S 期权定价"""
    S = np.maximum(S, 1e-9) 
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_risk_metrics(portfolio_values, initial_capital):
    """计算 CVaR 风险"""
    var = np.percentile(portfolio_values, 1)
    cvar = portfolio_values[portfolio_values <= var].mean()
    return initial_capital - cvar
