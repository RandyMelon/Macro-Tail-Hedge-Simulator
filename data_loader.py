import yfinance as yf
import numpy as np
import pandas as pd

def get_real_market_params(ticker="QQQ", period="3y"):
    """
    抓取真实数据，并估算 Merton Jump-Diffusion 模型的参数。
    """
    print(f"[*] 正在拉取 {ticker} 过去 {period} 的市场数据...")
    
    # 1. 获取数据 (静默模式，不显示冗长的进度条)
    data = yf.download(ticker, period=period, progress=False)
    
    if data.empty:
        raise ValueError(f"无法获取 {ticker} 的数据，请检查代码或网络。")
        
    # 2. 计算每日对数收益率
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    # 3. 计算基础 GBM 参数
    trading_days = 252
    daily_mean = data['Log_Ret'].mean()
    daily_std = data['Log_Ret'].std()
    
    mu_annual = daily_mean * trading_days
    sigma_annual = daily_std * np.sqrt(trading_days)
    
    # 4. 估算极端跳跃 (3个标准差以外视为极端宏观冲击)
    threshold = 3 * daily_std
    jumps = data[data['Log_Ret'].abs() > threshold]['Log_Ret']
    
    years_of_data = len(data) / trading_days
    lambda_j = len(jumps) / years_of_data 
    
    mu_j = jumps.mean() if len(jumps) > 0 else 0
    sigma_j = jumps.std() if len(jumps) > 1 else 0.01 
    
    # 提取最新价格
    current_price = float(data['Close'].iloc[-1].item())
    
    print(f"[+] 参数提取成功！")
    print(f"    - 最新价格 (S0): ${current_price:.2f}")
    print(f"    - 年化波动率 (Sigma): {sigma_annual:.2%}")
    print(f"    - 年均极端跳跃 (Lambda): {lambda_j:.2f} 次/年")
    
    return {
        "S0": current_price,
        "mu": mu_annual,
        "sigma": sigma_annual,
        "lambda_j": lambda_j,
        "mu_j": mu_j,
        "sigma_j": sigma_j
    }
