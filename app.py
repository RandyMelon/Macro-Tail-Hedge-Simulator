import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd  # 确保引入了 pandas
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 强力数据引擎 ---
@st.cache_data(ttl=3600)
def get_portfolio_data(tickers):
    try:
        # 下载数据
        data = yf.download(tickers, period="3y")['Close']
        if data.empty: return None
        
        # 强制转换为 DataFrame 格式，处理单资产返回 Series 的情况
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers
            
        returns = np.log(data / data.shift(1)).dropna()
        S0 = data.iloc[-1]
        vols = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        
        port_ret = returns.mean(axis=1)
        threshold = 2 * port_ret.std()
        jumps = port_ret[abs(port_ret) > threshold]
        lam = float(len(jumps) / 3)
        
        return S0, vols, corr_matrix, lam, tickers
    except Exception as e:
        st.error(f"数据处理出错: {e}")
        return None

# --- 2. 稳定模拟引擎 ---
def run_portfolio_simulation(S0, vols, corr_matrix, lam, days, iterations):
    T, dt, N = days / 252, 1 / 252, int(days)
    num_assets = len(S0)
    
    # 核心：确保相关性矩阵是 numpy 数组
    corr_array = corr_matrix.values if hasattr(corr_matrix, 'values') else corr_matrix
    
    if num_assets == 1:
        z_corr = np.random.standard_normal((1, iterations))
    else:
        # Cholesky 分解
        L = np.linalg.cholesky(corr_array)
        z_corr = L @ np.random.standard_normal((num_assets, iterations))
    
    drift = (0.05 - 0.5 * vols.values**2) * T
    diffusion = vols.values[:, np.newaxis] * np.sqrt(T) * z_corr
    
    n_jumps = np.random.poisson(lam * T, iterations)
    jump_impact = np.zeros(iterations)
    for i in range(iterations):
        if n_jumps[i] > 0:
            jump_impact[i] = np.sum(np.random.normal(-0.03, 0.05, n_jumps[i]))
            
    terminal_prices = S0.values[:, np.newaxis] * np.exp(drift[:, np.newaxis] + diffusion + jump_impact)
    return terminal_prices.T

# --- 3. UI 界面 ---
st.set_page_config(page_title="量化风控引擎 V4.1", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Portfolio V4.1")

st.sidebar.header("⚙️ 资产配置 (1-3 资产)")
t1 = st.sidebar.text_input("资产 1", value="QQQ").strip().upper()
t2 = st.sidebar.text_input("资产 2 (可选)", value="MSFT").strip().upper()
t3 = st.sidebar.text_input("资产 3 (可选)", value="").strip().upper()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

if num_active == 1:
    weights = np.array([1.0])
elif num_active == 2:
    w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.5)
    weights = np.array([w1, 1.0 - w1])
else:
    w1 = st.sidebar.slider(f"{active_tickers
