import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 核心数据与模拟引擎 ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    hist = yf.Ticker(ticker).history(period="3y")
    if hist.empty:
        raise ValueError(f"无法获取 {ticker} 的数据，请检查代码。")
    
    close_prices = hist['Close']
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    S0 = float(close_prices.iloc[-1])
    sigma = float(returns.std() * np.sqrt(252))
    
    threshold = 2 * returns.std()
    jumps = returns[abs(returns) > threshold]
    lam = float(len(jumps) / 3)
    mu_j = float(jumps.mean() if len(jumps) > 0 else 0.0)
    sigma_j = float(jumps.std() if len(jumps) > 0 else 0.05)
    
    return S0, sigma, lam, mu_j, sigma_j

def run_simulation(S0, sigma, lam, mu_j, sigma_j, days, iterations):
    T = days / 252
    dt = 1 / 252
    N = int(days)
    prices = np.zeros((iterations, N + 1))
    prices[:, 0] = S0
    for i in range(1, N + 1):
        z = np.random.standard_normal(iterations)
        n_jumps = np.random.poisson(lam * dt, iterations)
        jump_sum = np.random.normal(mu_j, sigma_j, iterations) * n_jumps
        drift = (0.05 - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        prices[:, i] = prices[:, i-1] * np.exp(drift + diffusion + jump_sum)
    return prices[:, -1]

# --- 网页前端 UI 设计 ---
st.set_page_config(page_title="量化风控引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge-Simulator")
st.markdown("基于真实市场数据的 MJD 蒙特卡洛压力测试与期权对冲模拟系统。")
st.divider()

st.sidebar.header("⚙️ 资产配置面板")
ticker = st.sidebar.text_input("输入美股/外汇代码 (如 QQQ, MSFT, JPY=X)", value="QQQ").upper()
capital = st.sidebar.number_input("投资组合规模 ($)", value=100000, step=10000)
days = st.sidebar.slider("压力测试天数", min_value=5, max_value=60, value=22)
iters = st.sidebar.selectbox("蒙特卡洛模拟路径数", [1000, 5000, 10000, 50000], index=2)

if st.sidebar.button("🚀 运行实盘压力测试", type="primary"):
    with st.spinner(f"正在抓取 {ticker} 实时数据并执行并行计算..."):
        try:
            S0, sigma, lam, mu_j, sigma_j = get_market_data(ticker)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.info(f"**最新标的价格**\n\n### ${S0:.2f}")
            col2.info(f"**历史年化波动率**\n\n### {sigma*100:.2f}%")
            col3.info(f"**极端跳跃频率**\n\n### {lam:.1f} 次/年")
            col4.info(f"**平均跳跃幅度**\n\n### {mu_j*100:.2f}%")
            
            terminal_prices = run_simulation(S0, sigma, lam, mu_j, sigma_j, days, iters)
            
            K = S0 * 0.90
            T = days / 252
            d1 = (np.log(S0/K) + (0.05 + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            put_price = K * np.exp(-0.05*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            
            naked_pnl = (terminal_prices - S0) / S0 * capital
            option_payoff = np.maximum(K - terminal_prices, 0)
            hedged_pnl = naked_pnl + (option_payoff - put_price) / S0 * capital
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(naked_pnl, bins=80, alpha=0.5, color='#ff9999', label='Naked Portfolio')
            ax.hist(hedged_pnl, bins=80, alpha=0.5, color='#66b3ff', label='Hedged with 10% OTM Put')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"Portfolio PnL Distribution ({ticker} - {days} Days)", fontsize=14)
            ax.set_xlabel("Profit / Loss ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            st.markdown("### 📊 压力测试结果报告")
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.error(f"**最大裸头寸回撤**: \n\n${abs(min(naked_pnl)):,.2f}")
                st.success(f"**对冲后最大回撤**: \n\n${abs(min(hedged_pnl)):,.2f}")
                st.info(f"**期权对冲总成本**: \n\n${(put_price/S0)*capital:,.2f}")
            with res_col2:
                st.pyplot(fig)
        except Exception as e:
            st.error(f"运行出错: {e}")
else:
    st.info("👈 请在左侧配置参数，并点击运行按钮启动引擎。")
