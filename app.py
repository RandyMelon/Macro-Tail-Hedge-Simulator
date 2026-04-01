import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. 抓取真实期权隐含波动率 (IV) ---
@st.cache_data(ttl=3600)
def get_real_iv(ticker, target_K):
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None 
        
        opt = tk.option_chain(exps[0])
        puts = opt.puts
        
        idx = (np.abs(puts['strike'] - target_K)).argmin()
        real_iv = puts.iloc[idx]['impliedVolatility']
        
        if real_iv > 0.01: 
            return float(real_iv)
        return None
    except:
        return None

# --- 2. 抓取基础数据与跳跃特征 ---
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

# --- 3. MJD 蒙特卡洛模拟核心 ---
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

# --- 4. Web App UI 界面 ---
st.set_page_config(page_title="量化风控引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge-Simulator")
st.markdown("基于真实市场数据的 MJD 蒙特卡洛压力测试与期权对冲模拟系统。")
st.divider()

st.sidebar.header("⚙️ 资产配置面板")
ticker = st.sidebar.text_input("输入美股代码 (如 QQQ, MSFT, AAPL)", value="QQQ").upper()
capital = st.sidebar.number_input("投资组合规模 ($)", value=100000, step=10000)
days = st.sidebar.slider("压力测试天数", min_value=5, max_value=60, value=22)
iters = st.sidebar.selectbox("蒙特卡洛模拟路径数", [1000, 5000, 10000, 50000], index=2)

if st.sidebar.button("🚀 运行实盘压力测试", type="primary"):
    with st.spinner(f"正在抓取 {ticker} 实时数据并执行计算..."):
        try:
            # 数据解包
            S0, sigma, lam, mu_j, sigma_j = get_market_data(ticker)
            
            # 顶部数据面板
            col1, col2, col3, col4 = st.columns(4)
            col1.info(f"**最新标的价格**\n\n### ${S0:.2f}")
            col2.info(f"**历史年化波动率**\n\n### {sigma*100:.2f}%")
            col3.info(f"**极端跳跃频率**\n\n### {lam:.1f} 次/年")
            col4.info(f"**平均跳跃幅度**\n\n### {mu_j*100:.2f}%")
            
            # 运行模拟
            terminal_prices = run_simulation(S0, sigma, lam, mu_j, sigma_j, days, iters)
            
            # 期权定价 (引入真实 IV)
            K = S0 * 0.90 
            T = days / 252
            
            real_iv = get_real_iv(ticker, K)
            pricing_vol = real_iv if real_iv else (sigma + 0.03) 
            
            if real_iv:
                st.toast(f"✅ 成功抓取 {ticker} 真实市场隐含波动率: {real_iv*100:.2f}%", icon="📈")
            else:
                st.toast(f"⚠️ 未抓取到期权链，使用历史波动率定价", icon="🧮")

            d1 = (np.log(S0/K) + (0.05 + 0.5*pricing_vol**2)*T) / (pricing_vol*np.sqrt(T))
            d2 = d1 - pricing_vol*np.sqrt(T)
            put_price = K * np.exp(-0.05*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            
            # 计算 PnL
            naked_pnl = (terminal_prices - S0) / S0 * capital
            option_payoff = np.maximum(K - terminal_prices, 0)
            hedged_pnl = naked_pnl + (option_payoff - put_price) / S0 * capital
            
            # CVaR 风险指标计算
            percentile = 1  
            naked_var = np.percentile(naked_pnl, percentile)
            naked_cvar = naked_pnl[naked_pnl <= naked_var].mean()
            
            hedged_var = np.percentile(hedged_pnl, percentile)
            hedged_cvar = hedged_pnl[hedged_pnl <= hedged_var].mean()

            # 画图
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(naked_pnl, bins=80, alpha=0.5, color='#ff9999', label='Naked Portfolio')
            ax.hist(hedged_pnl, bins=80, alpha=0.5, color='#66b3ff', label='Hedged with 10% OTM Put')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.set_title(f"Portfolio PnL Distribution ({ticker} - {days} Days)", fontsize=14)
            ax.set_xlabel("Profit / Loss ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            # 渲染底部排版
            st.markdown("### 📊 机构级尾部风险报告 (99% CVaR)")
            res_col1, res_col2 = st.columns([1, 2]) 
            
            with res_col1:
                st.error(f"**裸头寸预期极寒亏损 (Naked CVaR)**: \n\n### ${abs(naked_cvar):,.2f}")
                st.success(f"**对冲后预期亏损 (Hedged CVaR)**: \n\n### ${abs(hedged_cvar):,.2f}")
                st.info(f"**期权对冲总成本 (Hedging Cost)**: \n\n### ${((put_price/S0)*capital):,.2f}")
                
            with res_col2:
                st.pyplot(fig) 
                
        except ValueError as ve:
            st.error(f"⚠️ 数据获取失败: {ve}")
        except Exception as e:
            st.error(f"❌ 发生未知错误: {e}")
else:
    st.info("👈 请在左侧侧边栏配置参数，并点击运行按钮启动引擎。")
