import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 数据引擎 ---
@st.cache_data(ttl=3600)
def get_portfolio_data(tickers):
    try:
        data = yf.download(tickers, period="3y")['Close']
        if data.empty: return None
        # 如果是单资产，yf 返回的是 Series，转成 DataFrame 统一处理
        if isinstance(data, np.ndarray) or isinstance(data, pd.Series):
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
    except:
        return None

# --- 2. 模拟引擎 ---
def run_portfolio_simulation(S0, vols, corr_matrix, lam, days, iterations):
    T, dt, N = days / 252, 1 / 252, int(days)
    num_assets = len(S0)
    
    if num_assets == 1:
        z_corr = np.random.standard_normal((1, iterations))
    else:
        L = np.linalg.cholesky(corr_matrix)
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

# --- 3. UI 布局 ---
st.set_page_config(page_title="多资产风控引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Portfolio V4.0")

st.sidebar.header("⚙️ 资产配置 (支持 1-3 资产)")
t1 = st.sidebar.text_input("资产 1", value="QQQ").strip().upper()
t2 = st.sidebar.text_input("资产 2 (可选)", value="MSFT").strip().upper()
t3 = st.sidebar.text_input("资产 3 (可选)", value="").strip().upper()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

if num_active == 1:
    weights = np.array([1.0])
    st.sidebar.info(f"模式：单资产压力测试 ({active_tickers[0]})")
elif num_active == 2:
    w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.5)
    weights = np.array([w1, 1.0 - w1])
else:
    w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.4)
    w2 = st.sidebar.slider(f"{active_tickers[1]} 权重", 0.0, 1.0 - w1, 0.3)
    weights = np.array([w1, w2, round(1.0 - w1 - w2, 2)])

capital = st.sidebar.number_input("投资规模 ($)", value=100000)
days = st.sidebar.slider("测试天数", 5, 60, 22)

if st.sidebar.button("🚀 启动压力测试", type="primary"):
    result = get_portfolio_data(active_tickers)
    if result:
        S0, vols, corr_matrix, lam, tickers = result
        prices_matrix = run_portfolio_simulation(S0, vols, corr_matrix, lam, days, 10000)
        
        asset_returns = np.nan_to_num((prices_matrix - S0.values) / S0.values)
        port_returns = (asset_returns @ weights)
        combined_pnl = port_returns * capital
        
        var_99 = np.percentile(combined_pnl, 1)
        cvar_99 = combined_pnl[combined_pnl <= var_99].mean()
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("### 🧊 资产相关性矩阵")
            if num_active > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax_corr, square=True)
                plt.xticks(rotation=45); plt.yticks(rotation=0)
                st.pyplot(fig_corr)
            else:
                st.success("单资产模式：无需计算相关性。")
                
        with col_b:
            st.write("### 📉 组合盈亏分布")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
            ax_hist.hist(combined_pnl, bins=80, color='#66b3ff', alpha=0.7)
            ax_hist.axvline(var_99, color='red', linestyle='--', label=f'VaR 99%')
            ax_hist.legend(); st.pyplot(fig_hist)
            
        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("组合资产数", num_active)
        r2.error(f"**99% CVaR (预期极端亏损)**\n\n### ${abs(cvar_99):,.2f}")
        r3.info(f"**最大潜在回撤**\n\n### ${abs(np.min(combined_pnl)):,.2f}")
