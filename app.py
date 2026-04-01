import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 强力数据引擎 (兼容单/多资产结构) ---
@st.cache_data(ttl=300)
def get_portfolio_data(tickers):
    try:
        # 抓取数据，不再强制 group_by，让 yfinance 自行处理
        data = yf.download(tickers, period="3y")
        if data.empty:
            return None
        
        # 核心修复逻辑：智能提取收盘价
        if len(tickers) > 1:
            # 多资产时，'Close' 是 DataFrame 的一个 Level
            close_df = data['Close']
        else:
            # 单资产时，'Close' 是其中的一列，强制转为以 Ticker 命名的 DataFrame
            close_df = data['Close'].to_frame()
            close_df.columns = tickers
        
        close_df = close_df.dropna()
        returns = np.log(close_df / close_df.shift(1)).dropna()
        
        S0 = close_df.iloc[-1]
        vols = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        
        # 跳跃参数
        port_ret = returns.mean(axis=1)
        lam = float(len(port_ret[abs(port_ret) > 2 * port_ret.std()]) / 3)
        
        return S0, vols, corr_matrix, lam, tickers
    except Exception as e:
        st.error(f"❌ 数据获取环节出错: {str(e)}")
        return None

# --- 2. 模拟引擎 ---
def run_portfolio_simulation(S0, vols, corr_matrix, lam, days, iterations):
    try:
        num_assets = len(S0)
        T, dt = days / 252, 1 / 252
        
        if num_assets > 1:
            L = np.linalg.cholesky(corr_matrix.values)
            z = L @ np.random.standard_normal((num_assets, iterations))
        else:
            z = np.random.standard_normal((1, iterations))
            
        drift = (0.05 - 0.5 * vols.values**2) * T
        diffusion = vols.values[:, np.newaxis] * np.sqrt(T) * z
        
        # 模拟跳跃
        n_jumps = np.random.poisson(lam * T, iterations)
        jump_impact = np.array([np.sum(np.random.normal(-0.03, 0.05, n_jumps[i])) if n_jumps[i] > 0 else 0 for i in range(iterations)])
        
        prices = S0.values[:, np.newaxis] * np.exp(drift[:, np.newaxis] + diffusion + jump_impact)
        return prices.T
    except Exception as e:
        st.error(f"❌ 模拟计算环节出错: {str(e)}")
        return None

# --- 3. UI 界面 ---
st.set_page_config(page_title="量化风险管理引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Portfolio Simulator")
st.markdown("多资产联动压力测试系统：利用 **Cholesky Decomposition** 与 **MJD 模型** 捕捉尾部风险。")

st.sidebar.header("⚙️ 资产配置面板")
t1 = st.sidebar.text_input("资产 1 (必填)", value="QQQ").upper().strip()
t2 = st.sidebar.text_input("资产 2 (可选)", value="").upper().strip()
t3 = st.sidebar.text_input("资产 3 (可选)", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

# 动态权重逻辑
weights = []
if num_active > 0:
    if num_active == 1:
        weights = [1.0]
        st.sidebar.success(f"已锁定 {active_tickers[0]} 100% 权重")
    elif num_active == 2:
        w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.5)
        weights = [w1, 1.0 - w1]
    else:
        w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.4)
        w2 = st.sidebar.slider(f"{active_tickers[1]} 权重", 0.0, 1.0 - w1, 0.3)
        weights = [w1, w2, round(1.0 - w1 - w2, 2)]

capital = st.sidebar.number_input("投资规模 ($)", value=100000)
days = st.sidebar.slider("压力测试天数", 5, 60, 22)

# --- 运行逻辑 ---
if st.sidebar.button("🚀 启动压力测试", type="primary"):
    if not active_tickers:
        st.warning("⚠️ 请输入有效的股票代码。")
    else:
        with st.spinner("执行蒙特卡洛矩阵运算中..."):
            data_bundle = get_portfolio_data(active_tickers)
            
            if data_bundle:
                S0, vols, corr_matrix, lam, tickers = data_bundle
                prices_matrix = run_portfolio_simulation(S0, vols, corr_matrix, lam, days, 10000)
                
                if prices_matrix is not None:
                    # 计算 PnL
                    asset_rets = (prices_matrix - S0.values) / S0.values
                    port_rets = asset_rets @ np.array(weights)
                    combined_pnl = port_rets * capital
                    
                    var_99 = np.percentile(combined_pnl, 1)
                    cvar_99 = combined_pnl[combined_pnl <= var_99].mean()
                    
                    col_left, col_right = st.columns([1, 1])
                    with col_left:
                        st.write("### 🧊 资产相关性热力图")
                        if num_active > 1:
                            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                            sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax_corr, square=True)
                            plt.xticks(rotation=45); plt.yticks(rotation=0)
                            st.pyplot(fig_corr)
                        else:
                            st.info("💡 单资产模式下无需计算相关性矩阵。")
                            
                    with col_right:
                        st.write("### 📉 组合盈亏分布")
                        fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
                        ax_hist.hist(combined_pnl, bins=80, color='#66b3ff', alpha=0.7)
                        ax_hist.axvline(var_99, color='red', linestyle='--', label='99% VaR')
                        ax_hist.legend(); st.pyplot(fig_hist)
                    
                    st.divider()
                    r1, r2, r3 = st.columns(3)
                    r1.metric("活跃资产数", num_active)
                    r2.error(f"**99% CVaR (预期极端亏损)**\n\n### ${abs(cvar_99):,.2f}")
                    r3.info(f"**模拟内最大回撤**\n\n### ${abs(np.min(combined_pnl)):,.2f}")
            else:
                st.error("❌ 无法抓取数据，请检查拼写或尝试更换标的。")
