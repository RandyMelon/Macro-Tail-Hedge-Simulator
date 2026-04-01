import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 数据引擎：抓取三资产相关性 ---
@st.cache_data(ttl=3600)
def get_portfolio_data(tickers):
    try:
        data = yf.download(tickers, period="3y")['Close']
        if data.empty:
            raise ValueError("未能获取数据，请检查代码是否正确。")
        
        returns = np.log(data / data.shift(1)).dropna()
        S0 = data.iloc[-1]
        vols = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        
        # 提取组合层面的跳跃特征 (基于等权收益率)
        port_ret = returns.mean(axis=1)
        threshold = 2 * port_ret.std()
        jumps = port_ret[abs(port_ret) > threshold]
        lam = float(len(jumps) / 3)
        
        return S0, vols, corr_matrix, lam, tickers
    except Exception as e:
        st.error(f"数据抓取失败: {e}")
        return None

# --- 2. 核心模拟引擎：Cholesky 矩阵耦合 ---
def run_portfolio_simulation(S0, vols, corr_matrix, lam, days, iterations):
    T = days / 252
    dt = 1 / 252
    N = int(days)
    num_assets = len(S0)
    
    # Cholesky 分解：将相关性矩阵耦合
    L = np.linalg.cholesky(corr_matrix)
    
    # 模拟路径 (只取终点以节省计算资源)
    # 生成独立随机噪音 (num_assets, iterations)
    z_raw = np.random.standard_normal((num_assets, iterations))
    # 注入相关性
    z_corr = L @ z_raw
    
    # 计算几何布朗运动部分
    drift = (0.05 - 0.5 * vols.values**2) * T
    diffusion = vols.values[:, np.newaxis] * np.sqrt(T) * z_corr
    
    # 宏观跳跃冲击 (对全组合的一致性打击)
    n_jumps = np.random.poisson(lam * T, iterations)
    jump_impact = np.zeros(iterations)
    for i in range(iterations):
        if n_jumps[i] > 0:
            jump_impact[i] = np.sum(np.random.normal(-0.03, 0.05, n_jumps[i]))
            
    # 计算终点价格矩阵
    terminal_prices = S0.values[:, np.newaxis] * np.exp(drift[:, np.newaxis] + diffusion + jump_impact)
    return terminal_prices.T # 返回 (iterations, 3)

# --- 3. UI 布局 ---
st.set_page_config(page_title="多资产风控引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Portfolio V3.0")
st.markdown("三资产联动模拟系统：利用 **Cholesky Decomposition** 捕捉资产相关性崩塌风险。")

st.sidebar.header("⚙️ 组合配置 (3 Assets)")
t1 = st.sidebar.text_input("资产 1", value="QQQ").upper()
t2 = st.sidebar.text_input("资产 2", value="NVDA").upper()
t3 = st.sidebar.text_input("资产 3", value="TLT").upper()

st.sidebar.subheader("权重分配")
w1 = st.sidebar.slider(f"{t1} 权重", 0.0, 1.0, 0.4)
w2 = st.sidebar.slider(f"{t2} 权重", 0.0, 1.0 - w1, 0.4)
w3 = round(1.0 - w1 - w2, 2)
st.sidebar.info(f"当前比例: {t1}({w1}) | {t2}({w2}) | {t3}({w3})")

capital = st.sidebar.number_input("投资规模 ($)", value=100000)
days = st.sidebar.slider("测试天数", 5, 60, 22)
iters = 10000 # 保持 1万次模拟以平衡速度

if st.sidebar.button("🚀 启动投资组合压力测试", type="primary"):
    with st.spinner("矩阵运算中..."):
        result = get_portfolio_data([t1, t2, t3])
        if result:
            S0, vols, corr_matrix, lam, tickers = result
            weights = np.array([w1, w2, w3])
            
            # 运行模拟
            prices_matrix = run_portfolio_simulation(S0, vols, corr_matrix, lam, days, iters)
            
            # 计算组合收益
            # (iterations, 3) -> 各资产收益率
            asset_returns = (prices_matrix - S0.values) / S0.values
            # 加权平均得到组合收益率
            port_returns = asset_returns @ weights
            combined_pnl = port_returns * capital
            
            # 风险度量 (CVaR)
            var_99 = np.percentile(combined_pnl, 1)
            cvar_99 = combined_pnl[combined_pnl <= var_99].mean()
            
            # --- 渲染界面 ---
            col_a, col_b = st.columns([1, 1])
            
            with col_a:
                st.write("### 🧊 资产相关性矩阵")
                fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax_corr, fmt=".2f")
                st.pyplot(fig_corr)
                
            with col_b:
                st.write("### 📉 组合盈亏分布")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
                ax_hist.hist(combined_pnl, bins=80, color='#66b3ff', alpha=0.7)
                ax_hist.axvline(var_99, color='red', linestyle='--', label=f'VaR 99%')
                ax_hist.set_title("Portfolio PnL Projection")
                ax_hist.legend()
                st.pyplot(fig_hist)
            
            st.divider()
            
            # 结果看板
            res_1, res_2, res_3 = st.columns(3)
            res_1.metric("组合预期年化波动率", f"{np.sqrt(weights @ corr_matrix.values @ weights.T) * np.mean(vols):.2f}%")
            res_2.error(f"**组合 99% CVaR (极寒亏损)**\n\n### ${abs(cvar_99):,.2f}")
            res_3.info(f"**模拟天数内最大潜在回撤**\n\n### ${abs(np.min(combined_pnl)):,.2f}")

else:
    st.info("👈 在左侧配置三个资产及其权重，观察它们如何联动。")
