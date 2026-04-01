import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. 数据引擎 (鲁棒适配单/多资产) ---
@st.cache_data(ttl=300)
def get_market_data(tickers):
    try:
        data = yf.download(tickers, period="3y")
        if data.empty:
            return None
        
        # 智能提取数据
        if isinstance(data['Close'], pd.Series):
            # 单资产情况
            close_df = data['Close'].to_frame()
            close_df.columns = tickers
        else:
            # 多资产情况
            close_df = data['Close']
            
        close_df = close_df.dropna()
        returns = np.log(close_df / close_df.shift(1)).dropna()
        
        S0 = close_df.iloc[-1]
        sigma = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        
        # 跳跃参数 (用于顶部面板显示)
        lam_dict, mu_j_dict, sigma_j_dict = {}, {}, {}
        for t in tickers:
            ret = returns[t]
            thresh = 2 * ret.std()
            jumps = ret[abs(ret) > thresh]
            
            lam_dict[t] = float(len(jumps) / 3)
            mu_j_dict[t] = float(jumps.mean() if len(jumps) > 0 else 0.0)
            sigma_j_dict[t] = float(jumps.std() if len(jumps) > 0 else 0.05)
            
        return S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, tickers
    except Exception as e:
        st.error(f"❌ 数据抓取出错: {str(e)}")
        return None

# --- 2. 模拟引擎 (智能适配单/多资产) ---
def run_simulation(S0, vols, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, iterations):
    try:
        num_assets = len(S0)
        T, dt = days / 252, 1 / 252
        iters = iterations
        
        # 初始化价格矩阵 (iters, num_assets)
        terminal_prices = np.zeros((iters, num_assets))
        
        # 相关性耦合 (核心变化：改为向量化计算，提升速度)
        if num_assets > 1:
            L = np.linalg.cholesky(corr_matrix.values)
            z = L @ np.random.standard_normal((num_assets, iters)) # (3, iters)
        else:
            z = np.random.standard_normal((1, iters)) # (1, iters)
            
        # 遍历每个资产进行 MJD 模拟
        for idx, t in enumerate(S0.index):
            # 几何布朗运动部分
            vol, S0_val = vols[t], S0[t]
            drift = (0.05 - 0.5 * vol**2) * T
            diffusion = vol * np.sqrt(T) * z[idx, :]
            
            # 跳跃部分 (每个资产独立跳跃)
            lam, mu_j, sigma_j = lam_dict[t], mu_j_dict[t], sigma_j_dict[t]
            n_jumps = np.random.poisson(lam * T, iters)
            jump_impact = np.array([np.sum(np.random.normal(mu_j, sigma_j, n)) if n > 0 else 0 for n in n_jumps])
            
            # 计算终点价格
            terminal_prices[:, idx] = S0_val * np.exp(drift + diffusion + jump_impact)
            
        return terminal_prices
    except Exception as e:
        st.error(f"❌ 模拟计算出错: {str(e)}")
        return None

# --- 3. UI 界面 ---
st.set_page_config(page_title="量化风险管理引擎", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Portfolio Simulator")

# 侧边栏
st.sidebar.header("⚙️ 资产配置面板")
t1 = st.sidebar.text_input("资产 1 (必填)", value="QQQ").upper().strip()
t2 = st.sidebar.text_input("资产 2 (可选)", value="").upper().strip()
t3 = st.sidebar.text_input("资产 3 (可选)", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

# 动态权重
weights = []
if num_active > 0:
    if num_active == 1:
        weights = [1.0]
    elif num_active == 2:
        w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.5)
        weights = [w1, 1.0 - w1]
    else:
        w1 = st.sidebar.slider(f"{active_tickers[0]} 权重", 0.0, 1.0, 0.4)
        w2 = st.sidebar.slider(f"{active_tickers[1]} 权重", 0.0, 1.0 - w1, 0.3)
        weights = [w1, w2, round(1.0 - w1 - w2, 2)]

capital = st.sidebar.number_input("投资规模 ($)", value=100000)
days = st.sidebar.slider("压力测试天数", 5, 60, 22)

# --- 核心运行逻辑 ---
if st.sidebar.button("🚀 启动压力测试", type="primary"):
    if not active_tickers:
        st.warning("⚠️ 请输入有效的股票代码。")
    else:
        with st.spinner("数据抓取与矩阵运算中..."):
            result = get_market_data(active_tickers)
            
            if result:
                S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, tickers = result
                # 运行模拟
                terminal_matrix = run_simulation(S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, 10000)
                
                if terminal_matrix is not None:
                    
                    # =============== ✨ 分叉逻辑开始：智能适配 UI ===============
                    
                    if num_active == 1:
                        # ----- 模式 A：单一资产模式 (找回图 2 的感觉) -----
                        st.subheader(f"📊 {active_tickers[0]} 单一资产压力测试报告")
                        
                        # 顶部 4 块面板
                        t = active_tickers[0]
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("标的价格", f"${S0[t]:.2f}")
                        col2.metric("年化波动率", f"{sigma[t]*100:.2f}%")
                        col3.metric("极端跳跃频率", f"{lam_dict[t]:.1f} 次/年")
                        col4.metric("平均跳跃幅度", f"{mu_j_dict[t]*100:.2f}%")
                        
                        st.divider()
                        
                        # 计算对冲收益 (V1.0 逻辑回归)
                        K = S0[t] * 0.90 # 10% 虚值看跌
                        vol_bs = sigma[t] + 0.03 # 波动率风险溢价补偿
                        T_option = days / 252
                        
                        d1 = (np.log(S0[t]/K) + (0.05 + 0.5*vol_bs**2)*T_option) / (vol_bs*np.sqrt(T_option))
                        d2 = d1 - vol_bs*np.sqrt(T_option)
                        put_BS = K * np.exp(-0.05*T_option) * norm.cdf(-d2) - S0[t] * norm.cdf(-d1)
                        
                        # 计算 PnL (iters, 1)
                        prices = terminal_matrix[:, 0]
                        naked_pnl = (prices - S0[t]) / S0[t] * capital
                        option_payoff = np.maximum(K - prices, 0)
                        hedged_pnl = naked_pnl + (option_payoff - put_BS) / S0[t] * capital
                        
                        # 风险度量 (99% CVaR)
                        naked_cvar = naked_pnl[naked_pnl <= np.percentile(naked_pnl, 1)].mean()
                        hedged_cvar = hedged_pnl[hedged_pnl <= np.percentile(hedged_pnl, 1)].mean()
                        
                        # 下方两列
                        col_l, col_r = st.columns([1, 2])
                        with col_l:
                            st.error(f"**裸头寸预期极寒亏损 (CVaR)**\n\n### ${abs(naked_cvar):,.2f}")
                            st.success(f"**对冲后预期亏损 (Hedged CVaR)**\n\n### ${abs(hedged_cvar):,.2f}")
                            st.info(f"**期权对冲成本 (Hedging Cost)**\n\n### ${((put_BS/S0[t])*capital):,.2f}")
                            
                        with col_r:
                            st.write("### 📉 裸头寸 vs. 期权对冲分布图")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            ax.hist(naked_pnl, bins=80, alpha=0.5, color='#ff9999', label='Naked')
                            ax.hist(hedged_pnl, bins=80, alpha=0.5, color='#66b3ff', label='Hedged')
                            ax.axvline(x=0, color='black', linestyle='--')
                            ax.set_title("PnL Distribution Overlay")
                            ax.legend()
                            st.pyplot(fig)
                            
                    else:
                        # ----- 模式 B：组合模式 (保持 V4.0 的热力图) -----
                        st.subheader(f"🌐 投资组合多资产联动测试报告")
                        
                        # 计算组合收益
                        asset_rets = (terminal_matrix - S0.values) / S0.values
                        port_rets = asset_rets @ np.array(weights)
                        combined_pnl = port_rets * capital
                        
                        var_99 = np.percentile(combined_pnl, 1)
                        cvar_99 = combined_pnl[combined_pnl <= var_99].mean()
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.write("### 🧊 资产相关性热力图")
                            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                            sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax_corr, square=True)
                            plt.xticks(rotation=45); plt.yticks(rotation=0)
                            st.pyplot(fig_corr)
                            
                        with col_b:
                            st.write("### 📉 组合盈亏分布")
                            fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
                            ax_hist.hist(combined_pnl, bins=80, color='#66b3ff', alpha=0.7)
                            ax_hist.axvline(var_99, color='red', linestyle='--', label='VaR 99%')
                            ax_hist.legend(); st.pyplot(fig_hist)
                        
                        st.divider()
                        r1, r2, r3 = st.columns(3)
                        r1.metric("活跃资产数", num_active)
                        r2.error(f"**组合 99% CVaR (极寒亏损)**\n\n### ${abs(cvar_99):,.2f}")
                        r3.info(f"**组合模拟最大回撤**\n\n### ${abs(np.min(combined_pnl)):,.2f}")
