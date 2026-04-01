import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as sco

# --- 1. Data & Alpha Signal Engine ---
@st.cache_data(ttl=300)
def get_market_data(tickers):
    try:
        data = yf.download(tickers, period="3y")
        if data.empty: return None
        
        close_df = data['Close'].to_frame() if isinstance(data['Close'], pd.Series) else data['Close']
        vol_df = data['Volume'].to_frame() if isinstance(data['Volume'], pd.Series) else data['Volume']
        
        close_df = close_df.dropna()
        vol_df = vol_df.dropna()
        returns = np.log(close_df / close_df.shift(1)).dropna()
        
        S0 = close_df.iloc[-1]
        sigma = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        cov_matrix = returns.cov() * 252
        
        lam_dict, mu_j_dict, sigma_j_dict = {}, {}, {}
        for t in tickers:
            ret = returns[t]
            jumps = ret[abs(ret) > 2 * ret.std()]
            lam_dict[t] = float(len(jumps) / 3)
            mu_j_dict[t] = float(jumps.mean() if len(jumps) > 0 else 0.0)
            sigma_j_dict[t] = float(jumps.std() if len(jumps) > 0 else 0.05)
            
        # ================= 新增：量价结合的 Alpha 雷达 =================
        rsi_dict, boll_dict, vol_dict, alpha_boost = {}, {}, {}, {}
        
        for t in tickers:
            # 1. RSI
            delta = close_df[t].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            latest_rsi = (100 - (100 / (1 + rs))).iloc[-1]
            rsi_dict[t] = latest_rsi
            
            # 2. BOLL
            sma_20 = close_df[t].rolling(window=20).mean()
            std_20 = close_df[t].rolling(window=20).std()
            upper_band, lower_band = sma_20 + (std_20 * 2), sma_20 - (std_20 * 2)
            latest_price = S0[t]
            
            if latest_price > upper_band.iloc[-1]: boll_status = "Over Upper"
            elif latest_price < lower_band.iloc[-1]: boll_status = "Below Lower"
            else: boll_status = "Mid Band"
            boll_dict[t] = boll_status
            
            # 3. Volume Analysis (量价关系)
            vol_sma_20 = vol_df[t].rolling(window=20).mean().iloc[-1]
            latest_vol = vol_df[t].iloc[-1]
            vol_ratio = latest_vol / vol_sma_20 if vol_sma_20 > 0 else 1
            vol_dict[t] = vol_ratio
            
            # 4. 融合量价的 Drift 修正逻辑
            boost = 0.0
            signal_note = "Neutral"
            
            if latest_rsi < 35 and boll_status == "Below Lower":
                if vol_ratio > 1.5:
                    # 放量暴跌：指标钝化，绝对不接飞刀！
                    boost -= 0.05 
                    signal_note = "🚨 Falling Knife (High Vol)"
                else:
                    # 缩量企稳：真实的超卖反弹
                    boost += 0.08
                    signal_note = "✅ Bottom Sighted (Low Vol)"
                    
            elif latest_rsi > 65 and boll_status == "Over Upper":
                if vol_ratio > 1.5:
                    # 放量拉升：主升浪，可能继续逼空
                    boost += 0.03
                    signal_note = "🔥 Momentum Breakout"
                else:
                    # 缩量新高：量价背离，准备回调
                    boost -= 0.08
                    signal_note = "⚠️ Divergence (Top Warning)"
                    
            alpha_boost[t] = {"boost": boost, "note": signal_note}

        return S0, sigma, corr_matrix, cov_matrix, lam_dict, mu_j_dict, sigma_j_dict, returns, rsi_dict, boll_dict, vol_dict, alpha_boost
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return None

# --- 2. Simulation Engine ---
def run_simulation(S0, vols, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, alpha_boost, days, iterations, crash_scenario):
    num_assets, iters = len(S0), iterations
    T = days / 252
    terminal_prices = np.zeros((iters, num_assets))
    
    if num_assets > 1:
        L = np.linalg.cholesky(corr_matrix.values)
        z = L @ np.random.standard_normal((num_assets, iters))
    else:
        z = np.random.standard_normal((1, iters))
        
    for idx, t in enumerate(S0.index):
        vol, S0_val = vols[t], S0[t]
        
        # 应用 Alpha 修正
        base_drift = 0.05
        adjusted_drift = base_drift + alpha_boost[t]["boost"]
        drift = (adjusted_drift - 0.5 * vol**2) * T
        
        shock = 0
        if crash_scenario == "2008 Lehman": shock = -0.15
        elif crash_scenario == "2020 COVID": shock = -0.10
        elif crash_scenario == "Tech Meltdown" and ("AAPL" in t or "MSFT" in t or "QQQ" in t): shock = -0.12
            
        diffusion = vol * np.sqrt(T) * z[idx, :]
        lam, mu_j, sigma_j = lam_dict[t], mu_j_dict[t], sigma_j_dict[t]
        n_jumps = np.random.poisson(lam * T, iters)
        jump_impact = np.array([np.sum(np.random.normal(mu_j, sigma_j, n)) if n > 0 else 0 for n in n_jumps])
        
        terminal_prices[:, idx] = S0_val * np.exp(drift + diffusion + jump_impact + shock)
    return terminal_prices

# --- 3. UI Layout ---
st.set_page_config(page_title="Quant Risk Engine", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Alpha & Risk Engine")

st.sidebar.header("⚙️ Configuration")
t1 = st.sidebar.text_input("Asset 1 (Primary)", value="AAPL").upper().strip()
t2 = st.sidebar.text_input("Asset 2 (Optional)", value="GLD").upper().strip()
t3 = st.sidebar.text_input("Asset 3 (Optional)", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

st.sidebar.markdown("---")
auto_optimize = st.sidebar.checkbox("🤖 Auto-Optimize Weights (Max Sharpe)", value=False)

manual_weights = []
if num_active > 0:
    if num_active == 1: 
        manual_weights = np.array([1.0])
    elif not auto_optimize:
        if num_active == 2:
            w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.5)
            manual_weights = np.array([w1, 1.0 - w1])
        else:
            w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.4)
            w2 = st.sidebar.slider(f"{active_tickers[1]} Weight", 0.0, 1.0 - w1, 0.3)
            manual_weights = np.array([w1, w2, round(1.0 - w1 - w2, 2)])
    else:
        st.sidebar.success("🤖 AI Optimization Active")

capital = st.sidebar.number_input("Portfolio Size ($)", value=100000)
days = st.sidebar.slider("Test Horizon (Days)", 5, 60, 30)
scenario = st.sidebar.selectbox("Inject Macro Shock", ["None", "2008 Lehman", "2020 COVID", "Tech Meltdown"])

if st.sidebar.button("🚀 Run Live Analysis", type="primary"):
    if not active_tickers:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Executing Quant Models & Optimization..."):
            result = get_market_data(active_tickers)
            if result:
                S0, sigma, corr_matrix, cov_matrix, lam_dict, mu_j_dict, sigma_j_dict, hist_returns, rsi_dict, boll_dict, vol_dict, alpha_boost = result
                
                weights = manual_weights
                if auto_optimize and num_active > 1:
                    mean_returns_annual = hist_returns.mean() * 252
                    def neg_sharpe(w):
                        p_ret = np.sum(mean_returns_annual * w)
                        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                        return -(p_ret - 0.04) / p_vol
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bnds = tuple((0, 1) for _ in range(num_active))
                    init_guess = num_active * [1. / num_active]
                    opt_res = sco.minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bnds, constraints=cons)
                    weights = np.array(opt_res.x)
                    weight_strs = [f"**{t}**: {w*100:.1f}%" for t, w in zip(active_tickers, weights)]
                    st.success(f"🤖 **Optimal Allocation:** " + " | ".join(weight_strs))
                elif num_active == 1:
                    weights = np.array([1.0])

                terminal_matrix = run_simulation(S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, alpha_boost, days, 20000, scenario)
                
                # ================= Alpha 信号监控台 (加入量价关系) =================
                st.markdown("##### 📡 Volume-Price Alpha Signals")
                cols = st.columns(num_active)
                for i, t in enumerate(active_tickers):
                    with cols[i]:
                        vol_str = f"{(vol_dict[t]*100):.0f}% of 20d Avg"
                        vol_color = "🔴" if vol_dict[t] > 1.2 else ("🟢" if vol_dict[t] < 0.8 else "⚪")
                        st.info(f"**{t} Analytics:**\n\n"
                                f"**RSI (14):** {rsi_dict[t]:.1f}\n\n"
                                f"**BOLL:** {boll_dict[t]}\n\n"
                                f"**Volume:** {vol_str} {vol_color}\n\n"
                                f"**Action:** {alpha_boost[t]['note']}")
                
                # ================= UI 渲染部分 =================
                if num_active == 1:
                    t = active_tickers[0]
                    st.subheader(f"📊 {t} Tail Risk Report")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Latest Price", f"${S0[t]:.2f}")
                    c2.metric("Annual Volatility", f"{sigma[t]*100:.2f}%")
                    c3.metric("Jump Frequency", f"{lam_dict[t]:.1f} / yr")
                    c4.metric("Avg Jump Amplitude", f"{mu_j_dict[t]*100:.2f}%")
                    
                    K, vol_bs, T_opt = S0[t] * 0.90, sigma[t] + 0.03, days / 252
                    d1 = (np.log(S0[t]/K) + (0.05 + 0.5*vol_bs**2)*T_opt) / (vol_bs*np.sqrt(T_opt))
                    d2 = d1 - vol_bs*np.sqrt(T_opt)
                    put_px = K * np.exp(-0.05*T_opt) * norm.cdf(-d2) - S0[t] * norm.cdf(-d1)
                    
                    prices = terminal_matrix[:, 0]
                    naked_pnl = (prices - S0[t]) / S0[t] * capital
                    hedged_pnl = naked_pnl + (np.maximum(K - prices, 0) - put_px) / S0[t] * capital
                    
                    st.divider()
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.error(f"**Naked CVaR**\n\n### ${abs(naked_pnl[naked_pnl <= np.percentile(naked_pnl, 1)].mean()):,.0f}")
                        st.success(f"**Hedged CVaR**\n\n### ${abs(hedged_pnl[hedged_pnl <= np.percentile(hedged_pnl, 1)].mean()):,.0f}")
                        st.info(f"**10% OTM Put Cost**\n\n### ${((put_px/S0[t])*capital):,.0f}")
                    with col_r:
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
                        ax.hist(naked_pnl, bins=100, alpha=0.4, color='red', label='Naked Position')
                        ax.hist(hedged_pnl, bins=100, alpha=0.6, color='dodgerblue', label='Hedged Position')
                        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
                        ax.legend()
                        st.pyplot(fig)
                        
                else:
                    st.subheader("🌐 Institutional Risk Dashboard")
                    
                    asset_rets = (terminal_matrix - S0.values) / S0.values
                    port_rets = asset_rets @ weights
                    pnl = port_rets * capital
                    
                    annual_ret = hist_returns.mean() @ weights * 252
                    port_vol = np.sqrt(weights.T @ corr_matrix.values @ weights) * np.mean(sigma)
                    weighted_vol = np.sum(sigma.values * weights)
                    div_benefit = (weighted_vol - port_vol) * 100
                    sharpe = (annual_ret - 0.04) / port_vol if port_vol > 0 else 0
                    
                    var_99, cvar_99 = np.percentile(pnl, 1), pnl[pnl <= np.percentile(pnl, 1)].mean()
                    
                    st.divider()
                    st.markdown("##### 📈 Return & Efficiency")
                    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                    r1c1.metric("Exp. Annual Return", f"{annual_ret*100:.2f}%")
                    r1c2.metric("Portfolio Volatility", f"{port_vol*100:.2f}%")
                    r1c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    r1c4.metric("Diversification Benefit", f"+{div_benefit:.2f}%")
                    
                    st.markdown("##### ⚠️ Tail Risk & Simulation Extremes")
                    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                    r2c1.metric("Simulated Win Rate", f"{(len(pnl[pnl > 0]) / len(pnl)) * 100:.1f}%")
                    r2c2.metric("99% VaR (Threshold)", f"${abs(var_99):,.0f}")
                    r2c3.metric("99% CVaR (Expected)", f"${abs(cvar_99):,.0f}")
                    r2c4.metric("Max Simulated Drawdown", f"${abs(np.min(pnl)):,.0f}")
                    
                    st.divider()
                    col_l, col_r = st.columns([1, 1.5])
                    with col_l:
                        st.markdown("##### 🧊 Asset Correlation Matrix")
                        st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn_r', axis=None, vmin=-1.0, vmax=1.0).format("{:.2f}"), use_container_width=True)
                    with col_r:
                        st.markdown("##### 📉 Portfolio PnL Distribution (White)")
                        fig, ax = plt.subplots(figsize=(8, 4.2))
                        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
                        ax.hist(pnl, bins=100, color='dodgerblue', alpha=0.8)
                        ax.axvline(var_99, color='red', linestyle='--', label='99% VaR Threshold')
                        ax.tick_params(colors='black')
                        ax.legend()
                        st.pyplot(fig)
