import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Data Engine (Expanded for R_f and Benchmarking) ---
@st.cache_data(ttl=300)
def get_market_data(tickers):
    try:
        data = yf.download(tickers, period="3y")
        if data.empty: return None
        close_df = data['Close'].to_frame() if isinstance(data['Close'], pd.Series) else data['Close']
        close_df = close_df.dropna()
        returns = np.log(close_df / close_df.shift(1)).dropna()
        
        S0 = close_df.iloc[-1]
        sigma = returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        
        # Calculate individual jump stats
        lam_dict, mu_j_dict, sigma_j_dict = {}, {}, {}
        for t in tickers:
            ret = returns[t]
            jumps = ret[abs(ret) > 2 * ret.std()]
            lam_dict[t] = float(len(jumps) / 3)
            mu_j_dict[t] = float(jumps.mean() if len(jumps) > 0 else 0.0)
            sigma_j_dict[t] = float(jumps.std() if len(jumps) > 0 else 0.05)
            
        return S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, returns
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return None

# --- 2. Simulation Engine ---
def run_simulation(S0, vols, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, iterations, crash_scenario=None):
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
        drift = (0.05 - 0.5 * vol**2) * T
        
        # Apply Stress Scenarios
        shock = 0
        if crash_scenario == "2008 Lehman": shock = -0.15
        elif crash_scenario == "2020 COVID": shock = -0.10
        elif crash_scenario == "Tech Meltdown": shock = -0.08 if "AAPL" in t or "MSFT" in t or "NVDA" in t else 0
            
        diffusion = vol * np.sqrt(T) * z[idx, :]
        lam, mu_j, sigma_j = lam_dict[t], mu_j_dict[t], sigma_j_dict[t]
        n_jumps = np.random.poisson(lam * T, iters)
        jump_impact = np.array([np.sum(np.random.normal(mu_j, sigma_j, n)) if n > 0 else 0 for n in n_jumps])
        
        terminal_prices[:, idx] = S0_val * np.exp(drift + diffusion + jump_impact + shock)
    return terminal_prices

# --- 3. UI Layout ---
st.set_page_config(page_title="Portfolio Alpha Engine", layout="wide")
st.title("🌪️ Macro-Tail-Hedge: Alpha Engine V5.0")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
t1 = st.sidebar.text_input("Asset 1", value="AAPL").upper().strip()
t2 = st.sidebar.text_input("Asset 2", value="GLD").upper().strip()
t3 = st.sidebar.text_input("Asset 3", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

# Dynamic Weighting
if num_active > 0:
    if num_active == 1: weights = np.array([1.0])
    elif num_active == 2:
        w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.5)
        weights = np.array([w1, 1.0 - w1])
    else:
        w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.4)
        w2 = st.sidebar.slider(f"{active_tickers[1]} Weight", 0.0, 1.0 - w1, 0.3)
        weights = np.array([w1, w2, round(1.0 - w1 - w2, 2)])

capital = st.sidebar.number_input("Portfolio Size ($)", value=100000)
days = st.sidebar.slider("Test Horizon (Days)", 5, 60, 30)

st.sidebar.subheader("🔥 Stress Scenarios")
scenario = st.sidebar.selectbox("Select History Shock", ["None", "2008 Lehman", "2020 COVID", "Tech Meltdown"])

if st.sidebar.button("🚀 Run Live Analysis", type="primary"):
    result = get_market_data(active_tickers)
    if result:
        S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, hist_returns = result
        terminal_matrix = run_simulation(S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, 20000, scenario)
        
        # --- Quantitative Metrics Calculation ---
        asset_rets = (terminal_matrix - S0.values) / S0.values
        port_rets = asset_rets @ weights
        pnl = port_rets * capital
        
        # 1. Diversification Benefit
        weighted_vol = np.sum(sigma.values * weights)
        port_vol = np.sqrt(weights.T @ corr_matrix.values @ weights) * np.mean(sigma) # Approx
        div_benefit = (weighted_vol - port_vol) * 100
        
        # 2. Sharpe Ratio (Annualized)
        annual_ret = hist_returns.mean() @ weights * 252
        annual_vol = np.sqrt(weights.T @ corr_matrix.values @ weights) * np.mean(sigma) 
        sharpe = (annual_ret - 0.04) / annual_vol # Using 4% Risk-free rate
        
        # 3. Risk Metrics
        cvar_99 = pnl[pnl <= np.percentile(pnl, 1)].mean()
        
        # --- UI Rendering ---
        st.divider()
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Diversification Benefit", f"+{div_benefit:.2f}%", help="Risk reduction achieved by combining assets.")
        col_m2.metric("Portfolio Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted return (Benchmark: 4% R_f)")
        col_m3.error(f"Tail Risk (99% CVaR): ${abs(cvar_99):,.0f}")
        
        st.divider()
        c_l, c_r = st.columns([1, 1.2])
        with c_l:
            st.markdown("### 🧊 Correlation & Weights")
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
            st.write(f"Scenario Applied: **{scenario}**")
        with c_r:
            st.markdown("### 📉 PnL Stress Distribution (White)")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor('white'); ax.set_facecolor('white')
            ax.hist(pnl, bins=100, color='dodgerblue', alpha=0.7, edgecolor='white')
            ax.axvline(np.percentile(pnl, 1), color='red', linestyle='--', label='Tail Event')
            ax.set_title(f"Projected PnL in {days} Days", color='black')
            ax.tick_params(colors='black')
            st.pyplot(fig)
            
        r1, r2, r3 = st.columns(3)
        r1.info(f"**Max Potential Loss**\n\n### ${abs(np.min(pnl)):,.0f}")
        r2.warning(f"**Expected Scenario Hit**\n\n### ${abs(np.mean(pnl)):,.0f}")
        r3.success(f"**99% Confidence Cap**\n\n### ${np.percentile(pnl, 99):,.0f}")
