import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Data Engine ---
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
        
        lam_dict, mu_j_dict, sigma_j_dict = {}, {}, {}
        for t in tickers:
            ret = returns[t]
            jumps = ret[abs(ret) > 2 * ret.std()]
            lam_dict[t] = float(len(jumps) / 3)
            mu_j_dict[t] = float(jumps.mean() if len(jumps) > 0 else 0.0)
            sigma_j_dict[t] = float(jumps.std() if len(jumps) > 0 else 0.05)
            
        return S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, returns
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return None

# --- 2. Simulation Engine (with Stress Scenarios) ---
def run_simulation(S0, vols, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, iterations, crash_scenario):
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
        
        # Apply Macro Shocks
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

# Sidebar
st.sidebar.header("⚙️ Configuration")
t1 = st.sidebar.text_input("Asset 1 (Primary)", value="AAPL").upper().strip()
t2 = st.sidebar.text_input("Asset 2 (Optional)", value="").upper().strip()
t3 = st.sidebar.text_input("Asset 3 (Optional)", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

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
scenario = st.sidebar.selectbox("Inject Macro Shock", ["None", "2008 Lehman", "2020 COVID", "Tech Meltdown"])

if st.sidebar.button("🚀 Run Live Analysis", type="primary"):
    if not active_tickers:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Processing Quantitative Models..."):
            result = get_market_data(active_tickers)
            if result:
                S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, hist_returns = result
                terminal_matrix = run_simulation(S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, 20000, scenario)
                
                # ================= UI BIFURCATION (智能分流) =================
                
                if num_active == 1:
                    # --- MODE A: SINGLE ASSET OPTIONS HEDGING ---
                    t = active_tickers[0]
                    st.subheader(f"📊 {t} Tail Risk & Options Hedging Report")
                    st.caption(f"Scenario Applied: **{scenario}**")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Latest Price", f"${S0[t]:.2f}")
                    c2.metric("Annual Volatility", f"{sigma[t]*100:.2f}%")
                    c3.metric("Jump Frequency", f"{lam_dict[t]:.1f} / yr")
                    c4.metric("Avg Jump Amplitude", f"{mu_j_dict[t]*100:.2f}%")
                    
                    # Options Math
                    K, vol_bs, T_opt = S0[t] * 0.90, sigma[t] + 0.03, days / 252
                    d1 = (np.log(S0[t]/K) + (0.05 + 0.5*vol_bs**2)*T_opt) / (vol_bs*np.sqrt(T_opt))
                    d2 = d1 - vol_bs*np.sqrt(T_opt)
                    put_px = K * np.exp(-0.05*T_opt) * norm.cdf(-d2) - S0[t] * norm.cdf(-d1)
                    
                    prices = terminal_matrix[:, 0]
                    naked_pnl = (prices - S0[t]) / S0[t] * capital
                    hedged_pnl = naked_pnl + (np.maximum(K - prices, 0) - put_px) / S0[t] * capital
                    
                    naked_cvar = naked_pnl[naked_pnl <= np.percentile(naked_pnl, 1)].mean()
                    hedged_cvar = hedged_pnl[hedged_pnl <= np.percentile(hedged_pnl, 1)].mean()
                    
                    st.divider()
                    col_l, col_r = st.columns([1, 2])
                    with col_l:
                        st.error(f"**Naked Expected Loss (CVaR)**\n\n### ${abs(naked_cvar):,.0f}")
                        st.success(f"**Hedged Expected Loss (CVaR)**\n\n### ${abs(hedged_cvar):,.0f}")
                        st.info(f"**10% OTM Put Cost**\n\n### ${((put_px/S0[t])*capital):,.0f}")
                    with col_r:
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
                        ax.hist(naked_pnl, bins=100, alpha=0.4, color='red', label='Naked Position')
                        ax.hist(hedged_pnl, bins=100, alpha=0.6, color='dodgerblue', label='Hedged Position')
                        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
                        ax.set_title(f"PnL Distribution ({days} Days)", color='black')
                        ax.tick_params(colors='black')
                        ax.legend()
                        st.pyplot(fig)
                        
                else:
                    # --- MODE B: PORTFOLIO ALPHA & CORRELATION ---
                    st.subheader("🌐 Portfolio Alpha & Correlation Engine")
                    st.caption(f"Scenario Applied: **{scenario}**")
                    
                    asset_rets = (terminal_matrix - S0.values) / S0.values
                    port_rets = asset_rets @ weights
                    pnl = port_rets * capital
                    
                    # Math for Portfolio
                    weighted_vol = np.sum(sigma.values * weights)
                    port_vol = np.sqrt(weights.T @ corr_matrix.values @ weights) * np.mean(sigma)
                    div_benefit = (weighted_vol - port_vol) * 100
                    
                    annual_ret = hist_returns.mean() @ weights * 252
                    sharpe = (annual_ret - 0.04) / port_vol if port_vol > 0 else 0
                    cvar_99 = pnl[pnl <= np.percentile(pnl, 1)].mean()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Diversification Benefit", f"+{div_benefit:.2f}%")
                    c2.metric("Portfolio Sharpe Ratio", f"{sharpe:.2f}")
                    c3.metric("Tail Risk (99% CVaR)", f"${abs(cvar_99):,.0f}")
                    
                    st.divider()
                    col_l, col_r = st.columns([1, 1.2])
                    with col_l:
                        st.markdown("##### Correlation Matrix")
                        st.dataframe(corr_matrix.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), use_container_width=True)
                    with col_r:
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
                        ax.hist(pnl, bins=100, color='dodgerblue', alpha=0.8)
                        ax.axvline(np.percentile(pnl, 1), color='red', linestyle='--', label='99% VaR')
                        ax.set_title(f"Portfolio PnL Distribution ({days} Days)", color='black')
                        ax.tick_params(colors='black')
                        ax.legend()
                        st.pyplot(fig)
