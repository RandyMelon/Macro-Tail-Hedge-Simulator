import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- 1. Data Engine ---
@st.cache_data(ttl=300)
def get_market_data(tickers):
    try:
        data = yf.download(tickers, period="3y")
        if data.empty: return None
        if isinstance(data['Close'], pd.Series):
            close_df = data['Close'].to_frame()
            close_df.columns = tickers
        else:
            close_df = data['Close']
        close_df = close_df.dropna()
        returns = np.log(close_df / close_df.shift(1)).dropna()
        S0, sigma = close_df.iloc[-1], returns.std() * np.sqrt(252)
        corr_matrix = returns.corr()
        lam_dict, mu_j_dict, sigma_j_dict = {}, {}, {}
        for t in tickers:
            ret = returns[t]
            jumps = ret[abs(ret) > 2 * ret.std()]
            lam_dict[t] = float(len(jumps) / 3)
            mu_j_dict[t] = float(jumps.mean() if len(jumps) > 0 else 0.0)
            sigma_j_dict[t] = float(jumps.std() if len(jumps) > 0 else 0.05)
        return S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, tickers
    except Exception as e:
        st.error(f"Data Fetch Error: {str(e)}")
        return None

# --- 2. Simulation Engine ---
def run_simulation(S0, vols, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, iterations):
    try:
        num_assets, iters = len(S0), iterations
        T, dt = days / 252, 1 / 252
        terminal_prices = np.zeros((iters, num_assets))
        if num_assets > 1:
            L = np.linalg.cholesky(corr_matrix.values)
            z = L @ np.random.standard_normal((num_assets, iters))
        else:
            z = np.random.standard_normal((1, iters))
        for idx, t in enumerate(S0.index):
            vol, S0_val = vols[t], S0[t]
            drift = (0.05 - 0.5 * vol**2) * T
            diffusion = vol * np.sqrt(T) * z[idx, :]
            lam, mu_j, sigma_j = lam_dict[t], mu_j_dict[t], sigma_j_dict[t]
            n_jumps = np.random.poisson(lam * T, iters)
            jump_impact = np.array([np.sum(np.random.normal(mu_j, sigma_j, n)) if n > 0 else 0 for n in n_jumps])
            terminal_prices[:, idx] = S0_val * np.exp(drift + diffusion + jump_impact)
        return terminal_prices
    except: return None

# --- 3. UI Configuration ---
st.set_page_config(page_title="Quant Risk Engine", layout="wide")

st.title("🌪️ Macro-Tail-Hedge: Portfolio Simulator")
st.markdown("MJD Monte Carlo stress testing and options hedging simulation based on real market data.")

# Sidebar
st.sidebar.header("⚙️ Asset Configuration")
t1 = st.sidebar.text_input("Asset 1 (Primary)", value="AAPL").upper().strip()
t2 = st.sidebar.text_input("Asset 2 (Optional)", value="").upper().strip()
t3 = st.sidebar.text_input("Asset 3 (Optional)", value="").upper().strip()

active_tickers = [t for t in [t1, t2, t3] if t]
num_active = len(active_tickers)

# Dynamic Weighting
if num_active > 0:
    if num_active == 1: weights = [1.0]
    elif num_active == 2:
        w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.5)
        weights = [w1, 1.0 - w1]
    else:
        w1 = st.sidebar.slider(f"{active_tickers[0]} Weight", 0.0, 1.0, 0.4)
        w2 = st.sidebar.slider(f"{active_tickers[1]} Weight", 0.0, 1.0 - w1, 0.3)
        weights = [w1, w2, round(1.0 - w1 - w2, 2)]

capital = st.sidebar.number_input("Portfolio Size ($)", value=100000)
days = st.sidebar.slider("Stress Test Horizon (Days)", 5, 60, 30)

if st.sidebar.button("🚀 Run Live Stress Test", type="primary"):
    if not active_tickers:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Processing Market Dynamics..."):
            result = get_market_data(active_tickers)
            if result:
                S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, tickers = result
                terminal_matrix = run_simulation(S0, sigma, corr_matrix, lam_dict, mu_j_dict, sigma_j_dict, days, 20000)
                
                if terminal_matrix is not None:
                    if num_active == 1:
                        # ----- Single Asset Mode -----
                        t = active_tickers[0]
                        st.divider()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.info(f"**Latest Price**\n\n### ${S0[t]:.2f}")
                        c2.info(f"**Annual Volatility**\n\n### {sigma[t]*100:.2f}%")
                        c3.info(f"**Jump Frequency**\n\n### {lam_dict[t]:.1f} times/y")
                        c4.info(f"**Avg Jump Amp**\n\n### {mu_j_dict[t]*100:.2f}%")
                        
                        # Math for Hedging
                        K, vol_bs, T_opt = S0[t] * 0.90, sigma[t] + 0.03, days / 252
                        d1 = (np.log(S0[t]/K) + (0.05 + 0.5*vol_bs**2)*T_opt) / (vol_bs*np.sqrt(T_opt))
                        d2 = d1 - vol_bs*np.sqrt(T_opt)
                        put_px = K * np.exp(-0.05*T_opt) * norm.cdf(-d2) - S0[t] * norm.cdf(-d1)
                        
                        prices = terminal_matrix[:, 0]
                        naked_pnl = (prices - S0[t]) / S0[t] * capital
                        hedged_pnl = naked_pnl + (np.maximum(K - prices, 0) - put_px) / S0[t] * capital
                        
                        # Layout Adjustment
                        st.markdown(f"### 📊 {t} Tail Risk Report (99% CVaR)")
                        col_l, col_r = st.columns([1, 2])
                        with col_l:
                            st.error(f"**Naked CVaR**\n\n### ${abs(naked_pnl[naked_pnl <= np.percentile(naked_pnl, 1)].mean()):,.2f}")
                            st.success(f"**Hedged CVaR**\n\n### ${abs(hedged_pnl[hedged_pnl <= np.percentile(hedged_pnl, 1)].mean()):,.2f}")
                            st.info(f"**Hedging Cost**\n\n### ${((put_px/S0[t])*capital):,.2f}")
                            
                        with col_r:
                            fig, ax = plt.subplots(figsize=(8, 4.8)) # Adjusted size
                            fig.patch.set_facecolor('#0e1117')
                            ax.set_facecolor('#0e1117')
                            ax.hist(naked_pnl, bins=100, alpha=0.4, color='#ff4b4b', label='Naked Position')
                            ax.hist(hedged_pnl, bins=100, alpha=0.6, color='#00d4ff', label='Hedged (OTM Put)')
                            ax.axvline(0, color='white', linestyle='--', linewidth=0.8)
                            ax.set_title(f"PnL Distribution ({days} Days)", color='white', fontsize=12)
                            ax.tick_params(colors='white', labelsize=10)
                            for spine in ax.spines.values(): spine.set_color('#334155')
                            ax.legend(facecolor='#1e293b', labelcolor='white')
                            st.pyplot(fig)
                            
                    else:
                        # ----- Portfolio Mode -----
                        st.divider()
                        asset_rets = (terminal_matrix - S0.values) / S0.values
                        port_rets = asset_rets @ np.array(weights)
                        combined_pnl = port_rets * capital
                        cvar_99 = combined_pnl[combined_pnl <= np.percentile(combined_pnl, 1)].mean()
                        
                        st.markdown("### 🌐 Portfolio Risk Aggregation")
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                            sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', ax=ax_corr, square=True)
                            ax_corr.set_title("Correlation Matrix", fontsize=10)
                            st.pyplot(fig_corr)
                            
                        with col_b:
                            fig_hist, ax_hist = plt.subplots(figsize=(6, 4.5))
                            ax_hist.hist(combined_pnl, bins=80, color='#00d4ff', alpha=0.7)
                            ax_hist.axvline(np.percentile(combined_pnl, 1), color='red', linestyle='--')
                            ax_hist.set_title("Portfolio PnL Distribution", fontsize=10)
                            st.pyplot(fig_hist)
                        
                        st.divider()
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Active Assets", num_active)
                        r2.error(f"**Portfolio 99% CVaR**\n\n### ${abs(cvar_99):,.2f}")
                        r3.info(f"**Max Potential Drawdown**\n\n### ${abs(np.min(combined_pnl)):,.2f}")
