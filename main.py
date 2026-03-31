import numpy as np
from data_loader import get_real_market_params
import matplotlib.pyplot as plt
from utils import fetch_and_process_data, bs_put_price, calculate_risk_metrics

# 配置参数
TICKERS = ['MSFT', 'AAPL', 'GOOG', 'GLD', 'JPY=X']
WEIGHTS = np.array([0.30, 0.30, 0.15, 0.15, 0.10]) 
INITIAL_CAPITAL = 100000
MC_SIMS, DAYS = 10000, 30
SCENARIO_JUMP_PARAMS = {
    'MSFT': {'lambda': 12.0, 'mu_j': -0.08, 'sigma_j': 0.05},
    'AAPL': {'lambda': 12.0, 'mu_j': -0.08, 'sigma_j': 0.05},
    'GOOG': {'lambda': 12.0, 'mu_j': -0.08, 'sigma_j': 0.05},
    'GLD':  {'lambda': 5.0,  'mu_j': 0.03,  'sigma_j': 0.03},
    'JPY=X':{'lambda': 8.0,  'mu_j': -0.08, 'sigma_j': 0.06}, 
}

def main():
    returns, r_f = fetch_and_process_data(TICKERS, "2023-01-01", "2026-03-28")
    # 动态获取各个 Ticker 的真实跳跃参数，覆盖掉原本写死的 SCENARIO_JUMP_PARAMS
    SCENARIO_JUMP_PARAMS = {}
    for ticker in TICKERS:
        market_data = get_real_market_params(ticker=ticker, period="3y")
        SCENARIO_JUMP_PARAMS[ticker] = {
            'lambda': market_data['lambda_j'],
            'nu_j': market_data['mu_j'],
            'sigma_j': market_data['sigma_j']
        }
    L = np.linalg.cholesky(returns.cov())
    lambdas = np.array([SCENARIO_JUMP_PARAMS[t]['lambda'] for t in TICKERS])
    mu_js = np.array([SCENARIO_JUMP_PARAMS[t]['mu_j'] for t in TICKERS])
    sigma_js = np.array([SCENARIO_JUMP_PARAMS[t]['sigma_j'] for t in TICKERS])

    portfolio_sims = np.zeros(MC_SIMS)
    for i in range(MC_SIMS):
        Z = np.random.normal(size=(DAYS, len(TICKERS))).dot(L.T)
        jumps = np.random.poisson(lambdas * (1/252), size=(DAYS, len(TICKERS)))
        jump_m = np.ones((DAYS, len(TICKERS)))
        for t in range(DAYS):
            idx = np.where(jumps[t,:] > 0)[0]
            for a in idx:
                jump_m[t,a] = np.exp(np.random.normal(jumps[t,a]*mu_js[a], np.sqrt(jumps[t,a])*sigma_js[a]))

        paths = np.cumprod(np.exp(-0.006 + Z) * jump_m, axis=0)
        portfolio_sims[i] = INITIAL_CAPITAL * np.sum(WEIGHTS * paths[-1, :])

    strike = INITIAL_CAPITAL * 0.90
    cost = bs_put_price(INITIAL_CAPITAL, strike, 60/252, r_f, 0.25)
    hedged = portfolio_sims + np.maximum(strike - portfolio_sims, 0) - cost

    plt.figure(figsize=(10,6))
    plt.hist(portfolio_sims, bins=100, alpha=0.4, label='Naked', color='brown', density=True)
    plt.hist(hedged, bins=100, alpha=0.6, label='Hedged', color='dodgerblue', density=True)
    plt.ylim(0, 0.0005)
    plt.legend()
    plt.savefig('stress_test_results.png')
    print(f"Report: Naked Loss ${calculate_risk_metrics(portfolio_sims, INITIAL_CAPITAL):,.2f}")

if __name__ == "__main__":
    main()
