# Macro-Tail-Hedge-Sim: 日元流动性挤兑下的尾部风险管理模拟器
# Macro-Tail-Hedge-Sim: Quantitative Risk Management for JPY Carry Trade Shocks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 项目背景 | Project Abstract

**中文：**
在宏观波动剧烈的环境下，传统的正态分布模型往往低估了极端事件（Fat-tails）发生的概率。本项目通过模拟 **日元流动性挤兑 (JPY Liquidity Squeeze)** 场景，量化评估了高杠杆科技股组合在极端冲击下的脆弱性。

**English:**
Standard risk models often fail during regime shifts due to Gaussian assumptions. This project simulates a **JPY Liquidity Squeeze** to test the resilience of tech-heavy portfolios.

---

## 🚀 核心技术特性 | Core Features

* **Merton 跳跃扩散模型 (MJD)**：引入泊松跳跃项，真实还原“闪崩”特征。
* **动态期权定价引擎 (B-S Model)**：根据实时 $r$ 和 $\sigma$ 动态计算权利金。
* **极端风险度量 (Advanced Metrics)**：使用 **CVaR** 专注于损失分布最差 1% 场景。

---

## 📊 数学框架 | Mathematical Framework

### 1. 资产价格路径 (Jump Diffusion Process)
$$dS_t = (r - \lambda \kappa) S_t dt + \sigma S_t dW_t + (Y-1) S_t dN_t$$

### 2. 考虑成本的对冲损益 (Hedged Payoff)
$$Net\_Value = S_T + \max(K - S_T, 0) - Premium \cdot e^{rT}$$

---

## 📈 压力测试结论 | Simulation Results

| 指标 (Metrics) | 裸头寸 (Naked) | 对冲组合 (Hedged) |
| :--- | :--- | :--- |
| **CVaR (99% Confidence)** | ~$36,000 (Loss) | ~$12,000 (Loss) |
| **策略评价 (Comment)** | 尾部风险暴露严重 | 凸性保护生效 |

![Stress Test Results](stress_test_results.png)

---

## 🛠️ 快速开始 | Quick Start

1. **环境配置**: `pip install -r requirements.txt`
2. **运行模拟**: `python main.py`

---

## 💡 风险洞察 | Quantitative Insights

1. **波动率成本陷阱**：当 $\sigma > 0.5$ 时，对冲成本将显著侵蚀长期收益。
2. **凸性价值**：期权的非线性赔付是防止系统性风险中账户爆仓的唯一“安全带”。
