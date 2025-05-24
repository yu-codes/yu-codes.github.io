---
title: "強化式學習速查：MDP、Q-Learning、DQN、Policy Gradient 與探索-利用平衡"
date: 2025-05-18 22:00:00 +0800
categories: [機器學習理論]
tags: [強化學習, MDP, Q-Learning, DQN, Policy Gradient, 探索-利用, ε-Greedy, UCB]
---

# 強化式學習速查：MDP、Q-Learning、DQN、Policy Gradient 與探索-利用平衡

強化式學習（Reinforcement Learning, RL）是 AI 自主決策與控制的核心。從馬可夫決策過程（MDP）、Q-Learning、DQN，到 Policy Gradient、探索-利用平衡（ε-Greedy、UCB），這些理論與演算法是自動駕駛、遊戲 AI、機器人等領域的基石。本章將深入數學原理、直覺圖解、Python 實作、應用場景、面試熱點與常見誤區，幫助你快速掌握 RL 核心知識。

---

## MDP（馬可夫決策過程）

### 定義

- $S$：狀態空間，$A$：動作空間，$P$：轉移機率，$R$：獎勵函數，$\gamma$：折扣因子。
- 目標：學習策略 $\pi(a|s)$，最大化累積期望獎勵。

### 貝爾曼方程（Bellman Equation）

- 狀態價值函數 $V^\pi(s)$：
  $$
  V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \mid s_0 = s \right]
  $$
- 最優價值函數滿足：
  $$
  V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
  $$

---

## Policy / Value Function

- **策略（Policy）**：決定在每個狀態下採取哪個動作。
- **價值函數（Value Function）**：評估狀態或狀態-動作對的好壞。
  - 狀態價值 $V(s)$、動作價值 $Q(s,a)$。

---

## Q-Learning、DQN、Policy Gradient 對比

### Q-Learning

- 無模型、離線學習，更新規則：
  $$
  Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]
  $$
- 適合小型離散空間。

### DQN（Deep Q-Network）

- 用深度神經網路近似 Q 函數，適合高維狀態空間。
- 經典應用：Atari 遊戲 AI。

```python
import numpy as np

Q = np.zeros((5, 2))  # 5 狀態, 2 動作
alpha, gamma = 0.1, 0.99
s, a, r, s_next = 0, 1, 1, 2
Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
```

### Policy Gradient

- 直接優化策略參數，最大化期望獎勵。
- 適合連續動作空間，常用於機器人控制、AlphaGo。

```python
import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
```

---

## 探索-利用平衡（Exploration-Exploitation）

### ε-Greedy

- 以機率 ε 隨機探索，其餘時間選擇最佳動作。
- ε 可隨訓練逐步減小。

### UCB（Upper Confidence Bound）

- 根據置信上界選擇動作，兼顧平均回報與不確定性。
- 常用於多臂賭徒問題（Multi-Armed Bandit）。

---

## 理論直覺、應用場景與常見誤區

### 應用場景

- 遊戲 AI（AlphaGo、Atari）、機器人控制、自動駕駛、推薦系統
- 連續決策、動態規劃、資源分配

### 常見誤區

- 忽略探索，導致陷入次優策略
- Q-Learning 不適用於高維連續空間
- DQN 訓練不穩定，需經驗回放與目標網路
- Policy Gradient 易陷入高方差，需 baseline 技巧

---

## 面試熱點與經典問題

| 主題         | 常見問題 |
|--------------|----------|
| MDP          | 定義與貝爾曼方程？ |
| Q-Learning   | 更新規則與收斂性？ |
| DQN          | 如何穩定訓練？ |
| Policy Gradient | 優缺點與應用？ |
| 探索-利用    | ε-Greedy 與 UCB 差異？ |

---

## 使用注意事項

* 強化學習需大量互動資料，訓練成本高
* DQN 類方法需經驗回放（Replay Buffer）與目標網路
* Policy Gradient 需 variance reduction 技巧（如 baseline、GAE）

---

## 延伸閱讀與資源

* [Deep RL Book](https://www.deepreinforcementlearningbook.org/)
* [OpenAI Spinning Up RL](https://spinningup.openai.com/)
* [DQN 論文](https://www.nature.com/articles/nature14236)
* [Policy Gradient 論文](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)

---

## 經典面試題與解法提示

1. MDP 的五大元素與貝爾曼方程推導？
2. Q-Learning 如何更新？何時收斂？
3. DQN 如何解決 Q-Learning 的限制？
4. Policy Gradient 的數學推導與應用場景？
5. 探索-利用平衡有哪些策略？
6. ε-Greedy 如何設計 ε 衰減？
7. UCB 的數學原理與應用？
8. 強化學習在推薦系統的應用？
9. DQN 訓練不穩定的原因與解法？
10. Policy Gradient 如何降低方差？

---

## 結語

強化式學習是 AI 決策與控制的核心。熟悉 MDP、Q-Learning、DQN、Policy Gradient 與探索-利用平衡，能讓你在自動化決策、遊戲 AI、機器人等領域發揮專業實力。下一章將進入倫理、偏差與公平，敬請期待！
