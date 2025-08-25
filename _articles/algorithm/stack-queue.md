---
title: "Stack & Queue 全攻略：Python 語法大全 + 所有常見演算法題型解析"
date: 2025-05-13 17:20:00 +0800
categories: [Algorithm]
tags: ["stack", "queue", "data-structure"]
---

# Stack & Queue 全攻略：Python 語法大全 + 所有常見演算法題型解析

**Stack（堆疊）** 和 **Queue（佇列）** 是面試中經常出現的資料結構，廣泛應用於回溯、遞迴模擬、樹與圖的遍歷、以及各種序列處理問題。

> 本文包含兩大主軸：
> 1. 📘 Python 中 Stack / Queue 操作語法與內建工具
> 2. 🧠 面試常見題型與解題策略，含完整邏輯範例

---

## 📘 第一部分：Python 中的 Stack / Queue 操作語法

---

### 🧱 Stack（先進後出：LIFO）

```python
stack = []

stack.append(1)     # push
x = stack.pop()     # pop
peek = stack[-1]    # top
empty = len(stack) == 0
```

> ✅ `list` 即可當 stack 使用，或使用 `collections.deque()` 增強效能。

---

### 🧱 Queue（先進先出：FIFO）

```python
from collections import deque

queue = deque()

queue.append(1)     # enqueue
x = queue.popleft() # dequeue
peek = queue[0]
empty = len(queue) == 0
```

---

### 🔁 雙向佇列（Deque）

```python
queue.appendleft(x)
queue.pop()
```

---

### ⛓️ 模擬其他結構

* Queue with 2 Stacks
* Stack with 2 Queues
* Monotonic Stack / Queue
* 使用 `heapq` 實作 Priority Queue（補充於 Heap 篇）

---

## 🧠 第二部分：面試常見題型與解法策略

---

### 1. 📚 Stack 類題：回溯、括號、前綴後綴

* Valid Parentheses
* Evaluate Reverse Polish Notation
* Decode String（LeetCode #394）
* Min Stack / Max Stack

```python
# 有效括號配對
def isValid(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in mapping:
            if not stack or stack.pop() != mapping[c]:
                return False
        else:
            stack.append(c)
    return not stack
```

---

### 2. 🔁 Queue 類題：BFS 遍歷、滑動窗口、傳播擴散

* 二元樹的層序遍歷
* 二維矩陣的最短距離
* 滑動視窗最大值（Monotonic Queue）
* Rotten Oranges / Zombie in Matrix

```python
# 樹的 BFS
def levelOrder(root):
    res, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```

---

### 3. ⚙️ 單調堆疊 / 雙端滑窗

* Next Greater Element
* Sliding Window Maximum
* Stock Span Problem

```python
# 單調遞減堆疊解 Next Greater Element
def nextGreater(nums):
    res = [-1] * len(nums)
    stack = []
    for i in range(len(nums)-1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            res[i] = stack[-1]
        stack.append(nums[i])
    return res
```

---

## 📑 題型彙整表

| 類型            | 常見題目                      |
| --------------- | ----------------------------- |
| Stack 配對/回溯 | 括號配對、字串重構、Undo 模擬 |
| Stack 運算處理  | 逆波蘭式、計算器              |
| Queue 擴散/BFS  | 樹遍歷、網格傳播、感染計算    |
| Monotonic Stack | 下一個更大、最大矩形          |
| 滑動窗口 Queue  | 最大值、最短長度              |

---

## ✅ 面試應對與技巧

* Stack 擅長：**回溯、記錄狀態、括號結構、逆序處理**
* Queue 擅長：**序列遍歷、傳播、最短步數、最小天數**
* Monotonic Stack/Queue：**最大值維護、Window 題高頻利器**
* 遇到 **「對稱性」/「嵌套結構」** 想 Stack
* 遇到 **「層級擴展」/「依序遍歷」** 想 Queue / BFS

---

## 📘 推薦題庫資源

* [LeetCode - Stack & Queue 題庫](https://leetcode.com/tag/stack/)
* [NeetCode Patterns - Monotonic Stack](https://neetcode.io/)
* [GeeksForGeeks Stack/Queue Problems](https://www.geeksforgeeks.org/stack-data-structure/)

---

## ✅ 結語

Stack 與 Queue 題型雖然概念簡單，卻能延伸出大量面試挑戰題，從括號驗證到二元樹 BFS，從滑動窗口到單調堆疊。熟練操作技巧與辨識題型模式，是攻克中高階演算法題的關鍵。
