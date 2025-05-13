---
title: "Python Dict 語法大全 + 所有常見演算法題型解析"
date: 2025-05-13 17:10:00 +0800
categories: [Algorithm]
tags: [Hash Table, Dict, Python, 演算法, 面試]
---

# Python Dict 語法大全 + 所有常見演算法題型解析

Hash Table（雜湊表）是技術面試中最關鍵的資料結構之一，它能在常數時間內完成查找、插入與刪除，幾乎可用於優化任何涉及查找或記錄的題型。

> 本文分成兩大部分：
> 1. 📘 Python `dict` 語法完全整理
> 2. 🧠 Hash Table 類型面試題與策略總覽

---

## 📘 第一部分：Python 字典（dict）語法大全

Python 的字典（dict）是一個 key-value 對應資料結構，底層實作為 Hash Table。

### 🔑 基本操作

```python
d = {"a": 1, "b": 2}
d["a"]        # 取得值 → 1
d["c"] = 3    # 新增鍵
del d["b"]    # 刪除鍵
"a" in d      # 是否存在鍵
len(d)        # 鍵值數量
```

---

### 🧠 進階操作技巧

```python
d.get("x", 0)            # 安全取值，不存在回預設值
d.setdefault("k", [])    # 若不存在則設為預設值
d.keys()                 # 所有鍵
d.values()               # 所有值
d.items()                # 所有鍵值對
```

---

### 🔁 遍歷與反轉

```python
for key, val in d.items():
    print(key, val)

# 字典反轉
reversed_d = {v: k for k, v in d.items()}
```

---

### 🔧 常見輔助工具

```python
from collections import defaultdict, Counter

d = defaultdict(int)        # 預設為 0
freq = Counter("banana")    # 統計字元次數
```

---

## 🧠 第二部分：Hash Table 面試題型與解法

---

### 1. 🔍 出現次數統計

* 出現最多 / 最少的元素
* 統計字元 / 數字出現次數

```python
from collections import Counter
most_common = Counter(arr).most_common(1)
```

---

### 2. ✅ 判斷重複與唯一性

* 檢查是否有重複（如 `containsDuplicate`）
* 檢查兩陣列是否相同（元素順序無關）

```python
return len(set(arr)) != len(arr)
```

---

### 3. 🧩 兩數問題（Two Sum）系列

* 兩數加總為目標值
* 記錄值 → index 對應

```python
def two_sum(nums, target):
    lookup = {}
    for i, num in enumerate(nums):
        if target - num in lookup:
            return [lookup[target - num], i]
        lookup[num] = i
```

---

### 4. 🧠 字串對應 / 映射題

* 是否為同構字（isomorphic strings）
* Word pattern 比對
* 雜湊 map 建立雙向對應

```python
def is_isomorphic(s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))
```

---

### 5. 📦 Group 類題：Anagram、分群、分類

* Group Anagrams
* 分類字串 / 數字群組

```python
d = defaultdict(list)
for word in words:
    key = "".join(sorted(word))
    d[key].append(word)
```

---

### 6. 🧮 Prefix sum + Hash Table 結合

* Subarray sum 等於 k 的個數
* 記錄前綴和與出現次數

```python
def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    d = defaultdict(int)
    d[0] = 1

    for num in nums:
        prefix_sum += num
        count += d[prefix_sum - k]
        d[prefix_sum] += 1
    return count
```

---

### 📑 題型彙整表

| 類型      | 常見題目關鍵字            |
| ------- | ------------------ |
| 記錄出現次數  | 字元頻率、數字頻率          |
| 判斷唯一性   | 判重、比對、同構           |
| 快速查找    | 兩數和、補值比對           |
| 群組分類    | Anagram 題、Group By |
| 雜湊最佳化查詢 | 前綴和、滑動視窗查表加速       |

---

## 🛠 實作建議與最佳實踐

* 使用 `Counter` 做計數，`set()` 判重
* 雙雜湊對映可解決一對一對映問題
* `defaultdict` 讓你不用手動初始化 list / int
* 字典查詢時間複雜度為 O(1) → 非常適合優化暴力解法

---

## 🧾 面試應對建議

1. **當題目出現「最快」、「有沒有重複」、「配對」時，第一時間考慮 Hash Table**
2. **若題目需「分類」、「分組」、「查頻率」，考慮用 `defaultdict` 或 `Counter`**
3. **將雜湊表當作 cache / lookup table，搭配 prefix sum、滑窗常見於進階題**

---

## 📘 推薦資源

* [LeetCode Hash Table 題庫](https://leetcode.com/tag/hash-table/)
* [Python Counter 官方文件](https://docs.python.org/3/library/collections.html#collections.Counter)
* [NeetCode Hash Table Patterns](https://neetcode.io/)

---

## ✅ 結語

Hash Table 是面試中不可忽視的利器，只要你掌握 Python `dict` 的各種操作方式、常見的計數與對應技巧，以及題型轉換思維，就能應付超過 70% 的常見邏輯挑戰。