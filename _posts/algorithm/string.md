---
title: "Python 字串語法大全 + 所有常見演算法題型解析"
date: 2025-05-13 17:00:00 +0800
categories: [Algorithm]
tags: [String, Python, 演算法, 資料結構]
---

# Python 字串語法大全 + 所有常見演算法題型解析

字串是另一個在面試中極高機率出現的資料類型。從基本操作到語意處理、從雙指針到 KMP 演算法，字串處理考驗的是你對「序列、索引、轉換」的理解深度。

> 本文包含兩大部分：
> 1. 📘 Python 字串語法完全整理
> 2. 🧠 字串演算法題型全分類與實作範例

---

## 📘 第一部分：Python 字串語法大全（str 操作技巧）

Python 的 `str` 是不可變資料型別（immutable），可進行大量內建轉換與處理。

### 🧱 建立與基本操作

```python
s = "hello"
len(s)            # 長度
s[0]              # 索引取值
s[-1]             # 從後面取值
s[1:4]            # 切片，不包含尾端
s[::-1]           # 字串反轉
"e" in s          # 成員檢查
```

---

### 🔤 常見轉換與清理

```python
s.upper()         # 轉大寫
s.lower()         # 轉小寫
s.strip()         # 去除頭尾空白
s.replace("a", "b") # 替換字元
s.split(" ")      # 拆字串
"-".join(list)    # 合併為字串
s.startswith("he")
s.endswith("lo")
```

---

### 🔎 查找與比較

```python
s.find("l")        # 傳回第一個 l 的 index（找不到回 -1）
s.index("l")       # 與 find 類似，但找不到會報錯
s.count("l")       # 出現次數
s == "hello"       # 相等比較
s.isdigit()        # 是否為數字
s.isalpha()        # 是否為字母
```

---

### ✍️ 特殊處理技巧

```python
# 快速建字串
"".join([chr(97 + i) for i in range(26)])

# f-string 格式化
name = "John"
f"Hello, {name}!"
```

---

## 🧠 第二部分：常見字串演算法題型與解法

---

### 1. ✅ 基礎轉換與統計

* 字元統計
* 反轉字串
* 移除空白 / 特定符號

```python
collections.Counter(s)
s[::-1]
s.replace(" ", "")
```

---

### 2. 🧠 雙指針與滑動視窗技巧

* 判斷是否為回文
* 找出最長不重複子字串
* 子字串匹配

```python
def is_palindrome(s):
    l, r = 0, len(s)-1
    while l < r:
        if s[l] != s[r]: return False
        l += 1; r -= 1
    return True
```

---

### 3. 🔍 子字串搜尋與比較

* 子字串是否存在
* 最小覆蓋子串
* 模式匹配（暴力 / Rabin-Karp / KMP）

```python
# 最小覆蓋子串（LeetCode: Minimum Window Substring）
# 利用雙指針 + Counter
```

---

### 4. 📦 字串與資料結構結合

* 頻率對照（anagram 題）
* 雜湊表判重（字母順序、組合）
* 用 dict 記錄索引與狀態

```python
# Valid Anagram
return sorted(s) == sorted(t)
```

---

### 5. 🧮 分組與整理

* 字串壓縮（"aabb" → "a2b2"）
* 字母群組（Group Anagrams）
* 字元分類（母音 / 子音）

---

### 6. 🧾 分段與重建

* 重組句子（reverse words）
* 拆解為字典中詞語（word break）
* 加密 / 解密處理（base64 / 位移）

---

### 📑 題型彙整表

| 題型         | 常見關鍵字                             |
| ---------- | --------------------------------- |
| 基本處理       | 反轉、轉小寫、替換、切割                      |
| 回文         | 雙指針、切割、前後比較                       |
| 子字串處理      | 滑動視窗、子串比對、最長長度                    |
| Anagram 題型 | 頻率統計、雜湊、排序                        |
| 模式搜尋       | KMP、Rabin-Karp、find()             |
| 字元處理       | isalpha(), isdigit(), split, join |

---

## 🛠 面試中如何表現字串題解

* ✅ 明確界定「字元 vs 子字串 vs 單字」
* ✅ 用 `collections.Counter`、`defaultdict`、`set()` 實作統計題
* ✅ 多用 `in`, `s.count()`, `s.find()` 解決暴力解法
* ✅ 熟練模板類題如滑動視窗、字串對應比較

---

## 📘 推薦題庫資源

* [LeetCode - String 題庫](https://leetcode.com/tag/string/)
* [GeeksForGeeks String Problems](https://www.geeksforgeeks.org/python-strings/)
* [Python 字串官方文件](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str)

---

## ✅ 結語

字串題型範圍廣泛但規則性高，練熟各種基本操作與雙指針技巧，並學會搭配雜湊、集合等結構，就能快速解決大多數面試題。
