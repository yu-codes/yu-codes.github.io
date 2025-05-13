---
title: "Tree & Recursion 全攻略：Python 實作語法 + 經典題型分類解析"
date: 2025-05-13 17:40:00 +0800
categories: [Algorithm]
tags: [Binary Tree, Recursion, DFS, 資料結構, 演算法]
---

# Tree & Recursion 全攻略：Python 實作語法 + 經典題型分類解析

樹（Tree）是資料結構中最經典也最具挑戰性的題型之一，而遞迴（Recursion）幾乎是處理樹與圖題的標準技巧。無論是 DFS、BFS、樹高、遍歷還是構建，這類問題經常出現在中高階面試中。

---

## 📘 第一部分：Python 中樹的定義與遞迴語法

### 🌳 樹節點定義

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

---

### 🔁 遞迴的基本框架

```python
def traverse(node):
    if not node:
        return
    traverse(node.left)
    traverse(node.right)
```

遞迴三元素：

* 終止條件（base case）
* 拆解子問題
* 回傳（可選）

---

## 🧠 第二部分：經典樹題型與遞迴解法分類

---

### 1. 🌿 遍歷 Traversal（前中後層）

* Preorder / Inorder / Postorder
* Level Order Traversal（使用 queue）

```python
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []
```

---

### 2. 🧮 樹的性質計算

* 最大深度 / 最小深度
* 是否平衡樹
* 節點總數 / 葉子節點數

```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

---

### 3. 🔁 合併型遞迴

* 相同樹判斷
* 鏡像樹（symmetric tree）
* 合併兩棵樹（merge trees）

```python
def isSameTree(p, q):
    if not p and not q: return True
    if not p or not q: return False
    return p.val == q.val and isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

---

### 4. 📤 返回值型遞迴（重點！）

* Lowest Common Ancestor（LCA）
* 路徑和（Path Sum） / 所有路徑
* 子樹是否存在某結構

```python
def hasPathSum(root, target):
    if not root: return False
    if not root.left and not root.right:
        return root.val == target
    return hasPathSum(root.left, target - root.val) or hasPathSum(root.right, target - root.val)
```

---

### 5. 🧩 DFS + Backtracking

* Binary Tree Paths
* Sum of Root to Leaf Numbers
* 所有符合條件的路徑

---

### 6. 🧠 構建型題目（建構樹）

* 從 preorder/inorder 重建二元樹
* 有序陣列轉 BST

```python
def buildTree(preorder, inorder):
    if not preorder or not inorder: return None
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])
    return root
```

---

## 📑 題型彙整表

| 題型       | 常見題目關鍵詞                      |
| -------- | ---------------------------- |
| 遍歷       | Inorder, Preorder, Postorder |
| 計算性質     | 最大深度、葉子節點、路徑和                |
| 判斷結構     | 是否為同一棵、鏡像、平衡樹                |
| 構建 / 還原樹 | 從遍歷還原樹、BST 建構                |
| 回傳型遞迴    | 是否存在、回傳 LCA、找符合條件路徑          |

---

## 💼 面試建議與策略

* 遞迴邏輯要清晰說明「base case → 拆解 → 合併」
* 強調每個題目是「前序」、「中序」、「後序」哪一種場景
* 遞迴結構盡量簡潔明確，避免太多額外變數污染邏輯
* 若題目限制遞迴，可使用 Stack 模擬（視為延伸挑戰）

---

## 📘 題庫與練習推薦

* [LeetCode Tree 題庫](https://leetcode.com/tag/tree/)
* [NeetCode - Recursion & Tree Patterns](https://neetcode.io/)
* [Python Recursion Tutorial](https://realpython.com/python-thinking-recursively/)

---

## ✅ 結語

樹與遞迴是面試中「必會、必考、變化多」的類型。只要掌握基本遍歷、回傳型遞迴與結構構建技巧，就能應對 90% 的常見題。
