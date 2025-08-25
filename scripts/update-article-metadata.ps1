# Article metadata update script
# 1. Article category corresponds to directory folder
# 2. Merge deep learning into machine learning  
# 3. All tags in English, not too similar to category
# 4. Maximum 3 tags per article

# Define category mapping (directory name -> display name)
$categoryMapping = @{
    "algorithm" = "Algorithm"
    "backend" = "Backend"
    "cloud" = "Cloud"
    "data-engineering" = "Data Engineering"
    "devops" = "DevOps"
    "machine-learning" = "Machine Learning"
    "optimization" = "Optimization"
    "system-design" = "System Design"
}

# Define tag mapping for English conversion
$tagMapping = @{
    "Learning Rate" = "learning-rate"
    "Step Decay" = "step-decay"
    "Cosine Annealing" = "cosine-annealing"
    "Warm-up" = "warmup"
    "Cyclical" = "cyclical"
    "One-Cycle" = "one-cycle"
    "LR Finder" = "lr-finder"
    "CNN" = "cnn"
    "ResNet" = "resnet"
    "EfficientNet" = "efficientnet"
    "Inception" = "inception"
    "TCN" = "tcn"
    "WaveNet" = "wavenet"
    "RNN" = "rnn"
    "LSTM" = "lstm"
    "GRU" = "gru"
    "Transformer" = "transformer"
    "BERT" = "bert"
    "GPT" = "gpt"
    "Dropout" = "dropout"
    "BatchNorm" = "batch-norm"
    "PEFT" = "peft"
    "LoRA" = "lora"
    "GAN" = "gan"
    "VAE" = "vae"
}

function Update-ArticleMetadata {
    param(
        [string]$FilePath,
        [string]$Category
    )
    
    $content = Get-Content $FilePath -Raw -Encoding UTF8
    
    # 提取現有的 frontmatter
    if ($content -match '(?s)^---\s*\n(.*?)\n---') {
        $frontmatter = $matches[1]
        $bodyContent = $content -replace '(?s)^---\s*\n.*?\n---\s*\n?', ''
        
        # 解析現有 tags
        $existingTags = @()
        if ($frontmatter -match 'tags:\s*\[(.*?)\]') {
            $tagString = $matches[1]
            $existingTags = $tagString -split ',\s*' | ForEach-Object { $_.Trim().Trim('"').Trim("'") }
        }
        
        # 轉換 tags 為英文，限制最多3個
        $newTags = @()
        foreach ($tag in $existingTags) {
            if ($tagMapping.ContainsKey($tag)) {
                $englishTag = $tagMapping[$tag]
                if ($newTags -notcontains $englishTag -and $englishTag -ne $Category.ToLower()) {
                    $newTags += $englishTag
                }
            } elseif ($tag -match '^[a-zA-Z0-9\-]+$' -and $tag -ne $Category.ToLower()) {
                # 已經是英文的 tag
                if ($newTags -notcontains $tag.ToLower()) {
                    $newTags += $tag.ToLower()
                }
            }
            
            # 限制最多3個 tags
            if ($newTags.Count -ge 3) {
                break
            }
        }
        
        # 構建新的 frontmatter
        $title = ""
        $date = ""
        
        if ($frontmatter -match 'title:\s*["\']?(.*?)["\']?\s*$') {
            $title = $matches[1].Trim('"').Trim("'")
        }
        
        if ($frontmatter -match 'date:\s*(.*)$') {
            $date = $matches[1].Trim()
        }
        
        $tagsString = '["' + ($newTags -join '", "') + '"]'
        
        $newFrontmatter = @"
---
title: "$title"
date: $date
categories: [$Category]
tags: $tagsString
---
"@
        
        # 組合新內容
        $newContent = $newFrontmatter + "`n`n" + $bodyContent.TrimStart()
        
        # 寫回文件
        Set-Content -Path $FilePath -Value $newContent -Encoding UTF8
        
        Write-Host "Updated: $FilePath -> Category: $Category, Tags: $($newTags -join ', ')"
    }
}

# 處理所有文章
$articlesPath = "C:\Users\yuhan\workplace\yu-codes.github.io\_articles"

foreach ($categoryDir in $categoryMapping.Keys) {
    $fullPath = Join-Path $articlesPath $categoryDir
    if (Test-Path $fullPath) {
        $categoryName = $categoryMapping[$categoryDir]
        $mdFiles = Get-ChildItem -Path $fullPath -Filter "*.md"
        
        foreach ($file in $mdFiles) {
            Update-ArticleMetadata -FilePath $file.FullName -Category $categoryName
        }
    }
}

Write-Host "Article metadata update completed!"
