# Article metadata update script
param()

# Define category mapping
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

# Common tag mappings to English
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
    
    # Extract frontmatter
    if ($content -match '(?s)^---\s*\r?\n(.*?)\r?\n---') {
        $frontmatter = $matches[1]
        $bodyContent = $content -replace '(?s)^---\s*\r?\n.*?\r?\n---\s*\r?\n?', ''
        
        # Parse existing tags
        $existingTags = @()
        if ($frontmatter -match 'tags:\s*\[(.*?)\]') {
            $tagString = $matches[1]
            $existingTags = $tagString -split ',\s*' | ForEach-Object { 
                $_.Trim().Trim('"').Trim("'") 
            }
        }
        
        # Convert tags to English, max 3
        $newTags = @()
        foreach ($tag in $existingTags) {
            if ($tagMapping.ContainsKey($tag)) {
                $englishTag = $tagMapping[$tag]
                if ($newTags -notcontains $englishTag -and $englishTag -ne $Category.ToLower()) {
                    $newTags += $englishTag
                }
            } elseif ($tag -match '^[a-zA-Z0-9\-]+$' -and $tag -ne $Category.ToLower()) {
                if ($newTags -notcontains $tag.ToLower()) {
                    $newTags += $tag.ToLower()
                }
            }
            
            if ($newTags.Count -ge 3) {
                break
            }
        }
        
        # Extract title and date
        $title = ""
        $date = ""
        
        if ($frontmatter -match 'title:\s*"([^"]*)"') {
            $title = $matches[1]
        } elseif ($frontmatter -match "title:\s*'([^']*)'") {
            $title = $matches[1]
        } elseif ($frontmatter -match 'title:\s*(.+)$') {
            $title = $matches[1].Trim()
        }
        
        if ($frontmatter -match 'date:\s*(.+)$') {
            $date = $matches[1].Trim()
        }
        
        # Build new frontmatter
        $tagsString = if ($newTags.Count -gt 0) { 
            '[' + (($newTags | ForEach-Object { '"' + $_ + '"' }) -join ', ') + ']'
        } else { 
            '[]' 
        }
        
        $newFrontmatter = @"
---
title: "$title"
date: $date
categories: [$Category]
tags: $tagsString
---
"@
        
        # Combine new content
        $newContent = $newFrontmatter + "`r`n`r`n" + $bodyContent.TrimStart()
        
        # Write back to file
        Set-Content -Path $FilePath -Value $newContent -Encoding UTF8
        
        Write-Host "Updated: $FilePath -> Category: $Category, Tags: $($newTags -join ', ')"
    }
}

# Process all articles
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
