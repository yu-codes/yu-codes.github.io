# Convert Chinese tags to English tags
$tagMapping = @{
    "學習率" = "learning-rate"
    "卷積" = "convolution"
    "池化" = "pooling"
    "注意力機制" = "attention"
    "自監督學習" = "self-supervised"
    "遷移學習" = "transfer-learning"
    "強化學習" = "reinforcement-learning"
    "多模態" = "multimodal"
    "視覺語言" = "vision-language"
    "生成模型" = "generative-models"
    "擴散模型" = "diffusion"
    "正則化" = "regularization"
    "訓練技巧" = "training-tricks"
    "參數高效" = "parameter-efficient"
    "微調" = "fine-tuning"
    "分散式訓練" = "distributed-training"
    "模型壓縮" = "model-compression"
    "量化" = "quantization"
    "剪枝" = "pruning"
    "知識蒸餾" = "knowledge-distillation"
    "資料增強" = "data-augmentation"
    "課程學習" = "curriculum-learning"
    "超參數" = "hyperparameters"
    "優化器" = "optimizer"
    "梯度" = "gradient"
    "反向傳播" = "backpropagation"
}

function Convert-TagsToEnglish {
    param([string]$FilePath)
    
    $content = Get-Content $FilePath -Raw -Encoding UTF8
    
    # Extract and update tags in frontmatter
    if ($content -match '(?s)tags:\s*\[(.*?)\]') {
        $originalTags = $matches[1]
        $updatedTags = $originalTags
        
        foreach ($chineseTag in $tagMapping.Keys) {
            $englishTag = $tagMapping[$chineseTag]
            $updatedTags = $updatedTags -replace [regex]::Escape($chineseTag), $englishTag
        }
        
        # Replace the tags section
        $newContent = $content -replace [regex]::Escape("tags: [$originalTags]"), "tags: [$updatedTags]"
        
        if ($newContent -ne $content) {
            Set-Content -Path $FilePath -Value $newContent -Encoding UTF8
            Write-Host "Updated tags in: $FilePath"
        }
    }
}

# Process all markdown files
$articlesPath = "C:\Users\yuhan\workplace\yu-codes.github.io\_articles"
Get-ChildItem -Path $articlesPath -Recurse -Filter "*.md" | ForEach-Object {
    Convert-TagsToEnglish -FilePath $_.FullName
}

Write-Host "Tag conversion completed!"
