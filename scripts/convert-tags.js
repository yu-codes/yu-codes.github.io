const fs = require('fs');
const path = require('path');

// Chinese to English tag mapping
const tagMapping = {
  '學習率': 'learning-rate',
  '卷積': 'convolution',
  '池化': 'pooling',
  '注意力機制': 'attention',
  '自監督學習': 'self-supervised',
  '遷移學習': 'transfer-learning',
  '強化學習': 'reinforcement-learning',
  '多模態': 'multimodal',
  '視覺語言': 'vision-language',
  '生成模型': 'generative-models',
  '擴散模型': 'diffusion',
  '正則化': 'regularization',
  '訓練技巧': 'training-tricks',
  '參數高效': 'parameter-efficient',
  '微調': 'fine-tuning',
  '分散式訓練': 'distributed-training',
  '模型壓縮': 'model-compression',
  '量化': 'quantization',
  '剪枝': 'pruning',
  '知識蒸餾': 'knowledge-distillation',
  '資料增強': 'data-augmentation',
  '課程學習': 'curriculum-learning',
  '超參數': 'hyperparameters',
  '優化器': 'optimizer',
  '梯度': 'gradient',
  '反向傳播': 'backpropagation',
  '成本': 'cost',
  '安全': 'security',
  '合規': 'compliance',
  '監控': 'monitoring',
  '觀測性': 'observability',
  '線上離線分離': 'online-offline',
  '告警': 'alerting',
  '追蹤': 'tracing',
  '系統設計': 'system-design',
  '高可用': 'high-availability',
  '負載均衡': 'load-balancing',
  '熔斷': 'circuit-breaker',
  '降級': 'degradation',
  '重試': 'retry',
  '回溯': 'rollback',
  '多活': 'multi-active',
  '冷備': 'cold-standby',
  '熱備': 'hot-standby',
  '數據同步': 'data-sync',
  '管理': 'management',
  '套件化': 'packaging',
  '自動擴縮': 'auto-scaling',
  '模型版本': 'model-versioning',
  '推論 API': 'inference-api',
  '叢集調度': 'cluster-scheduling',
  '冷啟動': 'cold-start',
  '容器化': 'containerization',
  '面試題': 'interview-questions',
  '白板題': 'whiteboard',
  '解題技巧': 'problem-solving',
  '雲端': 'cloud',
  '成本最佳化': 'cost-optimization',
  '自動化部署': 'automated-deployment',
  '特徵一致性': 'feature-consistency',
  '向量化': 'vectorization'
};

function convertTagsToEnglish(content) {
  // Find tags line
  const tagMatch = content.match(/^tags:\s*\[(.*?)\]$/m);
  if (!tagMatch) return content;
  
  const tagsString = tagMatch[1];
  const tags = tagsString.split(',').map(tag => tag.trim().replace(/['"]/g, ''));
  
  // Convert tags to English and limit to 3
  const englishTags = [];
  for (const tag of tags) {
    if (tagMapping[tag]) {
      englishTags.push(tagMapping[tag]);
    } else if (/^[a-zA-Z0-9\-\s]+$/.test(tag)) {
      // Already English, convert to lowercase and replace spaces with hyphens
      englishTags.push(tag.toLowerCase().replace(/\s+/g, '-'));
    }
    
    if (englishTags.length >= 3) break;
  }
  
  // Create new tags string
  const newTagsString = englishTags.map(tag => `"${tag}"`).join(', ');
  const newTagsLine = `tags: [${newTagsString}]`;
  
  // Replace the tags line
  return content.replace(/^tags:\s*\[.*?\]$/m, newTagsLine);
}

function processFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const newContent = convertTagsToEnglish(content);
    
    if (newContent !== content) {
      fs.writeFileSync(filePath, newContent, 'utf8');
      console.log(`Updated: ${filePath}`);
    }
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
  }
}

function processDirectory(dirPath) {
  const items = fs.readdirSync(dirPath);
  
  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      processDirectory(fullPath);
    } else if (item.endsWith('.md')) {
      processFile(fullPath);
    }
  }
}

// Start processing
const articlesPath = 'C:\\Users\\yuhan\\workplace\\yu-codes.github.io\\_articles';
processDirectory(articlesPath);

console.log('Tag conversion completed!');
