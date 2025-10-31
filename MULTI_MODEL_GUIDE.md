# 多模型支持指南

RAG4Spice 现已支持多种AI模型，包括中国和国际的主流大语言模型。

## 🌍 支持的模型

### 国际模型
- **Google Gemini**: `gemini-2.5-flash`, `gemini-2.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- **Anthropic Claude**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- **Mistral AI**: `mistral-large-latest`, `mistral-small`
- **Cohere**: `command-r-plus`, `command`

### 中国模型 🇨🇳
- **阿里云通义千问**: `qwen-turbo`, `qwen-plus`, `qwen-max`
- **百度文心一言**: `ernie-bot-4`, `ernie-bot-turbo`
- **智谱清言**: `glm-4`, `glm-3-turbo`
- **月之暗面 Kimi**: `moonshot-v1-8k`, `moonshot-v1-32k`
- **深度求索 DeepSeek**: `deepseek-chat`, `deepseek-coder`

## 🚀 快速开始

### 1. 配置API密钥

复制 `.env.example` 为 `.env` 并配置所需API密钥：c

```bash
cp .env.example .env
```

编辑 `.env` 文件，添加你的API密钥：

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# 阿里云通义千问
DASHSCOPE_API_KEY=your_dashscope_api_key

# 智谱清言
ZHIPUAI_API_KEY=your_zhipuai_api_key

# Kimi
MOONSHOT_API_KEY=your_moonshot_api_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 2. 启动应用

```bash
streamlit run app.py
```

### 3. 选择模型

在应用的侧边栏中：
1. 选择你想要使用的AI模型
2. 配置对应的API密钥（如果未设置环境变量）
3. 测试API连接
4. 开始使用

## 📋 API密钥获取方式

| 提供商 | 获取地址 | 说明 |
|--------|----------|------|
| Google | [AI Studio](https://aistudio.google.com/) | 免费额度较高 |
| OpenAI | [Platform](https://platform.openai.com/) | 需要付费 |
| Anthropic | [Console](https://console.anthropic.com/) | 需要付费 |
| 阿里云 | [百炼平台](https://bailian.console.aliyun.com/) | 国内访问快 |
| 百度 | [智能云](https://cloud.baidu.com/) | 国内访问快 |
| 智谱 | [开放平台](https://open.bigmodel.cn/) | 国内访问快 |
| Kimi | [开放平台](https://platform.moonshot.cn/) | 国内访问快 |
| DeepSeek | [平台](https://platform.deepseek.com/) | 价格便宜 |
| Mistral | [Console](https://console.mistral.ai/) | 欧洲模型 |
| Cohere | [Dashboard](https://dashboard.cohere.com/) | 企业级 |

## 🎯 推荐配置

### 经济实用型
- **国际**: `gpt-4o-mini` + `claude-3-haiku-20240307`
- **国内**: `qwen-turbo` + `glm-3-turbo` + `deepseek-chat`

### 高性能型
- **国际**: `gemini-2.5-flash` + `gpt-4o` + `claude-3-5-sonnet-20241022`
- **国内**: `qwen-max` + `glm-4` + `moonshot-v1-32k`

### 代码生成专用
- **推荐**: `deepseek-coder` (专门优化代码生成)
- **备选**: `gpt-4o` + `claude-3-5-sonnet-20241022`

## 🔧 高级功能

### 1. 会话内API密钥管理
- 支持在应用界面直接输入API密钥
- API密钥仅保存在当前会话中，重启后清空
- 支持多个模型的API密钥同时保存

### 2. API连接测试
- 每个模型都提供连接测试功能
- 自动验证API密钥有效性
- 显示连接状态和错误信息

### 3. 模型信息展示
- 显示每个模型的详细参数
- 包含最大Token数、温度设置等
- 提供模型描述和适用场景

### 4. 智能推荐
- 根据使用场景推荐合适模型
- 区分经济型和高性能选项
- 提供中国模型和国际模型选择

## 🛠️ 技术特性

### 统一API接口
- 所有模型使用相同的调用接口
- 自动处理不同模型的参数差异
- 统一的错误处理和重试机制

### 智能降级
- 如果首选模型失败，自动尝试备选模型
- 支持模型间的无缝切换
- 保证服务可用性

### 性能优化
- 连接池管理，减少连接开销
- 智能缓存，提高响应速度
- 并发处理，支持批量任务

## 🔒 安全性

- API密钥本地存储，不上传服务器
- 支持代理配置，保护网络隐私
- 会话隔离，防止密钥泄露

## 📝 使用示例

### 基础使用
1. 选择模型：`qwen-turbo`
2. 输入API密钥
3. 测试连接
4. 上传实验截图
5. 生成HSPICE代码

### 高级使用
1. 配置多个模型的API密钥
2. 根据任务复杂度选择不同模型
3. 简单任务用经济模型，复杂任务用高性能模型
4. 对比不同模型的生成结果

## 🆘 故障排除

### 常见问题

**Q: API连接失败**
A: 检查网络连接、API密钥正确性、代理设置

**Q: 模型响应慢**
A: 尝试切换到更快的模型，如 `qwen-turbo` 或 `claude-3-haiku`

**Q: 生成质量不佳**
A: 切换到高性能模型，如 `gpt-4o` 或 `glm-4`

**Q: 国内访问慢**
A: 优先选择中国模型，如 `qwen`、`glm`、`kimi`

### 调试模式
在应用中查看详细的调试信息：
- API调用日志
- 错误详细信息
- 响应时间统计

## 📈 性能对比

| 模型 | 速度 | 质量 | 成本 | 推荐场景 |
|------|------|------|------|----------|
| gemini-2.5-flash | 快 | 高 | 低 | 日常使用 |
| gpt-4o-mini | 快 | 中 | 低 | 经济型 |
| claude-3-haiku | 快 | 中 | 中 | 平衡型 |
| qwen-turbo | 快 | 中 | 低 | 国内用户 |
| deepseek-chat | 中 | 高 | 低 | 代码生成 |
| glm-4 | 中 | 高 | 中 | 复杂任务 |

## 🔄 版本更新

多模型支持将持续更新：
- 新增更多模型提供商
- 优化模型性能
- 增强用户体验
- 改进错误处理

---

💡 **提示**: 首次使用建议从免费模型开始，如 `gemini-2.5-flash` 或 `qwen-turbo`，熟悉后再尝试付费模型。