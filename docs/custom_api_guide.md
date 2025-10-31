# 自定义API配置指南

本指南介绍如何使用RAG4Spice的自定义API功能，类似Cherry Studio的用户体验。

## 功能特性

- 🔧 **自定义API提供商**：支持任何OpenAI兼容的API端点
- 🤖 **自动模型发现**：自动获取API支持的模型列表
- 🔐 **安全密钥存储**：API密钥加密存储在本地
- 🧪 **连接测试**：实时验证API可用性
- 📊 **状态监控**：显示API连接状态和测试结果

## 支持的API类型

### 1. OpenAI兼容API
- ✅ OpenAI官方API
- ✅ Azure OpenAI
- ✅ LocalAI
- ✅ Ollama (通过OpenAI兼容端点)
- ✅ vLLM
- ✅ FastChat
- ✅ 其他OpenAI兼容的服务

### 2. 本地部署方案
- ✅ Ollama本地服务
- ✅ LM Studio
- ✅ Text Generation WebUI
- ✅ Open WebUI
- ✅ 其他本地大模型服务

## 使用步骤

### 1. 访问自定义API配置

在应用侧边栏中，点击 **"🔧 自定义API配置"** 按钮，进入配置页面。

### 2. 添加新的API配置

#### 基本配置
1. **提供商名称**：为API服务起一个容易识别的名称
2. **API基础URL**：API服务的基础URL
   - OpenAI: `https://api.openai.com/v1`
   - Ollama: `http://localhost:11434/v1`
   - LM Studio: `http://localhost:1234/v1`
3. **API密钥**：API访问密钥（如果需要）
4. **描述**：可选，简单描述此API的用途

#### 示例配置

**Ollama本地服务**
```
提供商名称: 我的Ollama
API基础URL: http://localhost:11434/v1
API密钥: (留空，如果不需要)
描述: 本地Ollama服务
```

**OpenAI官方API**
```
提供商名称: OpenAI官方
API基础URL: https://api.openai.com/v1
API密钥: sk-xxxxxxxxxxxxxxxxxxxxxxxx
描述: OpenAI GPT模型
```

### 3. 测试连接

保存配置后，系统会自动：
- 🔄 测试API连接
- 🔍 发现可用模型列表
- ✅ 显示测试结果

### 4. 使用自定义模型

配置完成后，返回主配置页面，在模型选择器中可以看到：
- 🔧 我的Ollama: llama2
- 🔧 我的Ollama: codellama
- 🔧 OpenAI官方: gpt-4o

选择自定义模型进行任务分析和代码生成。

## 常见配置示例

### 1. Ollama本地部署
```bash
# 启动Ollama服务
ollama serve

# 确保服务运行在 http://localhost:11434
```

配置参数：
- URL: `http://localhost:11434/v1`
- 密钥: 留空（Ollama通常不需要）

### 2. LM Studio
1. 启动LM Studio
2. 在服务器设置中启用OpenAI兼容模式
3. 查看本地服务器端口（通常是1234）

配置参数：
- URL: `http://localhost:1234/v1`
- 密钥: 留空或任意字符串

### 3. Azure OpenAI
```bash
# Azure OpenAI端点格式
https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-12-01-preview
```

配置参数：
- URL: `https://your-resource.openai.azure.com/openai/deployments/your-deployment`
- 密钥: Azure API密钥

## 故障排除

### 连接失败
1. **检查URL格式**：确保以 `/v1` 结尾
2. **验证服务运行**：确认API服务正在运行
3. **网络连接**：检查防火墙和网络设置
4. **API密钥**：验证密钥是否正确

### 模型发现失败
1. **检查API兼容性**：确保支持 `/models` 端点
2. **手动添加模型**：在配置中手动指定模型名称
3. **查看API文档**：参考具体服务的API文档

### 响应超时
1. **调整超时设置**：自定义API使用30秒超时
2. **检查服务性能**：本地服务可能需要更长时间
3. **简化请求**：使用较小的模型进行测试

## 安全注意事项

### API密钥安全
- 🔐 API密钥使用简单加密存储
- 📁 配置文件保存在本地 `config/custom_apis.json`
- 🔒 建议定期更新API密钥
- 🚫 不要分享配置文件

### 网络安全
- 🌐 仅在可信网络中使用自定义API
- 🔍 验证API端点的真实性
- 🛡️ 使用HTTPS连接（如果可用）

## 高级功能

### 批量配置
可以添加多个自定义API配置：
- 本地Ollama服务
- 云端OpenAI服务
- 公司内部API服务
- 测试环境API

### 模型选择优先级
1. 预定义模型（官方支持）
2. 自定义API模型（用户配置）
3. 按提供商分组显示

### 配置管理
- ✏️ 编辑现有配置
- 🗑️ 删除不需要的配置
- 🔄 刷新模型列表
- 🧪 重新测试连接

## 技术细节

### API兼容性要求
- 支持 `GET /models` 端点（用于模型发现）
- 支持 `POST /chat/completions` 端点
- 使用标准的OpenAI请求/响应格式

### 配置文件格式
```json
{
  "custom_apis": [
    {
      "provider_name": "我的Ollama",
      "base_url": "http://localhost:11434/v1",
      "api_key": "encrypted_key",
      "models": ["llama2", "codellama"],
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z",
      "last_tested": "2024-01-01T00:00:00Z",
      "test_status": "success",
      "description": "本地Ollama服务"
    }
  ]
}
```

### 模型ID格式
自定义模型的标识格式：`custom:provider_name:model_name`
- 示例：`custom:我的Ollama:llama2`
- 用于区分不同提供商的相同模型名称

---

如有问题，请查看应用日志或联系技术支持。