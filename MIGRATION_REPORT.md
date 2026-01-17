# aitest 项目迁移到 ai-lib-rust 报告

## 迁移概述

本次迁移将 `aitest` 项目从使用 `ai-lib` (0.4.0) 迁移到使用本地 `ai-lib-rust` (0.2.0) 库。

## 主要变更

### 1. 依赖变更

**Cargo.toml:**
- 移除: `ai-lib = { version = "0.4.0", features = ["transport", "streaming"] }`
- 添加: `ai-lib-rust = { path = "../ai-lib-rust" }`

### 2. 导入语句变更

**之前:**
```rust
use ai_lib::metrics::Metrics;
use ai_lib::{AiClient, AiClientBuilder, ChatCompletionRequest, Message, Provider};
use ai_lib::types::{Usage, UsageStatus};
```

**之后:**
```rust
use ai_lib_rust::prelude::*;
use crate::app_metrics::{Metrics, SimpleMetrics};
```

### 3. 客户端构建方式变更

**之前 (ai-lib):**
```rust
let mut builder = AiClientBuilder::new(provider)
    .with_timeout(Duration::from_secs(180));
if let Ok(proxy) = std::env::var("PROXY_URL") {
    builder = builder.with_proxy(Some(&proxy));
}
let client = builder.with_failover_chain(failover_chain)?.build()?;
```

**之后 (ai-lib-rust):**
```rust
let builder = AiClientBuilder::new()
    .with_fallbacks(failover_models[1..].to_vec());
let client = builder.build(&failover_models[0]).await?;
```

**关键差异:**
- `ai-lib-rust` 使用模型字符串（如 `"groq/llama-3.3-70b-versatile"`）而不是 Provider 枚举
- 构建是异步的（`.await`）
- 超时和代理通过环境变量配置，而不是 builder 方法
- 使用 `with_fallbacks()` 而不是 `with_failover_chain()`

### 4. Provider 枚举差异

**ai-lib-rust 的 Provider 枚举:**
- 没有 `Mistral` 变体，需要使用 `Provider::Custom("mistral".to_string())`
- 支持的变体: `OpenAI`, `Anthropic`, `Gemini`, `Groq`, `DeepSeek`, `Custom(String)`

### 5. 流式聊天 API 变更

**之前 (ai-lib):**
```rust
let req = ChatCompletionRequest::new(model.clone(), messages)
    .with_temperature(0.7)
    .with_max_tokens(4096);
let mut s = client.chat_completion_stream(req.clone()).await?;
// 处理 choices[0].delta.content
```

**之后 (ai-lib-rust):**
```rust
let req = ChatCompletionRequest::new(messages)
    .temperature(0.7)
    .max_tokens(4096)
    .stream();
let mut s = client.chat_completion_stream(req.clone()).await?;
// 处理 StreamingEvent::PartialContentDelta { content, .. }
```

**关键差异:**
- `ChatCompletionRequest::new()` 不再需要 model 参数（model 在构建 client 时指定）
- 使用 `StreamingEvent` 枚举而不是直接访问 `choices`
- 需要导入并使用 `ChatFacade` trait（通过 `prelude` 自动导入）

### 6. Message 构造方式

**保持不变:**
```rust
Message::system("...")
Message::user("...")
Message::assistant("...")
```

### 7. Usage 和 Token 统计变更

**之前:**
```rust
use ai_lib::types::{Usage, UsageStatus};
fn estimate_tokens(usage_and_status: Option<(Usage, UsageStatus)>, s: &str) -> (usize, bool)
```

**之后:**
```rust
fn estimate_tokens(usage_data: Option<serde_json::Value>, s: &str) -> (usize, bool)
// 从 StreamingEvent::Metadata { usage, .. } 获取
```

**关键差异:**
- `ai-lib-rust` 使用 `serde_json::Value` 而不是结构化的 `Usage` 类型
- 需要手动解析 JSON 来提取 `total_tokens`

### 8. Metrics Trait

**变更:**
- `ai-lib-rust` 没有内置的 `Metrics` trait
- 在 `app_metrics.rs` 中定义了本地的 `Metrics` trait 和 `SimpleMetrics` 实现
- 保持了与原有代码的兼容性

## 功能对比

| 功能 | ai-lib (0.4.0) | ai-lib-rust (0.2.0) | 状态 |
|------|----------------|---------------------|------|
| 流式聊天 | ✅ | ✅ | ✅ 已迁移 |
| Failover/Fallback | ✅ | ✅ | ✅ 已迁移 |
| 超时配置 | ✅ (builder) | ✅ (env var) | ✅ 已迁移 |
| 代理配置 | ✅ (builder) | ✅ (env var) | ✅ 已迁移 |
| Metrics | ✅ (内置) | ❌ | ⚠️ 本地实现 |
| Usage 统计 | ✅ (结构化) | ⚠️ (JSON) | ⚠️ 需要解析 |
| Provider 枚举 | ✅ (完整) | ⚠️ (部分) | ⚠️ 使用 Custom |

## 需要 ai-lib-rust 调整的建议

### 1. Provider 枚举扩展 ⭐ 建议

**问题:** `ai-lib-rust` 的 `Provider` 枚举缺少 `Mistral` 变体，需要使用 `Custom("mistral")`。

**建议:** 在 `ai-lib-rust/src/facade/provider.rs` 中添加 `Mistral` 变体：
```rust
pub enum Provider {
    OpenAI,
    Anthropic,
    Gemini,
    Groq,
    DeepSeek,
    Mistral,  // 添加这一行
    Custom(String),
}
```

**原因:** 
- 提高 API 一致性
- 减少用户需要记住的 provider ID 字符串
- 更好的类型安全

### 2. Usage 类型结构化 ⭐⭐ 建议

**问题:** `ai-lib-rust` 使用 `serde_json::Value` 表示 usage，需要手动解析。

**建议:** 定义结构化的 `Usage` 类型：
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}
```

**原因:**
- 更好的类型安全
- 减少运行时解析错误
- 与 ai-lib 保持 API 兼容性

### 3. Metrics Trait 支持 ⭐ 可选

**问题:** `ai-lib-rust` 没有内置的 Metrics trait。

**建议:** 可以考虑添加一个可选的 metrics 接口，但这不是必需的，因为：
- 应用层可以自己实现
- 保持库的轻量级特性

### 4. Builder 超时方法 ⭐ 可选

**问题:** `ai-lib-rust` 的 builder 没有 `with_timeout()` 方法，需要通过环境变量配置。

**建议:** 可以考虑添加 builder 方法：
```rust
pub fn with_timeout(mut self, timeout: Duration) -> Self {
    // 设置超时配置
}
```

**原因:**
- 更灵活的配置方式
- 与 ai-lib API 更接近

## 测试结果

✅ **编译成功:** `cargo check` 通过
✅ **Release 构建成功:** `cargo build --release` 通过
⚠️ **运行时测试:** 需要实际运行验证（需要 API keys）

## 迁移完成度

- ✅ 依赖更新
- ✅ 导入语句更新
- ✅ 客户端构建代码更新
- ✅ 流式聊天代码更新
- ✅ Message 构造保持不变
- ✅ Usage 统计代码适配
- ✅ Metrics trait 本地实现
- ⚠️ Provider 枚举使用 Custom（建议添加 Mistral）

## 下一步

1. **运行时测试:** 配置 API keys 并测试实际功能
2. **性能对比:** 对比迁移前后的性能表现
3. **错误处理验证:** 测试各种错误场景的处理
4. **考虑 ai-lib-rust 调整:** 根据上述建议评估是否需要调整库

## 总结

迁移基本完成，代码可以编译通过。主要差异在于：
1. API 设计理念不同（ai-lib-rust 更偏向协议驱动）
2. 一些类型需要手动解析（如 Usage）
3. Provider 枚举不完整（缺少 Mistral）

建议优先考虑添加 `Mistral` Provider 变体和结构化的 `Usage` 类型，这将显著改善开发体验。
