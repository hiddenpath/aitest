# ai-lib-rust Axum 示例：流式聊天（Phase 2 - 模型切换）

English: [README.md](./README.md)

本仓库是一个精简的示例项目，演示如何将 `ai-lib-rust` 与 Axum Web 服务及单文件前端集成。Phase 2 展示：

- 跨提供商/模型的统一客户端接口（如 DeepSeek + Groq）
- Server-Sent Events (SSE) 流式响应
- 基于 SQLite 的聊天历史
- 基础限流与安全校验
- 通过本地 `Metrics` trait（`src/app_metrics.rs`）实现可插拔指标

项目刻意保持简单，侧重清晰与正确性而非“智能路由”。目标是先保证单提供商下的流式聊天端到端可用，再逐步增加更多提供商。

## 本示例的定位

- **是**：一个小的 Axum 应用，从**选定模型**流式返回聊天补全，并与 `ai-lib-rust` 清晰集成。
- **不是**：完整的“智能路由”、多提供商合并或成本优化器；这些可在本示例之上扩展，此处省略以保持示例聚焦、可审计。

## 环境要求

- Rust 1.70+
- SQLite

## 构建

```bash
cargo build
```

Windows 下仅使用 mingw 工具链时：`rustup run stable-x86_64-pc-windows-gnu cargo build`。详细构建与环境配置见 [构建说明.md](./构建说明.md)。

## 配置环境

为计划使用的提供商设置 API 密钥。

```bash
# Windows PowerShell 示例
$env:DEEPSEEK_API_KEY = "..."

# 可选代理示例
$env:PROXY_URL = "http://127.0.0.1:7890"

# SQLite 路径（可选；默认：sqlite://D:/ai_data/groqchat.db）
$env:DATABASE_URL = "sqlite:///path/to/chat.db"
```

说明：
- 示例优先使用 `PROXY_URL`，若设置了 `AI_PROXY_URL` 也会回退使用。
- 若所选提供商未配置鉴权（环境变量或 keyring），请求会失败。
- 默认模型 ID：`MODEL_ID`（或旧版 `DEEPSEEK_MODEL_ID`）（默认：`deepseek/deepseek-chat`）。
- 可选白名单（逗号分隔）：`ALLOWED_MODEL_IDS`（或 `MODEL_IDS`）。
- 可选上下文上限：`MAX_CONTEXT_MESSAGES`（默认 6）— 每次请求携带的历史消息条数上限；较小值可避免 Groq 免费 tier（6000 TPM）等场景下的 TPM/413 错误。

## 运行

```bash
cargo run
```

访问 `http://127.0.0.1:3000`。

## 接口说明

- `GET /` – 静态单页聊天 UI（英文）
- `GET /health` – 服务健康、运行时间及活跃提供商
- `GET /models` – 允许的模型列表与默认模型 ID（供 UI 使用）
- `GET /history?user_id=...&session_id=...` – 某会话最近 50 条消息
- `POST /history` – 与 GET 相同，但接受 JSON `{ user_id, session_id }`
- `POST /chat/stream` – SSE 流式聊天（JSON 事件）；请求体示例：

```json
{
  "user_id": "user-abc",
  "session_id": "session-xyz",
  "message": "Hello",
  "model_id": "groq/llama-3.1-8b-instant"
}
```

SSE 事件负载（每个事件为通过 `data: { ... }` 发送的单个 JSON 对象）：

```json
{ "type": "provider", "provider": "deepseek", "model_id": "deepseek/deepseek-chat" }
{ "type": "delta", "content": "partial text" }
{ "type": "usage", "tokens": 123, "accurate": false }
{ "type": "error", "message": "..." }
{ "type": "rate_limited" }
{ "type": "done" }
```

完整 schema 与错误类型说明见 `SSE_EVENTS_SCHEMA.md`，可插拔 token 估算示例见 `TOKEN_ESTIMATION_PLUGGABLE.md`。

## 与 ai-lib-rust 的集成方式

- 客户端通过 `AiClientBuilder`（或 `AiClient::new()`）按 `model_id` 创建。
- 所有请求使用统一的运行时聊天构建 API（`client.chat()...stream().execute_stream()`）。
- 模型切换通过前端传入 `model_id` 或使用默认 `MODEL_ID` 实现。
- 指标通过本地 `Metrics` 实现（`src/app_metrics.rs`）记录，可替换为自有后端。

## 前端

UI 为单个文件 `static/index.html`：

- 纯 JavaScript，通过 `fetch` 使用 SSE，使用本地 `marked.js` 渲染 Markdown。
- 代码块由本地 Prism 高亮（若存在），否则使用简易回退。
- 用户可见文案为英文。

## Token 统计说明

流式响应通常不包含提供商用量。因此我们使用本地估算并标记为 `EST`。若提供商提供最终用量，请使用该值并标记为 `ACU`。所有数字仅供参考，不构成保证。

## 项目结构

- `src/main.rs` – Axum 服务、路由、SSE 流式聊天
- `src/app_metrics.rs` – ai-lib 指标 trait 示例（`SimpleMetrics`）
- `static/` – 单页 UI 与本地资源（marked、prism）

## 建议扩展（为保持示例精简未实现）

- 若对外公网暴露，在前端对渲染后的 Markdown 做消毒（如 DOMPurify）。
- 增加优雅关闭、可观测端点与结构化日志。
- 在非流式调用或摘要中提供提供商用量时予以展示。
- 用策略（延迟/成本/SLA）替代轮询，仍基于 ai-lib 统一接口。
- 将 SSE 事件负载规范为 JSON 便于客户端解析。

## 许可证

MIT
