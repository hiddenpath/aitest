# aitest 构建与运行说明

本文档说明如何使用**本地 mingw 工具链**编译、运行 aitest，并与最新的 **ai-lib-rust** 库对齐。

---

## 一、环境要求

- **Rust** 1.70+
- **SQLite**
- **mingw 工具链**：已安装 `stable-x86_64-pc-windows-gnu`，并确保本机有可用的 gcc（如 MSYS2 的 `mingw-w64-x86_64-gcc`）

检查已安装的 Rust 目标：

```powershell
rustup show
```

若未见 `x86_64-pc-windows-gnu`，可安装：

```powershell
rustup target add x86_64-pc-windows-gnu
```

若希望默认使用 gnu 工具链（推荐在仅 mingw 环境下）：

```powershell
rustup default stable-x86_64-pc-windows-gnu
```

---

## 二、使用 mingw 编译

在仅安装 mingw、未安装 MSVC（或无 `link.exe`）的环境中，**build script 会在宿主上运行**。若默认工具链为 `msvc`，build script 会因找不到 `link.exe` 而失败。

**推荐做法：用 gnu 工具链运行 cargo**

```powershell
cd d:\rustapp\aitest
rustup run stable-x86_64-pc-windows-gnu cargo build
```

这样宿主与目标均使用 gnu，build script 也会使用 gcc 链接，避免 `link.exe` 缺失。

若已执行 `rustup default stable-x86_64-pc-windows-gnu`，则可直接：

```powershell
cargo build
```

**发布构建：**

```powershell
rustup run stable-x86_64-pc-windows-gnu cargo build --release
```

---

## 三、环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（或通过 keyring 配置） | `sk-...` |
| `GROQ_API_KEY` | Groq API 密钥（或通过 keyring 配置） | `gsk_...` |
| `DATABASE_URL` | SQLite 库路径 | `sqlite://chat.db` |
| `MODEL_ID`（或兼容 `DEEPSEEK_MODEL_ID`） | 默认模型 ID，默认 `deepseek/deepseek-chat` | `deepseek/deepseek-chat` |
| `ALLOWED_MODEL_IDS`（或 `MODEL_IDS`） | 允许前端切换的模型列表（逗号分隔） | `deepseek/deepseek-chat,groq/llama-3.1-8b-instant` |
| `MAX_CONTEXT_MESSAGES` | 每次请求携带的历史消息条数上限（默认 6，避免 Groq 等 6000 TPM 限制导致 413） | `6` |
| `PROXY_URL` 或 `AI_PROXY_URL` | 可选代理 | `http://127.0.0.1:7890` |

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:DATABASE_URL = "sqlite://chat.db"
```

---

## 四、运行

```powershell
rustup run stable-x86_64-pc-windows-gnu cargo run
# 若已 default 到 gnu：cargo run
```

浏览器访问：`http://127.0.0.1:3000`。

---

## 五、与 ai-lib-rust 的对接说明

aitest 已按 **ai-lib-rust** 当前 API 对齐，主要包括：

- **客户端**：`AiClientBuilder::new().build(&model_id).await` 或 `AiClient::new(&model_id).await`
- **流式请求**：`client.chat().messages(...).temperature(...).max_tokens(...).stream().execute_stream()`
- **事件类型**：`StreamingEvent::PartialContentDelta`、`Metadata`、`StreamEnd`、`StreamError` 等；`StreamError` 的 `error` 为 `serde_json::Value`，已做 `message` / `error` 字段与字符串的兼容解析
- **消息类型**：`Message::system` / `user` / `assistant`；`MessageContent::Text`、`ContentBlock::Text` 等用于 token 估算与历史构建

---

## 六、常见问题

**Q: 报错 `linker 'link.exe' not found`**  
A: 请使用 `rustup run stable-x86_64-pc-windows-gnu cargo build`，或先将默认工具链设为 gnu。

**Q: 报错 `DEEPSEEK_API_KEY` 或鉴权失败**  
A: 设置对应 provider 的 `*_API_KEY`（如 `DEEPSEEK_API_KEY` / `GROQ_API_KEY`），或按 ai-lib-rust 文档配置 keyring。

**Q: 依赖的 ai-lib-rust 来源**  
A: `Cargo.toml` 中为 `ai-lib-rust = "0.6.0"`，从 **crates.io** 拉取线上版本；无需本地 ai-lib-rust 源码。如需本地调试可临时改为 `{ path = "../ai-lib-rust" }`。

---

## 七、项目结构简要

- `src/main.rs`：Axum 服务、路由、SSE 流式聊天、历史与鉴权逻辑
- `src/app_metrics.rs`：本地 `Metrics` 实现（与 ai-lib-rust 无关，可替换为自有监控）
- `static/`：前端页面与 marked.js 等资源

更多接口与 SSE 事件格式见 `README.md`、`SSE_EVENTS_SCHEMA.md`。
