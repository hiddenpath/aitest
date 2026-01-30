# ai-lib-rust Axum Example: Streaming Chat (Phase 2 - Model Switch)

中文说明见 [README_CN.md](./README_CN.md).

This is a minimal, production-lean example showing how to integrate `ai-lib-rust` with an Axum web server and a simple, single-file frontend. Phase 2 demonstrates:

- A unified client interface across providers/models (e.g., DeepSeek + Groq)
- Server-Sent Events (SSE) streaming
- SQLite-backed chat history
- Basic rate limiting and safety checks
- Pluggable metrics via a local `Metrics` trait (`src/app_metrics.rs`)

It is intentionally simple, focusing on clarity and correctness rather than “smart routing.” The goal is to first ensure basic streaming chat works end-to-end with one provider, then add more providers incrementally.

## What this example is (and isn’t)

- Is: a small Axum app that streams chat completions from a **selected model** with a clean `ai-lib-rust` integration.
- Is not: a full “smart router,” multi-provider merge, or cost optimizer. Those are easy to build on top, but omitted here to keep the example focused and auditable.

For detailed Chinese build instructions (e.g. mingw), see [构建说明.md](./构建说明.md).

## Prerequisites

- Rust 1.70+
- SQLite

## Build

```bash
cargo build
```

On Windows with mingw toolchain only: `rustup run stable-x86_64-pc-windows-gnu cargo build`. For detailed build and environment setup, see [构建说明.md](./构建说明.md).

## Configure environment

Set the API keys for the providers you plan to use.

```bash
# Windows PowerShell (example)
$env:DEEPSEEK_API_KEY = "..."

# Optional proxy (example)
$env:PROXY_URL = "http://127.0.0.1:7890"

# SQLite path (optional; default: sqlite://D:/ai_data/groqchat.db)
$env:DATABASE_URL = "sqlite:///path/to/chat.db"
```

Notes:
- The example prefers `PROXY_URL`, and will also fall back to `AI_PROXY_URL` if set.
- If auth is not configured (env or keyring) for the selected provider, requests will fail.
- Default model id: `MODEL_ID` (or legacy `DEEPSEEK_MODEL_ID`) (default: `deepseek/deepseek-chat`).
- Optional allowlist (comma-separated): `ALLOWED_MODEL_IDS` (or `MODEL_IDS`)
- Optional context cap: `MAX_CONTEXT_MESSAGES` (default: 6) — max history messages sent per request; lower values avoid TPM/413 errors on providers like Groq free tier (6000 TPM)

## Run

```bash
cargo run
```

Visit `http://127.0.0.1:3000`.

## Endpoints

- `GET /` – static single-page chat UI (English)
- `GET /health` – service health, uptime, and active providers
- `GET /models` – allowed model list and default model id (used by the UI)
- `GET /history?user_id=...&session_id=...` – latest 50 messages for a session
- `POST /history` – same as GET but accepts JSON `{ user_id, session_id }`
- `POST /chat/stream` – SSE streaming chat (JSON events); request body:

```json
{
  "user_id": "user-abc",
  "session_id": "session-xyz",
  "message": "Hello",
  "model_id": "groq/llama-3.1-8b-instant"
}
```

SSE event payloads (each event is a single JSON object sent via `data: { ... }`):

```json
{ "type": "provider", "provider": "deepseek", "model_id": "deepseek/deepseek-chat" }
{ "type": "delta", "content": "partial text" }
{ "type": "usage", "tokens": 123, "accurate": false }
{ "type": "error", "message": "..." }
{ "type": "rate_limited" }
{ "type": "done" }
```

See `SSE_EVENTS_SCHEMA.md` for the full schema and error-class guidance, and `TOKEN_ESTIMATION_PLUGGABLE.md` for a pluggable token estimator example.

## How this uses ai-lib-rust

- A client is instantiated via `AiClientBuilder` (or `AiClient::new()`), keyed by `model_id`.
- All requests use the unified runtime chat builder API (`client.chat()...stream().execute_stream()`).
- Model switching is done by passing `model_id` from the frontend (or defaulting to `MODEL_ID`).
- Metrics are recorded through a local `Metrics` implementation (`src/app_metrics.rs`), which you can swap with a real backend.

## Frontend

The UI is a single `static/index.html` file:

- Plain JavaScript, SSE via `fetch`, and Markdown rendering with local `marked.js`.
- Code blocks are highlighted with local Prism if present; otherwise a tiny fallback.
- All user-visible text is English.

## Token accounting disclaimer

Streaming responses rarely include provider usage in-band. We therefore report a local estimate labeled as `EST`. If a provider’s finalized usage is available, use that and label as `ACU`. Treat all numbers as indicative, not guarantees.

## Project structure

- `src/main.rs` – Axum server, routes, SSE streaming
- `src/app_metrics.rs` – `ai-lib` metrics trait demo (`SimpleMetrics`)
- `static/` – single-page UI and local assets (marked, prism)

## Suggested extensions (left out to keep this example small)

- Sanitize rendered Markdown on the frontend (e.g., DOMPurify) if you expose this on the public internet.
- Add graceful shutdown, observability endpoints, and structured logs.
- Surface provider usage when available after non-streaming calls, or from summaries.
- Replace round-robin with a policy (latency/cost/SLA-aware), still using `ai-lib`’s unified interfaces.
- Promote SSE event payloads to JSON for easier parsing on the client.

## License

MIT
