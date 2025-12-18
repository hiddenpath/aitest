# ai-lib Axum Example: Round-robin Chat across 3 Providers

This is a minimal, production-lean example showing how to integrate `ai-lib` with an Axum web server and a simple, single-file frontend. It demonstrates:

- A unified client interface to multiple providers (Groq, Mistral, DeepSeek)
- Round-robin routing per request (one provider per request, rotating)
- Server-Sent Events (SSE) streaming
- SQLite-backed chat history
- Basic rate limiting and safety checks
- Pluggable metrics via `ai-lib::metrics::Metrics` trait

It is intentionally simple, focusing on clarity and correctness rather than “smart routing.” Use it to verify that ai-lib is not “Groq-only” and that Axum integration is straightforward.

## What this example is (and isn’t)

- Is: a small Axum app that streams chat completions from three providers in a round-robin fashion, with a clean `ai-lib` integration.
- Is not: a full “smart router,” multi-provider merge, or cost optimizer. Those are easy to build on top, but omitted here to keep the example focused and auditable.

## Prerequisites

- Rust 1.70+
- SQLite

## Configure environment

Set at least one provider API key; any subset is fine. The app only initializes clients for providers whose keys are present.

```bash
# Windows PowerShell (example)
$env:GROQ_API_KEY = "..."
$env:MISTRAL_API_KEY = "..."
$env:DEEPSEEK_API_KEY = "..."

# Optional proxy (example)
$env:PROXY_URL = "http://127.0.0.1:7890"

# SQLite path (optional; default: sqlite://D:/ai_data/groqchat.db)
$env:DATABASE_URL = "sqlite:///path/to/chat.db"
```

Notes:
- The example prefers `PROXY_URL`, and will also fall back to `AI_PROXY_URL` if set.
- If no API keys are present, the server refuses to start.

## Run

```bash
cargo run
```

Visit `http://127.0.0.1:3000`.

## Endpoints

- `GET /` – static single-page chat UI (English)
- `GET /health` – service health, uptime, and active providers
- `GET /history?user_id=...&session_id=...` – latest 50 messages for a session
- `POST /history` – same as GET but accepts JSON `{ user_id, session_id }`
- `POST /rr-chat/stream` – SSE streaming chat (JSON events); request body:

```json
{
  "user_id": "user-abc",
  "session_id": "session-xyz",
  "message": "Hello"
}
```

SSE event payloads (each event is a single JSON object sent via `data: { ... }`):

```json
{ "type": "provider", "provider": "groq|mistral|deepseek" }
{ "type": "delta", "content": "partial text" }
{ "type": "usage", "tokens": 123, "accurate": false }
{ "type": "error", "message": "..." }
{ "type": "rate_limited" }
{ "type": "done" }
```

See `SSE_EVENTS_SCHEMA.md` for the full schema and error-class guidance, and `TOKEN_ESTIMATION_PLUGGABLE.md` for a pluggable token estimator example.

## How this uses ai-lib

- Each configured provider is instantiated via `AiClientBuilder`.
- All requests use the unified `ChatCompletionRequest` model and `chat_completion_stream` API.
- Failover (ai-lib 0.3.4): per-request cross-provider failover is configured. For example, if Groq is selected, the client will try Mistral then DeepSeek on retryable errors, transparently to the handler.
- Metrics are recorded through a `Metrics` implementation (`src/app_metrics.rs`), which you can swap with a real backend.

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
- `tests/` – minimal skeleton to extend later

## Suggested extensions (left out to keep this example small)

- Sanitize rendered Markdown on the frontend (e.g., DOMPurify) if you expose this on the public internet.
- Add graceful shutdown, observability endpoints, and structured logs.
- Surface provider usage when available after non-streaming calls, or from summaries.
- Replace round-robin with a policy (latency/cost/SLA-aware), still using `ai-lib`’s unified interfaces.
- Promote SSE event payloads to JSON for easier parsing on the client.

## License

MIT
