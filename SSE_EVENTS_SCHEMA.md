## Structured SSE events: schema and guidance

This document specifies the structured Server-Sent Events (SSE) format used by this example. Each SSE event carries exactly one JSON object in a single `data:` payload line (no multi-line JSON fragments), followed by a blank line, per SSE framing.

- Transport: `Content-Type: text/event-stream; charset=utf-8`
- Framing: `data: { ... }\n\n`
- Encoding: UTF-8
- Contract: Unknown fields must be ignored by clients for forward-compatibility

### Event envelope

All events share the envelope:

```json
{
  "type": "<event-kind>",
  "ts": 1730000000000 // optional milliseconds since epoch, if provided
}
```

The `type` determines required/optional fields below.

### Event kinds

1) provider
```json
{ "type": "provider", "provider": "groq|mistral|deepseek|..." }
```

2) delta (streamed content)
```json
{ "type": "delta", "content": "partial text chunk" }
```

3) usage (token accounting)
```json
{ "type": "usage", "tokens": 123, "accurate": false }
```
- accurate=true: provider-finalized total (ACU)
- accurate=false: locally estimated (EST)

4) error
```json
{
  "type": "error",
  "message": "human-readable",
  "code": "E_xxx",              // optional, implementation-defined
  "class": "retryable|non_retryable|provider_switch|chunk_timeout|request_timeout|client"
}
```

5) rate_limited
```json
{ "type": "rate_limited", "retry_after_ms": 10000 }
```

6) done
```json
{ "type": "done" }
```

### Error classification (recommended)

- retryable: transient provider/network error; client may retry (same provider)
- provider_switch: retryable but better to switch provider (server may already have failed over)
- chunk_timeout: per-chunk stall; stream aborted
- request_timeout: overall timeout; no further chunks
- client: client-originated (e.g., cancelled)
- non_retryable: input invalid / policy violation / permanent failure

Clients should treat `class` as a hint; implementations may add new classes.

### Ordering and semantics

- First event should be `provider`.
- Zero or more `delta` events follow.
- Optionally one `usage` near end-of-stream.
- Zero or more `error` events may occur; stream terminates after a terminal error.
- Stream ends with a single `done`.

### Client parsing reference (pseudo-JS)

```js
for (const frame of sseFrames) {
  const evt = JSON.parse(frame.data);
  switch (evt.type) {
    case 'provider': ui.setProvider(evt.provider); break;
    case 'delta':    ui.appendMarkdown(evt.content); break;
    case 'usage':    ui.showTokens(evt.tokens, evt.accurate); break;
    case 'rate_limited': ui.notifyRateLimit(evt.retry_after_ms); break;
    case 'error':    ui.notifyError(evt.message, evt.class); break;
    case 'done':     ui.finalize(); return;
    default:         /* ignore unknown */
  }
}
```

### Security & robustness

- Sanitize rendered HTML when transforming Markdown on the client.
- Disconnect idle SSE when app is backgrounded to save resources.
- Backoff on reconnect if the transport is dropped.

### Versioning

This schema is small by design. If you need to extend it:

- Add new fields/types conservatively.
- Clients must ignore unknown fields to remain compatible.
- If you add breaking changes, gate them behind a query param (`?sse_v=2`) or a different endpoint.


