## Pluggable token estimation: interface and example

Streaming responses often lack finalized usage. This note shows how to plug in a custom estimator while preserving ai-lib semantics (ACU vs EST).

### Goals

- Prefer provider-supplied finalized usage when available (`accurate=true`).
- Fall back to a local estimator when usage is missing/unsupported (`accurate=false`).
- Allow swapping estimators (e.g., model-specific heuristics) without changing handlers.

### Minimal trait (example)

```rust
/// Contract for token estimation given raw text.
pub trait TokenEstimator: Send + Sync + 'static {
    fn estimate_total_tokens(&self, text: &str) -> usize;
}

/// Default estimator: ASCII ~4 chars/token, non-ASCII ~1 char/token.
pub struct DefaultEstimator;

impl TokenEstimator for DefaultEstimator {
    fn estimate_total_tokens(&self, s: &str) -> usize {
        let mut tokens = 0usize;
        let mut ascii_run = 0usize;
        for ch in s.chars() {
            if ch.is_ascii() {
                ascii_run += 1;
            } else {
                if ascii_run > 0 { tokens += (ascii_run + 3) / 4; ascii_run = 0; }
                tokens += 1;
            }
        }
        if ascii_run > 0 { tokens += (ascii_run + 3) / 4; }
        tokens
    }
}
```

### Integrating with ai-lib usage

```rust
use ai_lib::types::{Usage, UsageStatus};

pub fn tokens_from_usage_or_estimate(
    usage_and_status: Option<(Usage, UsageStatus)>,
    text: &str,
    estimator: &dyn TokenEstimator,
) -> (usize, bool) {
    if let Some((usage, status))) = usage_and_status {
        match status {
            UsageStatus::Finalized if usage.total_tokens > 0 => {
                return (usage.total_tokens as usize, true);
            }
            UsageStatus::Estimated if usage.total_tokens > 0 => {
                return (usage.total_tokens as usize, false);
            }
            UsageStatus::Pending | UsageStatus::Unsupported => { /* fall through */ }
        }
    }
    (estimator.estimate_total_tokens(text), false)
}
```

### Wiring into the handler (excerpt)

```rust
let (tokens, accurate) = tokens_from_usage_or_estimate(
    None,             // streaming often has no usage; pass Some(...) if you have it
    &full_response,
    &DefaultEstimator,
);
```

### Notes

- If you maintain per-model estimators, wrap them in a registry:

```rust
struct ModelAwareEstimator { default: DefaultEstimator }
impl TokenEstimator for ModelAwareEstimator {
    fn estimate_total_tokens(&self, text: &str) -> usize { self.default.estimate_total_tokens(text) }
}
```

- Mark UI with `ACU` (accurate) vs `EST` (estimated) to set expectations.
- For non-streaming endpoints that return usage, prefer provider totals and show both input/output breakdown where available.


