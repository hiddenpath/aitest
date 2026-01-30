use ai_lib_rust::{AiClient, AiClientBuilder, Message, StreamingEvent};
use axum::body::Body;
use axum::http::{HeaderValue, Method};
use axum::{
    extract::{Query, State},
    response::{Html, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, timeout::TimeoutLayer};
mod app_metrics;
use crate::app_metrics::{Metrics, SimpleMetrics};

// App state
struct AppState {
    db: SqlitePool,
    default_model_id: String,
    allowed_models: Vec<ModelInfo>,
    clients: tokio::sync::RwLock<HashMap<String, Arc<AiClient>>>,
    start_time: Instant,
    rate_limits: tokio::sync::RwLock<HashMap<String, Vec<Instant>>>,
    metrics: Arc<SimpleMetrics>,
}

// Request/response types
#[derive(Deserialize)]
struct ChatRequest {
    user_id: String,
    session_id: String,
    message: String,
    /// Optional override; when absent, server uses `default_model_id`.
    model_id: Option<String>,
}

#[derive(Deserialize)]
struct HistoryQuery {
    user_id: String,
    session_id: String,
}

#[derive(Deserialize)]
struct HistoryRequest {
    user_id: String,
    session_id: String,
}

#[derive(Serialize, sqlx::FromRow)]
struct HistoryMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct ModelInfo {
    id: String,
    provider: String,
    display_name: String,
}

#[derive(Debug, Clone, Serialize)]
struct TokenUsageSummary {
    // Provider-supplied (if available)
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,

    // Accuracy flags
    accurate_total: bool,
    accurate_breakdown: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize database (default path if env not provided)
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite://chat.db".into());
    let db = SqlitePool::connect(&db_url).await?;
    sqlx::query("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)").execute(&db).await?;

    // Phase 2: allow switching provider/model from the frontend.
    //
    // Canonical model id format is `provider/model`. Auth is resolved automatically via:
    // 1) OS keyring entry ("ai-protocol", provider_id)
    // 2) env var: <PROVIDER_ID>_API_KEY (e.g., DEEPSEEK_API_KEY, GROQ_API_KEY)
    let default_model_id = std::env::var("MODEL_ID")
        .or_else(|_| std::env::var("DEEPSEEK_MODEL_ID"))
        .unwrap_or_else(|_| "deepseek/deepseek-chat".to_string());

    let mut allowed_models = allowed_models_from_env()
        .unwrap_or_else(|| vec![
            model_info_from_id("deepseek/deepseek-chat"),
            // Groq current models (llama3-8b-8192 / llama3-70b-8192 are decommissioned)
            model_info_from_id("groq/llama-3.1-8b-instant"),
            model_info_from_id("groq/llama-3.3-70b-versatile"),
        ]);
    // Ensure the default model is selectable (even if the allowlist is provided).
    if !allowed_models.iter().any(|m| m.id == default_model_id) {
        allowed_models.insert(0, model_info_from_id(&default_model_id));
    }

    // Print auth hints for all allowed providers (best-effort).
    for m in &allowed_models {
        let env = provider_api_key_env(&m.id);
        if std::env::var(&env).is_ok() {
            println!("🔑 {} found for {}", env, m.provider);
        } else {
            println!(
                "ℹ️ {} not set for {} (OK if keyring is configured), continuing",
                env, m.provider
            );
        }
    }

    let state = Arc::new(AppState {
        db,
        default_model_id: default_model_id.clone(),
        allowed_models,
        clients: tokio::sync::RwLock::new(HashMap::new()),
        start_time: Instant::now(),
        rate_limits: tokio::sync::RwLock::new(HashMap::new()),
        metrics: SimpleMetrics::new(),
    });

    // Warm up the default model client at startup so we fail fast if misconfigured.
    let default_client = AiClientBuilder::new().build(&default_model_id).await?;
    state
        .clients
        .write()
        .await
        .insert(default_model_id.clone(), Arc::new(default_client));
    println!("✅ default client initialized (model_id={})", default_model_id);

    let app = Router::new()
        .route("/", get(index))
        .route("/test-md", get(test_md))
        .route("/js/marked.min.js", get(serve_marked_js))
        .route("/models", get(get_models))
        .route("/chat/stream", post(chat_stream))
        // Support both GET (query) and POST (JSON) for history for convenience
        .route("/history", get(get_history).post(get_history_post))
        .route("/health", get(health))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(300),
        ))
        .layer(
            CorsLayer::new()
                .allow_origin("*".parse::<HeaderValue>().unwrap())
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([axum::http::header::CONTENT_TYPE]),
        )
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:3000").await?;

    // Resolve local IP (best-effort)
    let local_ip = std::env::var("LOCAL_IP").unwrap_or_else(|_| {
        // Try to get local IP on Windows
        if let Ok(output) = std::process::Command::new("ipconfig")
            .args(&["/all"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("IPv4") && line.contains("192.168") {
                    if let Some(ip) = line.split_whitespace().last() {
                        return ip.to_string();
                    }
                }
            }
        }
        "127.0.0.1".to_string()
    });

    println!("🚀 Server running:");
    println!("   Local:      http://127.0.0.1:3000");
    println!("   LAN (ifc):  http://{}:3000", local_ip);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn test_md() -> Html<&'static str> {
    const PAGE: &str = r#"<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Markdown Rendering Test</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; padding: 24px; }
        .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        pre { background:#f6f8fa; padding:12px; border-radius:8px; overflow:auto; }
        .markdown p { margin: 0.8em 0; }
        .markdown ul, .markdown ol { margin: 0.8em 0; padding-left: 1.5em; }
        .markdown li { margin: 0.3em 0; line-height: 1.5; }
        .markdown table { width: 100%; border-collapse: collapse; margin: 0.8em 0; }
        .markdown th, .markdown td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }
        .markdown thead th { background: #f9fafb; font-weight: 600; }
    </style>
    <script src=\"/js/marked.min.js\"></script>
    <script>
        marked.setOptions({ gfm:true, breaks:true, smartLists:true, headerIds:false, mangle:false });
        function render(md){ return marked.parse(md.replace(/\r\n/g,'\n').replace(/\r/g,'\n')); }
        window.addEventListener('DOMContentLoaded', ()=>{
            const samples = [
                '# Heading\n\nFirst paragraph with single newline\nSecond line (should break).\n\n- List A\n- List B\n  - Sub B1\n  - Sub B2\n\n1) One\n2) Two\n\nThis is the ending paragraph.',
                'Paragraph one\n\nParagraph two (separated by blank line).\n- List item 1\n- List item 2\n\nParagraph three' ,
                'Paragraph a\nParagraph b (single newline should be <br>).\n\n### Subtitle\nContent paragraph\n\n* italic *test*\n** bold **test**'
            ];
            const raw = document.getElementById('raw');
            const html = document.getElementById('html');
            raw.textContent = samples.join('\n\n---\n\n');
            html.innerHTML = render(raw.textContent);
        });
    </script>
    </head>
<body>
    <h1>Markdown Rendering Test</h1>
    <p>This page verifies paragraph breaks, list rendering and numbering (breaks + smartLists enabled).</p>
    <div class=\"row\">
        <div class=\"card\">
            <h3>Raw Markdown</h3>
            <pre id=\"raw\"></pre>
        </div>
        <div class=\"card markdown\">
            <h3>Rendered Result</h3>
            <div id=\"html\"></div>
        </div>
    </div>
    <p><a href=\"/\">Back to Chat</a></p>
</body>
</html>"#;
    Html(PAGE)
}

async fn serve_marked_js() -> Response<String> {
    Response::builder()
        .header("Content-Type", "application/javascript")
        .header("Cache-Control", "public, max-age=86400")
        .body(include_str!("../static/js/marked.min.js").to_string())
        .unwrap()
}

// removed: prism/font static file helpers to keep the example minimal

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let uptime = state.start_time.elapsed().as_secs();
    let mut providers: Vec<String> = state
        .allowed_models
        .iter()
        .map(|m| m.provider.clone())
        .collect();
    providers.sort();
    providers.dedup();
    let provider_count = providers.len();
    Json(json!({
        "status": "ok",
        "uptime_secs": uptime,
        "default_model_id": state.default_model_id.as_str(),
        "active_providers": providers,
        "provider_count": provider_count,
    }))
}

async fn get_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "default_model_id": state.default_model_id.as_str(),
        "models": &state.allowed_models
    }))
}

// GET /history?user_id=...&session_id=...
async fn get_history(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HistoryQuery>,
) -> Json<serde_json::Value> {
    let mut rows = sqlx::query_as::<_, HistoryMessage>(
        "SELECT role, content FROM messages WHERE user_id = ?1 AND session_id = ?2 ORDER BY id DESC LIMIT 50"
    )
    .bind(&q.user_id)
    .bind(&q.session_id)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    rows.reverse(); // ensure chronological order oldest -> newest for the UI

    Json(json!({ "history": rows }))
}

// POST /history with JSON body
async fn get_history_post(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<HistoryRequest>,
) -> Json<serde_json::Value> {
    let mut rows = sqlx::query_as::<_, HistoryMessage>(
        "SELECT role, content FROM messages WHERE user_id = ?1 AND session_id = ?2 ORDER BY id DESC LIMIT 50"
    )
    .bind(&payload.user_id)
    .bind(&payload.session_id)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default();

    rows.reverse();

    Json(json!({ "history": rows }))
}

// Simple rate limit and basic safety guard
async fn is_rate_limited(state: &Arc<AppState>, user_id: &str) -> bool {
    let now = Instant::now();
    let window = Duration::from_secs(10);
    let max_requests = 15;

    let mut guard = state.rate_limits.write().await;
    let entry = guard.entry(user_id.to_string()).or_insert_with(Vec::new);
    entry.retain(|t| now.duration_since(*t) < window);

    if entry.len() >= max_requests {
        return true;
    }
    entry.push(now);
    false
}

// Basic safety check
fn is_safe_message(message: &str) -> bool {
    message.len() <= 5000 && !message.to_lowercase().contains("hack")
}

// Streaming chat (single provider: deepseek)
async fn chat_stream(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatRequest>,
) -> Response {
    use futures::StreamExt;
    let timer = state.metrics.start_timer("chat_stream").await;

    if is_rate_limited(&state, &payload.user_id).await {
        state
            .metrics
            .incr_counter("ai_requests_rate_limited", 1)
            .await;
        return Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream")
            .body(Body::from("data: {\"type\":\"rate_limited\"}\n\n"))
            .unwrap();
    }

    // safety check
    if !is_safe_message(&payload.message) {
        state.metrics.incr_counter("ai_errors_total", 1).await;
        let evt = json!({"type":"error","message":"Request content rejected"}).to_string();
        let resp = Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream")
            .body(Body::from(format!("data: {}\n\n", evt)))
            .unwrap();
        if let Some(t) = timer {
            t.stop();
        }
        return resp;
    }

    // Build message history
    let messages = build_message_sequence(
        &state.db,
        &payload.user_id,
        &payload.session_id,
        &payload.message,
    )
    .await;

    let state_clone = state.clone();
    let user_id = payload.user_id.clone();
    let session_id = payload.session_id.clone();
    let user_message = payload.message.clone();
    let model_id = payload
        .model_id
        .clone()
        .unwrap_or_else(|| state.default_model_id.clone());

    if !is_allowed_model_id(&state, &model_id) {
        state.metrics.incr_counter("ai_errors_total", 1).await;
        let evt = json!({"type":"error","message": format!("Model not allowed: {}", model_id)}).to_string();
        let resp = Response::builder()
            .status(200)
            .header("Content-Type", "text/event-stream")
            .body(Body::from(format!("data: {}\n\n", evt)))
            .unwrap();
        if let Some(t) = timer {
            t.stop();
        }
        return resp;
    }

    let provider_name = model_id.split('/').next().unwrap_or("unknown").to_string();

    let metrics = state.metrics.clone();
    let stream = async_stream::stream! {
        let prov_evt = json!({"type":"provider","provider": provider_name, "model_id": model_id});
        yield format!("data: {}\n\n", prov_evt.to_string());

        // Resolve (or lazily create) a client for the requested model id.
        let client = match get_or_create_client(&state_clone, &model_id).await {
            Ok(c) => c,
            Err(e) => {
                metrics.incr_counter("ai_errors_total", 1).await;
                let evt = json!({"type":"error","message": format!("Client init failed: {}", e)});
                yield format!("data: {}\n\n", evt.to_string());
                yield "data: {\"type\":\"done\"}\n\n".to_string();
                return;
            }
        };

        // Runtime chat: builder-based request (no Provider enum, no SDK request structs).
        // Keep a copy for token estimation after the stream completes.
        let messages_for_usage = messages.clone();
        let build_stream = client
            .chat()
            .messages(messages)
            .temperature(0.7)
            .max_tokens(4096)
            .stream()
            .execute_stream();

        // timeouts: 420s for entire request, 300s between chunks (DeepSeek can be very slow)
        use futures::StreamExt;
        match tokio::time::timeout(Duration::from_secs(420), build_stream).await {
            Ok(Ok(mut s)) => {
                let mut full_response = String::new();
                let mut usage_data: Option<serde_json::Value> = None;
                let mut chunk_timeout = tokio::time::interval(Duration::from_secs(300));
                chunk_timeout.tick().await; // skip the first immediate tick

                loop {
                    tokio::select! {
                        chunk_result = s.next() => {
                            match chunk_result {
                                Some(Ok(event)) => {
                                    match event {
                                        StreamingEvent::PartialContentDelta { content, .. } => {
                                            if !content.is_empty() {
                                                full_response.push_str(&content);
                                                let evt = json!({"type":"delta","content": content});
                                                yield format!("data: {}\n\n", evt.to_string());
                                                chunk_timeout.reset();
                                            }
                                        }
                                        StreamingEvent::Metadata { usage, .. } => {
                                            usage_data = usage;
                                        }
                                        StreamingEvent::StreamEnd { .. } => {
                                            break;
                                        }
                                        StreamingEvent::StreamError { error, .. } => {
                                            metrics.incr_counter("ai_errors_total", 1).await;
                                            let msg: String = error
                                                .get("message")
                                                .or_else(|| error.get("error"))
                                                .and_then(|v| v.as_str())
                                                .or_else(|| error.as_str())
                                                .map(String::from)
                                                .unwrap_or_else(|| error.to_string());
                                            let evt = json!({"type":"error","message": format!("Stream error: {}", msg)});
                                            yield format!("data: {}\n\n", evt.to_string());
                                            break;
                                        }
                                        _ => {} // Ignore other event types (ThinkingDelta, ToolCallStarted, etc.)
                                    }
                                }
                                Some(Err(e)) => {
                                    metrics.incr_counter("ai_errors_total", 1).await;
                                    let evt = json!({"type":"error","message": e.to_string()});
                                    yield format!("data: {}\n\n", evt.to_string());
                                    break;
                                }
                                None => break, // end of stream
                            }
                        }
                        _ = chunk_timeout.tick() => {
                            let evt = json!({"type":"error","message": "Stream chunk timeout"});
                            yield format!("data: {}\n\n", evt.to_string());
                            break;
                        }
                    }
                }

                // Check for empty response and log warning
                if full_response.is_empty() {
                    metrics.incr_counter("ai_errors_total", 1).await;
                    let warning_msg = "Received empty response from provider. Potential causes: \
                                       1) Provider returned an empty completion. \
                                       2) Request was cancelled or handle was dropped prematurely. \
                                       3) Protocol manifest rules failed to extract content from frames.";
                    let evt = json!({"type":"warning","message": warning_msg});
                    yield format!("data: {}\n\n", evt.to_string());
                }

                // Save history (only if response is not empty)
                if !full_response.is_empty() {
                    let _ = save_chat_history(&state_clone.db, &user_id, &session_id, &user_message, &full_response).await;
                }

                // Token accounting: prefer provider usage if provided; else local estimate
                let usage = estimate_tokens(usage_data, &messages_for_usage, &full_response);
                if usage.accurate_total {
                    metrics
                        .incr_counter("ai_tokens_used_accurate", usage.total_tokens as u64)
                        .await;
                } else {
                    metrics
                        .incr_counter("ai_tokens_used_estimated", usage.total_tokens as u64)
                        .await;
                }
                // Backward-compatible: keep `tokens`/`accurate` while adding breakdown.
                let usage_evt = json!({
                    "type":"usage",
                    "tokens": usage.total_tokens,
                    "accurate": usage.accurate_total,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "accurate_total": usage.accurate_total,
                    "accurate_breakdown": usage.accurate_breakdown
                });
                yield format!("data: {}\n\n", usage_evt.to_string());
            }
            Ok(Err(e)) => {
                // Let ai-lib-rust's configured failover handle retryable errors.
                // Record the error and surface a short message to the client.
                metrics.incr_counter("ai_errors_total", 1).await;
                let evt = json!({"type":"error","message": format!("Stream init failed: {}", e)});
                yield format!("data: {}\n\n", evt.to_string());
            }
            Err(_) => {
                metrics.incr_counter("ai_errors_total", 1).await;
                let evt = json!({"type":"error","message": "Request timeout"});
                yield format!("data: {}\n\n", evt.to_string());
            }
        }
        yield "data: {\"type\":\"done\"}\n\n".to_string();
    };

    let body_stream = stream.map(|chunk| Ok::<_, std::io::Error>(chunk));
    Response::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(Body::from_stream(body_stream))
        .unwrap()
}

// Helpers
/// Max number of history messages to include in context (reduces request size for providers with low TPM, e.g. Groq free tier 6000).
fn max_context_messages() -> i32 {
    std::env::var("MAX_CONTEXT_MESSAGES")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|n| n.clamp(2, 50))
        .unwrap_or(6)
}

async fn build_message_sequence(
    db: &SqlitePool,
    user_id: &str,
    session_id: &str,
    new_message: &str,
) -> Vec<Message> {
    let limit = max_context_messages();
    let history = sqlx::query_as::<_, HistoryMessage>(
        "SELECT role, content FROM messages WHERE user_id = ?1 AND session_id = ?2 ORDER BY id DESC LIMIT ?3",
    )
    .bind(user_id)
    .bind(session_id)
    .bind(limit)
    .fetch_all(db)
    .await
    .unwrap_or_default();

    // Use ai-lib's convenient Message constructors (accepts &str or String)
    let mut messages = vec![Message::system(
        "You are a helpful assistant. Answer the user's question accurately and concisely.",
    )];

    for msg in history.into_iter().rev() {
        // Skip empty messages to avoid invalid assistant messages (e.g., Mistral requires content or tool_calls)
        if msg.content.is_empty() {
            continue;
        }
        if msg.role == "user" {
            messages.push(Message::user(msg.content));
        } else {
            messages.push(Message::assistant(msg.content));
        }
    }

    messages.push(Message::user(new_message));

    messages
}

async fn save_chat_history(
    db: &SqlitePool,
    user_id: &str,
    session_id: &str,
    user_message: &str,
    assistant_reply: &str,
) -> anyhow::Result<()> {
    // Save user message
    sqlx::query(
        "INSERT INTO messages (user_id, session_id, role, content) VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(user_id)
    .bind(session_id)
    .bind("user")
    .bind(user_message)
    .execute(db)
    .await?;

    // Save assistant reply
    sqlx::query(
        "INSERT INTO messages (user_id, session_id, role, content) VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(user_id)
    .bind(session_id)
    .bind("assistant")
    .bind(assistant_reply)
    .execute(db)
    .await?;

    Ok(())
}

/// Compute token count and whether it is provider-supplied (accurate) or locally estimated.
/// - If usage JSON is present with total_tokens>0, return that (marked as accurate);
/// - Otherwise, fall back to local estimate.
fn estimate_tokens(
    usage_data: Option<serde_json::Value>,
    messages: &[Message],
    assistant_text: &str,
) -> TokenUsageSummary {
    // 1) Prefer provider-supplied usage fields (prompt/completion/total)
    if let Some(usage) = usage_data {
        let prompt = usage
            .get("prompt_tokens")
            .or_else(|| usage.get("promptTokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let completion = usage
            .get("completion_tokens")
            .or_else(|| usage.get("completionTokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let total = usage
            .get("total_tokens")
            .or_else(|| usage.get("totalTokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // If we have full breakdown, it's accurate.
        if let (Some(p), Some(c), Some(t)) = (prompt, completion, total) {
            if t > 0 {
                return TokenUsageSummary {
                    prompt_tokens: p,
                    completion_tokens: c,
                    total_tokens: t,
                    accurate_total: true,
                    accurate_breakdown: true,
                };
            }
        }

        // If we only have total, keep total as accurate but estimate breakdown.
        if let Some(t) = total.filter(|t| *t > 0) {
            let est_completion = estimate_text_tokens(assistant_text);
            let est_prompt = t.saturating_sub(est_completion);
            return TokenUsageSummary {
                prompt_tokens: est_prompt,
                completion_tokens: est_completion,
                total_tokens: t,
                accurate_total: true,
                accurate_breakdown: false,
            };
        }
    }

    // 2) Local estimation fallback (both prompt + completion)
    let prompt_text = messages_text_for_estimation(messages);
    let prompt_tokens = estimate_text_tokens(&prompt_text);
    let completion_tokens = estimate_text_tokens(assistant_text);
    TokenUsageSummary {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        accurate_total: false,
        accurate_breakdown: false,
    }
}

fn messages_text_for_estimation(messages: &[Message]) -> String {
    // Best-effort: count only textual content.
    // For multimodal blocks (image/audio), we don't try to estimate tokenization cost here.
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!("{:?}: ", m.role));
        match &m.content {
            ai_lib_rust::types::message::MessageContent::Text(t) => {
                out.push_str(t);
            }
            ai_lib_rust::types::message::MessageContent::Blocks(blocks) => {
                for b in blocks {
                    match b {
                        ai_lib_rust::types::message::ContentBlock::Text { text } => {
                            out.push_str(text);
                            out.push('\n');
                        }
                        _ => {
                            // ignore non-text blocks in estimation
                        }
                    }
                }
            }
        }
        out.push('\n');
    }
    out
}

fn estimate_text_tokens(s: &str) -> usize {
    // Simple heuristic:
    // - ASCII: ~4 chars per token
    // - Non-ASCII: ~1 char per token
    let mut tokens = 0usize;
    let mut ascii_run = 0usize;

    for ch in s.chars() {
        if ch.is_ascii() {
            ascii_run += 1;
        } else {
            if ascii_run > 0 {
                tokens += (ascii_run + 3) / 4;
                ascii_run = 0;
            }
            tokens += 1;
        }
    }

    if ascii_run > 0 {
        tokens += (ascii_run + 3) / 4;
    }

    tokens
}

fn provider_api_key_env(model_id: &str) -> String {
    let provider = model_id.split('/').next().unwrap_or(model_id);
    format!("{}_API_KEY", provider.to_uppercase())
}

fn is_allowed_model_id(state: &AppState, model_id: &str) -> bool {
    state.allowed_models.iter().any(|m| m.id == model_id)
}

fn allowed_models_from_env() -> Option<Vec<ModelInfo>> {
    // Comma-separated list of `provider/model` ids, e.g.:
    // ALLOWED_MODEL_IDS="deepseek/deepseek-chat,groq/llama-3.1-8b-instant"
    let raw = std::env::var("ALLOWED_MODEL_IDS")
        .or_else(|_| std::env::var("MODEL_IDS"))
        .ok()?;

    let mut out = Vec::new();
    for part in raw.split(',') {
        let id = part.trim();
        if id.is_empty() {
            continue;
        }
        out.push(model_info_from_id(id));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn model_info_from_id(model_id: &str) -> ModelInfo {
    let provider = model_id
        .split('/')
        .next()
        .unwrap_or(model_id)
        .to_string();

    let display_name = match model_id {
        "deepseek/deepseek-chat" => "DeepSeek Chat".to_string(),
        "groq/llama-3.1-8b-instant" => "Groq • Llama 3.1 8B Instant".to_string(),
        "groq/llama-3.3-70b-versatile" => "Groq • Llama 3.3 70B Versatile".to_string(),
        "groq/llama3-8b-8192" => "Groq • Llama 3 8B (deprecated)".to_string(),
        "groq/llama3-70b-8192" => "Groq • Llama 3 70B (deprecated)".to_string(),
        _ => model_id.to_string(),
    };

    ModelInfo {
        id: model_id.to_string(),
        provider,
        display_name,
    }
}

async fn get_or_create_client(
    state: &Arc<AppState>,
    model_id: &str,
) -> anyhow::Result<Arc<AiClient>> {
    {
        let guard = state.clients.read().await;
        if let Some(c) = guard.get(model_id) {
            return Ok(c.clone());
        }
    }

    // Build without holding any locks.
    let built = AiClientBuilder::new().build(model_id).await?;
    let built = Arc::new(built);

    let mut guard = state.clients.write().await;
    let entry = guard.entry(model_id.to_string()).or_insert_with(|| built.clone());
    Ok(entry.clone())
}

// Simple validation function for fixed code examples
#[allow(dead_code)]
fn validate_fixed_examples() {
    // Test Message construction
    let _user_msg = Message::user("Hello, how are you?");
    let _system_msg = Message::system("You are a helpful assistant.");
    let _assistant_msg = Message::assistant("I'm doing well, thank you!");

    println!("All fixed code examples compile successfully!");
}
