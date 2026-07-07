//! sglang-lite thin control plane library (OpenAI API layer + Python execution core client).
//!
//! This crate provides the minimal control point for the sglang-lite engine:
//! - Strict OpenAI-compatible surface (minimal)
//! - Early rejection of out-of-scope features
//! - Clean internal request protocol to the engine core
//!
//! NOTE: Full OpenAI surface, drivers, and serving logic MUST be implemented in
//! Unigateway (external Rust). This crate is only a thin adapter.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::Stream;
use std::convert::Infallible;
use tokio::time::sleep;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;
use uuid::Uuid;

pub mod openai;
pub mod protocol;
pub mod stub_engine;

pub use openai::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Delta, Role};
pub use protocol::GenerationRequest;
pub use stub_engine::StubEngineClient;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<StubEngineClient>,
    pub model_list: Arc<Vec<String>>,
    pub core_url: Option<String>,
}

/// Build the axum router for the sglang-lite control plane.
pub fn build_router(
    engine: Arc<StubEngineClient>,
    model_list: Arc<Vec<String>>,
    core_url: Option<String>,
) -> Router {
    let state = AppState {
        engine,
        model_list,
        core_url,
    };

    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/healthz", get(healthz))
        .route("/v1/health", get(healthz))
        .route("/metrics", get(metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
}

/// Run a standalone HTTP server with the given configuration.
///
/// This is provided as a convenience for local testing and standalone deployments.
/// Production serving should be handled by Unigateway.
pub async fn serve(
    engine: Arc<StubEngineClient>,
    model_list: Arc<Vec<String>>,
    core_url: Option<String>,
    port: u16,
) -> Result<()> {
    let app = build_router(engine, model_list, core_url);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("sglang-lite control plane listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// POST /v1/chat/completions — the primary control surface.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, String)> {
    // === CONTROL POINT: early validation + scope enforcement ===
    if !state.model_list.iter().any(|m| m == &req.model) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            json_error(
                "invalid_request_error",
                &format!(
                    "model '{}' not supported in sglang-lite (see GET /v1/models).",
                    req.model
                ),
                "model_not_found",
            ),
        ));
    }

    if req.messages.iter().any(|m| {
        matches!(m, ChatMessage::User { content: c, .. } if c.contains("data:image") || c.contains("<image>"))
    }) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            json_error("invalid_request_error", "Multimodal content is not supported in sglang-lite core.", "multimodal_not_supported"),
        ));
    }

    if req.response_format.is_some() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            json_error(
                "invalid_request_error",
                "response_format / structured output is not supported inside the engine.",
                "structured_output_not_supported",
            ),
        ));
    }

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    let gen_req = GenerationRequest {
        request_id: request_id.clone(),
        model: req.model.clone(),
        messages: req.messages.clone(),
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.7),
        top_p: req.top_p.unwrap_or(0.95),
        top_k: req.top_k,
        stop: req.stop.clone(),
        stream: req.stream.unwrap_or(false),
    };

    if gen_req.stream {
        // Streaming path — SSE
        let stream = stream_chat(state.engine, gen_req, created, request_id);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming — collect then return
        let resp = state.engine.generate_blocking(gen_req).await.map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("engine error: {}", e),
            )
        })?;

        let choice = openai::ChatChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(resp.text),
                tool_calls: None,
            },
            finish_reason: Some(resp.finish_reason),
        };

        // Convert protocol Usage -> openai Usage for the response surface
        let openai_usage = openai::Usage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
            cache_hit_tokens: resp.usage.cache_hit_tokens,
        };
        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
            choices: vec![choice],
            usage: Some(openai_usage),
        };
        Ok(Json(response).into_response())
    }
}

/// SSE stream helper
pub fn stream_chat(
    engine: Arc<StubEngineClient>,
    gen_req: GenerationRequest,
    created: i64,
    request_id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let model = gen_req.model.clone();
        let mut token_stream = engine.generate_stream(gen_req).await;

        // First chunk usually contains role
        let first = Event::default()
            .json_data(openai::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![openai::ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some(Role::Assistant),
                        content: None,
                    },
                    finish_reason: None,
                }],
            })
            .unwrap_or_else(|_| Event::default().data("data: [ERROR]"));
        yield Ok(first);

        while let Some(delta) = token_stream.recv().await {
            let finish = delta.finish_reason.clone();
            let chunk = openai::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![openai::ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(delta.text),
                    },
                    finish_reason: finish.clone(),
                }],
            };

            match Event::default().json_data(chunk) {
                Ok(ev) => yield Ok(ev),
                Err(_) => break,
            }

            if finish.is_some() {
                break;
            }

            // tiny delay to make streaming visible in demos
            sleep(Duration::from_millis(12)).await;
        }

        // OpenAI style final empty chunk (some clients expect it)
        let done = Event::default()
            .json_data(serde_json::json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }))
            .unwrap_or_else(|_| Event::default().data("data: [DONE]"));
        yield Ok(done);

        // The official terminator
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<openai::ModelsResponse> {
    let data = state
        .model_list
        .iter()
        .map(|id| openai::ModelObject {
            id: id.clone(),
            object: "model".to_string(),
            owned_by: "sglang-lite".to_string(),
        })
        .collect();

    Json(openai::ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

/// GET /healthz
pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "sglang-lite",
        "phase": "1",
        "note": "Production shell in progress"
    }))
}

/// GET /metrics — Phase 1 observability
pub async fn metrics(
    State(state): State<AppState>,
) -> Result<String, (axum::http::StatusCode, String)> {
    if let Some(base) = state.core_url.as_ref() {
        let client = reqwest::Client::new();
        let url = format!("{}/metrics", base.trim_end_matches('/'));
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(body) = resp.text().await {
                    return Ok(body);
                }
            }
        }
    }

    // Fallback basic metrics
    let mut output = String::from("# sglang-lite metrics (Phase 1)\n");
    output.push_str("sglang_lite_phase 1\n");
    output.push_str("sglang_lite_up 1\n");
    Ok(output)
}

fn json_error(typ: &str, message: &str, code: &str) -> String {
    serde_json::json!({
        "error": {
            "message": message,
            "type": typ,
            "param": null,
            "code": code
        }
    })
    .to_string()
}
