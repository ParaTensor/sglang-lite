//! sglang-lite-serving
//!
//! **Thin Rust wrapper** for the sglang-lite inference engine.
//!
//! - This binary is the official minimal standalone service.
//! - Broad OpenAI protocol support, multi-backend routing, auth, and policy stay in an optional gateway.
//! - This crate only composes the `sglang-lite-control` library into a runnable standalone server.
//! - See `control/` for the actual control plane logic.

use std::sync::Arc;

use sglang_lite_control::StubEngineClient;
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sglang_lite=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let core_url = std::env::var("SGLANG_LITE_PYTHON_CORE").ok();

    // Supported models (very limited in lite mode — this is intentional)
    let supported_models: Vec<String> = vec![
        "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "Qwen/Qwen2.5-72B-Instruct".to_string(),
        "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
        "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
        "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
        // Add more only after explicit support + test
    ];

    let engine = Arc::new(StubEngineClient::new());

    info!("sglang-lite-serving starting");
    info!("Phase 1 — thin wrapper over sglang-lite-control");
    info!(
        "Try: curl http://localhost:{}/v1/chat/completions -d '{{...}}'",
        port
    );
    info!("Metrics: curl http://localhost:{}/metrics", port);

    sglang_lite_control::serve(engine, Arc::new(supported_models), core_url, port).await
}
