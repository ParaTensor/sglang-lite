//! Execution client (Phase 0).
//!
//! Default: pure in-process stub (great for fast iteration on the Rust control plane).
//! When SGLANG_LITE_PYTHON_CORE=http://... is set, it will forward to the
//! Python execution core over HTTP. This exercises the real control boundary.

use anyhow::Result;
use tokio::sync::mpsc;

use crate::protocol::{GenerationRequest, GenerationResult, TokenDelta, Usage};

use once_cell::sync::Lazy;

/// If set, forward GenerationRequest to this Python core HTTP endpoint.
static PYTHON_CORE_URL: Lazy<Option<String>> = Lazy::new(|| {
    std::env::var("SGLANG_LITE_PYTHON_CORE").ok()
});

/// Very simple stub that produces deterministic but realistic-looking output.
/// It simulates prefill + decode time and can pretend to do prefix cache hits.
pub struct StubEngineClient {
    // In real version this will hold connection to python core or in-process handle.
}

impl StubEngineClient {
    pub fn new() -> Self {
        Self {}
    }

    /// Non-streaming path.
    pub async fn generate_blocking(&self, req: GenerationRequest) -> Result<GenerationResult> {
        if let Some(base) = PYTHON_CORE_URL.as_ref() {
            return forward_blocking(base, req).await;
        }

        // Pure stub path
        tokio::time::sleep(std::time::Duration::from_millis(35)).await;

        let prompt_tokens = estimate_prompt_tokens(&req);
        let mut completion = simulate_completion(&req, 0);

        let max_t = req.max_tokens as usize;
        if completion.len() > max_t {
            completion.truncate(max_t);
        }

        let completion_tokens = completion.len() as u32;
        let text = completion.join(" ");

        Ok(GenerationResult {
            text,
            finish_reason: "stop".to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        })
    }

    /// Streaming path. Returns an mpsc receiver the handler drains as SSE.
    pub async fn generate_stream(&self, req: GenerationRequest) -> mpsc::Receiver<TokenDelta> {
        let (tx, rx) = mpsc::channel(128);

        if let Some(base) = PYTHON_CORE_URL.as_ref() {
            // Forward streaming to Python core (simplified: we call non-stream and chunk it)
            let base = base.clone();
            tokio::spawn(async move {
                match forward_blocking(&base, req).await {
                    Ok(res) => {
                        let words: Vec<&str> = res.text.split_whitespace().collect();
                        let n = words.len();
                        for (i, w) in words.into_iter().enumerate() {
                            let is_last = i + 1 == n;
                            let _ = tx
                                .send(TokenDelta {
                                    text: format!("{} ", w),
                                    finish_reason: if is_last { Some("stop".to_string()) } else { None },
                                    usage: if is_last { Some(res.usage.clone()) } else { None },
                                })
                                .await;
                            tokio::time::sleep(std::time::Duration::from_millis(6)).await;
                        }
                    }
                    Err(e) => {
                        let _ = tx
                            .send(TokenDelta {
                                text: format!("[engine error: {}]", e),
                                finish_reason: Some("error".to_string()),
                                usage: None,
                            })
                            .await;
                    }
                }
            });
            return rx;
        }

        // Pure stub path
        tokio::spawn(async move {
            let prompt_tokens = estimate_prompt_tokens(&req);
            let mut tokens = simulate_completion(&req, 42);
            if tokens.len() > req.max_tokens as usize {
                tokens.truncate(req.max_tokens as usize);
            }
            let n = tokens.len();

            tokio::time::sleep(std::time::Duration::from_millis(18)).await;

            let mut completion_tokens = 0u32;

            for (i, tok) in tokens.into_iter().enumerate() {
                completion_tokens += 1;

                let is_last = i + 1 == n;

                let delta = TokenDelta {
                    text: format!("{} ", tok),
                    finish_reason: if is_last { Some("stop".to_string()) } else { None },
                    usage: if is_last {
                        Some(Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                        })
                    } else {
                        None
                    },
                };

                if tx.send(delta).await.is_err() {
                    break;
                }

                let delay = if i < 4 { 8 } else { 4 };
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            }
        });

        rx
    }
}

fn estimate_prompt_tokens(req: &GenerationRequest) -> u32 {
    // Very rough: count chars / 3. Good enough for stub and demo.
    let mut n = 0u32;
    for m in &req.messages {
        match m {
            crate::openai::ChatMessage::System { content, .. } => n += (content.len() as u32) / 3 + 4,
            crate::openai::ChatMessage::User { content, .. } => n += (content.len() as u32) / 3 + 4,
            crate::openai::ChatMessage::Assistant { content, .. } => {
                if let Some(c) = content {
                    n += (c.len() as u32) / 3 + 4;
                }
            }
            crate::openai::ChatMessage::Tool { content, .. } => n += (content.len() as u32) / 3 + 4,
        }
    }
    n.max(12)
}

/// Produce a fake but plausible token sequence.
/// The stub tries to look like it understood the last user message.
fn simulate_completion(req: &GenerationRequest, seed: u64) -> Vec<String> {
    let last_user = req
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            crate::openai::ChatMessage::User { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .unwrap_or("Hello");

    let lower = last_user.to_lowercase();

    if lower.contains("hello") || lower.contains("hi ") || lower == "hi" {
        return "Hello! How can I help you today? I'm running in sglang-lite Phase 0 stub mode.".split_whitespace().map(|s| s.to_string()).collect();
    }
    if lower.contains("who are you") || lower.contains("what are you") {
        return "I am sglang-lite, a minimal high-coherence inference engine. Radix KV cache and continuous batching are the focus.".split_whitespace().map(|s| s.to_string()).collect();
    }
    if lower.contains("rust") {
        return "Rust is used for the control plane (OpenAI surface, validation, streaming). The execution core (KV + scheduler) lives in Python with Triton for now.".split_whitespace().map(|s| s.to_string()).collect();
    }

    // Generic continuation
    let base = "This is a simulated response from sglang-lite. In a real run the tokens would come from the Radix cache + scheduler + model runner with CUDA graph decode.";
    let words: Vec<&str> = base.split_whitespace().collect();

    // Vary a little based on "seed"
    let mut out: Vec<String> = words.iter().map(|w| w.to_string()).collect();
    if seed % 3 == 0 {
        out.push("The prefix cache would have helped on repeated system prompts.".into());
    }
    out
}

/// Forward a request to the Python core HTTP server (loose coupling for Phase 0).
async fn forward_blocking(base: &str, req: GenerationRequest) -> Result<GenerationResult> {
    let client = reqwest::Client::new();
    let url = format!("{}/generate", base.trim_end_matches('/'));

    // The internal Python core currently accepts a simplified shape.
    // We map our clean GenerationRequest into what it expects.
    let body = serde_json::json!({
        "request_id": req.request_id,
        "model": req.model,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stop": req.stop,
        "stream": false,
    });

    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("python core error: {}", txt);
    }

    let v: serde_json::Value = resp.json().await?;
    let text = v["text"].as_str().unwrap_or("").to_string();
    let fr = v["finish_reason"].as_str().unwrap_or("stop").to_string();
    let usage = if let Some(u) = v.get("usage") {
        Usage {
            prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
            total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
        }
    } else {
        Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    Ok(GenerationResult { text, finish_reason: fr, usage })
}
