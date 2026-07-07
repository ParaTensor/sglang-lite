//! Internal protocol between Rust control plane and Python execution core.
//!
//! This is the clean boundary. The Rust layer translates the (potentially messy)
//! OpenAI request into this, and the core only ever sees this.
//!
//! Note: Full/complex OpenAI parsing and serving drivers live in Unigateway.
//! This file is minimal adapter only.

use serde::{Deserialize, Serialize};

use crate::openai::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub request_id: String,
    pub model: String,

    /// The chat messages (Rust layer can also pre-tokenize later if desired).
    /// Keeping messages here keeps tokenization on the Python side for now
    /// (transformers is authoritative).
    pub messages: Vec<ChatMessage>,

    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<i32>,
    pub stop: Option<Vec<String>>,

    pub stream: bool,
}

/// A single generated token delta (or final).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDelta {
    pub text: String,
    /// When present, this is the last delta for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Optional usage on the final message (non-streaming or last chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Number of prompt tokens served from cache (for sglang-lite / unigateway passthrough)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_tokens: Option<u32>,
}

/// Blocking result used by non-stream path.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
}
