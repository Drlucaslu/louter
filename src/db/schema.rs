use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ProviderRow {
    pub id: String,
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub api_key: String,
    pub is_enabled: bool,
    pub config_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ModelRow {
    pub id: String,
    pub provider_id: String,
    pub model_id: String,
    pub is_enabled: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct KeyRow {
    pub id: String,
    pub key_value: String,
    pub name: String,
    pub default_provider_id: Option<String>,
    pub routing_mode: String, // "rules", "hybrid", or "smart"
    pub is_enabled: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RoutingRuleRow {
    pub id: String,
    pub key_id: String,
    pub model_pattern: String,
    pub target_provider_id: String,
    pub priority: i32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct UsageLogRow {
    pub id: String,
    pub key_id: Option<String>,
    pub provider_id: Option<String>,
    pub request_model: String,
    pub model_id: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    pub latency_ms: i32,
    pub status_code: i32,
    pub created_at: String,
}

// ── Distillation: Training Samples ──

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TrainingSampleRow {
    pub id: String,
    pub request_messages: String,
    pub request_tools: Option<String>,
    pub response_content: String,
    pub request_model: String,
    pub actual_model: String,
    pub provider_type: String,
    pub task_type: String,
    pub has_tool_calls: bool,
    pub is_successful: bool,
    pub source: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    pub latency_ms: i32,
    pub is_exported: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RoutingHistoryRow {
    pub id: String,
    pub task_type: String,
    pub routed_to: String,
    pub was_successful: bool,
    pub was_fallback: bool,
    pub latency_ms: i32,
    pub created_at: String,
}

// ── RL: Episodes ──

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct RlEpisodeRow {
    pub id: String,
    pub sample_id: String,
    pub prompt_messages: String,
    pub completion: String,
    pub source: String,
    pub reward: Option<f64>,
    pub reward_source: Option<String>,
    pub reward_details: Option<String>,
    pub is_used_for_training: bool,
    pub created_at: String,
}
