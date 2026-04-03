use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::db;
use crate::db::schema::{KeyRow, ProviderRow, RoutingRuleRow};
use crate::error::{AppError, AppResult};
use crate::providers;
use crate::AppState;

// ── Provider CRUD ──

pub async fn list_providers(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<Vec<ProviderRow>>> {
    let providers = db::list_providers(&state.db).await?;
    Ok(Json(providers))
}

pub async fn get_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> AppResult<Json<ProviderRow>> {
    let provider = db::get_provider(&state.db, &id)
        .await?
        .ok_or_else(|| AppError::NotFound("Provider not found".to_string()))?;
    Ok(Json(provider))
}

#[derive(Debug, Deserialize)]
pub struct CreateProviderRequest {
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    #[serde(default)]
    pub config_json: Option<String>,
}

pub async fn create_provider(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateProviderRequest>,
) -> AppResult<Json<ProviderRow>> {
    let now = chrono::Utc::now().to_rfc3339();
    let row = ProviderRow {
        id: uuid::Uuid::new_v4().to_string(),
        name: req.name,
        provider_type: req.provider_type,
        base_url: req.base_url,
        api_key: req.api_key,
        is_enabled: true,
        config_json: req.config_json.unwrap_or_else(|| "{}".to_string()),
        created_at: now.clone(),
        updated_at: now,
    };
    db::insert_provider(&state.db, &row).await?;

    // Register in-memory provider instance
    if let Some(provider) = providers::create_provider_from_row(&row) {
        state.registry.register(row.id.clone(), provider).await;
        tracing::info!("Provider '{}' registered", row.name);
    }

    Ok(Json(row))
}

#[derive(Debug, Deserialize)]
pub struct UpdateProviderRequest {
    pub name: Option<String>,
    pub provider_type: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub is_enabled: Option<bool>,
    pub config_json: Option<String>,
}

pub async fn update_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateProviderRequest>,
) -> AppResult<Json<ProviderRow>> {
    let mut row = db::get_provider(&state.db, &id)
        .await?
        .ok_or_else(|| AppError::NotFound("Provider not found".to_string()))?;

    if let Some(name) = req.name {
        row.name = name;
    }
    if let Some(pt) = req.provider_type {
        row.provider_type = pt;
    }
    if let Some(url) = req.base_url {
        row.base_url = url;
    }
    if let Some(key) = req.api_key {
        row.api_key = key;
    }
    if let Some(enabled) = req.is_enabled {
        row.is_enabled = enabled;
    }
    if let Some(config) = req.config_json {
        row.config_json = config;
    }

    db::update_provider(&state.db, &row).await?;

    // Update in-memory registry
    if row.is_enabled {
        if let Some(provider) = providers::create_provider_from_row(&row) {
            state.registry.register(row.id.clone(), provider).await;
            tracing::info!("Provider '{}' updated in registry", row.name);
        }
    } else {
        state.registry.remove(&row.id).await;
        tracing::info!("Provider '{}' disabled, removed from registry", row.name);
    }

    Ok(Json(row))
}

pub async fn delete_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    // Remove from registry first
    state.registry.remove(&id).await;
    db::delete_provider(&state.db, &id).await?;
    tracing::info!("Provider '{}' deleted", id);
    Ok(Json(serde_json::json!({"deleted": true})))
}

// ── Provider Test ──

pub async fn test_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    let row = db::get_provider(&state.db, &id)
        .await?
        .ok_or_else(|| AppError::NotFound("Provider not found".to_string()))?;

    let provider = providers::create_provider_from_row(&row)
        .ok_or_else(|| AppError::BadRequest(format!("Unknown provider type: {}", row.provider_type)))?;

    match provider.list_models().await {
        Ok(models) => Ok(Json(serde_json::json!({
            "success": true,
            "message": format!("Connected successfully. Found {} models.", models.len()),
            "models": models,
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "success": false,
            "message": format!("Connection failed: {e}"),
        }))),
    }
}

// ── Key CRUD ──

pub async fn list_keys(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<Vec<KeyRow>>> {
    let keys = db::list_keys(&state.db).await?;
    Ok(Json(keys))
}

fn default_routing_mode() -> String {
    "failover".to_string()
}

fn validate_routing_mode(mode: &str) -> Result<String, AppError> {
    match mode {
        "failover" | "content" => Ok(mode.to_string()),
        _ => Err(AppError::BadRequest(format!(
            "Invalid routing_mode '{}'. Must be 'failover' or 'content'",
            mode
        ))),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    #[serde(default)]
    pub name: String,
    pub default_provider_id: String,
    #[serde(default = "default_routing_mode")]
    pub routing_mode: String,
}

pub async fn create_key(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateKeyRequest>,
) -> AppResult<Json<KeyRow>> {
    let routing_mode = validate_routing_mode(&req.routing_mode)?;

    // Validate default_provider_id exists
    if req.default_provider_id.is_empty() {
        return Err(AppError::BadRequest(
            "default_provider_id is required — every key must have at least one provider".to_string(),
        ));
    }
    db::get_provider(&state.db, &req.default_provider_id)
        .await?
        .ok_or_else(|| {
            AppError::BadRequest(format!(
                "Provider '{}' not found",
                req.default_provider_id
            ))
        })?;

    let now = chrono::Utc::now().to_rfc3339();
    let key_value = format!("lot_{}", generate_key());
    let row = KeyRow {
        id: uuid::Uuid::new_v4().to_string(),
        key_value,
        name: req.name,
        default_provider_id: Some(req.default_provider_id),
        routing_mode,
        is_enabled: true,
        created_at: now.clone(),
        updated_at: now,
    };
    db::insert_key(&state.db, &row).await?;
    Ok(Json(row))
}

pub async fn update_key(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<serde_json::Value>,
) -> AppResult<Json<KeyRow>> {
    let mut row = db::list_keys(&state.db)
        .await?
        .into_iter()
        .find(|k| k.id == id)
        .ok_or_else(|| AppError::NotFound("Key not found".to_string()))?;

    if let Some(name) = req.get("name").and_then(|v| v.as_str()) {
        row.name = name.to_string();
    }
    if let Some(dp) = req.get("default_provider_id") {
        row.default_provider_id = dp.as_str().map(|s| s.to_string());
    }
    if let Some(mode) = req.get("routing_mode").and_then(|v| v.as_str()) {
        row.routing_mode = validate_routing_mode(mode)?;
    }
    if let Some(enabled) = req.get("is_enabled").and_then(|v| v.as_bool()) {
        row.is_enabled = enabled;
    }

    db::update_key(&state.db, &row).await?;
    Ok(Json(row))
}

pub async fn delete_key(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    db::delete_key(&state.db, &id).await?;
    Ok(Json(serde_json::json!({"deleted": true})))
}

// ── Routing Rules ──

pub async fn list_routing_rules(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<Vec<RoutingRuleRow>>> {
    let rules = db::list_routing_rules(&state.db).await?;
    Ok(Json(rules))
}

#[derive(Debug, Deserialize)]
pub struct CreateRoutingRuleRequest {
    pub key_id: String,
    pub model_pattern: String,
    pub target_provider_id: String,
    #[serde(default)]
    pub priority: i32,
}

pub async fn create_routing_rule(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRoutingRuleRequest>,
) -> AppResult<Json<RoutingRuleRow>> {
    let pattern = req.model_pattern.trim().to_string();
    if pattern.is_empty() {
        return Err(AppError::BadRequest("model_pattern cannot be empty".to_string()));
    }
    if req.target_provider_id.is_empty() {
        return Err(AppError::BadRequest("target_provider_id cannot be empty".to_string()));
    }

    let now = chrono::Utc::now().to_rfc3339();
    let row = RoutingRuleRow {
        id: uuid::Uuid::new_v4().to_string(),
        key_id: req.key_id,
        model_pattern: pattern,
        target_provider_id: req.target_provider_id,
        priority: req.priority,
        created_at: now,
    };
    db::insert_routing_rule(&state.db, &row).await?;
    Ok(Json(row))
}

pub async fn delete_routing_rule(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    db::delete_routing_rule(&state.db, &id).await?;
    Ok(Json(serde_json::json!({"deleted": true})))
}

// ── Usage Logs ──

pub async fn list_usage_logs(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult<Json<serde_json::Value>> {
    let page: i32 = params.get("page").and_then(|p| p.parse().ok()).unwrap_or(1).max(1);
    let page_size: i32 = params.get("page_size").and_then(|p| p.parse().ok()).unwrap_or(100).min(100).max(1);
    let total = db::count_usage_logs(&state.db).await?;
    let total_pages = ((total as f64) / (page_size as f64)).ceil() as i64;
    let logs = db::list_usage_logs_paged(&state.db, page, page_size).await?;
    Ok(Json(serde_json::json!({
        "data": logs,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
    })))
}

pub async fn usage_stats(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult<Json<serde_json::Value>> {
    let days: i32 = params.get("days").and_then(|d| d.parse().ok()).unwrap_or(30);

    let daily = db::get_daily_usage(&state.db, days).await?;
    let by_model = db::get_model_usage(&state.db, days).await?;

    // Compute totals
    let total_requests: i32 = daily.iter().map(|d| d.requests).sum();
    let total_tokens: i64 = daily.iter().map(|d| d.total_tokens).sum();
    let total_prompt: i64 = daily.iter().map(|d| d.prompt_tokens).sum();
    let total_completion: i64 = daily.iter().map(|d| d.completion_tokens).sum();

    Ok(Json(serde_json::json!({
        "days": days,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "daily": daily,
        "by_model": by_model,
    })))
}

fn generate_key() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..24).map(|_| rng.gen()).collect();
    hex::encode(bytes)
}
