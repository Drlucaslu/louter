use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::db;
use crate::error::AppResult;
use crate::AppState;

/// GET /api/admin/distill/stats — Training sample statistics
pub async fn distill_stats(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<serde_json::Value>> {
    let total = db::count_training_samples(&state.db).await?;
    let unexported = db::count_unexported_samples(&state.db).await?;
    let by_type = db::get_training_sample_stats(&state.db).await?;

    Ok(Json(serde_json::json!({
        "total_samples": total,
        "unexported_samples": unexported,
        "exported_samples": total - unexported,
        "by_task_type": by_type,
    })))
}

/// GET /api/admin/distill/routing — Routing statistics (local vs cloud)
pub async fn routing_stats(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> AppResult<Json<serde_json::Value>> {
    let days: i32 = params
        .get("days")
        .and_then(|d| d.parse().ok())
        .unwrap_or(7);

    let stats = db::get_routing_stats(&state.db, days).await?;
    let success_rates = db::get_local_success_rates(&state.db, days).await?;

    let total_requests: i64 = stats.iter().map(|s| s.total).sum();
    let local_requests: i64 = stats
        .iter()
        .filter(|s| s.routed_to == "local")
        .map(|s| s.total)
        .sum();
    let local_hit_rate = if total_requests > 0 {
        local_requests as f64 / total_requests as f64
    } else {
        0.0
    };

    Ok(Json(serde_json::json!({
        "days": days,
        "total_requests": total_requests,
        "local_requests": local_requests,
        "cloud_requests": total_requests - local_requests,
        "local_hit_rate": local_hit_rate,
        "by_destination": stats,
        "local_success_by_type": success_rates,
    })))
}

/// POST /api/admin/distill/export — Export training samples as JSONL
pub async fn export_samples(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> AppResult<Json<serde_json::Value>> {
    let task_type = req.task_type.as_deref();
    let limit = req.limit.unwrap_or(1000).min(10000);
    let format = req.format.as_deref().unwrap_or("openai");

    let samples = db::list_training_samples_for_export(&state.db, task_type, limit).await?;

    if samples.is_empty() {
        return Ok(Json(serde_json::json!({
            "count": 0,
            "data": [],
            "message": "No unexported samples found",
        })));
    }

    let mut output = Vec::new();
    let mut exported_ids = Vec::new();

    for sample in &samples {
        let converted = match format {
            "openai" => convert_to_openai_format(sample),
            "sharegpt" => convert_to_sharegpt_format(sample),
            _ => convert_to_openai_format(sample),
        };
        if let Some(entry) = converted {
            output.push(entry);
            exported_ids.push(sample.id.clone());
        }
    }

    // Mark as exported if requested
    if req.mark_exported.unwrap_or(false) {
        db::mark_samples_exported(&state.db, &exported_ids).await?;
    }

    Ok(Json(serde_json::json!({
        "count": output.len(),
        "format": format,
        "data": output,
    })))
}

#[derive(Debug, serde::Deserialize)]
pub struct ExportRequest {
    pub task_type: Option<String>,
    pub limit: Option<i32>,
    pub format: Option<String>,
    pub mark_exported: Option<bool>,
}

/// GET /api/admin/distill/config — Get current distillation config
pub async fn get_distill_config(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<serde_json::Value>> {
    let distill = &state.config.distillation;

    Ok(Json(serde_json::json!({
        "distillation": {
            "collect_training_data": distill.collect_training_data,
            "max_samples": distill.max_samples,
            "only_successful": distill.only_successful,
        },
    })))
}

/// PUT /api/admin/distill/config — placeholder (config is static for now)
pub async fn update_distill_config(
    State(_state): State<Arc<AppState>>,
    Json(_req): Json<serde_json::Value>,
) -> AppResult<Json<serde_json::Value>> {
    Ok(Json(serde_json::json!({
        "message": "Distillation config is currently static. Edit louter.toml and restart.",
    })))
}

fn convert_to_openai_format(
    sample: &db::schema::TrainingSampleRow,
) -> Option<serde_json::Value> {
    let messages: Vec<serde_json::Value> =
        serde_json::from_str(&sample.request_messages).ok()?;
    let response: serde_json::Value = serde_json::from_str(&sample.response_content).ok()?;

    let mut conversation = messages;
    conversation.push(response);

    if let Some(ref tools_json) = sample.request_tools {
        let tools: serde_json::Value = serde_json::from_str(tools_json).ok()?;
        return Some(serde_json::json!({
            "messages": conversation,
            "tools": tools,
        }));
    }

    Some(serde_json::json!({
        "messages": conversation,
    }))
}

fn convert_to_sharegpt_format(
    sample: &db::schema::TrainingSampleRow,
) -> Option<serde_json::Value> {
    let messages: Vec<serde_json::Value> =
        serde_json::from_str(&sample.request_messages).ok()?;
    let response: serde_json::Value = serde_json::from_str(&sample.response_content).ok()?;

    let mut conversations = Vec::new();

    for msg in &messages {
        let role = msg.get("role")?.as_str()?;
        let content = msg
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("");

        let from = match role {
            "system" => "system",
            "user" => "human",
            "assistant" => "gpt",
            "tool" => "tool",
            _ => continue,
        };

        conversations.push(serde_json::json!({
            "from": from,
            "value": content,
        }));
    }

    let response_content = response
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    conversations.push(serde_json::json!({
        "from": "gpt",
        "value": response_content,
    }));

    Some(serde_json::json!({
        "conversations": conversations,
    }))
}
