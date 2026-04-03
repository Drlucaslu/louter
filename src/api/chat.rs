use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::db;
use crate::db::schema::{RoutingHistoryRow, TrainingSampleRow, UsageLogRow};
use crate::error::AppError;
use crate::feedback;
use crate::router;
use crate::router::smart_router;
use crate::types::chat::ChatCompletionRequest;
use crate::AppState;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    // Auth: extract Bearer token
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let key_value = auth_header
        .strip_prefix("Bearer ")
        .unwrap_or(auth_header);

    // Look up the key
    let key = db::get_key_by_value(&state.db, key_value)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Invalid API key".to_string()))?;

    // Capture the original requested model before routing
    let request_model = req.model.clone();

    // Classify the request for training data collection
    let classification = smart_router::classify_request(&req);
    let task_type = classification.category.to_string();

    // Implicit feedback: detect if this is a retry of a recent request.
    let content_hash = feedback::hash_last_user_message(&req.messages);
    let failed_sample = state
        .feedback
        .record_and_check_retry(content_hash, &task_type, None, "pending")
        .await;
    if let Some(sample_id) = failed_sample {
        let db = state.db.clone();
        tokio::spawn(async move {
            let _ = sqlx::query(
                "UPDATE training_samples SET is_successful = 0 WHERE id = ?",
            )
            .bind(&sample_id)
            .execute(&db)
            .await;
            tracing::info!("Marked sample {} as failed (retry detected)", sample_id);
        });
    }

    // Resolve routing targets (ordered by priority, for failover)
    let targets = router::resolve_targets(&state.registry, &state.db, &key, &req).await?;
    let key_id = key.id.clone();

    if req.stream {
        // Streaming: use the first target (no failover possible mid-stream)
        let target = targets.into_iter().next().ok_or_else(|| {
            AppError::NoRoute("No routing target available".into())
        })?;
        let mut req = req;
        req.model = target.model.clone();
        execute_and_track(
            state, req, key_id, request_model, target.model, task_type, target.provider, "cloud",
        )
        .await
    } else {
        // Non-streaming: try each target in order (failover on error)
        let mut last_err = None;
        for (i, target) in targets.iter().enumerate() {
            let mut req = req.clone();
            req.model = target.model.clone();

            let start = Instant::now();
            match target.provider.complete(&req).await {
                Ok(response) => {
                    let latency = start.elapsed().as_millis() as i32;
                    let usage = response.usage.as_ref();
                    let pt = usage.map(|u| u.prompt_tokens as i32).unwrap_or(0);
                    let ct = usage.map(|u| u.completion_tokens as i32).unwrap_or(0);

                    let db = state.db.clone();
                    let key_id_c = key_id.clone();
                    let request_model_c = request_model.clone();
                    let model_c = target.model.clone();
                    let task_type_c = task_type.clone();
                    let was_fallback = i > 0;

                    // Collect training sample
                    let collect_data = state.config.distillation.collect_training_data;
                    let sample_id = if collect_data {
                        collect_sample(
                            &state.db, &req, &response, &request_model, &target.model,
                            &target.provider.provider_type().to_string(),
                            &task_type, "cloud", latency,
                        )
                        .await
                    } else {
                        None
                    };

                    // Update feedback tracker
                    let content_hash = feedback::hash_last_user_message(&req.messages);
                    state
                        .feedback
                        .record_and_check_retry(content_hash, &task_type, sample_id, "cloud")
                        .await;

                    tokio::spawn(async move {
                        log_usage(
                            &db, &key_id_c, &request_model_c, &model_c, pt, ct, latency,
                        ).await;
                        log_routing(&db, &task_type_c, "cloud", true, was_fallback, latency).await;
                    });

                    return Ok(Json(response).into_response());
                }
                Err(e) => {
                    let latency = start.elapsed().as_millis() as i32;
                    tracing::warn!(
                        "Provider '{}' failed (target {}/{}): {e}",
                        target.provider.name(),
                        i + 1,
                        targets.len()
                    );
                    let db = state.db.clone();
                    let task_type_c = task_type.clone();
                    tokio::spawn(async move {
                        log_routing(&db, &task_type_c, "cloud", false, i > 0, latency).await;
                    });
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            AppError::NoRoute("No routing target available".into())
        }))
    }
}

/// Execute a request and track usage + training data (used for streaming).
async fn execute_and_track(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
    key_id: String,
    request_model: String,
    model: String,
    task_type: String,
    provider: Arc<dyn crate::providers::Provider>,
    source: &str,
) -> Result<Response, AppError> {
    let source = source.to_string();

    if req.stream {
        // Streaming response
        let start = Instant::now();
        let chunk_stream = provider.stream(&req).await?;

        let db = state.db.clone();
        let collect_data = state.config.distillation.collect_training_data;
        let req_for_sample = if collect_data { Some(req.clone()) } else { None };
        let task_type_clone = task_type.clone();
        let request_model_clone = request_model.clone();
        let model_clone = model.clone();
        let source_clone = source.clone();

        let (tx, rx) = mpsc::channel(64);

        tokio::spawn(async move {
            let mut prompt_tokens: u32 = 0;
            let mut completion_tokens: u32 = 0;
            let mut accumulated_content = String::new();
            let mut has_tool_calls = false;

            futures::pin_mut!(chunk_stream);
            while let Some(result) = chunk_stream.next().await {
                match result {
                    Ok(chunk) => {
                        if let Some(ref u) = chunk.usage {
                            if u.prompt_tokens > 0 {
                                prompt_tokens = u.prompt_tokens;
                            }
                            if u.completion_tokens > 0 {
                                completion_tokens = u.completion_tokens;
                            }
                        }

                        // Accumulate content for training sample
                        if collect_data {
                            for choice in &chunk.choices {
                                if let Some(ref content) = choice.delta.content {
                                    accumulated_content.push_str(content);
                                }
                                if choice.delta.tool_calls.is_some() {
                                    has_tool_calls = true;
                                }
                            }
                        }

                        let data = serde_json::to_string(&chunk).unwrap_or_default();
                        let event = Event::default().data(data);
                        if tx.send(Ok::<_, std::convert::Infallible>(event)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {e}");
                        let event = Event::default().data(format!(r#"{{"error": "{e}"}}"#));
                        let _ = tx.send(Ok(event)).await;
                        break;
                    }
                }
            }

            // Send [DONE]
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

            let latency = start.elapsed().as_millis() as i32;
            let (pt, ct, tt) = (
                prompt_tokens as i32,
                completion_tokens as i32,
                (prompt_tokens + completion_tokens) as i32,
            );

            log_usage(&db, &key_id, &request_model_clone, &model_clone, pt, ct, latency).await;
            log_routing(&db, &task_type_clone, &source_clone, true, false, latency).await;

            // Collect streaming response as training sample
            if collect_data && !accumulated_content.is_empty() {
                if let Some(req_data) = req_for_sample {
                    let response_json = serde_json::json!({
                        "role": "assistant",
                        "content": accumulated_content,
                    });
                    let messages_json =
                        serde_json::to_string(&req_data.messages).unwrap_or_default();
                    let tools_json = req_data.tools.as_ref().map(|t| {
                        serde_json::to_string(t).unwrap_or_default()
                    });

                    let sample = TrainingSampleRow {
                        id: uuid::Uuid::new_v4().to_string(),
                        request_messages: messages_json,
                        request_tools: tools_json,
                        response_content: response_json.to_string(),
                        request_model: request_model_clone,
                        actual_model: model_clone,
                        provider_type: "cloud".to_string(),
                        task_type: task_type_clone,
                        has_tool_calls,
                        is_successful: true,
                        source: "cloud".to_string(),
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        total_tokens: tt,
                        latency_ms: latency,
                        is_exported: false,
                        created_at: chrono::Utc::now().to_rfc3339(),
                    };
                    let _ = db::insert_training_sample(&db, &sample).await;
                }
            }
        });

        let event_stream = ReceiverStream::new(rx);
        Ok(Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming response
        let start = Instant::now();
        let response = provider.complete(&req).await?;
        let latency = start.elapsed().as_millis() as i32;

        let usage = response.usage.as_ref();
        let pt = usage.map(|u| u.prompt_tokens as i32).unwrap_or(0);
        let ct = usage.map(|u| u.completion_tokens as i32).unwrap_or(0);

        let db = state.db.clone();
        let key_id_clone = key_id.clone();
        let request_model_clone = request_model.clone();
        let model_clone = model.clone();
        let task_type_clone = task_type.clone();
        let source_clone = source.clone();

        // Collect training sample
        let collect_data = state.config.distillation.collect_training_data;
        let sample_id = if collect_data {
            collect_sample(
                &db, &req, &response, &request_model, &model,
                "cloud", &task_type, &source, latency,
            )
            .await
        } else {
            None
        };

        // Update feedback tracker
        let content_hash = feedback::hash_last_user_message(&req.messages);
        state
            .feedback
            .record_and_check_retry(content_hash, &task_type, sample_id, &source)
            .await;

        tokio::spawn(async move {
            log_usage(
                &db, &key_id_clone, &request_model_clone, &model_clone, pt, ct, latency,
            ).await;
            log_routing(&db, &task_type_clone, &source_clone, true, false, latency).await;
        });

        Ok(Json(response).into_response())
    }
}

/// Log usage to usage_logs table.
async fn log_usage(
    db: &sqlx::SqlitePool,
    key_id: &str,
    request_model: &str,
    model: &str,
    prompt_tokens: i32,
    completion_tokens: i32,
    latency_ms: i32,
) {
    let total_tokens = prompt_tokens + completion_tokens;
    let log = UsageLogRow {
        id: uuid::Uuid::new_v4().to_string(),
        key_id: Some(key_id.to_string()),
        provider_id: None,
        request_model: request_model.to_string(),
        model_id: model.to_string(),
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        status_code: 200,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    let _ = db::insert_usage_log(db, &log).await;
}

/// Log routing decision to routing_history table.
async fn log_routing(
    db: &sqlx::SqlitePool,
    task_type: &str,
    routed_to: &str,
    was_successful: bool,
    was_fallback: bool,
    latency_ms: i32,
) {
    let row = RoutingHistoryRow {
        id: uuid::Uuid::new_v4().to_string(),
        task_type: task_type.to_string(),
        routed_to: routed_to.to_string(),
        was_successful,
        was_fallback,
        latency_ms,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    let _ = db::insert_routing_history(db, &row).await;
}

/// Collect a training sample from a completed request/response pair.
async fn collect_sample(
    db: &sqlx::SqlitePool,
    req: &ChatCompletionRequest,
    response: &crate::types::chat::ChatCompletionResponse,
    request_model: &str,
    actual_model: &str,
    provider_type: &str,
    task_type: &str,
    source: &str,
    latency_ms: i32,
) -> Option<String> {
    let messages_json = serde_json::to_string(&req.messages).unwrap_or_default();
    let tools_json = req
        .tools
        .as_ref()
        .map(|t| serde_json::to_string(t).unwrap_or_default());

    let has_tool_calls = response
        .choices
        .first()
        .and_then(|c| c.message.tool_calls.as_ref())
        .is_some_and(|tc| !tc.is_empty());

    let response_json = response
        .choices
        .first()
        .map(|c| serde_json::to_string(&c.message).unwrap_or_default())
        .unwrap_or_default();

    let usage = response.usage.as_ref();
    let pt = usage.map(|u| u.prompt_tokens as i32).unwrap_or(0);
    let ct = usage.map(|u| u.completion_tokens as i32).unwrap_or(0);
    let tt = pt + ct;

    let sample_id = uuid::Uuid::new_v4().to_string();
    let sample = TrainingSampleRow {
        id: sample_id.clone(),
        request_messages: messages_json,
        request_tools: tools_json,
        response_content: response_json,
        request_model: request_model.to_string(),
        actual_model: actual_model.to_string(),
        provider_type: provider_type.to_string(),
        task_type: task_type.to_string(),
        has_tool_calls,
        is_successful: true,
        source: source.to_string(),
        prompt_tokens: pt,
        completion_tokens: ct,
        total_tokens: tt,
        latency_ms,
        is_exported: false,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    match db::insert_training_sample(db, &sample).await {
        Ok(_) => Some(sample_id),
        Err(_) => None,
    }
}
