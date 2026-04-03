use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;

use crate::db;
use crate::error::AppError;
use crate::router::static_router;
use crate::types::provider::ProviderType;
use crate::AppState;

/// Generic proxy for non-chat OpenAI-compatible endpoints.
/// Forwards the request body to the resolved provider's matching endpoint path.
///
/// Handles: /v1/images/generations, /v1/embeddings, /v1/audio/speech,
///          /v1/audio/transcriptions, etc.
pub async fn proxy_passthrough(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
) -> Result<Response, AppError> {
    // Auth
    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let key_value = auth_header.strip_prefix("Bearer ").unwrap_or(auth_header);

    let key = db::get_key_by_value(&state.db, key_value)
        .await?
        .ok_or_else(|| AppError::Unauthorized("Invalid API key".to_string()))?;

    // Try to extract "model" from the JSON body to resolve provider
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let model = body_json["model"].as_str().unwrap_or("");

    // Resolve provider: routing rules → default provider
    let provider = if !model.is_empty() {
        let rules = db::get_routing_rules_for_key(&state.db, &key.id).await?;
        let mut found = None;
        for rule in &rules {
            if static_router::match_pattern(&rule.model_pattern, model) {
                if let Some(p) = state.registry.get(&rule.target_provider_id).await {
                    found = Some(p);
                    break;
                }
            }
        }
        if let Some(p) = found {
            p
        } else if let Some(dp_id) = &key.default_provider_id {
            state.registry.get(dp_id).await
                .ok_or_else(|| AppError::NoRoute("Default provider not found".to_string()))?
        } else {
            return Err(AppError::NoRoute(format!("No provider found for model '{model}'")));
        }
    } else if let Some(dp_id) = &key.default_provider_id {
        state
            .registry
            .get(dp_id)
            .await
            .ok_or_else(|| AppError::NoRoute("No provider resolved".to_string()))?
    } else {
        return Err(AppError::NoRoute(
            "Cannot determine provider: no model field and no default provider".to_string(),
        ));
    };

    // Build upstream URL: provider_base_url + the path after /v1
    let path = uri.path(); // e.g. "/v1/images/generations"
    let sub_path = path.strip_prefix("/v1").unwrap_or(path);

    // Determine the base URL based on provider type
    let providers_list = db::list_providers(&state.db).await?;
    let provider_row = providers_list
        .iter()
        .find(|p| p.name == provider.name())
        .ok_or_else(|| AppError::Internal("Provider not found in DB".to_string()))?;

    let base_url = provider_row.base_url.trim_end_matches('/');
    let api_key = &provider_row.api_key;

    // For Ollama, the OpenAI-compatible API is under /v1
    let upstream_url = match provider.provider_type() {
        ProviderType::Ollama => format!("{}/v1{}", base_url, sub_path),
        ProviderType::Azure => {
            // Azure has a different URL scheme; for non-chat endpoints, best-effort
            format!("{}/openai{}", base_url, sub_path)
        }
        ProviderType::Anthropic => {
            return Err(AppError::BadRequest(
                "Anthropic does not support this endpoint type".to_string(),
            ));
        }
        _ => format!("{}{}", base_url, sub_path),
    };

    // Forward the request
    let client = reqwest::Client::new();
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json");

    let mut req_builder = client
        .post(&upstream_url)
        .header("Content-Type", content_type);

    // Set auth header based on provider type
    match provider.provider_type() {
        ProviderType::Azure => {
            req_builder = req_builder.header("api-key", api_key);
        }
        _ => {
            if !api_key.is_empty() {
                req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
            }
        }
    }

    let upstream_resp = req_builder.body(body.to_vec()).send().await?;

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let resp_content_type = upstream_resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let resp_body = upstream_resp.bytes().await?;

    Ok(Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, resp_content_type)
        .body(Body::from(resp_body))
        .unwrap())
}
