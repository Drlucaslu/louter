pub mod smart_router;
pub mod static_router;

use std::sync::Arc;

use crate::db;
use crate::db::schema::KeyRow;
use crate::error::{AppError, AppResult};
use crate::providers::Provider;
use crate::types::chat::ChatCompletionRequest;
use smart_router::classify_request;
use static_router::{match_pattern, ProviderRegistry};
use sqlx::SqlitePool;

/// A single routing target (provider + model to use).
#[derive(Clone)]
pub struct RouteTarget {
    pub provider: Arc<dyn Provider>,
    pub model: String,
}

/// Resolve an ordered list of routing targets for a request.
///
/// - `failover` mode: match model name against routing rules, return providers by priority.
/// - `content` mode: classify request content, match `@category` rules, return providers by priority.
///
/// The caller should try each target in order (failover on error).
pub async fn resolve_targets(
    registry: &ProviderRegistry,
    pool: &SqlitePool,
    key: &KeyRow,
    req: &ChatCompletionRequest,
) -> AppResult<Vec<RouteTarget>> {
    match key.routing_mode.as_str() {
        "content" => resolve_content_targets(registry, pool, key, req).await,
        _ => resolve_failover_targets(registry, pool, key, &req.model).await,
    }
}

/// Failover mode: match model pattern → collect providers by priority DESC → default provider.
async fn resolve_failover_targets(
    registry: &ProviderRegistry,
    pool: &SqlitePool,
    key: &KeyRow,
    model: &str,
) -> AppResult<Vec<RouteTarget>> {
    let rules = db::get_routing_rules_for_key(pool, &key.id).await?;
    let mut targets = Vec::new();

    // 1. Collect all matching rules (already sorted by priority DESC)
    for rule in &rules {
        if match_pattern(&rule.model_pattern, model) {
            if let Some(p) = registry.get(&rule.target_provider_id).await {
                targets.push(RouteTarget {
                    provider: p,
                    model: model.to_string(),
                });
            }
        }
    }

    if !targets.is_empty() {
        return Ok(targets);
    }

    // 2. Fall back to default provider
    if let Some(ref dp_id) = key.default_provider_id {
        if let Some(p) = registry.get(dp_id).await {
            return Ok(vec![RouteTarget {
                provider: p,
                model: model.to_string(),
            }]);
        }
    }

    Err(AppError::NoRoute(format!(
        "No provider found for model '{model}'"
    )))
}

/// Content mode: classify request → match @category rules → collect providers by priority DESC.
async fn resolve_content_targets(
    registry: &ProviderRegistry,
    pool: &SqlitePool,
    key: &KeyRow,
    req: &ChatCompletionRequest,
) -> AppResult<Vec<RouteTarget>> {
    let classification = classify_request(req);
    let category = classification.category; // e.g. "code", "math", "general"
    let pattern = format!("@{category}");

    let rules = db::get_routing_rules_for_key(pool, &key.id).await?;
    let mut targets = Vec::new();

    // 1. Collect rules matching @category
    for rule in &rules {
        if rule.model_pattern == pattern {
            if let Some(p) = registry.get(&rule.target_provider_id).await {
                targets.push(RouteTarget {
                    provider: p,
                    model: req.model.clone(),
                });
            }
        }
    }

    // 2. If no match, try @general as fallback
    if targets.is_empty() && category != "general" {
        for rule in &rules {
            if rule.model_pattern == "@general" {
                if let Some(p) = registry.get(&rule.target_provider_id).await {
                    targets.push(RouteTarget {
                        provider: p,
                        model: req.model.clone(),
                    });
                }
            }
        }
    }

    if !targets.is_empty() {
        return Ok(targets);
    }

    // 3. Fall back to default provider
    if let Some(ref dp_id) = key.default_provider_id {
        if let Some(p) = registry.get(dp_id).await {
            return Ok(vec![RouteTarget {
                provider: p,
                model: req.model.clone(),
            }]);
        }
    }

    Err(AppError::NoRoute(format!(
        "No provider found for content type '@{category}'"
    )))
}
