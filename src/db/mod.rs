pub mod schema;

use crate::error::AppResult;
use schema::*;
use sqlx::SqlitePool;

pub async fn init_db(database_url: &str) -> Result<SqlitePool, String> {
    let pool = SqlitePool::connect(database_url)
        .await
        .map_err(|e| format!("Failed to connect to database: {e}"))?;

    // Run migrations
    let migration_sql = include_str!("../../migrations/001_initial.sql");
    sqlx::raw_sql(migration_sql)
        .execute(&pool)
        .await
        .map_err(|e| format!("Failed to run migrations: {e}"))?;

    // Run incremental migrations (ignore errors for already-applied columns)
    let m002 = include_str!("../../migrations/002_add_request_model.sql");
    let _ = sqlx::raw_sql(m002).execute(&pool).await;

    let m003 = include_str!("../../migrations/003_training_samples.sql");
    let _ = sqlx::raw_sql(m003).execute(&pool).await;

    let m004 = include_str!("../../migrations/004_key_routing_mode.sql");
    let _ = sqlx::raw_sql(m004).execute(&pool).await;

    let m005 = include_str!("../../migrations/005_rl_episodes.sql");
    let _ = sqlx::raw_sql(m005).execute(&pool).await;

    Ok(pool)
}

// ── Provider CRUD ──

pub async fn list_providers(pool: &SqlitePool) -> AppResult<Vec<ProviderRow>> {
    let rows = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers ORDER BY name")
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

pub async fn get_provider(pool: &SqlitePool, id: &str) -> AppResult<Option<ProviderRow>> {
    let row = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn get_provider_by_name(pool: &SqlitePool, name: &str) -> AppResult<Option<ProviderRow>> {
    let row = sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers WHERE name = ?")
        .bind(name)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn insert_provider(pool: &SqlitePool, row: &ProviderRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO providers (id, name, provider_type, base_url, api_key, is_enabled, config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.name)
    .bind(&row.provider_type)
    .bind(&row.base_url)
    .bind(&row.api_key)
    .bind(row.is_enabled)
    .bind(&row.config_json)
    .bind(&row.created_at)
    .bind(&row.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_provider(pool: &SqlitePool, row: &ProviderRow) -> AppResult<()> {
    sqlx::query(
        "UPDATE providers SET name = ?, provider_type = ?, base_url = ?, api_key = ?, is_enabled = ?, config_json = ?, updated_at = datetime('now') WHERE id = ?"
    )
    .bind(&row.name)
    .bind(&row.provider_type)
    .bind(&row.base_url)
    .bind(&row.api_key)
    .bind(row.is_enabled)
    .bind(&row.config_json)
    .bind(&row.id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_provider(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM providers WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Key CRUD ──

pub async fn list_keys(pool: &SqlitePool) -> AppResult<Vec<KeyRow>> {
    let rows = sqlx::query_as::<_, KeyRow>("SELECT * FROM keys ORDER BY name")
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

pub async fn get_key_by_value(pool: &SqlitePool, key_value: &str) -> AppResult<Option<KeyRow>> {
    let row = sqlx::query_as::<_, KeyRow>("SELECT * FROM keys WHERE key_value = ? AND is_enabled = 1")
        .bind(key_value)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

pub async fn insert_key(pool: &SqlitePool, row: &KeyRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO keys (id, key_value, name, default_provider_id, routing_mode, is_enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_value)
    .bind(&row.name)
    .bind(&row.default_provider_id)
    .bind(&row.routing_mode)
    .bind(row.is_enabled)
    .bind(&row.created_at)
    .bind(&row.updated_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_key(pool: &SqlitePool, row: &KeyRow) -> AppResult<()> {
    sqlx::query(
        "UPDATE keys SET name = ?, default_provider_id = ?, routing_mode = ?, is_enabled = ?, updated_at = datetime('now') WHERE id = ?"
    )
    .bind(&row.name)
    .bind(&row.default_provider_id)
    .bind(&row.routing_mode)
    .bind(row.is_enabled)
    .bind(&row.id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_key(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM keys WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Routing Rules ──

pub async fn get_routing_rules_for_key(pool: &SqlitePool, key_id: &str) -> AppResult<Vec<RoutingRuleRow>> {
    let rows = sqlx::query_as::<_, RoutingRuleRow>(
        "SELECT * FROM routing_rules WHERE key_id = ? ORDER BY priority DESC"
    )
    .bind(key_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn list_routing_rules(pool: &SqlitePool) -> AppResult<Vec<RoutingRuleRow>> {
    let rows = sqlx::query_as::<_, RoutingRuleRow>(
        "SELECT * FROM routing_rules ORDER BY priority DESC"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn insert_routing_rule(pool: &SqlitePool, row: &RoutingRuleRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO routing_rules (id, key_id, model_pattern, target_provider_id, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_id)
    .bind(&row.model_pattern)
    .bind(&row.target_provider_id)
    .bind(row.priority)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_routing_rule(pool: &SqlitePool, id: &str) -> AppResult<()> {
    sqlx::query("DELETE FROM routing_rules WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn delete_routing_rule_by_key_and_pattern(
    pool: &SqlitePool,
    key_id: &str,
    model_pattern: &str,
) -> AppResult<()> {
    sqlx::query("DELETE FROM routing_rules WHERE key_id = ? AND model_pattern = ?")
        .bind(key_id)
        .bind(model_pattern)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Usage Logs ──

pub async fn insert_usage_log(pool: &SqlitePool, row: &UsageLogRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO usage_logs (id, key_id, provider_id, request_model, model_id, prompt_tokens, completion_tokens, total_tokens, latency_ms, status_code, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.key_id)
    .bind(&row.provider_id)
    .bind(&row.request_model)
    .bind(&row.model_id)
    .bind(row.prompt_tokens)
    .bind(row.completion_tokens)
    .bind(row.total_tokens)
    .bind(row.latency_ms)
    .bind(row.status_code)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn list_usage_logs(pool: &SqlitePool, limit: i32) -> AppResult<Vec<UsageLogRow>> {
    let rows = sqlx::query_as::<_, UsageLogRow>(
        "SELECT * FROM usage_logs ORDER BY created_at DESC LIMIT ?"
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn list_usage_logs_paged(pool: &SqlitePool, page: i32, page_size: i32) -> AppResult<Vec<UsageLogRow>> {
    let offset = (page - 1) * page_size;
    let rows = sqlx::query_as::<_, UsageLogRow>(
        "SELECT * FROM usage_logs ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    .bind(page_size)
    .bind(offset)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn count_usage_logs(pool: &SqlitePool) -> AppResult<i64> {
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM usage_logs")
        .fetch_one(pool)
        .await?;
    Ok(row.0)
}

/// Daily token usage aggregation
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct DailyUsage {
    pub day: String,
    pub requests: i32,
    pub total_tokens: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub avg_latency_ms: f64,
}

pub async fn get_daily_usage(pool: &SqlitePool, days: i32) -> AppResult<Vec<DailyUsage>> {
    let rows = sqlx::query_as::<_, DailyUsage>(
        "SELECT date(created_at) as day, \
         COUNT(*) as requests, \
         COALESCE(SUM(total_tokens), 0) as total_tokens, \
         COALESCE(SUM(prompt_tokens), 0) as prompt_tokens, \
         COALESCE(SUM(completion_tokens), 0) as completion_tokens, \
         COALESCE(AVG(latency_ms), 0) as avg_latency_ms \
         FROM usage_logs \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY date(created_at) \
         ORDER BY day ASC"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Per-model token usage aggregation
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct ModelUsage {
    pub model_id: String,
    pub requests: i32,
    pub total_tokens: i64,
}

pub async fn get_model_usage(pool: &SqlitePool, days: i32) -> AppResult<Vec<ModelUsage>> {
    let rows = sqlx::query_as::<_, ModelUsage>(
        "SELECT model_id, COUNT(*) as requests, \
         COALESCE(SUM(total_tokens), 0) as total_tokens \
         FROM usage_logs \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY model_id \
         ORDER BY total_tokens DESC"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

// ── Models ──

pub async fn list_models_for_provider(pool: &SqlitePool, provider_id: &str) -> AppResult<Vec<ModelRow>> {
    let rows = sqlx::query_as::<_, ModelRow>(
        "SELECT * FROM models WHERE provider_id = ? AND is_enabled = 1 ORDER BY model_id"
    )
    .bind(provider_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn upsert_model(pool: &SqlitePool, row: &ModelRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO models (id, provider_id, model_id, is_enabled, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(provider_id, model_id) DO UPDATE SET is_enabled = excluded.is_enabled"
    )
    .bind(&row.id)
    .bind(&row.provider_id)
    .bind(&row.model_id)
    .bind(row.is_enabled)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

// ── Training Samples (Distillation) ──

pub async fn insert_training_sample(pool: &SqlitePool, row: &TrainingSampleRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO training_samples (id, request_messages, request_tools, response_content, request_model, actual_model, provider_type, task_type, has_tool_calls, is_successful, source, prompt_tokens, completion_tokens, total_tokens, latency_ms, is_exported, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.request_messages)
    .bind(&row.request_tools)
    .bind(&row.response_content)
    .bind(&row.request_model)
    .bind(&row.actual_model)
    .bind(&row.provider_type)
    .bind(&row.task_type)
    .bind(row.has_tool_calls)
    .bind(row.is_successful)
    .bind(&row.source)
    .bind(row.prompt_tokens)
    .bind(row.completion_tokens)
    .bind(row.total_tokens)
    .bind(row.latency_ms)
    .bind(row.is_exported)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn list_training_samples_for_export(
    pool: &SqlitePool,
    task_type: Option<&str>,
    limit: i32,
) -> AppResult<Vec<TrainingSampleRow>> {
    let rows = if let Some(tt) = task_type {
        sqlx::query_as::<_, TrainingSampleRow>(
            "SELECT * FROM training_samples WHERE is_successful = 1 AND is_exported = 0 AND task_type = ? ORDER BY created_at ASC LIMIT ?"
        )
        .bind(tt)
        .bind(limit)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query_as::<_, TrainingSampleRow>(
            "SELECT * FROM training_samples WHERE is_successful = 1 AND is_exported = 0 ORDER BY created_at ASC LIMIT ?"
        )
        .bind(limit)
        .fetch_all(pool)
        .await?
    };
    Ok(rows)
}

pub async fn mark_samples_exported(pool: &SqlitePool, ids: &[String]) -> AppResult<()> {
    for id in ids {
        sqlx::query("UPDATE training_samples SET is_exported = 1 WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;
    }
    Ok(())
}

pub async fn count_training_samples(pool: &SqlitePool) -> AppResult<i64> {
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM training_samples")
        .fetch_one(pool)
        .await?;
    Ok(row.0)
}

pub async fn count_unexported_samples(pool: &SqlitePool) -> AppResult<i64> {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM training_samples WHERE is_successful = 1 AND is_exported = 0",
    )
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Per-task-type sample counts
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct TaskTypeStat {
    pub task_type: String,
    pub total: i64,
    pub successful: i64,
    pub with_tool_calls: i64,
}

pub async fn get_training_sample_stats(pool: &SqlitePool) -> AppResult<Vec<TaskTypeStat>> {
    let rows = sqlx::query_as::<_, TaskTypeStat>(
        "SELECT task_type, \
         COUNT(*) as total, \
         SUM(CASE WHEN is_successful = 1 THEN 1 ELSE 0 END) as successful, \
         SUM(CASE WHEN has_tool_calls = 1 THEN 1 ELSE 0 END) as with_tool_calls \
         FROM training_samples \
         GROUP BY task_type \
         ORDER BY total DESC"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

// ── Routing History ──

pub async fn insert_routing_history(pool: &SqlitePool, row: &RoutingHistoryRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO routing_history (id, task_type, routed_to, was_successful, was_fallback, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.task_type)
    .bind(&row.routed_to)
    .bind(row.was_successful)
    .bind(row.was_fallback)
    .bind(row.latency_ms)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Success rate for local routing by task type (last N days)
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct LocalSuccessRate {
    pub task_type: String,
    pub total: i64,
    pub successful: i64,
    pub success_rate: f64,
}

pub async fn get_local_success_rates(pool: &SqlitePool, days: i32) -> AppResult<Vec<LocalSuccessRate>> {
    let rows = sqlx::query_as::<_, LocalSuccessRate>(
        "SELECT task_type, \
         COUNT(*) as total, \
         SUM(CASE WHEN was_successful = 1 THEN 1 ELSE 0 END) as successful, \
         CAST(SUM(CASE WHEN was_successful = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as success_rate \
         FROM routing_history \
         WHERE routed_to = 'local' AND created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY task_type"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Overall routing stats
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct RoutingStats {
    pub routed_to: String,
    pub total: i64,
    pub successful: i64,
    pub fallbacks: i64,
    pub avg_latency_ms: f64,
}

// ── RL Episodes ──

pub async fn insert_rl_episode(pool: &SqlitePool, row: &RlEpisodeRow) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO rl_episodes (id, sample_id, prompt_messages, completion, source, reward, reward_source, reward_details, is_used_for_training, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    .bind(&row.id)
    .bind(&row.sample_id)
    .bind(&row.prompt_messages)
    .bind(&row.completion)
    .bind(&row.source)
    .bind(row.reward)
    .bind(&row.reward_source)
    .bind(&row.reward_details)
    .bind(row.is_used_for_training)
    .bind(&row.created_at)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_rl_episode_reward(
    pool: &SqlitePool,
    id: &str,
    reward: f64,
    reward_source: &str,
    reward_details: Option<&str>,
) -> AppResult<()> {
    sqlx::query(
        "UPDATE rl_episodes SET reward = ?, reward_source = ?, reward_details = ? WHERE id = ?",
    )
    .bind(reward)
    .bind(reward_source)
    .bind(reward_details)
    .bind(id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn mark_rl_episodes_used(pool: &SqlitePool, ids: &[String]) -> AppResult<()> {
    for id in ids {
        sqlx::query("UPDATE rl_episodes SET is_used_for_training = 1 WHERE id = ?")
            .bind(id)
            .execute(pool)
            .await?;
    }
    Ok(())
}

pub async fn count_rl_episodes_unused(pool: &SqlitePool) -> AppResult<i64> {
    let row: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM rl_episodes WHERE is_used_for_training = 0",
    )
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// RL episode statistics by source and reward status
#[derive(Debug, Clone, serde::Serialize, sqlx::FromRow)]
pub struct RlEpisodeStats {
    pub source: String,
    pub total: i64,
    pub scored: i64,
    pub used_for_training: i64,
    pub avg_reward: Option<f64>,
}

pub async fn get_rl_episode_stats(pool: &SqlitePool) -> AppResult<Vec<RlEpisodeStats>> {
    let rows = sqlx::query_as::<_, RlEpisodeStats>(
        "SELECT source, \
         COUNT(*) as total, \
         SUM(CASE WHEN reward IS NOT NULL THEN 1 ELSE 0 END) as scored, \
         SUM(CASE WHEN is_used_for_training = 1 THEN 1 ELSE 0 END) as used_for_training, \
         AVG(reward) as avg_reward \
         FROM rl_episodes \
         GROUP BY source"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

pub async fn get_routing_stats(pool: &SqlitePool, days: i32) -> AppResult<Vec<RoutingStats>> {
    let rows = sqlx::query_as::<_, RoutingStats>(
        "SELECT routed_to, \
         COUNT(*) as total, \
         SUM(CASE WHEN was_successful = 1 THEN 1 ELSE 0 END) as successful, \
         SUM(CASE WHEN was_fallback = 1 THEN 1 ELSE 0 END) as fallbacks, \
         COALESCE(AVG(latency_ms), 0) as avg_latency_ms \
         FROM routing_history \
         WHERE created_at >= datetime('now', '-' || ? || ' days') \
         GROUP BY routed_to"
    )
    .bind(days)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}
