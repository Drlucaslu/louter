mod api;
mod config;
#[allow(dead_code)]
mod db;
mod error;
mod feedback;
mod providers;
mod router;
mod server;
mod types;

use std::path::PathBuf;
use std::sync::Arc;

use config::AppConfig;
use router::static_router::ProviderRegistry;
use sqlx::SqlitePool;

pub struct AppState {
    pub db: SqlitePool,
    pub registry: ProviderRegistry,
    pub config: AppConfig,
    pub feedback: feedback::FeedbackTracker,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "louter=info".into()),
        )
        .init();

    // Load config — check env var first (for container deployments), then CLI arg, then default
    let config_path = std::env::var("LOUTER_CONFIG")
        .map(PathBuf::from)
        .ok()
        .or_else(|| std::env::args().nth(1).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("louter.toml"));

    let config = if config_path.exists() {
        match AppConfig::load(&config_path) {
            Ok(c) => {
                tracing::info!("Loaded config from {}", config_path.display());
                c
            }
            Err(e) => {
                tracing::error!("Failed to load config: {e}");
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!("No config file found, using defaults");
        AppConfig {
            server: Default::default(),
            database: Default::default(),
            distillation: Default::default(),
        }
    };

    // Apply environment variable overrides (for container deployments)
    let server_host = std::env::var("LOUTER_HOST").unwrap_or(config.server.host.clone());
    let server_port: u16 = std::env::var("LOUTER_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(config.server.port);
    let db_path = std::env::var("LOUTER_DB_PATH").unwrap_or(config.database.path.clone());

    // Initialize database
    let db_url = format!("sqlite:{}?mode=rwc", db_path);
    let pool = match db::init_db(&db_url).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to initialize database: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("Database initialized at {}", db_path);

    // Load providers from database
    let registry = ProviderRegistry::new();
    match db::list_providers(&pool).await {
        Ok(rows) => {
            for row in &rows {
                if !row.is_enabled {
                    continue;
                }
                match providers::create_provider_from_row(row) {
                    Some(provider) => {
                        tracing::info!("Loaded provider: {} ({})", row.name, row.provider_type);
                        registry.register(row.id.clone(), provider).await;
                    }
                    None => {
                        tracing::warn!(
                            "Unknown provider type '{}' for '{}', skipping",
                            row.provider_type,
                            row.name
                        );
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!("Failed to load providers from DB: {e}");
        }
    }

    let addr = format!("{}:{}", server_host, server_port);
    let state = Arc::new(AppState {
        db: pool,
        registry,
        config,
        feedback: feedback::FeedbackTracker::new(),
    });

    let app = server::build_router(state);

    tracing::info!("Louter listening on http://{addr}");
    tracing::info!("Open http://{addr} to configure providers and keys via Web UI");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app).await.expect("Server error");
}
