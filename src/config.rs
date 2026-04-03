use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub database: DatabaseConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    6188
}

#[derive(Debug, Deserialize)]
pub struct DatabaseConfig {
    #[serde(default = "default_db_path")]
    pub path: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: default_db_path(),
        }
    }
}

fn default_db_path() -> String {
    "louter.db".to_string()
}

/// Distillation data collection configuration
#[derive(Debug, Deserialize)]
pub struct DistillationConfig {
    /// Enable automatic collection of training samples from cloud responses
    #[serde(default = "default_true")]
    pub collect_training_data: bool,

    /// Maximum number of training samples to keep (oldest are pruned)
    #[serde(default = "default_max_samples")]
    pub max_samples: i64,

    /// Only collect samples from successful responses (status 200)
    #[serde(default = "default_true")]
    pub only_successful: bool,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            collect_training_data: true,
            max_samples: 100_000,
            only_successful: true,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_max_samples() -> i64 {
    100_000
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file {}: {e}", path.display()))?;

        let config: AppConfig =
            toml::from_str(&content).map_err(|e| format!("Failed to parse config: {e}"))?;

        Ok(config)
    }
}
