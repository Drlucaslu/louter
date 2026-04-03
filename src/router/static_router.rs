use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::providers::Provider;

/// ProviderRegistry holds all initialized Provider instances.
/// Wrapped in RwLock to allow runtime updates from admin API.
pub struct ProviderRegistry {
    providers: RwLock<HashMap<String, Arc<dyn Provider>>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, id: String, provider: Arc<dyn Provider>) {
        self.providers.write().await.insert(id, provider);
    }

    pub async fn remove(&self, id: &str) {
        self.providers.write().await.remove(id);
    }

    pub async fn get(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers.read().await.get(id).cloned()
    }

    pub async fn all(&self) -> Vec<(String, Arc<dyn Provider>)> {
        self.providers
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Match model name against a glob-like pattern.
/// Supports `*` as wildcard.
pub fn match_pattern(pattern: &str, model: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix('*') {
        return model.starts_with(prefix);
    }

    if let Some(suffix) = pattern.strip_prefix('*') {
        return model.ends_with(suffix);
    }

    pattern == model
}

