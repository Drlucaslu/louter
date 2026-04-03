-- Migrate routing_mode values: rules/hybrid/smart → failover/content
UPDATE keys SET routing_mode = 'failover' WHERE routing_mode IN ('rules', 'hybrid');
UPDATE keys SET routing_mode = 'content' WHERE routing_mode = 'smart';

-- Create unique index to allow multiple providers per pattern but prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_routing_rules_key_pattern_provider
  ON routing_rules(key_id, model_pattern, target_provider_id);
