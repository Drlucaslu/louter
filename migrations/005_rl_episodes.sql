-- RL episodes for reinforcement learning pipeline
-- Each episode represents a prompt-completion pair with a reward signal.
CREATE TABLE IF NOT EXISTS rl_episodes (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL,                    -- FK to training_samples.id
    prompt_messages TEXT NOT NULL,              -- JSON: input messages (without response)
    completion TEXT NOT NULL,                   -- JSON: model response
    source TEXT NOT NULL DEFAULT 'cloud',       -- 'local' or 'cloud'
    reward REAL,                                -- scalar reward [-1.0, 1.0], NULL = unscored
    reward_source TEXT,                         -- 'implicit', 'judge', 'environment', 'manual'
    reward_details TEXT,                        -- JSON: breakdown of reward components
    is_used_for_training INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_rl_sample_id ON rl_episodes(sample_id);
CREATE INDEX IF NOT EXISTS idx_rl_source ON rl_episodes(source);
CREATE INDEX IF NOT EXISTS idx_rl_reward ON rl_episodes(reward);
CREATE INDEX IF NOT EXISTS idx_rl_reward_source ON rl_episodes(reward_source);
CREATE INDEX IF NOT EXISTS idx_rl_is_used ON rl_episodes(is_used_for_training);
CREATE INDEX IF NOT EXISTS idx_rl_created_at ON rl_episodes(created_at);
