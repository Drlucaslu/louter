import { useEffect, useState } from 'react'

interface Key {
  id: string
  key_value: string
  name: string
  default_provider_id: string | null
  routing_mode: string // "failover" | "content"
  is_enabled: boolean
  created_at: string
}

interface RoutingRule {
  id: string
  key_id: string
  model_pattern: string
  target_provider_id: string
  priority: number
}

interface Provider {
  id: string
  name: string
  provider_type: string
}

const MODEL_PATTERN_PRESETS: Record<string, string[]> = {
  openai: ['gpt-*', 'o*', 'gpt-4o*', 'gpt-4*', '*'],
  anthropic: ['claude-*', 'claude-sonnet-*', 'claude-opus-*', 'claude-haiku-*', '*'],
  deepseek: ['deepseek-*', 'deepseek-chat', 'deepseek-reasoner', '*'],
  azure: ['gpt-*', 'gpt-4o*', 'gpt-4*', '*'],
  ollama: ['*', 'llama*', 'mistral*', 'qwen*'],
  qwen: ['qwen-*', 'qwen-plus*', 'qwen-turbo*', '*'],
  groq: ['*', 'llama*', 'mixtral*'],
  together: ['*'],
  fireworks: ['*'],
}

const CONTENT_TYPES = ['@code', '@math', '@translation', '@tool_call', '@general']

function getPatternOptions(provider: Provider | undefined, mode: string): string[] {
  if (mode === 'content') return CONTENT_TYPES
  if (!provider) return ['*']
  const byType = MODEL_PATTERN_PRESETS[provider.provider_type]
  if (byType) return byType
  const name = provider.name.toLowerCase()
  for (const [key, patterns] of Object.entries(MODEL_PATTERN_PRESETS)) {
    if (name.includes(key)) return patterns
  }
  return ['*']
}

export default function Keys() {
  const [keys, setKeys] = useState<Key[]>([])
  const [rules, setRules] = useState<RoutingRule[]>([])
  const [providers, setProviders] = useState<Provider[]>([])
  const [showKeyForm, setShowKeyForm] = useState(false)
  const [keyForm, setKeyForm] = useState(() => ({ name: '', default_provider_id: '', routing_mode: 'failover' }))
  const [showRuleForm, setShowRuleForm] = useState<string | null>(null)
  const [ruleForm, setRuleForm] = useState({ model_pattern: '', target_provider_id: '', priority: 0 })
  const [copied, setCopied] = useState<string | null>(null)

  const load = () => {
    Promise.all([
      fetch('/api/admin/keys').then(r => r.json()),
      fetch('/api/admin/rules').then(r => r.json()),
      fetch('/api/admin/providers').then(r => r.json()),
    ]).then(([k, r, p]) => {
      setKeys(k)
      setRules(r)
      setProviders(p)
    }).catch(() => {})
  }

  useEffect(() => { load() }, [])

  const createKey = async () => {
    if (!keyForm.default_provider_id) {
      alert('Please select a default provider — every key must have at least one provider.')
      return
    }
    await fetch('/api/admin/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: keyForm.name,
        default_provider_id: keyForm.default_provider_id,
        routing_mode: keyForm.routing_mode,
      }),
    })
    setShowKeyForm(false)
    setKeyForm({ name: '', default_provider_id: '', routing_mode: 'failover' })
    load()
  }

  const deleteKey = async (id: string) => {
    if (!confirm('Delete this key?')) return
    await fetch(`/api/admin/keys/${id}`, { method: 'DELETE' })
    load()
  }

  const copyKey = async (keyValue: string) => {
    await navigator.clipboard.writeText(keyValue)
    setCopied(keyValue)
    setTimeout(() => setCopied(null), 2000)
  }

  const createRule = async (keyId: string) => {
    const missingFields: string[] = []
    if (!ruleForm.model_pattern.trim()) missingFields.push('model pattern')
    if (!ruleForm.target_provider_id) missingFields.push('target provider')
    if (missingFields.length > 0) {
      alert(`Please fill in: ${missingFields.join(', ')}`)
      return
    }
    await fetch('/api/admin/rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        key_id: keyId,
        model_pattern: ruleForm.model_pattern.trim(),
        target_provider_id: ruleForm.target_provider_id,
        priority: ruleForm.priority,
      }),
    })
    setShowRuleForm(null)
    setRuleForm({ model_pattern: '', target_provider_id: '', priority: 0 })
    load()
  }

  const deleteRule = async (ruleId: string) => {
    try {
      const resp = await fetch(`/api/admin/rules/${ruleId}`, { method: 'DELETE' })
      if (!resp.ok) {
        console.error('Delete rule failed:', resp.status, await resp.text())
      }
    } catch (e) {
      console.error('Delete rule error:', e)
    }
    load()
  }

  const providerName = (id: string) => providers.find(p => p.id === id)?.name || id

  // Get the routing mode for the key that the rule form is currently open for
  const currentKeyMode = showRuleForm ? keys.find(k => k.id === showRuleForm)?.routing_mode || 'failover' : 'failover'

  // Auto-fill model pattern when provider changes
  const [prevProviderId, setPrevProviderId] = useState('')
  useEffect(() => {
    if (ruleForm.target_provider_id && ruleForm.target_provider_id !== prevProviderId) {
      setPrevProviderId(ruleForm.target_provider_id)
      const provider = providers.find(p => p.id === ruleForm.target_provider_id)
      const options = getPatternOptions(provider, currentKeyMode)
      setRuleForm(prev => ({ ...prev, model_pattern: options[0] }))
    }
  }, [ruleForm.target_provider_id, prevProviderId, providers, currentKeyMode])

  const selectedProvider = providers.find(p => p.id === ruleForm.target_provider_id)
  const patternOptions = getPatternOptions(selectedProvider, currentKeyMode)
  const isCustomPattern = !!(ruleForm.target_provider_id && !patternOptions.includes(ruleForm.model_pattern))

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">API Keys & Routing</h1>
        <button
          onClick={() => {
            if (!showKeyForm) {
              setKeyForm({ name: '', default_provider_id: providers[0]?.id || '', routing_mode: 'failover' })
            }
            setShowKeyForm(!showKeyForm)
          }}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
        >
          Create Key
        </button>
      </div>

      {showKeyForm && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
          <div className="grid grid-cols-3 gap-4">
            <input
              placeholder="Key name"
              value={keyForm.name}
              onChange={e => setKeyForm({ ...keyForm, name: e.target.value })}
              className="bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
            />
            <select
              value={keyForm.routing_mode}
              onChange={e => setKeyForm({ ...keyForm, routing_mode: e.target.value })}
              className="bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
            >
              <option value="failover">Failover (priority-based)</option>
              <option value="content">Content (type-based routing)</option>
            </select>
            <select
              value={keyForm.default_provider_id}
              onChange={e => setKeyForm({ ...keyForm, default_provider_id: e.target.value })}
              className="bg-gray-950 border border-gray-700 rounded px-3 py-2 text-sm"
            >
              {providers.length === 0 && (
                <option value="">No providers available</option>
              )}
              {providers.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>
          <div className="mt-4 flex gap-2">
            <button
              onClick={createKey}
              disabled={providers.length === 0}
              className={`px-4 py-2 rounded text-sm text-white ${providers.length === 0 ? 'bg-gray-600 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'}`}
            >Create</button>
            <button onClick={() => setShowKeyForm(false)} className="px-4 py-2 bg-gray-700 text-white rounded text-sm hover:bg-gray-600">Cancel</button>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {keys.map(k => (
          <div key={k.id} className="bg-gray-900 border border-gray-800 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-medium">{k.name || 'Unnamed Key'}</span>
                <select
                  value={k.routing_mode || 'failover'}
                  onChange={async (e) => {
                    await fetch(`/api/admin/keys/${k.id}`, {
                      method: 'PUT',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ routing_mode: e.target.value }),
                    })
                    load()
                  }}
                  className={`text-xs px-2 py-0.5 rounded border-0 cursor-pointer ${
                    k.routing_mode === 'content' ? 'bg-blue-900 text-blue-300' :
                    'bg-gray-800 text-gray-400'
                  }`}
                >
                  <option value="failover">failover</option>
                  <option value="content">content</option>
                </select>
                <code className="text-xs bg-gray-950 px-2 py-1 rounded text-yellow-400">{k.key_value}</code>
                <button
                  onClick={() => copyKey(k.key_value)}
                  className="px-2 py-0.5 text-xs rounded bg-gray-800 hover:bg-gray-700 text-gray-300"
                >
                  {copied === k.key_value ? 'Copied!' : 'Copy'}
                </button>
                {k.default_provider_id && (
                  <span className="text-xs text-gray-500">default: {providerName(k.default_provider_id)}</span>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    if (showRuleForm === k.id) {
                      setShowRuleForm(null)
                    } else {
                      setRuleForm({ model_pattern: '', target_provider_id: '', priority: 0 })
                      setPrevProviderId('')
                      setShowRuleForm(k.id)
                    }
                  }}
                  className="px-3 py-1 text-sm rounded bg-gray-800 hover:bg-gray-700"
                >
                  Add Rule
                </button>
                <button
                  onClick={() => deleteKey(k.id)}
                  className="px-3 py-1 text-sm rounded bg-red-900 hover:bg-red-800 text-red-300"
                >
                  Delete
                </button>
              </div>
            </div>

            {showRuleForm === k.id && (
              <div className="bg-gray-950 rounded p-3 mb-3 grid grid-cols-4 gap-2 items-end">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Route to</label>
                  <select
                    value={ruleForm.target_provider_id}
                    onChange={e => setRuleForm(prev => ({ ...prev, target_provider_id: e.target.value }))}
                    className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm"
                  >
                    <option value="">Select provider</option>
                    {providers.map(p => (
                      <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    {k.routing_mode === 'content' ? 'Content type' : 'Model pattern'}
                  </label>
                  {ruleForm.target_provider_id ? (
                    <>
                      <select
                        value={isCustomPattern ? '__custom__' : ruleForm.model_pattern}
                        onChange={e => {
                          const v = e.target.value
                          if (v === '__custom__') {
                            setRuleForm(prev => ({ ...prev, model_pattern: '' }))
                          } else {
                            setRuleForm(prev => ({ ...prev, model_pattern: v }))
                          }
                        }}
                        className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm"
                      >
                        {patternOptions.map(p => (
                          <option key={p} value={p}>{p}</option>
                        ))}
                        {k.routing_mode !== 'content' && (
                          <option value="__custom__">Custom...</option>
                        )}
                      </select>
                      {isCustomPattern && k.routing_mode !== 'content' && (
                        <input
                          placeholder="e.g. gpt-4o* or my-model"
                          value={ruleForm.model_pattern}
                          onChange={e => { const v = e.target.value; setRuleForm(prev => ({ ...prev, model_pattern: v })) }}
                          autoFocus
                          className="w-full mt-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm"
                        />
                      )}
                    </>
                  ) : (
                    <input
                      placeholder="Select a provider first"
                      disabled
                      className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-600"
                    />
                  )}
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Priority (higher = first)</label>
                  <input
                    type="number"
                    value={ruleForm.priority}
                    onChange={e => { const v = parseInt(e.target.value) || 0; setRuleForm(prev => ({ ...prev, priority: v })) }}
                    className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm"
                  />
                </div>
                <button
                  onClick={() => createRule(k.id)}
                  className="px-3 py-1 text-sm rounded bg-green-600 hover:bg-green-700 text-white self-end"
                >
                  Add
                </button>
              </div>
            )}

            {rules.filter(r => r.key_id === k.id).length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-gray-500 mb-1">Routing Rules:</div>
                {rules.filter(r => r.key_id === k.id).map(r => (
                  <div key={r.id} className="flex items-center gap-2 text-sm bg-gray-950 rounded px-3 py-1">
                    <code className={r.model_pattern.startsWith('@') ? 'text-purple-400' : 'text-blue-400'}>
                      {r.model_pattern}
                    </code>
                    <span className="text-gray-600">&rarr;</span>
                    <span className="text-green-400">{providerName(r.target_provider_id)}</span>
                    <span className="text-gray-600 text-xs">priority: {r.priority}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteRule(r.id); }}
                      className="ml-auto text-red-400 hover:text-red-300 text-xs cursor-pointer"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {keys.length === 0 && (
          <p className="text-gray-500 text-center py-8">No API keys created yet.</p>
        )}
      </div>
    </div>
  )
}
