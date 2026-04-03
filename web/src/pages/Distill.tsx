import { useEffect, useState } from 'react'

interface TaskTypeStat {
  task_type: string
  total: number
  successful: number
  with_tool_calls: number
}

interface DistillStats {
  total_samples: number
  unexported_samples: number
  exported_samples: number
  by_task_type: TaskTypeStat[]
}

interface RoutingStatEntry {
  routed_to: string
  total: number
  successful: number
  fallbacks: number
  avg_latency_ms: number
}

interface LocalSuccessRate {
  task_type: string
  total: number
  successful: number
  success_rate: number
}

interface RoutingStats {
  days: number
  total_requests: number
  local_requests: number
  cloud_requests: number
  local_hit_rate: number
  by_destination: RoutingStatEntry[]
  local_success_by_type: LocalSuccessRate[]
}

interface DistillConfig {
  distillation: {
    collect_training_data: boolean
    max_samples: number
    only_successful: boolean
  }
}

export default function Distill() {
  const [stats, setStats] = useState<DistillStats | null>(null)
  const [routing, setRouting] = useState<RoutingStats | null>(null)
  const [config, setConfig] = useState<DistillConfig | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/api/admin/distill/stats').then(r => r.json()),
      fetch('/api/admin/distill/routing?days=7').then(r => r.json()),
      fetch('/api/admin/distill/config').then(r => r.json()),
    ]).then(([s, r, c]) => {
      setStats(s)
      setRouting(r)
      setConfig(c)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-gray-400">Loading...</div>
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Distillation</h1>

      {/* Data Collection Status */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Data Collection</h2>
        {config && stats ? (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Collection</span>
              <span className={config.distillation.collect_training_data ? 'text-green-400' : 'text-gray-500'}>
                {config.distillation.collect_training_data ? 'Active' : 'Inactive'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Samples</span>
              <span className="text-white font-mono">{stats.total_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Ready for Export</span>
              <span className="text-yellow-400 font-mono">{stats.unexported_samples.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Already Exported</span>
              <span className="text-green-400 font-mono">{stats.exported_samples.toLocaleString()}</span>
            </div>
          </div>
        ) : (
          <p className="text-gray-500 text-sm">No data</p>
        )}
      </div>

      {/* Routing Stats */}
      {routing && routing.total_requests > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">
            Routing Stats (Last {routing.days} days)
          </h2>

          {/* Summary bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-blue-400">
                Local: {routing.local_requests} ({(routing.local_hit_rate * 100).toFixed(1)}%)
              </span>
              <span className="text-purple-400">
                Cloud: {routing.cloud_requests} ({((1 - routing.local_hit_rate) * 100).toFixed(1)}%)
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
              <div
                className="bg-blue-500 h-full rounded-full transition-all"
                style={{ width: `${routing.local_hit_rate * 100}%` }}
              />
            </div>
          </div>

          {/* By destination */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            {routing.by_destination.map(d => (
              <div key={d.routed_to} className="bg-gray-800/50 rounded p-3">
                <div className="text-xs text-gray-400 mb-1 uppercase">{d.routed_to}</div>
                <div className="text-lg font-mono">{d.total}</div>
                <div className="text-xs text-gray-400">
                  {d.successful} successful · {d.fallbacks} fallbacks · {d.avg_latency_ms.toFixed(0)}ms avg
                </div>
              </div>
            ))}
          </div>

          {/* Local success by task type */}
          {routing.local_success_by_type.length > 0 && (
            <div>
              <h3 className="text-xs text-gray-400 mb-2 uppercase">Local Success by Task Type</h3>
              <div className="space-y-2">
                {routing.local_success_by_type.map(r => (
                  <div key={r.task_type} className="flex items-center gap-3">
                    <span className="text-sm w-24 text-gray-300">{r.task_type}</span>
                    <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${r.success_rate >= 0.7 ? 'bg-green-500' : r.success_rate >= 0.4 ? 'bg-yellow-500' : 'bg-red-500'}`}
                        style={{ width: `${r.success_rate * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-16 text-right">
                      {(r.success_rate * 100).toFixed(1)}% ({r.total})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Samples by Type */}
      {stats && stats.by_task_type.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
          <h2 className="text-sm font-medium text-gray-400 mb-3">Training Samples by Task Type</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-xs uppercase">
                <th className="text-left py-2">Type</th>
                <th className="text-right py-2">Total</th>
                <th className="text-right py-2">Successful</th>
                <th className="text-right py-2">With Tools</th>
                <th className="text-right py-2">Rate</th>
              </tr>
            </thead>
            <tbody>
              {stats.by_task_type.map(t => (
                <tr key={t.task_type} className="border-t border-gray-800">
                  <td className="py-2 text-gray-200">{t.task_type}</td>
                  <td className="py-2 text-right font-mono">{t.total}</td>
                  <td className="py-2 text-right font-mono text-green-400">{t.successful}</td>
                  <td className="py-2 text-right font-mono text-blue-400">{t.with_tool_calls}</td>
                  <td className="py-2 text-right font-mono">
                    {t.total > 0 ? ((t.successful / t.total) * 100).toFixed(1) : 0}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Quick Start Guide */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h2 className="text-sm font-medium text-gray-400 mb-3">Quick Start</h2>
        <div className="text-sm text-gray-300 space-y-3">
          <div>
            <span className="text-blue-400 font-mono">1.</span> Collect data: Use Louter normally — cloud responses are automatically saved as training samples.
          </div>
          <div>
            <span className="text-blue-400 font-mono">2.</span> Export & train:
            <pre className="mt-1 bg-gray-800 rounded p-2 text-xs overflow-x-auto">
{`cd distill
./run_distill.sh`}
            </pre>
          </div>
          <div>
            <span className="text-blue-400 font-mono">3.</span> Iterate: As you use the system, more data is collected. Re-run distillation periodically to improve the local model.
          </div>
        </div>
      </div>
    </div>
  )
}
