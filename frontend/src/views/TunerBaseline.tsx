import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'
import type { RunDecision, SweepParam } from '../api/types'
import { Banner, Card } from '../components/ui'
import { downloadJSON } from '../lib/exportData'

const BUTTON =
  'rounded-md border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink hover:bg-accent-wash disabled:opacity-50'
const INPUT =
  'tabular w-20 rounded-lg border border-line bg-page px-2 py-1 text-sm text-ink outline-none focus:border-series-1'

/** "Save run": the server's record of this sweep and decision, as a JSON download. */
export function SaveRunButton({ jobId, param, decision }: { jobId: string; param: SweepParam; decision: RunDecision }) {
  const save = useMutation({
    mutationFn: () => api.sweepRun(jobId, decision),
    onSuccess: (run) => downloadJSON(`faissight-run-${param}-${new Date().toISOString().slice(0, 10)}.json`, run),
  })
  return (
    <button
      type="button"
      className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink disabled:opacity-50"
      disabled={save.isPending}
      title={save.error?.message ?? 'Settings, measurements, index and query identity, to compare with later'}
      onClick={() => save.mutate()}
    >
      Save run
    </button>
  )
}

/** Compare this sweep with a run saved earlier, setting by setting, with regression limits. */
export function BaselineCompare({ jobId, param, decision }: { jobId: string; param: SweepParam; decision: RunDecision }) {
  const [fileName, setFileName] = useState<string | null>(null)
  const [recallDrop, setRecallDrop] = useState(0.01)
  const [p95Increase, setP95Increase] = useState(20)
  const [p95Floor, setP95Floor] = useState(0.05)
  const [readError, setReadError] = useState<string | null>(null)
  const compare = useMutation({
    mutationFn: (baseline: unknown) =>
      api.sweepBaseline(jobId, {
        ...decision,
        baseline,
        max_recall_drop: recallDrop,
        max_p95_increase: p95Increase / 100,
        min_p95_increase_ms: p95Floor,
      }),
  })
  const [baseline, setBaseline] = useState<unknown>(null)
  // Snapshot of the thresholds the shown result was actually compared with: if the inputs
  // below have since changed, the result on screen no longer reflects them.
  const [comparedWith, setComparedWith] = useState<{ recallDrop: number; p95Increase: number; p95Floor: number } | null>(null)
  const r = compare.data
  const stale =
    r !== undefined &&
    comparedWith !== null &&
    (comparedWith.recallDrop !== recallDrop ||
      comparedWith.p95Increase !== p95Increase ||
      comparedWith.p95Floor !== p95Floor)

  const runCompare = (data: unknown) => {
    setComparedWith({ recallDrop, p95Increase, p95Floor })
    compare.mutate(data)
  }

  const onFile = async (file: File | undefined) => {
    if (!file) return
    // Clear whatever's on screen before the new file is even read: a bad file, or one
    // that's still loading, must never leave an earlier file's verdict looking current.
    compare.reset()
    setBaseline(null)
    setComparedWith(null)
    setFileName(file.name)
    setReadError(null)
    try {
      const data: unknown = JSON.parse(await file.text())
      setBaseline(data)
      runCompare(data)
    } catch {
      setReadError(`${file.name} isn't JSON. Use a file saved with "Save run" or sweep --save.`)
    }
  }

  return (
    <Card
      title="Compare with a saved run"
      subtitle="Load a run saved earlier (Save run, or faissight sweep --save) to see what changed since"
    >
      <div className="flex flex-wrap items-end gap-3 text-xs text-ink-2">
        <label className={`${BUTTON} cursor-pointer`}>
          {fileName ? 'Load another run…' : 'Load a saved run…'}
          <input
            type="file"
            accept="application/json,.json"
            className="sr-only"
            aria-label="Saved run file"
            onChange={(e) => onFile(e.target.files?.[0])}
          />
        </label>
        {fileName && <span className="pb-1 text-ink">{fileName}</span>}
        <label className="ml-auto flex flex-col gap-1">
          Recall drop allowed
          <input
            type="number"
            min={0}
            max={1}
            step={0.005}
            value={recallDrop}
            onChange={(e) => setRecallDrop(Math.max(0, Number(e.target.value) || 0))}
            className={INPUT}
          />
        </label>
        <label className="flex flex-col gap-1">
          p95 growth allowed (%)
          <input
            type="number"
            min={0}
            step={5}
            value={p95Increase}
            onChange={(e) => setP95Increase(Math.max(0, Number(e.target.value) || 0))}
            className={INPUT}
          />
        </label>
        <label className="flex flex-col gap-1" title="Sub-millisecond timings vary between runs">
          and at least (ms)
          <input
            type="number"
            min={0}
            step={0.01}
            value={p95Floor}
            onChange={(e) => setP95Floor(Math.max(0, Number(e.target.value) || 0))}
            className={INPUT}
          />
        </label>
        <button type="button" className={BUTTON} disabled={baseline === null || compare.isPending} onClick={() => runCompare(baseline)}>
          Compare again
        </button>
      </div>

      {readError && <p className="mt-3 text-sm text-critical">{readError}</p>}
      {compare.error && <p className="mt-3 text-sm text-critical">{compare.error.message}</p>}
      {r && (
        <div className={`mt-3 flex flex-col gap-3 ${compare.isPending ? 'opacity-60' : ''}`}>
          {stale && (
            <Banner tone="warning">
              The thresholds below changed since this comparison ran. Click "Compare again" to check against them.
            </Banner>
          )}
          <Banner tone={r.regressed ? 'error' : r.comparable ? 'info' : 'warning'}>
            {r.regressed
              ? `Regressed at ${r.points.filter((p) => p.recall_regressed || p.latency_regressed).length} of ${r.points.length} settings`
              : r.comparable
                ? 'No regressions against this run'
                : "Nothing was compared — see why below, this isn't a pass"}
            {r.baseline_label || r.baseline_created_at
              ? ` (baseline ${[r.baseline_label, r.baseline_created_at?.slice(0, 16).replace('T', ' ')].filter(Boolean).join(', ')})`
              : ''}
            .
            {r.baseline_recommended !== r.recommended &&
              ` Recommended ${param}: ${r.baseline_recommended ?? 'none'} then, ${r.recommended ?? 'none'} now.`}
          </Banner>
          {r.notes.length > 0 && (
            <ul className="list-disc pl-5 text-sm text-ink-2">
              {r.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
          {r.points.length > 0 && (
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr>
                  <th className="py-1 pr-3 font-normal">{param}</th>
                  <th className="py-1 pr-3 text-right font-normal">Recall (then → now)</th>
                  <th className="py-1 pr-3 text-right font-normal">p95 ms (then → now)</th>
                  <th className="py-1 font-normal" />
                </tr>
              </thead>
              <tbody>
                {r.points.map((p) => (
                  <tr key={p.value} className="border-t border-line">
                    <td className="tabular py-1.5 pr-3 text-ink">{p.value}</td>
                    <td className={`tabular py-1.5 pr-3 text-right ${p.recall_regressed ? 'text-critical' : 'text-ink'}`}>
                      {p.baseline_recall.toFixed(3)} → {p.recall.toFixed(3)}
                    </td>
                    <td className={`tabular py-1.5 pr-3 text-right ${p.latency_regressed ? 'text-critical' : 'text-ink-2'}`}>
                      {p.baseline_p95_ms.toPrecision(3)} → {p.p95_ms.toPrecision(3)}
                      {r.latency_comparable && ` (${p.p95_change >= 0 ? '+' : ''}${(p.p95_change * 100).toFixed(0)}%)`}
                    </td>
                    <td className="py-1.5 text-xs">
                      {p.recall_regressed || p.latency_regressed ? (
                        <span className="text-critical">
                          regressed: {[p.recall_regressed && 'recall', p.latency_regressed && 'p95'].filter(Boolean).join(', ')}
                        </span>
                      ) : (
                        <span className="text-ink-2">ok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </Card>
  )
}
