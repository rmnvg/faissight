import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../api/client'
import type { HistoryComparison, HistorySummary } from '../api/types'
import { Banner, Card, EmptyState, Spinner } from '../components/ui'
import { downloadJSON } from '../lib/exportData'

const BUTTON =
  'rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink-2 hover:text-ink disabled:opacity-50'
const KIND_LABELS: Record<string, string> = {
  sweep: 'Sweep',
  comparison: 'Comparison',
  evaluation: 'Relevance',
}

function when(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

function HistoryRow({ run, onChanged }: { run: HistorySummary; onChanged: () => void }) {
  const [label, setLabel] = useState(run.label)
  const rename = useMutation({ mutationFn: () => api.renameRun(run.id, label.trim()), onSuccess: onChanged })
  const remove = useMutation({ mutationFn: () => api.deleteRun(run.id), onSuccess: onChanged })
  const download = useMutation({
    mutationFn: () => api.historyRecord(run.id),
    onSuccess: (record) => downloadJSON(`faissight-${run.kind}-${run.id.slice(0, 8)}.json`, record.data),
  })
  const error = rename.error ?? remove.error ?? download.error
  return (
    <li className="border-t border-line py-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="w-20 shrink-0 rounded bg-surface-2 px-1.5 py-0.5 text-center text-[11px] text-ink">
          {KIND_LABELS[run.kind] ?? run.kind}
        </span>
        <input
          aria-label="Run name"
          value={label}
          maxLength={200}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && label.trim() && label.trim() !== run.label) rename.mutate()
          }}
          className="min-w-48 flex-1 rounded-md border border-line bg-page px-2 py-1 text-sm text-ink"
        />
        {label.trim() !== run.label && (
          <button
            type="button"
            className={BUTTON}
            disabled={rename.isPending || !label.trim()}
            onClick={() => rename.mutate()}
          >
            Rename
          </button>
        )}
        <span className="text-xs text-muted">{when(run.created_at)}</span>
        <button type="button" className={BUTTON} disabled={download.isPending} onClick={() => download.mutate()}>
          Download
        </button>
        <button
          type="button"
          className={BUTTON}
          disabled={remove.isPending}
          onClick={() => {
            if (window.confirm(`Delete "${run.label}"? This can't be undone.`)) remove.mutate()
          }}
        >
          Delete
        </button>
      </div>
      {error && (
        <div className="mt-2">
          <Banner tone="error">{error.message}</Banner>
        </div>
      )}
    </li>
  )
}

function ComparisonTable({ data }: { data: HistoryComparison }) {
  return (
    <div className="mt-3 flex flex-col gap-2">
      <Banner tone={data.regressed ? 'error' : data.comparable ? 'info' : 'warning'}>
        {data.regressed
          ? 'Regression: recall dropped or p95 latency grew beyond the thresholds.'
          : data.comparable
            ? 'No regressions at the shared settings.'
            : 'These runs are not comparable.'}
      </Banner>
      {data.notes.map((note) => (
        <p key={note} className="text-sm text-ink-2">
          {note}
        </p>
      ))}
      {data.points.length > 0 && (
        <table className="w-full text-sm">
          <caption className="sr-only">Saved sweep comparison per setting</caption>
          <thead className="text-left text-xs text-muted">
            <tr>
              <th scope="col" className="py-1 pr-3 font-normal">
                Setting
              </th>
              <th scope="col" className="py-1 pr-3 text-right font-normal">
                Recall (baseline → current)
              </th>
              <th scope="col" className="py-1 text-right font-normal">
                p95 ms (baseline → current)
              </th>
            </tr>
          </thead>
          <tbody>
            {data.points.map((p) => (
              <tr key={p.value} className="border-t border-line">
                <th scope="row" className="tabular py-1.5 pr-3 text-left font-normal text-ink">
                  {p.value}
                </th>
                <td className={`tabular py-1.5 pr-3 text-right ${p.recall_regressed ? 'text-critical' : 'text-ink'}`}>
                  {p.baseline_recall.toFixed(3)} → {p.recall.toFixed(3)}
                </td>
                <td className={`tabular py-1.5 text-right ${p.latency_regressed ? 'text-critical' : 'text-ink'}`}>
                  {p.baseline_p95_ms.toFixed(3)} → {p.p95_ms.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function History() {
  const client = useQueryClient()
  const history = useQuery({
    queryKey: ['history'],
    queryFn: ({ signal }) => api.history(signal),
    refetchInterval: 5000,
  })
  const [baseline, setBaseline] = useState('')
  const [current, setCurrent] = useState('')
  const compare = useMutation({ mutationFn: () => api.compareHistory(baseline, current) })
  const refresh = () => void client.invalidateQueries({ queryKey: ['history'] })

  const data = history.data
  const runs = data?.runs ?? []
  const here = runs.filter((r) => r.index === data?.current_index)
  const elsewhere = runs.filter((r) => r.index !== data?.current_index)
  const sweeps = runs.filter((r) => r.kind === 'sweep')

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-ink">Run history</h1>
        <p className="mt-1 text-sm text-ink-2">
          Finished sweeps, index comparisons and relevance evaluations are saved on this machine and survive
          restarts. The newest 100 are kept; download any you need for longer.
        </p>
      </header>
      {history.isPending && <Spinner label="Loading saved runs…" />}
      {history.error && <Banner tone="error">{history.error.message}</Banner>}
      {data && !data.enabled && (
        <Banner>History is off for this session (read-only demo, or the disk cache is disabled).</Banner>
      )}
      {data?.warning && <Banner tone="warning">{data.warning}</Banner>}
      {data?.enabled && runs.length === 0 && (
        <Card>
          <EmptyState title="No saved runs yet">
            Run a sweep in the Tuner, a comparison, or a relevance evaluation; it appears here when it finishes.
          </EmptyState>
        </Card>
      )}

      {sweeps.length > 1 && (
        <Card
          title="Compare two saved sweeps"
          subtitle="Checks that queries and ground truth match, then flags recall drops and p95 growth per setting"
        >
          <div className="flex flex-wrap items-end gap-3">
            {(
              [
                ['Baseline', baseline, setBaseline],
                ['Current', current, setCurrent],
              ] as const
            ).map(([label, value, setValue]) => (
              <label key={label} className="flex flex-col gap-1 text-xs text-ink-2">
                {label}
                <select
                  value={value}
                  onChange={(e) => {
                    setValue(e.target.value)
                    compare.reset()
                  }}
                  className="rounded-md border border-line bg-page px-2 py-1.5 text-sm text-ink"
                >
                  <option value="">Choose a sweep</option>
                  {sweeps.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label} · {when(r.created_at)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            <button
              type="button"
              className="rounded-md border border-line bg-surface px-3 py-1.5 text-sm text-ink hover:bg-accent-wash disabled:opacity-50"
              disabled={!baseline || !current || baseline === current || compare.isPending}
              onClick={() => compare.mutate()}
            >
              Compare
            </button>
          </div>
          {compare.error && (
            <div className="mt-3">
              <Banner tone="error">{compare.error.message}</Banner>
            </div>
          )}
          {compare.data && <ComparisonTable data={compare.data} />}
        </Card>
      )}

      {here.length > 0 && (
        <Card title="This index">
          <ul>
            {here.map((run) => (
              <HistoryRow key={run.id} run={run} onChanged={refresh} />
            ))}
          </ul>
        </Card>
      )}
      {elsewhere.length > 0 && (
        <Card title="Other indexes" subtitle="Saved while a different index file was open">
          <ul>
            {elsewhere.map((run) => (
              <HistoryRow key={run.id} run={run} onChanged={refresh} />
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}
