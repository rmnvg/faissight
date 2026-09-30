import { useState } from 'react'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { CompareCandidate, CompareMeasurement, CompareRequest, CompareResult, Info, SweepParam } from '../api/types'
import { Field, NumberInput } from '../components/form'
import { JobStatus } from '../components/JobStatus'
import { Banner, Card, EmptyState, StatTile } from '../components/ui'
import { fmtBytes, idList, recallIntervalsOverlap, relative, signedRecall, verdict } from '../lib/compare'
import { downloadCSV, downloadJSON } from '../lib/exportData'
import { fmtNum } from '../lib/format'
import { compareParams, runSettingsFromParams } from '../lib/jobParams'
import { intParam, navigate } from '../lib/route'

const EF_MAX = 1 << 16

/** The search knob's current value on an index, from its reported params. */
function currentValue(param: SweepParam | null, params: Record<string, number | boolean | string>): number | null {
  if (param === 'nprobe') return Number(params.nprobe ?? 1)
  if (param === 'efSearch') return Number(params.ef_search ?? 16)
  return null
}

export default function Compare({ info, params }: { info: Info; params: URLSearchParams }) {
  const candidates = info.compare
  const jobId = params.get('job')
  const [candidate, setCandidate] = useState(() => {
    const c = intParam(params, 'candidate')
    return c !== null && c >= 0 && c < candidates.length ? c : 0
  })
  const right: CompareCandidate | undefined = candidates[candidate]
  const leftParam = info.sweep?.param ?? null
  // The form starts from the URL, so a shared or reloaded link can be run again as-is.
  const [initial] = useState(() => runSettingsFromParams(params))
  const urlKnob = (key: string) => {
    const v = intParam(params, key)
    return v !== null && v >= 1 ? v : null
  }
  const [leftValue, setLeftValue] = useState(() => urlKnob('left') ?? currentValue(leftParam, info.params))
  const [rightValues, setRightValues] = useState<Record<number, number | null>>(() => {
    const r = urlKnob('right')
    return r !== null ? { [candidate]: r } : {}
  })
  const rightValue = right ? (rightValues[candidate] ?? currentValue(right.search_param, right.params)) : null
  const [k, setK] = useState(initial.k)
  const [nQueries, setNQueries] = useState(initial.nQueries)
  const [repeats, setRepeats] = useState(initial.repeats)
  const [seed, setSeed] = useState(initial.seed)

  const { job, poll, start, cancel, missing } = useJob({
    kind: 'compare',
    jobId,
    start: api.startCompare,
    fetch: api.compare,
    cancel: api.cancelCompare,
    onStarted: (j, req: CompareRequest) => navigate('compare', compareParams(j.job_id, req)),
  })

  if (candidates.length === 0) {
    return (
      <div className="mx-auto flex max-w-3xl flex-col gap-4 p-6">
        <Header />
        <Card>
          <EmptyState title="No other index to compare with">
            Start faissight with one or more <code>--compare</code> indexes built from the same vectors, for
            example an IVF-PQ and an HNSW version of this index:
            <pre className="mt-3 overflow-auto rounded-lg bg-surface-2 p-3 text-left text-xs text-ink">
              {`faissight serve ${info.name} --vectors vectors.npy \\\n  --compare ivf_pq.index --compare hnsw.index`}
            </pre>
            <p className="mt-2">
              In Python: <code>faissight.launch(index, vectors, compare=[other_index])</code>. Raw vectors are
              needed so both indexes are scored against the same exact ground truth.
            </p>
          </EmptyState>
        </Card>
      </div>
    )
  }

  const run = () => {
    if (!right) return
    const req: CompareRequest = { candidate, k, n_queries: nQueries, repeats, seed }
    if (leftParam === 'nprobe' && leftValue !== null) req.left_nprobe = leftValue
    if (leftParam === 'efSearch' && leftValue !== null) req.left_ef_search = leftValue
    if (right.search_param === 'nprobe' && rightValue !== null) req.right_nprobe = rightValue
    if (right.search_param === 'efSearch' && rightValue !== null) req.right_ef_search = rightValue
    start.mutate(req)
  }
  const result = job?.status === 'done' ? job.result : null
  const lim = info.demo_limits

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <Header />
      <Card>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <Field label="Main index" className="min-w-40">
            <div className="truncate py-2 text-sm text-ink" title={info.name}>
              {info.name} <span className="text-muted">({info.kind})</span>
            </div>
          </Field>
          {leftParam && leftValue !== null && (
            <Field label={`${leftParam} (main)`} className="w-28">
              <NumberInput
                value={leftValue}
                onChange={setLeftValue}
                min={1}
                max={leftParam === 'nprobe' ? (info.sweep?.max_value ?? 1) : (lim?.max_ef_search ?? EF_MAX)}
              />
            </Field>
          )}
          <Field label="Compare with" className="min-w-48">
            <select
              value={candidate}
              onChange={(e) => setCandidate(Number(e.target.value))}
              className="rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
            >
              {candidates.map((c) => (
                <option key={c.index} value={c.index}>
                  {c.name} ({c.kind})
                </option>
              ))}
            </select>
          </Field>
          {right?.search_param && rightValue !== null && (
            <Field label={`${right.search_param} (other)`} className="w-28">
              <NumberInput
                value={rightValue}
                onChange={(v) => setRightValues((r) => ({ ...r, [candidate]: v }))}
                min={1}
                max={right.search_param === 'nprobe' ? (right.max_value ?? 1) : (lim?.max_ef_search ?? EF_MAX)}
              />
            </Field>
          )}
          <Field label="k" className="w-20">
            <NumberInput value={k} onChange={setK} min={1} max={lim?.max_sweep_k ?? 1000} />
          </Field>
          <Field
            label={info.inputs.queries ? `queries (of ${info.inputs.queries} given)` : 'queries (sampled)'}
            className="w-40"
          >
            <NumberInput value={nQueries} onChange={setNQueries} min={1} max={lim?.max_sweep_queries ?? 10000} />
          </Field>
          <Field label="Timing repeats" className="w-28">
            <NumberInput value={repeats} onChange={setRepeats} min={1} max={lim ? 3 : 20} />
          </Field>
          <Field label="Seed" className="w-28">
            <NumberInput value={seed} onChange={setSeed} min={0} max={4294967295} />
          </Field>
          <button
            type="submit"
            disabled={start.isPending || job?.status === 'running'}
            className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {job?.status === 'running' ? 'Comparing…' : 'Compare'}
          </button>
          {job?.status === 'running' && jobId && (
            <button
              type="button"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate(jobId)}
              className="rounded-lg border border-line px-4 py-2 text-sm text-ink"
            >
              Cancel
            </button>
          )}
        </form>
        <p className="mt-2 text-xs text-muted">
          Both indexes answer the same queries against the same exact ground truth. Each query is timed on one
          FAISS thread, alternating which index goes first.
        </p>
      </Card>

      {!jobId && (
        <Card>
          <EmptyState title="Run a comparison to begin">
            Recall, latency, size and which neighbours change, for identical queries.
          </EmptyState>
        </Card>
      )}
      <JobStatus
        noun="Comparison"
        job={job}
        missing={missing}
        errors={[start.error, cancel.error, missing ? null : poll.error]}
        onRunAgain={run}
        running={start.isPending}
      />

      {result && <Results result={result} leftParam={leftParam} />}
    </div>
  )
}

function Header() {
  return (
    <header>
      <h1 className="text-xl font-semibold text-ink">Compare</h1>
      <p className="text-sm text-ink-2">
        Choose between indexes: recall, tail latency, size and changed neighbours on the same queries.
      </p>
    </header>
  )
}

function paramText(m: CompareMeasurement): string {
  const entries = Object.entries(m.params)
  return entries.length ? entries.map(([k, v]) => `${k} ${v}`).join(', ') : 'no search parameter'
}

function Results({ result, leftParam }: { result: CompareResult; leftParam: SweepParam | null }) {
  const { left, right, k } = result
  const dRecall = right.recall - left.recall
  const p95 = relative(left.latency_p95_ms, right.latency_p95_ms, ['faster', 'slower'])
  const mean = relative(left.latency_mean_ms, right.latency_mean_ms, ['faster', 'slower'])
  const size = relative(left.serialized_bytes, right.serialized_bytes, ['smaller', 'larger'])
  const noisy = Math.abs(dRecall) >= 0.0005 && recallIntervalsOverlap(left, right)
  const ci = (m: CompareMeasurement) =>
    m.recall_ci_low !== null && m.recall_ci_high !== null
      ? `${m.recall_ci_low.toFixed(3)}–${m.recall_ci_high.toFixed(3)}`
      : '—'
  const unchanged = result.n_queries - result.n_changed
  const leftValue = leftParam === 'nprobe' ? left.params.nprobe : leftParam === 'efSearch' ? left.params.efSearch : null

  const rows: [string, string, string, string][] = [
    ['Kind', left.kind, right.kind, ''],
    ['Search parameters', paramText(left), paramText(right), ''],
    [`Recall@${k}`, left.recall.toFixed(3), right.recall.toFixed(3), signedRecall(dRecall)],
    ['95% interval', ci(left), ci(right), noisy ? 'intervals overlap' : ''],
    ['Mean latency', `${left.latency_mean_ms.toFixed(3)} ms`, `${right.latency_mean_ms.toFixed(3)} ms`, mean.text],
    ['p95 latency', `${left.latency_p95_ms.toFixed(3)} ms`, `${right.latency_p95_ms.toFixed(3)} ms`, p95.text],
    ['Serialized size', fmtBytes(left.serialized_bytes), fmtBytes(right.serialized_bytes), size.text],
  ]

  return (
    <>
      <Card>
        <p className="text-sm text-ink">{verdict(left, right, k)}</p>
        <p className="mt-1 text-xs text-muted">
          {fmtNum(result.n_queries)} {result.query_origin === 'given' ? 'given' : 'sampled stored-vector'} queries
          {result.query_origin === 'sampled' && ' (each excludes itself)'} · {result.metric} · {result.repeats}{' '}
          timing repeats · seed {result.seed} · query set {result.query_sha256.slice(0, 12)}
        </p>
      </Card>
      {noisy && (
        <Banner tone="warning">
          The recall intervals overlap: with {fmtNum(result.n_queries)} queries the recall gap may be sampling noise.
          Compare more queries before deciding on recall alone.
        </Banner>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label={`Recall@${k} change`}
          value={signedRecall(dRecall)}
          tone={dRecall > 0.0005 ? 'good' : dRecall < -0.0005 ? 'bad' : 'default'}
          caption={`${left.recall.toFixed(3)} → ${right.recall.toFixed(3)}`}
        />
        <StatTile
          label="p95 latency"
          value={p95.text}
          tone={p95.better === null ? 'default' : p95.better ? 'good' : 'bad'}
          caption={`${left.latency_p95_ms.toFixed(3)} → ${right.latency_p95_ms.toFixed(3)} ms per query`}
        />
        <StatTile
          label="Serialized size"
          value={size.text}
          tone={size.better === null ? 'default' : size.better ? 'good' : 'bad'}
          caption={`${fmtBytes(left.serialized_bytes)} → ${fmtBytes(right.serialized_bytes)}`}
        />
        <StatTile
          label="Queries with different results"
          value={`${fmtNum(result.n_changed)} of ${fmtNum(result.n_queries)}`}
          caption={`${fmtNum(result.n_improved)} better · ${fmtNum(result.n_worsened)} worse · ${fmtNum(unchanged)} identical`}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-5">
        <Card
          className="xl:col-span-2"
          title="Side by side"
          subtitle="Size is faiss.serialize_index bytes, not process memory"
          actions={
            <button
              className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink print:hidden"
              onClick={() => downloadJSON('faissight-compare.json', result)}
            >
              JSON
            </button>
          }
        >
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr>
                <th className="py-1 pr-3 font-normal" />
                <th className="py-1 pr-3 font-normal break-all">{left.name}</th>
                <th className="py-1 pr-3 font-normal break-all">{right.name}</th>
                <th className="py-1 font-normal">Change</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, a, b, change]) => (
                <tr key={label} className="border-t border-line">
                  <td className="py-1.5 pr-3 text-ink-2">{label}</td>
                  <td className="tabular py-1.5 pr-3 text-ink">{a}</td>
                  <td className="tabular py-1.5 pr-3 text-ink">{b}</td>
                  <td className="py-1.5 text-xs text-ink-2">{change}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card
          className="xl:col-span-3"
          title="Changed neighbours"
          subtitle={
            result.n_changed === 0
              ? 'Both indexes returned the same neighbours for every query'
              : `Largest recall change first${result.changes_truncated ? ` (top ${result.changes.length} of ${fmtNum(result.n_changed)})` : ''}`
          }
          actions={
            result.changes.length > 0 && (
              <button
                className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink print:hidden"
                onClick={() =>
                  downloadCSV(
                    'faissight-compare-changes.csv',
                    result.changes.map((c) => ({
                      query_no: c.query_no,
                      id: c.id,
                      left_recall: c.left_recall,
                      right_recall: c.right_recall,
                      left_only: c.left_only.join(' '),
                      right_only: c.right_only.join(' '),
                      overlap: c.overlap,
                    })) as never,
                  )
                }
              >
                CSV
              </button>
            )
          }
        >
          {result.changes.length > 0 && (
            <div className="max-h-96 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface text-left text-xs text-muted">
                  <tr>
                    <th className="py-1 pr-3 font-normal">Query</th>
                    <th className="py-1 pr-3 text-right font-normal">Main</th>
                    <th className="py-1 pr-3 text-right font-normal">Other</th>
                    <th className="py-1 pr-3 font-normal">Only in main</th>
                    <th className="py-1 pr-3 font-normal">Only in other</th>
                    <th className="py-1 font-normal" />
                  </tr>
                </thead>
                <tbody>
                  {result.changes.map((c) => (
                    <tr key={c.query_no} className="border-t border-line align-top">
                      <td className="tabular py-1.5 pr-3 whitespace-nowrap text-ink">
                        {c.id !== null ? `id ${c.id}` : `query #${c.query_no}`}
                      </td>
                      <td className="tabular py-1.5 pr-3 text-right text-ink">{c.left_recall.toFixed(2)}</td>
                      <td
                        className={`tabular py-1.5 pr-3 text-right ${
                          c.right_recall < c.left_recall ? 'text-critical' : c.right_recall > c.left_recall ? 'text-good-text' : 'text-ink'
                        }`}
                      >
                        {c.right_recall.toFixed(2)}
                      </td>
                      <td className="tabular py-1.5 pr-3 text-xs text-ink-2">{idList(c.left_only)}</td>
                      <td className="tabular py-1.5 pr-3 text-xs text-ink-2">{idList(c.right_only)}</td>
                      <td className="py-1.5 text-right">
                        {c.id !== null && (
                          <button
                            title="Open this query on the main index in the Query Explorer"
                            className="rounded-md border border-line px-2 py-0.5 text-xs whitespace-nowrap text-ink-2 hover:text-ink"
                            onClick={() =>
                              navigate('query', {
                                id: String(c.id),
                                k,
                                ...(leftValue !== null && leftValue !== undefined
                                  ? { [leftParam === 'nprobe' ? 'nprobe' : 'ef']: leftValue }
                                  : {}),
                              })
                            }
                          >
                            Explain
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
