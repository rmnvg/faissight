import { useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { api } from '../api/client'
import { useJob } from '../api/jobs'
import type { Info, SweepParam, SweepPoint, SweepRequest, SweepResult } from '../api/types'
import { Field, NumberInput } from '../components/form'
import { JobStatus } from '../components/JobStatus'
import { Banner, Card, EmptyState, Segmented, StatTile } from '../components/ui'
import { downloadCSV, downloadJSON } from '../lib/exportData'
import { runSettingsFromParams, sweepParams, sweepValuesText } from '../lib/jobParams'
import { navigate } from '../lib/route'
import {
  codeSnippet,
  fastest,
  fractionBelow,
  meets,
  parseValues,
  prefersLogAxis,
  recommend,
  speedup,
} from '../lib/tuner'
import { RecallBreakdown, WorstQueries } from './TunerDiagnostics'

const AXIS = { fill: 'var(--muted)', fontSize: 11 }

export default function Tuner({ info, params }: { info: Info; params: URLSearchParams }) {
  const defaults = info.sweep
  // The form starts from the URL, so a shared or reloaded link can be run again as-is.
  const [initial] = useState(() => runSettingsFromParams(params))
  const [valuesText, setValuesText] = useState(sweepValuesText(params) ?? defaults?.values.join(', ') ?? '')
  const [k, setK] = useState(initial.k)
  const [nQueries, setNQueries] = useState(initial.nQueries)
  const [repeats, setRepeats] = useState(initial.repeats)
  const [seed, setSeed] = useState(initial.seed)
  const [target, setTarget] = useState(0.95)
  const [formError, setFormError] = useState<string | null>(null)
  const jobId = params.get('job')

  const { job, poll, start, cancel, missing } = useJob({
    kind: 'sweep',
    jobId,
    start: api.startSweep,
    fetch: api.sweep,
    cancel: api.cancelSweep,
    onStarted: (j, req: SweepRequest) => navigate('tuner', sweepParams(j.job_id, req)),
  })

  if (!defaults) {
    return (
      <div className="p-6">
        <EmptyState title="Nothing to tune">
          Only IVF (nprobe) and HNSW (efSearch) indexes have a speed/recall knob.
        </EmptyState>
      </div>
    )
  }
  const param: SweepParam = defaults.param
  const givenQueries = info.inputs.queries

  const run = () => {
    const values = parseValues(valuesText, defaults.max_value)
    if (!values) {
      setFormError(
        `Enter positive integers${defaults.max_value ? ` up to ${defaults.max_value}` : ''}, separated by commas.`,
      )
      return
    }
    setFormError(null)
    start.mutate({ param, values, k, n_queries: nQueries, repeats, seed })
  }

  const result = job?.status === 'done' ? job.result : null

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Tuner</h1>
        <p className="text-sm text-ink-2">
          Sweep <code>{param}</code> over a query set and pick the cheapest setting that reaches your
          target recall.
        </p>
      </header>

      <Card>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(e) => {
            e.preventDefault()
            run()
          }}
        >
          <Field label={`${param} values`} className="min-w-64 flex-1">
            <input
              value={valuesText}
              onChange={(e) => setValuesText(e.target.value)}
              className="tabular rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
            />
          </Field>
          <Field label="k" className="w-24">
            <NumberInput value={k} onChange={setK} min={1} max={info.demo_limits?.max_sweep_k ?? 1000} />
          </Field>
          <Field
            label={givenQueries ? `queries (of ${givenQueries} given)` : 'queries (sampled)'}
            className="w-44"
          >
            <NumberInput
              value={nQueries}
              onChange={setNQueries}
              min={1}
              max={info.demo_limits?.max_sweep_queries ?? 10000}
            />
          </Field>
          <Field label="Timing repeats" className="w-28">
            <NumberInput value={repeats} onChange={setRepeats} min={1} max={info.demo_limits ? 3 : 20} />
          </Field>
          <Field label="Seed" className="w-28">
            <NumberInput value={seed} onChange={setSeed} min={0} max={4294967295} />
          </Field>
          <button
            type="submit"
            disabled={start.isPending || job?.status === 'running'}
            className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {job?.status === 'running' ? 'Sweeping…' : 'Run sweep'}
          </button>
          {job?.status === 'running' && jobId && (
            <button type="button" disabled={cancel.isPending} onClick={() => cancel.mutate(jobId)}
              className="rounded-lg border border-line px-4 py-2 text-sm text-ink">
              Cancel sweep
            </button>
          )}
        </form>
        {formError && <p className="mt-2 text-sm text-critical">{formError}</p>}
        <p className="mt-2 text-xs text-muted">
          Each query runs repeatedly on one FAISS thread after warming up each setting. Latencies are
          comparable across values (not your production throughput).
        </p>
      </Card>

      {!jobId && (
        <Card>
          <EmptyState title="Run a sweep to begin">
            The defaults cover {defaults.values[0]}–{defaults.values[defaults.values.length - 1]}.
            Recall is measured against exact ground truth.
          </EmptyState>
        </Card>
      )}
      <JobStatus
        noun="Sweep"
        job={job}
        missing={missing}
        errors={[start.error, cancel.error, missing ? null : poll.error]}
        onRunAgain={run}
        running={start.isPending}
      />

      {result && <p className="text-xs text-muted">
        {result.repeats} timing repeats · seed {result.seed} · query set {result.query_sha256.slice(0, 12)}.
        JSON exports include measurement settings and environment versions.
      </p>}
      {result && (
        <Results result={result} target={target} setTarget={setTarget} hasRefine={info.has_refine} />
      )}
    </div>
  )
}

function Results({
  result,
  target,
  setTarget,
  hasRefine,
}: {
  result: SweepResult
  target: number
  setTarget: (t: number) => void
  hasRefine: boolean
}) {
  const [confident, setConfident] = useState(false)
  const [focusValue, setFocusValue] = useState<number | null>(null)
  const pts = result.points
  const rec = recommend(pts, target, confident)
  const quickest = fastest(pts, target, confident)
  const fast = rec ? speedup(pts, rec) : null
  const best = pts.reduce((a, b) => (b.recall > a.recall ? b : a))
  const pareto = new Set(result.pareto_values)
  const values = pts.map((p) => p.value)
  const logX = prefersLogAxis(values)
  const snippet = rec ? codeSnippet(result.param, rec.value, hasRefine) : null
  const recallFloor = Math.max(
    0,
    Math.min(Math.floor(Math.min(...pts.map((p) => p.recall_ci_low ?? p.recall)) * 10) / 10, target - 0.05),
  )
  // Recharts draws a band from a [low, high] pair.
  const chartPts = pts.map((p) => ({
    ...p,
    ci: p.recall_ci_low !== null && p.recall_ci_high !== null ? [p.recall_ci_low, p.recall_ci_high] : null,
  }))
  const focus = pts.find((p) => p.value === focusValue) ?? rec ?? best
  const hasDiagnostics = pts.some((p) => p.recall_distribution.length > 0)

  return (
    <>
      {result.truth_source === 'reconstructed' && (
        <Banner tone="warning">
          Ground truth computed on reconstructed vectors; PQ/SQ error is not measured.
        </Banner>
      )}

      <Card>
        <label className="flex flex-wrap items-center gap-3 text-sm text-ink-2">
          Target recall@{result.k}
          <input
            type="range"
            min={0.5}
            max={1}
            step={0.01}
            value={target}
            onChange={(e) => setTarget(Number(e.target.value))}
            className="w-64 accent-series-1"
          />
          <span className="tabular font-medium text-ink">{target.toFixed(2)}</span>
          <span className="text-xs text-muted">
            {result.n_queries} {result.query_origin === 'given' ? 'given' : 'sampled stored-vector'}{' '}
            queries
            {result.query_origin === 'sampled' && ' (each excludes itself)'}
          </span>
        </label>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-ink-2">
          Meets the target when
          <Segmented
            label="Recommendation rule"
            value={confident ? 'confident' : 'mean'}
            onChange={(v) => setConfident(v === 'confident')}
            options={[
              { value: 'mean', label: 'mean recall ≥ target' },
              {
                value: 'confident',
                label: '95% lower bound ≥ target',
                title: 'Guards against a small query set overstating recall',
              },
            ]}
          />
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label={`Recommended ${result.param}`}
          value={rec ? rec.value : 'Not reached'}
          tone={rec ? 'good' : 'bad'}
          caption={
            rec
              ? `smallest value ${confident ? 'confidently ' : ''}meeting the target`
              : `best was ${best.recall.toFixed(3)} at ${best.value}; try larger values`
          }
        />
        <StatTile
          label={`Recall@${result.k} there`}
          value={rec ? rec.recall.toFixed(3) : '—'}
          caption={
            rec
              ? rec.recall_ci_low !== null && rec.recall_ci_high !== null
                ? `95% interval ${rec.recall_ci_low.toFixed(3)}–${rec.recall_ci_high.toFixed(3)}`
                : `target ${target.toFixed(2)}`
              : undefined
          }
        />
        <StatTile
          label="Latency there"
          value={rec ? `${rec.latency_mean_ms.toFixed(3)} ms` : '—'}
          caption={rec ? `p95 ${rec.latency_p95_ms.toFixed(3)} ms per query` : undefined}
        />
        <StatTile
          label="Speed-up"
          value={fast ? `${fast.toFixed(1)}×` : '—'}
          caption={fast ? `vs ${result.param} ${Math.max(...values)}` : 'already the largest value'}
        />
      </div>
      {rec && quickest && quickest.value !== rec.value && (
        <Banner>
          Fastest measured setting that meets the target: <strong>{result.param} {quickest.value}</strong> at{' '}
          {quickest.latency_mean_ms.toPrecision(3)} ms, vs {rec.latency_mean_ms.toPrecision(3)} ms for{' '}
          {rec.value} ({((1 - quickest.latency_mean_ms / rec.latency_mean_ms) * 100).toFixed(0)}% faster). A larger
          value measuring faster is usually timing noise; rerun with more timing repeats before preferring it.
        </Banner>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card
          title={`Recall@${result.k} by ${result.param}`}
          subtitle="Mean over the query set; band = approximate 95% interval"
        >
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartPts} margin={{ top: 12, right: 12, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis
                  dataKey="value"
                  type="number"
                  scale={logX ? 'log' : 'linear'}
                  domain={['dataMin', 'dataMax']}
                  ticks={values}
                  tick={AXIS}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--axis)' }}
                  label={{ value: result.param, position: 'insideBottom', offset: -12, ...AXIS }}
                />
                <YAxis
                  domain={[recallFloor, 1]}
                  tick={AXIS}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <Tooltip content={<PointTooltip param={result.param} />} />
                <ReferenceLine
                  y={target}
                  stroke="var(--ink-2)"
                  label={{ value: `target ${target.toFixed(2)}`, position: 'insideBottomRight', ...AXIS }}
                />
                {rec && <ReferenceLine x={rec.value} stroke="var(--good)" />}
                <Area
                  dataKey="ci"
                  stroke="none"
                  fill="var(--series-1)"
                  fillOpacity={0.15}
                  isAnimationActive={false}
                  activeDot={false}
                />
                <Line
                  dataKey="recall"
                  stroke="var(--series-1)"
                  strokeWidth={2}
                  dot={{ r: 4, fill: 'var(--series-1)', stroke: 'var(--surface)', strokeWidth: 2 }}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title={`Latency by ${result.param}`} subtitle="Per query, single-threaded">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pts} margin={{ top: 12, right: 12, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis
                  dataKey="value"
                  type="number"
                  scale={logX ? 'log' : 'linear'}
                  domain={['dataMin', 'dataMax']}
                  ticks={values}
                  tick={AXIS}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--axis)' }}
                  label={{ value: result.param, position: 'insideBottom', offset: -12, ...AXIS }}
                />
                <YAxis
                  tick={AXIS}
                  tickLine={false}
                  axisLine={false}
                  width={48}
                  tickFormatter={(v: number) => `${Number(v.toPrecision(2))}`}
                  label={{ value: 'ms', angle: -90, position: 'insideLeft', ...AXIS }}
                />
                <Tooltip content={<PointTooltip param={result.param} />} />
                <Legend
                  verticalAlign="top"
                  height={24}
                  iconType="plainline"
                  wrapperStyle={{ fontSize: 11, color: 'var(--ink-2)' }}
                />
                {rec && <ReferenceLine x={rec.value} stroke="var(--good)" />}
                <Line
                  name="mean"
                  dataKey="latency_mean_ms"
                  stroke="var(--series-1)"
                  strokeWidth={2}
                  dot={{ r: 4, fill: 'var(--series-1)', stroke: 'var(--surface)', strokeWidth: 2 }}
                  isAnimationActive={false}
                />
                <Line
                  name="p95"
                  dataKey="latency_p95_ms"
                  stroke="var(--series-2)"
                  strokeWidth={2}
                  dot={{ r: 4, fill: 'var(--series-2)', stroke: 'var(--surface)', strokeWidth: 2 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Recall vs latency" subtitle="Up and to the left is better; ringed = Pareto-optimal">
          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 16, bottom: 20, left: 0 }}>
                <CartesianGrid stroke="var(--grid)" />
                <XAxis
                  dataKey="latency_mean_ms"
                  type="number"
                  name="mean latency"
                  scale={prefersLogAxis(pts.map((p) => p.latency_mean_ms)) ? 'log' : 'linear'}
                  domain={['dataMin', 'dataMax']}
                  tick={AXIS}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--axis)' }}
                  tickFormatter={(v: number) => `${Number(v.toPrecision(2))}`}
                  label={{ value: 'mean latency (ms)', position: 'insideBottom', offset: -12, ...AXIS }}
                />
                <YAxis
                  dataKey="recall"
                  type="number"
                  domain={[recallFloor, 1]}
                  tick={AXIS}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <ZAxis range={[60, 60]} />
                <Tooltip content={<PointTooltip param={result.param} />} />
                <ReferenceLine y={target} stroke="var(--ink-2)" />
                <Scatter
                  data={pts}
                  line={{ stroke: 'var(--series-1)', strokeWidth: 2 }}
                  isAnimationActive={false}
                  shape={(props: { cx?: number; cy?: number; payload?: SweepPoint }) => {
                    const p = props.payload
                    if (props.cx === undefined || props.cy === undefined || !p) return <g />
                    const isRec = rec?.value === p.value
                    return (
                      <g>
                        {pareto.has(p.value) && (
                          <circle cx={props.cx} cy={props.cy} r={8} fill="none" stroke="var(--series-1)" strokeWidth={1.5} />
                        )}
                        <circle
                          cx={props.cx}
                          cy={props.cy}
                          r={4}
                          fill={isRec ? 'var(--good)' : 'var(--series-1)'}
                          stroke="var(--surface)"
                          strokeWidth={2}
                        />
                        {(isRec || p.value === values[0] || p.value === values[values.length - 1]) && (
                          <text
                            // The rightmost point's label would clip at the plot edge: anchor it left.
                            x={p.value === values[values.length - 1] ? props.cx - 10 : props.cx + 10}
                            // Near the top of the plot, put the label under the point instead.
                            y={props.cy < 30 ? props.cy + 18 : props.cy - 10}
                            textAnchor={p.value === values[values.length - 1] ? 'end' : 'start'}
                            fontSize={11}
                            fill="var(--ink-2)"
                          >
                            {result.param} {p.value}
                          </text>
                        )}
                      </g>
                    )
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {hasDiagnostics && (
        <div className="grid gap-4 lg:grid-cols-2">
          <RecallBreakdown point={focus} param={result.param} k={result.k} target={target} />
          <WorstQueries point={focus} param={result.param} k={result.k} target={target} />
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-5">
        <Card
          className="xl:col-span-3"
          title="All measurements"
          subtitle={hasDiagnostics ? 'Click a row to see its per-query recall and worst queries' : undefined}
          actions={
            <div className="flex gap-2 print:hidden">
              <button
                className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink"
                onClick={() =>
                  downloadCSV(
                    `faissight-sweep-${result.param}.csv`,
                    pts.map((p) => ({
                      value: p.value,
                      recall: p.recall,
                      recall_ci_low: p.recall_ci_low,
                      recall_ci_high: p.recall_ci_high,
                      fraction_below_target: fractionBelow(p, target),
                      latency_mean_ms: p.latency_mean_ms,
                      latency_p95_ms: p.latency_p95_ms,
                    })) as never,
                  )
                }
              >
                CSV
              </button>
              <button
                className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink"
                onClick={() => downloadJSON(`faissight-sweep-${result.param}.json`, {
                    ...result,
                    target_recall: target,
                    recommendation_rule: confident ? 'ci_low' : 'mean',
                    recommended: rec?.value ?? null,
                    fastest_meeting_target: quickest?.value ?? null,
                  })}
              >
                JSON
              </button>
            </div>
          }
        >
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr>
                <th className="py-1 pr-3 font-normal">{result.param}</th>
                <th className="py-1 pr-3 text-right font-normal">Recall@{result.k}</th>
                <th className="py-1 pr-3 text-right font-normal whitespace-nowrap">95% interval</th>
                <th className="py-1 pr-3 text-right font-normal">Queries &lt; target</th>
                <th className="py-1 pr-3 text-right font-normal">Mean ms</th>
                <th className="py-1 pr-3 text-right font-normal">p95 ms</th>
                <th className="py-1 font-normal">Notes</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p) => {
                const below = fractionBelow(p, target)
                return (
                  <tr
                    key={p.value}
                    onClick={hasDiagnostics ? () => setFocusValue(p.value) : undefined}
                    title={hasDiagnostics ? 'Show per-query recall for this value' : undefined}
                    className={`border-t border-line ${hasDiagnostics ? 'cursor-pointer hover:bg-surface-2' : ''} ${
                      rec?.value === p.value ? 'bg-accent-wash' : ''
                    } ${focus.value === p.value && hasDiagnostics ? 'outline outline-1 -outline-offset-1 outline-series-1' : ''}`}
                  >
                    <td className="tabular py-1.5 pr-3 text-ink">{p.value}</td>
                    <td className="tabular py-1.5 pr-3 text-right text-ink">{p.recall.toFixed(3)}</td>
                    <td className="tabular py-1.5 pr-3 text-right whitespace-nowrap text-ink-2">
                      {p.recall_ci_low !== null && p.recall_ci_high !== null
                        ? `${p.recall_ci_low.toFixed(3)}–${p.recall_ci_high.toFixed(3)}`
                        : '—'}
                    </td>
                    <td className="tabular py-1.5 pr-3 text-right text-ink-2">
                      {below === null ? '—' : `${(below * 100).toFixed(0)}%`}
                    </td>
                    <td className="tabular py-1.5 pr-3 text-right text-ink-2">{p.latency_mean_ms.toFixed(3)}</td>
                    <td className="tabular py-1.5 pr-3 text-right text-ink-2">{p.latency_p95_ms.toFixed(3)}</td>
                    <td className="py-1.5 text-xs text-ink-2">
                      {rec?.value === p.value && (
                        <span className="mr-2 inline-flex items-center gap-1 text-ink">
                          <span className="h-2 w-2 rounded-full bg-good" />✓ recommended
                        </span>
                      )}
                      {meets(p, target, confident) ? 'meets target' : 'below target'}
                      {pareto.has(p.value) && ' · Pareto-optimal'}
                      {quickest?.value === p.value && quickest.value !== rec?.value && ' · fastest measured'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </Card>
        <Card className="xl:col-span-2" title="Apply it" subtitle={rec ? `${result.param} = ${rec.value}` : undefined}>
          {snippet ? <CodeBlock code={snippet} /> : <p className="text-sm text-ink-2">No value met the target. Add larger values and re-run.</p>}
        </Card>
      </div>
    </>
  )
}

function PointTooltip({
  active,
  payload,
  param,
}: {
  active?: boolean
  payload?: { payload: SweepPoint }[]
  param: SweepParam
}) {
  const p = active ? payload?.[0]?.payload : undefined
  if (!p) return null
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <div className="font-medium text-ink">
        {param} {p.value}
      </div>
      <div className="tabular text-ink-2">recall {p.recall.toFixed(3)}</div>
      <div className="tabular text-ink-2">
        mean {p.latency_mean_ms.toFixed(3)} ms · p95 {p.latency_p95_ms.toFixed(3)} ms
      </div>
    </div>
  )
}

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="relative">
      <pre className="overflow-auto rounded-lg bg-surface-2 p-3 pt-9 text-xs leading-relaxed whitespace-pre-wrap text-ink">
        <code>{code}</code>
      </pre>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(code).then(
            () => {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1500)
            },
            () => setCopied(false),
          )
        }}
        className="absolute top-2 right-2 rounded-md border border-line bg-surface px-2 py-0.5 text-xs text-ink-2 hover:text-ink"
      >
        {copied ? 'Copied ✓' : 'Copy'}
      </button>
    </div>
  )
}
