import { useMutation } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import type {
  Info,
  IvfTrace,
  MissReason,
  ProbeRow,
  SearchRequest,
  SearchResponse,
} from '../api/types'
import { ScatterMap, type HoverTarget, type Marker } from '../components/ScatterMap'
import { fillColors } from '../lib/points'
import { Banner, Card, EmptyState, Progress, ReasonBadge, Segmented, SnippetText, Spinner, StatTile } from '../components/ui'
import { fmtDist, fmtNum } from '../lib/format'
import { useMapData } from '../lib/mapData'
import { intParam, navigate } from '../lib/route'
import { CANVAS, type ThemeMode } from '../lib/theme'

const IVF_KINDS = new Set(['IVF_FLAT', 'IVF_PQ', 'IVF_SQ'])
type QueryMode = 'text' | 'id' | 'vector'

interface RunResult {
  search: SearchResponse
  trace: IvfTrace | null
  request: SearchRequest
}

export default function QueryExplorer({
  info,
  params,
  mode,
}: {
  info: Info
  params: URLSearchParams
  mode: ThemeMode
}) {
  const isIvf = IVF_KINDS.has(info.kind)
  const isHnsw = info.kind.startsWith('HNSW')
  const nlist = Number(info.params.nlist ?? 1)
  const hasEmbedder = info.inputs.embedder !== null
  const embedderReady = info.inputs.embedder_status === 'ready'

  const urlId = intParam(params, 'id')
  const [queryMode, setQueryMode] = useState<QueryMode>(
    urlId !== null ? 'id' : hasEmbedder ? 'text' : 'id',
  )
  const [text, setText] = useState(params.get('text') ?? '')
  const [idInput, setIdInput] = useState(urlId !== null ? String(urlId) : '')
  const [vectorInput, setVectorInput] = useState('')
  const [k, setK] = useState(intParam(params, 'k') ?? 10)
  const [nprobe, setNprobe] = useState(intParam(params, 'nprobe') ?? Number(info.params.nprobe ?? 1))
  const [efSearch, setEfSearch] = useState(intParam(params, 'ef') ?? Number(info.params.ef_search ?? 16))
  const [compare, setCompare] = useState(true)
  const [formError, setFormError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement & HTMLTextAreaElement>(null)

  const run = useMutation({
    mutationFn: async (req: SearchRequest): Promise<RunResult> => {
      const [search, trace] = await Promise.all([
        api.search(req),
        isIvf ? api.traceIvf({ ...req, compare: true }) : Promise.resolve(null),
      ])
      return { search, trace, request: req }
    },
  })

  const buildRequest = (): SearchRequest | null => {
    let query: SearchRequest['query']
    if (queryMode === 'text') {
      if (!text.trim()) return fail('Type some query text.')
      query = { text: text.trim() }
    } else if (queryMode === 'id') {
      const id = Number(idInput)
      if (!idInput.trim() || !Number.isInteger(id)) return fail('Enter an integer id.')
      query = { id }
    } else {
      const nums = vectorInput
        .replace(/[[\]\s]+/g, ' ')
        .split(/[ ,]+/)
        .filter(Boolean)
        .map(Number)
      if (nums.length !== info.d || nums.some((n) => !Number.isFinite(n)))
        return fail(`Paste ${info.d} numbers separated by commas or spaces (got ${nums.length}).`)
      query = { vector: nums }
    }
    setFormError(null)
    return {
      query,
      k,
      compare,
      ...(isIvf ? { nprobe } : {}),
      ...(isHnsw ? { efSearch } : {}),
      projection: { method: 'pca', dims: 2 },
    }
  }

  function fail(msg: string): null {
    setFormError(msg)
    return null
  }

  const submit = (e?: FormEvent) => {
    e?.preventDefault()
    const req = buildRequest()
    if (req) {
      run.mutate(req)
      if (req.query.id !== undefined)
        navigate('query', { id: req.query.id, k, nprobe: isIvf ? nprobe : null, ef: isHnsw ? efSearch : null })
    }
  }

  // Auto-run a query that arrived via the URL (e.g. "Query →" from the cluster map).
  const autoRan = useRef<string | null>(null)
  useEffect(() => {
    if (urlId === null || autoRan.current === String(urlId)) return
    autoRan.current = String(urlId)
    setQueryMode('id')
    setIdInput(String(urlId))
    run.mutate({
      query: { id: urlId },
      k,
      compare: true,
      ...(isIvf ? { nprobe } : {}),
      ...(isHnsw ? { efSearch } : {}),
      projection: { method: 'pca', dims: 2 },
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlId])

  // "/" focuses the query box.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const result = run.data

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Query explorer</h1>
        <p className="text-sm text-ink-2">
          Search the index, compare with exact ground truth, and see why true neighbours were missed.
        </p>
      </header>

      <Card>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Segmented
              label="Query type"
              value={queryMode}
              onChange={setQueryMode}
              options={[
                {
                  value: 'text',
                  label: 'Text',
                  disabled: !hasEmbedder,
                  title: hasEmbedder ? undefined : 'Start faissight with --embedder to query by text',
                },
                { value: 'id', label: 'Stored id' },
                { value: 'vector', label: 'Vector' },
              ]}
            />
            {queryMode === 'text' && hasEmbedder && !embedderReady && (
              <span className="text-xs text-muted">
                {info.inputs.embedder_status === 'failed'
                  ? 'Embedder failed to load; see the server log.'
                  : `Loading ${info.inputs.embedder}… the first query waits for it.`}
              </span>
            )}
            <span className="ml-auto text-xs text-muted">
              Press <kbd className="rounded border border-line px-1">/</kbd> to focus
            </span>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <label className="flex min-w-64 flex-1 flex-col gap-1 text-xs text-ink-2">
              {queryMode === 'text' ? 'Query text' : queryMode === 'id' ? 'Stored vector id' : `Vector (${info.d} numbers)`}
              {queryMode === 'vector' ? (
                <textarea
                  ref={inputRef}
                  value={vectorInput}
                  onChange={(e) => setVectorInput(e.target.value)}
                  rows={2}
                  placeholder="0.12, -0.5, …"
                  className="rounded-lg border border-line bg-page px-3 py-2 font-mono text-sm text-ink outline-none focus:border-series-1"
                />
              ) : (
                <input
                  ref={inputRef}
                  value={queryMode === 'text' ? text : idInput}
                  onChange={(e) => (queryMode === 'text' ? setText(e.target.value) : setIdInput(e.target.value))}
                  inputMode={queryMode === 'id' ? 'numeric' : undefined}
                  placeholder={queryMode === 'text' ? 'what is inverted file indexing?' : 'e.g. 42'}
                  className="rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
                />
              )}
            </label>
            <NumberField label="k" value={k} onChange={setK} min={1} max={1000} />
            {isIvf && (
              <label className="flex w-56 flex-col gap-1 text-xs text-ink-2">
                <span>
                  nprobe <span className="tabular text-ink">{nprobe}</span> / {nlist}
                </span>
                <input
                  type="range"
                  min={1}
                  max={nlist}
                  value={nprobe}
                  onChange={(e) => setNprobe(Number(e.target.value))}
                  className="accent-series-1"
                />
              </label>
            )}
            {isHnsw && <NumberField label="efSearch" value={efSearch} onChange={setEfSearch} min={1} max={4096} />}
            <label className="flex items-center gap-2 pb-2 text-sm text-ink-2">
              <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} className="accent-series-1" />
              Compare with exact
            </label>
            <button
              type="submit"
              disabled={run.isPending}
              className="rounded-lg bg-series-1 px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
            >
              {run.isPending ? 'Searching…' : 'Search'}
            </button>
          </div>
          {formError && <p className="text-sm text-critical">{formError}</p>}
        </form>
      </Card>

      {!result && !run.isPending && (
        <Card>
          <EmptyState title="Run a query to begin">
            Try a stored id (every vector in the index can be a query), or pick one from the cluster
            map with “Query →”.
          </EmptyState>
        </Card>
      )}
      {run.isPending && !result && <Spinner label="Searching…" />}

      {result && (
        <div className={`flex flex-col gap-4 ${run.isPending ? 'opacity-60' : ''}`}>
          <Headline result={result} isIvf={isIvf} />
          {result.search.truth_source === 'reconstructed' && (
            <Banner tone="warning">
              Ground truth computed on reconstructed vectors; PQ/SQ error is not measured.
            </Banner>
          )}
          <div className="grid gap-4 xl:grid-cols-2">
            <QueryMap info={info} result={result} mode={mode} isIvf={isIvf} />
            {isIvf && result.trace ? (
              <ProbeChart trace={result.trace} />
            ) : (
              <Card title={isHnsw ? 'How HNSW found these' : 'Search parameters'}>
                {isHnsw ? (
                  <div className="flex flex-col items-start gap-3 text-sm text-ink-2">
                    <p>
                      Replay this search step by step: the greedy descent through the upper layers,
                      then the candidate expansion on level 0, and which true neighbours were never
                      reached. Raising efSearch widens the search.
                    </p>
                    {result.request.query.id !== undefined ? (
                      <button
                        onClick={() =>
                          navigate('hnsw', { id: result.request.query.id ?? null, ef: result.request.efSearch ?? null })
                        }
                        className="rounded-lg bg-series-1 px-3 py-1.5 font-medium text-white hover:opacity-90"
                      >
                        Open the search trace →
                      </button>
                    ) : (
                      <p className="text-xs text-muted">The trace view takes a stored id as the query.</p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-ink-2">
                    Flat indexes search exhaustively, so every true neighbour is found.
                  </p>
                )}
              </Card>
            )}
          </div>
          <ResultsTable result={result.search} />
          {result.search.truth && <TruthTable result={result.search} isIvf={isIvf} />}
        </div>
      )}
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
}) {
  return (
    <label className="flex w-24 flex-col gap-1 text-xs text-ink-2">
      {label}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))}
        className="tabular rounded-lg border border-line bg-page px-3 py-2 text-sm text-ink outline-none focus:border-series-1"
      />
    </label>
  )
}

function Headline({ result, isIvf }: { result: RunResult; isIvf: boolean }) {
  const s = result.search
  const counts = s.reason_counts
  const misses = counts
    ? (Object.entries(counts) as [MissReason, number][]).filter(([r, n]) => r !== 'FOUND' && n > 0)
    : []
  const used = s.params.nprobe
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <StatTile
        label={`Recall@${s.k}`}
        value={s.recall === null ? '—' : s.recall.toFixed(2)}
        tone={s.recall === null ? 'default' : s.recall >= 1 ? 'good' : s.recall < 0.9 ? 'bad' : 'default'}
        caption={s.recall === null ? 'enable “compare with exact”' : `${Math.round(s.recall * s.k)} of ${s.k} true neighbours found`}
      />
      {isIvf ? (
        <StatTile
          label="Min nprobe for all true neighbours"
          value={result.trace?.min_nprobe ?? s.min_nprobe ?? '—'}
          caption={used !== undefined ? `you searched with nprobe ${used}` : undefined}
        />
      ) : (
        <StatTile label="Parameters" value={Object.entries(s.params).map(([k, v]) => `${k} ${v}`).join(', ') || 'none'} />
      )}
      <StatTile label="Latency" value={`${s.latency_ms.toFixed(2)} ms`} caption="single query, server side" />
      <div className="rounded-xl border border-line bg-surface px-4 py-3">
        <div className="text-xs text-ink-2">Misses by reason</div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {misses.length === 0 ? (
            <span className="text-sm text-ink-2">{s.truth ? 'No misses' : '—'}</span>
          ) : (
            misses.map(([r, n]) => (
              <span key={r} className="inline-flex items-center gap-1 text-sm">
                <ReasonBadge reason={r} /> <span className="tabular text-ink">×{n}</span>
              </span>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function QueryMap({
  info,
  result,
  mode,
  isIvf,
}: {
  info: Info
  result: RunResult
  mode: ThemeMode
  isIvf: boolean
}) {
  const { data, status, error } = useMapData('pca', 2, isIvf)
  const [hover, setHover] = useState<HoverTarget | null>(null)
  const colors = CANVAS[mode]
  const probed = useMemo(
    () => new Set(result.trace?.probes.filter((p) => p.probed).map((p) => p.list_no) ?? []),
    [result.trace],
  )

  const pointColors = useMemo(() => {
    if (!data) return new Uint8Array()
    const lnos = data.points.list_nos
    return fillColors(data.points.ids.length, (i) =>
      lnos && probed.has(lnos[i]) ? colors.pointProbed : colors.pointDim,
    )
  }, [data, probed, colors])

  const { markers, missingFromSample } = useMemo(() => {
    const out: Marker[] = []
    let missing = 0
    if (!data) return { markers: out, missingFromSample: 0 }
    const pos = (i: number): [number, number, number] => [data.positions[i * 3], data.positions[i * 3 + 1], 0]
    const s = result.search
    for (const t of s.truth ?? []) {
      if (t.reason === 'FOUND') continue
      const i = data.indexOf.get(t.id)
      if (i === undefined) missing++
      else out.push({ id: `miss-${t.id}`, position: pos(i), color: colors.series2, radius: 5, pointIndex: i })
    }
    for (const r of s.results) {
      const i = data.indexOf.get(r.id)
      if (i === undefined) missing++
      else out.push({ id: `res-${r.id}`, position: pos(i), color: colors.series1, radius: 5, pointIndex: i })
    }
    if (s.query_coords)
      out.push({ id: 'query', position: [s.query_coords[0], s.query_coords[1], 0], color: colors.ink, radius: 4, target: true })
    return { markers: out, missingFromSample: missing }
  }, [data, result, colors])

  const hoveredMarker = hover?.layer === 'markers' ? hover.marker : null
  const tooltip = hoveredMarker ? (
    <div className="text-ink">
      {hoveredMarker.id === 'query'
        ? 'Query'
        : hoveredMarker.id.startsWith('res-')
          ? `Returned · id ${hoveredMarker.id.slice(4)}`
          : `Missed true neighbour · id ${hoveredMarker.id.slice(5)}`}
    </div>
  ) : hover?.layer === 'points' && data ? (
    <div className="text-ink">
      id {data.points.ids[hover.index]}
      {data.points.list_nos && <span className="text-ink-2"> · list #{data.points.list_nos[hover.index]}</span>}
      <div className="text-muted">click to query this vector</div>
    </div>
  ) : null

  return (
    <Card
      title="Where the query landed"
      subtitle={
        isIvf
          ? 'PCA of stored vectors; darker points are in the cells that were probed'
          : 'PCA of stored vectors'
      }
    >
      <div className="relative h-96 overflow-hidden rounded-lg border border-line">
        {data ? (
          <ScatterMap
            mode={mode}
            dims={2}
            points={{ positions: data.positions, colors: pointColors, length: data.points.ids.length }}
            pointRadius={1.5}
            markers={markers}
            fitTo={markers.map((m) => m.position)}
            hover={hover}
            onHover={setHover}
            tooltip={tooltip}
            onClick={(t) => {
              const i = t.layer === 'points' ? t.index : t.marker?.pointIndex
              if (i !== undefined) navigate('query', { id: data.points.ids[i] })
            }}
          />
        ) : error ? (
          <EmptyState title="Map unavailable">{error.message}</EmptyState>
        ) : status ? (
          <div className="flex h-full items-center justify-center">
            <Progress value={status.progress} message={status.message} />
          </div>
        ) : (
          <div className="flex h-full items-center justify-center">
            <Spinner label="Loading map…" />
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-3 rounded-full border-2 border-ink" /> query
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-series-1" /> returned
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-series-2" /> missed true neighbour
        </span>
        {isIvf && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-ink-2" /> in a probed cell
          </span>
        )}
        {missingFromSample > 0 && <span className="text-muted">{missingFromSample} not in the sampled map</span>}
        {!result.search.query_coords && data && <span className="text-muted">query position unavailable (map still computing)</span>}
      </div>
      {info.transforms.length > 0 && (
        <p className="mt-1 text-xs text-muted">Shown in the transformed {info.core_d}-d space.</p>
      )}
    </Card>
  )
}

function ProbeChart({ trace }: { trace: IvfTrace }) {
  // Show the interesting head of the probe order: past both nprobe and the last needed cell.
  const needed = Math.max(trace.nprobe, trace.min_nprobe)
  const shown = Math.min(trace.nlist, Math.max(16, Math.ceil(needed * 1.6) + 2))
  // One dataset for line + dots (a separate scatter dataset would drive the y-domain alone
  // and clip the line). `truth_dist` is set only for cells holding true neighbours.
  const rows = trace.probes.slice(0, shown).map((p) => ({
    ...p,
    truth_dist: p.n_true_neighbours > 0 ? p.centroid_distance : null,
  }))
  const distLabel = trace.higher_is_closer ? 'centroid similarity' : 'centroid distance'

  return (
    <Card
      title="Probe order"
      subtitle={`Cells ranked by ${distLabel} to the query; the index scans the first nprobe`}
    >
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 28, right: 16, bottom: 24, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis
              dataKey="probe_rank"
              type="number"
              domain={[0, shown - 1]}
              allowDecimals={false}
              tick={{ fill: 'var(--muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: 'var(--axis)' }}
              label={{ value: 'probe rank', position: 'insideBottom', offset: -14, fill: 'var(--muted)', fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: 'var(--muted)', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={52}
              tickFormatter={(v: number) => String(Number(v.toPrecision(3)))}
              domain={trace.higher_is_closer ? ['auto', 'auto'] : [0, 'auto']}
            />
            <Tooltip
              cursor={{ stroke: 'var(--axis)' }}
              content={({ active, payload }) => {
                const p = active ? (payload?.[0]?.payload as ProbeRow | undefined) : undefined
                if (!p) return null
                return (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                    <div className="font-medium text-ink">
                      Rank {p.probe_rank} · list #{p.list_no}
                    </div>
                    <div className="text-ink-2">
                      {distLabel} {fmtDist(p.centroid_distance)} · {fmtNum(p.size)} vectors
                    </div>
                    <div className="text-ink-2">
                      {p.probed ? 'probed' : 'not probed'}
                      {p.n_true_neighbours > 0 && ` · holds ${p.n_true_neighbours} true neighbour(s)`}
                    </div>
                  </div>
                )
              }}
            />
            <ReferenceLine
              x={trace.nprobe - 0.5}
              stroke="var(--ink-2)"
              label={{ value: `nprobe = ${trace.nprobe}`, position: 'top', offset: 10, fill: 'var(--ink-2)', fontSize: 11 }}
            />
            <Line
              dataKey="centroid_distance"
              stroke="var(--series-1)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Scatter
              dataKey="truth_dist"
              fill="var(--series-2)"
              stroke="var(--surface)"
              strokeWidth={2}
              shape="circle"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded bg-series-1" /> {distLabel}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-series-2" /> cell holding true neighbours
        </span>
        {shown < trace.nlist && <span className="text-muted">first {shown} of {trace.nlist} cells</span>}
      </div>
    </Card>
  )
}

function ResultsTable({ result }: { result: SearchResponse }) {
  return (
    <Card title="Approximate results" subtitle={`What the index returned (${result.metric}, ${result.higher_is_closer ? 'higher' : 'lower'} = closer)`}>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Id</th>
              <th className="py-1 pr-3 text-right font-normal">Distance</th>
              <th className="py-1 pr-3 text-right font-normal">List</th>
              <th className="py-1 pr-3 font-normal">In ground truth</th>
              <th className="py-1 font-normal">Text</th>
            </tr>
          </thead>
          <tbody>
            {result.results.map((r) => (
              <tr key={r.id} className="border-t border-line align-top">
                <td className="tabular py-1.5 pr-3 text-muted">{r.rank + 1}</td>
                <td className="tabular py-1.5 pr-3">
                  <button onClick={() => navigate('query', { id: r.id })} className="text-ink hover:text-series-1 hover:underline" title="Query with this vector">
                    {r.id}
                  </button>
                </td>
                <td className="tabular py-1.5 pr-3 text-right text-ink">{fmtDist(r.distance)}</td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{r.list_no ?? '—'}</td>
                <td className="py-1.5 pr-3">
                  {r.in_truth === null ? (
                    <span className="text-muted">—</span>
                  ) : r.in_truth ? (
                    <span className="inline-flex items-center gap-1 text-xs text-ink"><span className="h-2 w-2 rounded-full bg-good" />✓ yes</span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs text-ink-2"><span className="h-2 w-2 rounded-full bg-warning" />✕ no</span>
                  )}
                </td>
                <td className="py-1.5 text-xs"><SnippetText snippet={r.snippet} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function TruthTable({ result, isIvf }: { result: SearchResponse; isIvf: boolean }) {
  return (
    <Card title="Exact nearest neighbours" subtitle="Ground truth, and what happened to each one">
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">#</th>
              <th className="py-1 pr-3 font-normal">Id</th>
              <th className="py-1 pr-3 text-right font-normal">Distance</th>
              {isIvf && <th className="py-1 pr-3 text-right font-normal">List</th>}
              {isIvf && <th className="py-1 pr-3 text-right font-normal">Probe rank</th>}
              <th className="py-1 pr-3 font-normal">Outcome</th>
              <th className="py-1 pr-3 text-right font-normal">Found at</th>
              <th className="py-1 font-normal">Text</th>
            </tr>
          </thead>
          <tbody>
            {result.truth!.map((t) => (
              <tr key={t.id} className="border-t border-line align-top">
                <td className="tabular py-1.5 pr-3 text-muted">{t.rank + 1}</td>
                <td className="tabular py-1.5 pr-3">
                  <button onClick={() => navigate('query', { id: t.id })} className="text-ink hover:text-series-1 hover:underline" title="Query with this vector">
                    {t.id}
                  </button>
                </td>
                <td className="tabular py-1.5 pr-3 text-right text-ink">{fmtDist(t.distance)}</td>
                {isIvf && <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.list_no ?? '—'}</td>}
                {isIvf && <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.probe_rank ?? '—'}</td>}
                <td className="py-1.5 pr-3"><ReasonBadge reason={t.reason} /></td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{t.found_rank === null ? '—' : t.found_rank + 1}</td>
                <td className="py-1.5 text-xs"><SnippetText snippet={t.snippet} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
