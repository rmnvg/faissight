import { parseId } from '../lib/ids'
import type { UserId } from '../api/types'
import { COORDINATE_SYSTEM, OrbitView, type Layer, type OrbitViewState } from '@deck.gl/core'
import { LineLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import DeckGL from '@deck.gl/react'
import { useMutation, useQueries, useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import type { HnswGraph, HnswOutcome, HnswStats, HnswTrace, Info } from '../api/types'
import { Banner, Card, EmptyState, Segmented, SnippetText, Spinner, StatTile } from '../components/ui'
import { fmtDist, fmtNum } from '../lib/format'
import { buildFrames, frameLabel, stateAt, type FrameState } from '../lib/hnswFrames'
import { intParam, navigate } from '../lib/route'
import { CANVAS, type RGBA, type ThemeMode } from '../lib/theme'

const LEVEL0_LIMIT = 1500
const UPPER_LIMIT = 4000
const SPEEDS = [1, 2, 4, 8]

type XY = [number, number]
type ViewState = Record<string, unknown>

export default function Hnsw({
  info,
  params,
  mode,
}: {
  info: Info
  params: URLSearchParams
  mode: ThemeMode
}) {
  const stats = useQuery({ queryKey: ['hnsw-stats'], queryFn: api.hnswStats, staleTime: Infinity })
  const urlId = parseId(params.get('id'))
  const [idInput, setIdInput] = useState(urlId !== null ? String(urlId) : '')
  const [k, setK] = useState(10)
  const [ef, setEf] = useState(intParam(params, 'ef') ?? Number(info.params.ef_search ?? 16))
  const [formError, setFormError] = useState<string | null>(null)

  const trace = useMutation({
    mutationFn: (req: { id: UserId; k: number; ef: number }) =>
      api.traceHnsw({ query: { id: req.id }, k: req.k, efSearch: req.ef, compare: true }),
  })

  const run = (id: UserId) => trace.mutate({ id, k, ef })
  const autoRan = useRef<UserId | null>(null)
  useEffect(() => {
    if (urlId === null || autoRan.current === urlId) return
    autoRan.current = urlId
    trace.mutate({ id: urlId, k, ef })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [urlId])

  const submit = (e: FormEvent) => {
    e.preventDefault()
    const id = parseId(idInput)
    if (id === null) {
      setFormError('Enter an integer id of a stored vector.')
      return
    }
    setFormError(null)
    navigate('hnsw', { id, ef })
    if (urlId === id) run(id)
  }

  const t = trace.data

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-end gap-3 border-b border-line bg-surface px-6 py-3">
        <div className="mr-2">
          <h1 className="text-base font-semibold text-ink">HNSW graph</h1>
          <p className="text-xs text-ink-2">Layers of the graph and a step-by-step search</p>
        </div>
        <form onSubmit={submit} className="flex flex-wrap items-end gap-3">
          <label className="flex w-40 flex-col gap-1 text-xs text-ink-2">
            Query: stored id
            <input
              value={idInput}
              onChange={(e) => setIdInput(e.target.value)}
              inputMode="numeric"
              placeholder="e.g. 42"
              className="rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink outline-none focus:border-series-1"
            />
          </label>
          <NumberField label="k" value={k} onChange={setK} max={1000} />
          <NumberField label="efSearch" value={ef} onChange={setEf} max={4096} />
          <button
            type="submit"
            disabled={trace.isPending}
            className="rounded-lg bg-series-1 px-4 py-1.5 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50"
          >
            {trace.isPending ? 'Tracing…' : 'Trace search'}
          </button>
        </form>
        {formError && <span className="text-sm text-critical">{formError}</span>}
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1 bg-surface">
          {stats.data ? (
            <Scene stats={stats.data} trace={t ?? null} mode={mode} />
          ) : stats.isError ? (
            <EmptyState title="Graph unavailable">{stats.error.message}</EmptyState>
          ) : (
            <div className="flex h-full items-center justify-center">
              <Spinner label="Reading the graph…" />
            </div>
          )}
        </div>
        <aside className="w-96 shrink-0 overflow-auto border-l border-line bg-page p-3">
          <div className="flex flex-col gap-3">
            {t && <TracePanel trace={t} />}
            {!t && (
              <Card>
                <EmptyState title="Trace a search">
                  Enter a stored id to replay how HNSW descends the layers and expands its
                  candidates on level 0.
                </EmptyState>
              </Card>
            )}
            {stats.data && <StatsPanel stats={stats.data} />}
          </div>
        </aside>
      </div>
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  max,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  max: number
}) {
  return (
    <label className="flex w-24 flex-col gap-1 text-xs text-ink-2">
      {label}
      <input
        type="number"
        value={value}
        min={1}
        max={max}
        onChange={(e) => onChange(Math.max(1, Math.min(max, Number(e.target.value) || 1)))}
        className="tabular rounded-lg border border-line bg-page px-3 py-1.5 text-sm text-ink outline-none focus:border-series-1"
      />
    </label>
  )
}

// --- 3D scene ----------------------------------------------------------------------------

function Scene({ stats, trace, mode }: { stats: HnswStats; trace: HnswTrace | null; mode: ThemeMode }) {
  const colors = CANVAS[mode]
  const levels = Array.from({ length: stats.max_level + 1 }, (_, i) => i)
  const level0Entry = trace?.levels.find((l) => l.level === 0)?.entry ?? null

  const graphs = useQueries({
    queries: levels.map((level) => ({
      queryKey: ['hnsw-graph', level, level === 0 ? level0Entry : null],
      queryFn: () =>
        api.hnswGraph(level, level === 0 ? LEVEL0_LIMIT : UPPER_LIMIT, level === 0 ? level0Entry : null),
      staleTime: Infinity,
    })),
  })
  const ready = graphs.every((g) => g.data)
  const graphData = graphs.map((g) => g.data).filter(Boolean) as HnswGraph[]

  // Position of every node we know about: graph samples plus every node in the trace.
  const pos = useMemo(() => {
    const m = new Map<UserId, XY>()
    for (const g of graphData) g.ids.forEach((id, i) => m.set(id, [g.x[i], g.y[i]]))
    if (trace) trace.nodes.ids.forEach((id, i) => m.set(id, [trace.nodes.x[i], trace.nodes.y[i]]))
    return m
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, graphData.length, trace])

  const geom = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const [x, y] of pos.values()) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
    if (!Number.isFinite(minX)) return null
    const span = Math.max(maxX - minX, maxY - minY, 1e-6)
    const gap = span * 0.55
    const pad = span * 0.06
    return { minX: minX - pad, maxX: maxX + pad, minY: minY - pad, maxY: maxY + pad, span, gap }
  }, [pos])

  const frames = useMemo(() => (trace ? buildFrames(trace) : []), [trace])
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(2)
  const [traceSeen, setTraceSeen] = useState<HnswTrace | null>(null)
  if (trace !== traceSeen) {
    // New trace: start from the top and play.
    setTraceSeen(trace)
    setFrame(0)
    setPlaying(trace !== null)
  }
  useEffect(() => {
    if (!playing || frames.length === 0) return
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f + 1 >= frames.length) {
          setPlaying(false)
          return f
        }
        return f + 1
      })
    }, 600 / speed)
    return () => window.clearInterval(id)
  }, [playing, speed, frames.length])

  const state: FrameState | null = trace && frames.length ? stateAt(trace, frames, frame) : null
  const finished = state?.isLast && !playing
  const [camera, setCamera] = useState<'auto' | 'layers' | 'level0'>('auto')
  const closeUp = trace !== null && (camera === 'level0' || (camera === 'auto' && state?.frame.level === 0))

  // Bounds of everything the level-0 search touched, for the close-up camera.
  const level0Box = useMemo(() => {
    if (!trace) return null
    const ids = new Set<UserId>()
    for (const l of trace.levels.filter((l) => l.level === 0))
      for (const st of l.steps) {
        ids.add(st.expanded)
        st.visits.forEach((v) => ids.add(v.node))
      }
    trace.truth?.forEach((t) => ids.add(t.id))
    const xs: number[] = []
    const ys: number[] = []
    for (const id of ids) {
      const p = pos.get(id)
      if (p) {
        xs.push(p[0])
        ys.push(p[1])
      }
    }
    if (trace.query_xy) {
      xs.push(trace.query_xy[0])
      ys.push(trace.query_xy[1])
    }
    if (!xs.length) return null
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) }
  }, [trace, pos])

  const [size, setSize] = useState({ w: 0, h: 0 })
  const wrap = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const fitted = useMemo<ViewState | null>(() => {
    if (!geom || size.w === 0) return null
    if (closeUp && level0Box) {
      const span = Math.max(level0Box.maxX - level0Box.minX, level0Box.maxY - level0Box.minY, geom.span * 0.02)
      return {
        target: [(level0Box.minX + level0Box.maxX) / 2, (level0Box.minY + level0Box.maxY) / 2, 0],
        rotationX: 90,
        rotationOrbit: 0,
        zoom: Math.log2((Math.min(size.w, size.h) * 0.75) / span),
      }
    }
    const height = stats.max_level * geom.gap
    const extent = Math.max(geom.span, height)
    return {
      target: [(geom.minX + geom.maxX) / 2, (geom.minY + geom.maxY) / 2, height / 2],
      rotationX: 22,
      rotationOrbit: -25,
      zoom: Math.log2((Math.min(size.w, size.h) * 0.7) / extent),
    }
  }, [geom, size, stats.max_level, closeUp, level0Box])
  const fitKey = fitted ? `${(fitted.target as number[]).join(',')}:${fitted.zoom}` : ''
  const [userView, setUserView] = useState<{ key: string; vs: ViewState } | null>(null)
  const viewState = userView && userView.key === fitKey ? userView.vs : fitted

  const z = (level: number) => (geom ? level * geom.gap : 0)
  const at = (id: UserId, level: number): [number, number, number] | null => {
    const p = pos.get(id)
    return p ? [p[0], p[1], z(level)] : null
  }

  const baseLayers = useMemo<Layer[]>(() => {
    if (!geom || !ready) return []
    const coord = { coordinateSystem: COORDINATE_SYSTEM.CARTESIAN }
    const planes = levels.map((level) => ({
      level,
      path: [
        [geom.minX, geom.minY, z(level)],
        [geom.maxX, geom.minY, z(level)],
        [geom.maxX, geom.maxY, z(level)],
        [geom.minX, geom.maxY, z(level)],
        [geom.minX, geom.minY, z(level)],
      ],
    }))
    const nodes: { p: [number, number, number]; level: number }[] = []
    const edges: { a: [number, number, number]; b: [number, number, number] }[] = []
    for (const g of graphData) {
      if (closeUp && g.level > 0) continue
      g.ids.forEach((_, i) => nodes.push({ p: [g.x[i], g.y[i], z(g.level)], level: g.level }))
      for (let e = 0; e < g.edges_src.length; e++) {
        const a = g.edges_src[e]
        const b = g.edges_dst[e]
        edges.push({ a: [g.x[a], g.y[a], z(g.level)], b: [g.x[b], g.y[b], z(g.level)] })
      }
    }
    return [
      new PathLayer({
        id: 'planes',
        ...coord,
        data: planes,
        getPath: (d: { path: number[][] }) => d.path as [number, number, number][],
        getColor: [...colors.pointDim.slice(0, 3), 160] as RGBA,
        widthUnits: 'pixels',
        getWidth: 1,
      }),
      new TextLayer({
        id: 'plane-labels',
        ...coord,
        data: planes,
        getPosition: (d: { level: number }) => [geom.minX, geom.maxY, z(d.level)],
        getText: (d: { level: number }) => `level ${d.level}`,
        getSize: 12,
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        getColor: colors.point,
        getTextAnchor: 'start',
        getAlignmentBaseline: 'bottom',
      }),
      new LineLayer({
        id: 'edges',
        ...coord,
        data: edges,
        getSourcePosition: (d: { a: [number, number, number] }) => d.a,
        getTargetPosition: (d: { b: [number, number, number] }) => d.b,
        getColor: [...colors.pointDim.slice(0, 3), mode === 'dark' ? 90 : 70] as RGBA,
        widthUnits: 'pixels',
        getWidth: 1,
      }),
      new ScatterplotLayer({
        id: 'nodes',
        ...coord,
        data: nodes,
        getPosition: (d: { p: [number, number, number] }) => d.p,
        getFillColor: (d: { level: number }) => (d.level > 0 ? colors.point : colors.pointDim),
        radiusUnits: 'pixels',
        getRadius: (d: { level: number }) => (d.level > 0 ? 2.5 : 1.8),
      }),
    ]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geom, ready, graphData.length, colors, mode, stats.max_level, closeUp])

  const overlayLayers = useMemo<Layer[]>(() => {
    if (!geom || !trace || !state) return []
    const coord = { coordinateSystem: COORDINATE_SYSTEM.CARTESIAN }
    const onTop = { ...coord, parameters: { depthCompare: 'always' as const } }
    const lvl = state.frame.level
    const dot = (id: UserId, level: number, color: RGBA, radius: number) => {
      const p = at(id, level)
      return p ? [{ p, color, radius }] : []
    }
    const expandedP = at(state.frame.expanded, lvl)
    const links = expandedP
      ? state.frame.visits.flatMap((v) => {
          const b = at(v.node, lvl)
          return b ? [{ a: expandedP, b, accepted: v.accepted }] : []
        })
      : []
    // Drop lines: where the search moved down a level (same node on both planes).
    const drops = state.entries
      .filter((e) => e.level > lvl)
      .flatMap((e) => {
        const next = trace.levels.find((l) => l.level === e.level - 1)
        const a = at(next?.entry ?? e.node, e.level)
        const b = at(next?.entry ?? e.node, e.level - 1)
        return a && b ? [{ a, b }] : []
      })
    const pts = [
      ...[...state.visited].flatMap((id) => dot(id, lvl, [...colors.series1.slice(0, 3), 110] as RGBA, 3)),
      ...[...state.frontier].flatMap((id) => dot(id, lvl, colors.series1, 4)),
      ...state.entries.flatMap((e) => dot(e.node, e.level, colors.ink, 3.5)),
    ]
    const finals = finishedOverlay()
    function finishedOverlay() {
      if (!finished || !trace) return []
      const found = trace.results.flatMap((r) => dot(r.id, 0, colors.series1, 5.5))
      const missed = (trace.truth ?? [])
        .filter((t) => t.outcome !== 'FOUND')
        .flatMap((t) => dot(t.id, 0, colors.series2, 5.5))
      return [...found, ...missed]
    }
    const query = trace.query_xy
      ? [{ p: [trace.query_xy[0], trace.query_xy[1], 0] as [number, number, number] }]
      : []
    return [
      new LineLayer({
        id: 'drops',
        ...onTop,
        data: drops,
        getSourcePosition: (d: { a: [number, number, number] }) => d.a,
        getTargetPosition: (d: { b: [number, number, number] }) => d.b,
        getColor: colors.ink,
        widthUnits: 'pixels',
        getWidth: 1.5,
      }),
      new LineLayer({
        id: 'links',
        ...onTop,
        data: links,
        getSourcePosition: (d: { a: [number, number, number] }) => d.a,
        getTargetPosition: (d: { b: [number, number, number] }) => d.b,
        getColor: (d: { accepted: boolean }) =>
          d.accepted ? colors.series1 : ([...colors.point.slice(0, 3), 140] as RGBA),
        widthUnits: 'pixels',
        getWidth: (d: { accepted: boolean }) => (d.accepted ? 2 : 1),
      }),
      new ScatterplotLayer({
        id: 'state',
        ...onTop,
        data: [...pts, ...finals],
        getPosition: (d: { p: [number, number, number] }) => d.p,
        getFillColor: (d: { color: RGBA }) => d.color,
        getRadius: (d: { radius: number }) => d.radius,
        radiusUnits: 'pixels',
        stroked: true,
        getLineColor: colors.surface,
        lineWidthUnits: 'pixels',
        getLineWidth: 1.5,
      }),
      new ScatterplotLayer({
        id: 'current',
        ...onTop,
        data: expandedP ? [{ p: expandedP }] : [],
        getPosition: (d: { p: [number, number, number] }) => d.p,
        radiusUnits: 'pixels',
        getRadius: 8,
        filled: false,
        stroked: true,
        getLineColor: colors.ink,
        lineWidthUnits: 'pixels',
        getLineWidth: 2,
      }),
      new ScatterplotLayer({
        id: 'query',
        ...onTop,
        data: query,
        getPosition: (d: { p: [number, number, number] }) => d.p,
        radiusUnits: 'pixels',
        getRadius: 7,
        filled: false,
        stroked: true,
        getLineColor: colors.series2,
        lineWidthUnits: 'pixels',
        getLineWidth: 2,
      }),
    ]
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geom, trace, state?.frame, frame, finished, colors, pos])

  return (
    <div ref={wrap} className="absolute inset-0">
      {!ready || !viewState ? (
        <div className="flex h-full items-center justify-center">
          <Spinner label="Loading layers…" />
        </div>
      ) : (
        <DeckGL
          views={new OrbitView({ id: 'orbit', orbitAxis: 'Z' })}
          viewState={viewState as unknown as OrbitViewState}
          onViewStateChange={({ viewState: vs }) => setUserView({ key: fitKey, vs: vs as ViewState })}
          controller={{ inertia: true }}
          layers={[...baseLayers, ...overlayLayers].map((l) =>
            closeUp && ['planes', 'plane-labels', 'drops'].includes(l.id) ? l.clone({ visible: false }) : l,
          )}
        />
      )}
      <button
        onClick={() => setUserView(null)}
        className="absolute top-2 right-2 rounded-md border border-line bg-surface/90 px-2 py-1 text-xs text-ink-2 hover:text-ink"
      >
        Reset view
      </button>

      {trace && frames.length > 0 && state && (
        <div className="absolute right-3 bottom-3 left-3 flex flex-wrap items-center gap-3 rounded-xl border border-line bg-surface/95 px-3 py-2 text-xs text-ink-2 shadow-sm">
          <button
            onClick={() => {
              if (state.isLast) setFrame(0)
              setPlaying(!playing)
            }}
            className="w-16 rounded-md bg-series-1 px-2 py-1 font-medium text-white"
            aria-label={playing ? 'Pause' : 'Play'}
          >
            {playing ? 'Pause' : state.isLast ? 'Replay' : 'Play'}
          </button>
          <button onClick={() => { setPlaying(false); setFrame(Math.max(0, frame - 1)) }} className="rounded-md border border-line px-2 py-1 hover:text-ink" aria-label="Previous step">◀</button>
          <button onClick={() => { setPlaying(false); setFrame(Math.min(frames.length - 1, frame + 1)) }} className="rounded-md border border-line px-2 py-1 hover:text-ink" aria-label="Next step">▶</button>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={frame}
            onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)) }}
            className="min-w-40 flex-1 accent-series-1"
            aria-label="Search step"
          />
          <span className="tabular w-44 text-ink">{frameLabel(state.frame)}</span>
          <Segmented
            label="Speed"
            value={speed}
            onChange={setSpeed}
            options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))}
          />
          <Segmented
            label="Camera"
            value={camera}
            onChange={setCamera}
            options={[
              { value: 'auto', label: 'Auto', title: 'Layers, then close in on level 0' },
              { value: 'layers', label: 'Layers' },
              { value: 'level0', label: 'Level 0' },
            ]}
          />
        </div>
      )}

      <div className="pointer-events-none absolute top-3 left-3 flex flex-col gap-1 rounded-lg border border-line bg-surface/90 px-3 py-2 text-xs text-ink-2">
        <LegendDot className="border-2 border-series-2" hollow label="query" />
        <LegendDot className="border-2 border-ink" hollow label="node being expanded" />
        <LegendDot className="bg-series-1" label={trace ? 'candidate (queued)' : 'node'} />
        {trace && <LegendDot className="bg-series-1/40" label="visited" />}
        {trace && <LegendDot className="bg-ink" label="level entry point" />}
        {finished && (trace?.truth ?? []).some((t) => t.outcome !== 'FOUND') && (
          <LegendDot className="bg-series-2" label="missed true neighbour" />
        )}
        {finished && <LegendDot className="bg-series-1" label="returned (final)" />}
      </div>
    </div>
  )
}

function LegendDot({ className, label, hollow }: { className: string; label: string; hollow?: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <span className={`h-2.5 w-2.5 rounded-full ${hollow ? '' : ''} ${className}`} />
      {label}
    </span>
  )
}

// --- side panels -------------------------------------------------------------------------

const OUTCOME: Record<HnswOutcome, { label: string; dot: string; help: string }> = {
  FOUND: { label: 'Found', dot: 'bg-good', help: 'Returned by the search.' },
  NOT_REACHED: {
    label: 'Not reached',
    dot: 'bg-critical',
    help: 'Not reached from the visited frontier: no expanded node linked to it before the search stopped. Raise efSearch.',
  },
  VISITED_NOT_KEPT: {
    label: 'Visited, ranked out',
    dot: 'bg-serious',
    help: 'Its distance was computed but approximate (SQ/PQ) distances ranked it out of the top-k.',
  },
}

function TracePanel({ trace }: { trace: HnswTrace }) {
  const perLevel = trace.levels.map((l) => ({
    level: l.level,
    expansions: l.steps.length,
    distances: l.steps.reduce((a, s) => a + s.visits.length, 0),
  }))
  const totalDist = perLevel.reduce((a, l) => a + l.distances, 0)
  const misses = (trace.truth ?? []).filter((t) => t.outcome !== 'FOUND')
  return (
    <>
      <Banner>
        <strong>Reconstructed trace.</strong> Replayed in Python over the graph; it returns{' '}
        {Math.round(trace.overlap_with_faiss * 100)}% of the ids FAISS itself returned.
      </Banner>
      <div className="grid grid-cols-2 gap-2">
        <StatTile
          label={`Recall@${trace.k}`}
          value={trace.recall === null ? '—' : trace.recall.toFixed(2)}
          tone={trace.recall === null ? 'default' : trace.recall >= 1 ? 'good' : trace.recall < 0.9 ? 'bad' : 'default'}
        />
        <StatTile label="Distance computations" value={fmtNum(totalDist)} caption={`efSearch ${trace.ef_search}`} />
      </div>
      <Card title="Work per level">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 font-normal">Level</th>
              <th className="py-1 text-right font-normal">Nodes expanded</th>
              <th className="py-1 text-right font-normal">Distances</th>
            </tr>
          </thead>
          <tbody>
            {perLevel.map((l) => (
              <tr key={l.level} className="border-t border-line">
                <td className="tabular py-1 text-ink">{l.level}</td>
                <td className="tabular py-1 text-right text-ink">{l.expansions}</td>
                <td className="tabular py-1 text-right text-ink-2">{l.distances}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      {trace.truth && (
        <Card title="Exact nearest neighbours" subtitle={misses.length ? `${misses.length} missed` : 'All found'}>
          <ul className="flex flex-col divide-y divide-line text-sm">
            {trace.truth.map((t) => {
              const o = OUTCOME[t.outcome]
              return (
                <li key={t.id} className="flex items-start gap-2 py-1.5" title={o.help}>
                  <span className="tabular w-6 text-muted">{t.rank + 1}</span>
                  <span className="tabular w-14 text-ink">{t.id}</span>
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-full border border-line px-2 py-0.5 text-xs text-ink">
                    <span className={`h-2 w-2 rounded-full ${o.dot}`} />
                    {t.outcome === 'FOUND' ? '✓' : '✕'} {o.label}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-xs">
                    <SnippetText snippet={t.snippet} />
                  </span>
                </li>
              )
            })}
          </ul>
          {misses.some((m) => m.outcome === 'NOT_REACHED') && (
            <p className="mt-2 text-xs text-ink-2">
              “Not reached”: no node the search expanded linked to it before the stopping rule
              fired. A larger efSearch widens the frontier.
            </p>
          )}
          <p className="mt-2 text-xs text-muted">
            Distances: {trace.higher_is_closer ? 'inner product (higher = closer)' : 'squared L2'}; nearest{' '}
            {fmtDist(trace.truth[0]?.distance ?? 0)}.
          </p>
        </Card>
      )}
    </>
  )
}

function StatsPanel({ stats }: { stats: HnswStats }) {
  const [level, setLevel] = useState(0)
  const lv = stats.levels.find((l) => l.level === level) ?? stats.levels[stats.levels.length - 1]
  const hist = lv.degree_hist.map((n, degree) => ({ degree, n }))
  return (
    <Card title="Graph structure" subtitle={`M = ${stats.m} · efConstruction ${stats.ef_construction}`}>
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">Level</th>
            <th className="py-1 text-right font-normal">Nodes</th>
            <th className="py-1 text-right font-normal">Mean links</th>
          </tr>
        </thead>
        <tbody>
          {stats.levels.map((l) => (
            <tr
              key={l.level}
              onClick={() => setLevel(l.level)}
              className={`cursor-pointer border-t border-line hover:bg-surface-2 ${l.level === level ? 'bg-accent-wash' : ''}`}
            >
              <td className="tabular py-1 text-ink">{l.level}</td>
              <td className="tabular py-1 text-right text-ink">{fmtNum(l.n_nodes)}</td>
              <td className="tabular py-1 text-right text-ink-2">
                {l.degree_mean.toFixed(1)} / {l.max_links}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 text-xs text-ink-2">Links per node on level {lv.level} (click a row)</div>
      <div className="h-32">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={hist} margin={{ top: 6, right: 4, bottom: 0, left: 0 }} barCategoryGap={2}>
            <XAxis dataKey="degree" tick={{ fill: 'var(--muted)', fontSize: 10 }} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} interval="preserveStartEnd" />
            <YAxis tick={{ fill: 'var(--muted)', fontSize: 10 }} tickLine={false} axisLine={false} width={34} allowDecimals={false} />
            <Tooltip
              cursor={{ fill: 'var(--accent-wash)' }}
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-lg border border-line bg-surface px-2 py-1 text-xs shadow">
                    {payload[0].payload.degree} links: {fmtNum(payload[0].payload.n)} nodes
                  </div>
                ) : null
              }
            />
            <Bar dataKey="n" fill="var(--series-1)" radius={[3, 3, 0, 0]} maxBarSize={24} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-muted">
        Entry point id {stats.entry_point} on level {stats.max_level}. Level 0 shows a sample of{' '}
        {fmtNum(LEVEL0_LIMIT)} nodes around where the search entered it.
      </p>
    </Card>
  )
}
