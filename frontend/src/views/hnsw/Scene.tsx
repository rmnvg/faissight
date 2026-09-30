import { COORDINATE_SYSTEM, type Layer, OrbitView, type OrbitViewState } from '@deck.gl/core'
import { LineLayer, PathLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers'
import DeckGL from '@deck.gl/react'
import { useQueries } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { HnswGraph, HnswStats, HnswTrace, UserId } from '../../api/types'
import { Spinner } from '../../components/ui'
import { CANVAS, type RGBA, type ThemeMode } from '../../lib/theme'
import { LEVEL0_LIMIT, UPPER_LIMIT } from './constants'
import { PlaybackControls } from './PlaybackControls'
import { usePlayback, type Camera } from './usePlayback'


type XY = [number, number]

type ViewState = Record<string, unknown>

export function Scene({ stats, trace, mode }: { stats: HnswStats; trace: HnswTrace | null; mode: ThemeMode }) {
  const colors = CANVAS[mode]
  const levels = Array.from({ length: stats.max_level + 1 }, (_, i) => i)
  const level0Entry = trace?.levels.find((l) => l.level === 0)?.entry ?? null

  const graphs = useQueries({
    queries: levels.map((level) => ({
      queryKey: ['hnsw-graph', level, level === 0 ? level0Entry : null],
      queryFn: ({ signal }) =>
        api.hnswGraph(
          level,
          level === 0 ? LEVEL0_LIMIT : UPPER_LIMIT,
          level === 0 ? level0Entry : null,
          signal,
        ),
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

  const playback = usePlayback(trace)
  const { frames, frame, state, finished } = playback
  const [camera, setCamera] = useState<Camera>('auto')
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
        <PlaybackControls playback={playback} camera={camera} onCamera={setCamera} />
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
