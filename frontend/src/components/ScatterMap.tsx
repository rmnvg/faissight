import {
  COORDINATE_SYSTEM,
  OrbitView,
  OrthographicView,
  type Layer,
  type PickingInfo,
} from '@deck.gl/core'
import { ScatterplotLayer } from '@deck.gl/layers'
import DeckGL from '@deck.gl/react'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { CANVAS, type RGBA, type ThemeMode } from '../lib/theme'

export interface PointCloud {
  /** xyz per point (z = 0 in 2D). */
  positions: Float32Array
  /** RGBA per point. */
  colors: Uint8Array
  length: number
}

export interface Marker {
  id: string
  position: [number, number, number]
  color: RGBA
  radius: number
  /** Draw as a hollow target ring (used for the query). */
  target?: boolean
  pointIndex?: number
}

export interface HoverTarget {
  layer: 'points' | 'centroids' | 'markers'
  index: number
  x: number
  y: number
  marker?: Marker
}

interface Props {
  mode: ThemeMode
  dims: 2 | 3
  points: PointCloud
  pointRadius?: number
  centroids?: PointCloud
  selectedCentroid?: number | null
  markers?: Marker[]
  onHover?: (t: HoverTarget | null) => void
  onClick?: (t: HoverTarget) => void
  tooltip?: ReactNode
  hover?: HoverTarget | null
  /** Initially frame these positions (e.g. a query and its neighbours) instead of all points. */
  fitTo?: [number, number, number][]
}

type ViewState = Record<string, unknown>

function bounds(p: PointCloud): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < p.length; i++) {
    for (let d = 0; d < 3; d++) {
      const v = p.positions[i * 3 + d]
      if (v < min[d]) min[d] = v
      if (v > max[d]) max[d] = v
    }
  }
  return { min, max }
}

function fitView(p: PointCloud, dims: 2 | 3, w: number, h: number): ViewState {
  if (p.length === 0 || w === 0 || h === 0) return { target: [0, 0, 0], zoom: 0 }
  const { min, max } = bounds(p)
  const center = [0, 1, 2].map((d) => (min[d] + max[d]) / 2)
  const span = Math.max(max[0] - min[0], max[1] - min[1], dims === 3 ? max[2] - min[2] : 0, 1e-6)
  const zoom = Math.log2((Math.min(w, h) * (dims === 3 ? 0.6 : 0.88)) / span)
  return dims === 3
    ? { target: center, zoom, rotationX: 25, rotationOrbit: 30, minZoom: zoom - 4, maxZoom: zoom + 10 }
    : { target: [center[0], center[1], 0], zoom, minZoom: zoom - 4, maxZoom: zoom + 12 }
}

export function ScatterMap({
  mode,
  dims,
  points,
  pointRadius = 2,
  centroids,
  selectedCentroid = null,
  markers = [],
  onHover,
  onClick,
  tooltip,
  hover,
  fitTo,
}: Props) {
  const colors = CANVAS[mode]
  const wrap = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  // The user's pan/zoom, valid only for the framing it started from (fitKey).
  const [userView, setUserView] = useState<{ key: string; vs: ViewState } | null>(null)

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(([e]) =>
      setSize({ w: e.contentRect.width, h: e.contentRect.height }),
    )
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Frame the focus positions if given (padded so context stays visible), else all points.
  const fitted = useMemo<ViewState | null>(() => {
    if (size.w === 0) return null
    const full = fitView(points, dims, size.w, size.h)
    if (!fitTo || fitTo.length === 0) return full
    const box = fitView(
      { positions: new Float32Array(fitTo.flat()), colors: new Uint8Array(), length: fitTo.length },
      dims,
      size.w,
      size.h,
    )
    // Keep a margin around the focus, and never zoom in more than 8x the full view.
    const zoom = Math.min((box.zoom as number) - 1.2, (full.zoom as number) + 3)
    return { ...full, target: box.target, zoom }
  }, [points, dims, size, fitTo])

  const fitKey = `${dims}:${points.length}:${points.positions[0]}:${
    fitted ? `${(fitted.target as number[]).join(',')}:${fitted.zoom}` : ''
  }`
  const viewState = userView && userView.key === fitKey ? userView.vs : fitted

  const view = useMemo(
    () =>
      dims === 3
        ? new OrbitView({ id: 'orbit', orbitAxis: 'Y' })
        : new OrthographicView({ id: 'ortho', flipY: false }),
    [dims],
  )

  const layers = useMemo(() => {
    const coord = { coordinateSystem: COORDINATE_SYSTEM.CARTESIAN }
    // Centroids and markers always draw over the point cloud, even when it's in front in 3D.
    const onTop = { ...coord, parameters: { depthCompare: 'always' as const } }
    const out: Layer[] = [
      new ScatterplotLayer({
        id: 'points',
        ...coord,
        data: {
          length: points.length,
          attributes: {
            getPosition: { value: points.positions, size: 3 },
            getFillColor: { value: points.colors, size: 4, normalized: true },
          },
        },
        radiusUnits: 'pixels',
        getRadius: pointRadius,
        radiusMinPixels: 1,
        pickable: true,
        updateTriggers: { getFillColor: points.colors },
      }),
    ]
    if (centroids && centroids.length > 0) {
      out.push(
        new ScatterplotLayer({
          id: 'centroids',
          ...onTop,
          data: {
            length: centroids.length,
            attributes: {
              getPosition: { value: centroids.positions, size: 3 },
              getLineColor: { value: centroids.colors, size: 4, normalized: true },
            },
          },
          radiusUnits: 'pixels',
          getRadius: 5,
          stroked: true,
          filled: true,
          getFillColor: [...colors.surface.slice(0, 3), 200] as RGBA,
          lineWidthUnits: 'pixels',
          getLineWidth: 1.5,
          pickable: true,
          updateTriggers: { getLineColor: centroids.colors, getFillColor: mode },
        }),
      )
    }
    if (selectedCentroid !== null && centroids) {
      const i = selectedCentroid
      out.push(
        new ScatterplotLayer({
          id: 'selected-centroid',
          ...onTop,
          data: [[centroids.positions[i * 3], centroids.positions[i * 3 + 1], centroids.positions[i * 3 + 2]]],
          getPosition: (d: number[]) => d as [number, number, number],
          radiusUnits: 'pixels',
          getRadius: 7,
          getFillColor: colors.series1,
          stroked: true,
          getLineColor: colors.surface,
          lineWidthUnits: 'pixels',
          getLineWidth: 2,
        }),
      )
    }
    if (markers.length > 0) {
      const dots = markers.filter((m) => !m.target)
      const targets = markers.filter((m) => m.target)
      out.push(
        new ScatterplotLayer<Marker>({
          id: 'markers',
          ...onTop,
          data: dots,
          getPosition: (m) => m.position,
          getFillColor: (m) => m.color,
          getRadius: (m) => m.radius,
          radiusUnits: 'pixels',
          stroked: true,
          getLineColor: colors.surface, // 2px surface ring keeps overlapping markers legible
          lineWidthUnits: 'pixels',
          getLineWidth: 2,
          pickable: true,
          updateTriggers: { getLineColor: mode },
        }),
        new ScatterplotLayer<Marker>({
          id: 'targets',
          ...onTop,
          data: targets.flatMap((m) => [
            { ...m, id: `${m.id}-outer`, radius: m.radius + 6, target: true },
            { ...m, target: false },
          ]),
          getPosition: (m) => m.position,
          getFillColor: (m) => (m.target ? [0, 0, 0, 0] : m.color),
          getLineColor: (m) => (m.target ? m.color : colors.surface),
          getRadius: (m) => m.radius,
          radiusUnits: 'pixels',
          stroked: true,
          lineWidthUnits: 'pixels',
          getLineWidth: (m) => (m.target ? 2 : 2),
          updateTriggers: { getLineColor: mode },
        }),
      )
    }
    return out
  }, [points, pointRadius, centroids, selectedCentroid, markers, colors, mode])

  const toTarget = (info: PickingInfo): HoverTarget | null => {
    if (!info.layer || info.index < 0) return null
    const layer = info.layer.id as HoverTarget['layer']
    if (layer !== 'points' && layer !== 'centroids' && layer !== 'markers') return null
    return {
      layer,
      index: info.index,
      x: info.x,
      y: info.y,
      marker: layer === 'markers' ? (info.object as Marker) : undefined,
    }
  }

  return (
    <div ref={wrap} className="relative h-full w-full overflow-hidden">
      {viewState && (
        <DeckGL
          views={view}
          viewState={viewState}
          onViewStateChange={({ viewState: vs }) => setUserView({ key: fitKey, vs: vs as ViewState })}
          controller={{ doubleClickZoom: true, inertia: true }}
          layers={layers}
          pickingRadius={6}
          onHover={(info) => onHover?.(toTarget(info))}
          onClick={(info) => {
            const t = toTarget(info)
            if (t) onClick?.(t)
          }}
          getCursor={({ isHovering, isDragging }) =>
            isDragging ? 'grabbing' : isHovering ? 'pointer' : 'grab'
          }
        />
      )}
      <button
        onClick={() => setUserView({ key: fitKey, vs: fitView(points, dims, size.w, size.h) })}
        className="absolute top-2 right-2 rounded-md border border-line bg-surface/90 px-2 py-1 text-xs text-ink-2 hover:text-ink"
        title="Fit all points in view"
      >
        Reset view
      </button>
      {hover && tooltip && (
        <div
          className="pointer-events-none absolute z-10 max-w-xs rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(hover.x + 14, Math.max(0, size.w - 300)),
            top: Math.min(hover.y + 14, Math.max(0, size.h - 120)),
          }}
        >
          {tooltip}
        </div>
      )}
    </div>
  )
}
