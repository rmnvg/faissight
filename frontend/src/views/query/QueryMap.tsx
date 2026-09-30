import { useMemo, useState } from 'react'
import type { Info } from '../../api/types'
import { type HoverTarget, type Marker, ScatterMap } from '../../components/ScatterMap'
import { Card, EmptyState, Progress, RetryButton, Spinner } from '../../components/ui'
import { useMapData } from '../../lib/mapData'
import { fillColors } from '../../lib/points'
import { navigate } from '../../lib/route'
import { CANVAS, type ThemeMode } from '../../lib/theme'
import type { RunResult } from './types'

export function QueryMap({
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
  const { data, status, error, retry } = useMapData('pca', 2, isIvf)
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
          <EmptyState title="Map unavailable">
            {error.message}
            <div>
              <RetryButton onClick={retry} />
            </div>
          </EmptyState>
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
