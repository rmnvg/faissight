import { useMemo, useState } from 'react'
import { useIvfLists, useListMembers, useMetadata } from '../api/hooks'
import type { Dims, Info, ProjectionMethod } from '../api/types'
import { ScatterMap, type HoverTarget } from '../components/ScatterMap'
import { fillColors } from '../lib/points'
import { Banner, Card, EmptyState, Progress, Segmented, SnippetText, Spinner } from '../components/ui'
import { fmtNum } from '../lib/format'
import { useMapData } from '../lib/mapData'
import { intParam, navigate } from '../lib/route'
import { CANVAS, rampColor, type ThemeMode } from '../lib/theme'

const IVF_KINDS = new Set(['IVF_FLAT', 'IVF_PQ', 'IVF_SQ'])
type ColorBy = 'none' | 'size'

export default function ClusterMap({
  info,
  params,
  mode,
}: {
  info: Info
  params: URLSearchParams
  mode: ThemeMode
}) {
  const isIvf = IVF_KINDS.has(info.kind)
  const [method, setMethod] = useState<ProjectionMethod>(
    params.get('method') === 'umap' ? 'umap' : 'pca',
  )
  const [dims, setDims] = useState<Dims>(params.get('dims') === '3' ? 3 : 2)
  const [colorBy, setColorBy] = useState<ColorBy>('none')
  const selected = intParam(params, 'list')
  const [hover, setHover] = useState<HoverTarget | null>(null)

  const { data, status, error } = useMapData(method, dims, isIvf)
  const lists = useIvfLists(isIvf)
  const sizes = lists.data?.sizes ?? null
  const hasMeta = info.inputs.metadata_rows !== null

  const select = (list: number | null) =>
    navigate('map', { list, method: method === 'pca' ? null : method, dims: dims === 2 ? null : dims })

  const hoveredList =
    hover?.layer === 'centroids'
      ? hover.index
      : hover?.layer === 'points' && data?.points.list_nos
        ? data.points.list_nos[hover.index]
        : null
  const focusList = selected ?? hoveredList

  const colors = CANVAS[mode]
  const pointColors = useMemo(() => {
    if (!data) return new Uint8Array()
    const lnos = data.points.list_nos
    const maxSize = sizes ? Math.max(...sizes, 1) : 1
    return fillColors(data.points.ids.length, (i) => {
      const l = lnos ? lnos[i] : -1
      if (focusList !== null && lnos) {
        return l === focusList ? colors.series1 : colors.pointDim
      }
      if (colorBy === 'size' && sizes && l >= 0) return rampColor(mode, sizes[l] / maxSize, 190)
      return colors.point
    })
  }, [data, focusList, colorBy, sizes, colors, mode])

  const centroidColors = useMemo(() => {
    if (!data?.centroids) return undefined
    return fillColors(data.centroids.ids.length, (i) =>
      focusList === null || i === focusList ? colors.centroid : colors.pointDim,
    )
  }, [data, focusList, colors])

  const hoveredPointId =
    hover?.layer === 'points' && data ? data.points.ids[hover.index] : null
  const meta = useMetadata(hoveredPointId, hasMeta)

  const tooltip =
    hover && data ? (
      hover.layer === 'centroids' ? (
        <div>
          <div className="font-medium text-ink">Centroid of list #{hover.index}</div>
          {sizes && <div className="text-ink-2">{fmtNum(sizes[hover.index])} vectors · click to inspect</div>}
        </div>
      ) : hover.layer === 'points' ? (
        <div>
          <div className="font-medium text-ink">
            id {data.points.ids[hover.index]}
            {data.points.list_nos && (
              <span className="font-normal text-ink-2"> · list #{data.points.list_nos[hover.index]}</span>
            )}
          </div>
          {hasMeta && (
            <div className="mt-1 text-ink-2">
              {meta.data ? <MetaPreview row={meta.data.row} /> : meta.isError ? 'No metadata' : '…'}
            </div>
          )}
        </div>
      ) : null
    ) : null

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-line bg-surface px-6 py-3">
        <h1 className="mr-2 text-base font-semibold text-ink">Cluster map</h1>
        <Segmented
          label="Projection method"
          value={method}
          onChange={setMethod}
          options={[
            { value: 'pca', label: 'PCA' },
            { value: 'umap', label: 'UMAP', title: 'Slower; needs faissight[umap]' },
          ]}
        />
        <Segmented
          label="Dimensions"
          value={dims}
          onChange={setDims}
          options={[
            { value: 2, label: '2D' },
            { value: 3, label: '3D' },
          ]}
        />
        {isIvf && (
          <Segmented
            label="Point colour"
            value={colorBy}
            onChange={setColorBy}
            options={[
              { value: 'none', label: 'Plain' },
              { value: 'size', label: 'By list size' },
            ]}
          />
        )}
        {data && (
          <span className="ml-auto text-xs text-muted">
            {data.points.sampled
              ? `${fmtNum(data.points.ids.length)} of ${fmtNum(data.points.n_total)} points (stratified sample)`
              : `${fmtNum(data.points.n_total)} points`}
            {data.points.explained_variance &&
              ` · PCA explains ${(data.points.explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of variance`}
          </span>
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1 bg-surface">
          {data ? (
            <ScatterMap
              mode={mode}
              dims={dims}
              points={{ positions: data.positions, colors: pointColors, length: data.points.ids.length }}
              centroids={
                data.centroidPositions && centroidColors
                  ? { positions: data.centroidPositions, colors: centroidColors, length: data.centroids!.ids.length }
                  : undefined
              }
              selectedCentroid={focusList}
              hover={hover}
              onHover={setHover}
              onClick={(t) => {
                if (t.layer === 'centroids') select(t.index === selected ? null : t.index)
                else if (t.layer === 'points' && data.points.list_nos) select(data.points.list_nos[t.index])
              }}
              tooltip={tooltip}
            />
          ) : error ? (
            <EmptyState title="Projection unavailable">{error.message}</EmptyState>
          ) : status ? (
            <div className="flex h-full items-center justify-center">
              <Progress value={status.progress} message={status.message} />
            </div>
          ) : (
            <div className="flex h-full items-center justify-center">
              <Spinner label="Loading projection…" />
            </div>
          )}
          {data && <MapLegend isIvf={isIvf} colorBy={colorBy} mode={mode} focusList={focusList} />}
        </div>

        {isIvf && selected !== null && (
          <ListPanel listNo={selected} size={sizes?.[selected]} onClose={() => select(null)} hasMeta={hasMeta} />
        )}
      </div>
      {info.transforms.length > 0 && (
        <div className="border-t border-line px-6 py-2">
          <Banner>Map shows the transformed {info.core_d}-d space the index actually searches.</Banner>
        </div>
      )}
    </div>
  )
}

function MetaPreview({ row }: { row: Record<string, unknown> }) {
  const text = ['text', 'content', 'chunk', 'page_content', 'passage', 'body']
    .map((k) => row[k])
    .find((v) => typeof v === 'string') as string | undefined
  const title = ['title', 'source', 'name'].map((k) => row[k]).find((v) => typeof v === 'string') as
    | string
    | undefined
  return (
    <SnippetText
      snippet={{ title, text: text && text.length > 160 ? `${text.slice(0, 159)}…` : text }}
    />
  )
}

function MapLegend({
  isIvf,
  colorBy,
  mode,
  focusList,
}: {
  isIvf: boolean
  colorBy: ColorBy
  mode: ThemeMode
  focusList: number | null
}) {
  const ramp = CANVAS[mode].ramp
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col gap-1 rounded-lg border border-line bg-surface/90 px-3 py-2 text-xs text-ink-2">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-muted" /> stored vector
      </div>
      {isIvf && (
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-ink" /> list centroid
        </div>
      )}
      {focusList !== null && (
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-series-1" /> list #{focusList}
        </div>
      )}
      {colorBy === 'size' && focusList === null && (
        <div className="flex items-center gap-2">
          <span>small</span>
          <span
            className="h-2 w-20 rounded-full"
            style={{ background: `linear-gradient(to right, ${ramp.join(',')})` }}
          />
          <span>large list</span>
        </div>
      )}
    </div>
  )
}

const PAGE = 50

function ListPanel({
  listNo,
  size,
  onClose,
  hasMeta,
}: {
  listNo: number
  size: number | undefined
  onClose: () => void
  hasMeta: boolean
}) {
  const [page, setPage] = useState(0)
  const [prevList, setPrevList] = useState(listNo)
  if (prevList !== listNo) {
    setPrevList(listNo)
    setPage(0)
  }
  const members = useListMembers(listNo, page * PAGE, PAGE)
  const total = members.data?.size ?? size ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))

  return (
    <aside className="flex w-96 shrink-0 flex-col border-l border-line bg-surface">
      <Card
        className="m-3 flex min-h-0 flex-1 flex-col !p-0"
        title={<span className="px-4 pt-4 inline-block">List #{listNo}</span>}
        subtitle={<span className="px-4 inline-block">{fmtNum(total)} vectors</span>}
        actions={
          <button onClick={onClose} className="px-4 pt-4 text-muted hover:text-ink" aria-label="Close">
            ✕
          </button>
        }
      >
        <div className={`min-h-0 flex-1 overflow-auto px-4 ${members.isPlaceholderData ? 'opacity-60' : ''}`}>
          {members.isPending ? (
            <Spinner />
          ) : (
            <table className="w-full text-sm">
              <tbody>
                {members.data?.members.map((m) => (
                  <tr key={m.id} className="border-t border-line align-top">
                    <td className="tabular py-1.5 pr-3 text-ink">{m.id}</td>
                    <td className="py-1.5 text-xs">{hasMeta ? <SnippetText snippet={m.snippet} /> : null}</td>
                    <td className="py-1.5 pl-2 text-right">
                      <button
                        onClick={() => navigate('query', { id: m.id })}
                        className="text-xs whitespace-nowrap text-series-1 hover:underline"
                        title="Run this vector as a query"
                      >
                        Query →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <footer className="flex items-center justify-between border-t border-line px-4 py-2 text-xs text-ink-2">
          <button disabled={page === 0} onClick={() => setPage(page - 1)} className="disabled:opacity-30">
            ← Prev
          </button>
          <span className="tabular">
            page {page + 1} / {pages}
          </span>
          <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)} className="disabled:opacity-30">
            Next →
          </button>
        </footer>
      </Card>
    </aside>
  )
}
