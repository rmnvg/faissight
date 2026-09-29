import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
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
import type { Info, PqError } from '../api/types'
import { Banner, Card, EmptyState, Progress, Segmented, SnippetText, Spinner, StatTile } from '../components/ui'
import { fmtNum } from '../lib/format'
import { downloadCSV, downloadJSON } from '../lib/exportData'
import { navigate } from '../lib/route'

const AXIS = { fill: 'var(--muted)', fontSize: 11 }
const sig = (v: number, p = 3) => String(Number(v.toPrecision(p)))

export default function Quantization({ info }: { info: Info }) {
  const q = useQuery({
    queryKey: ['pq-error'],
    queryFn: ({ signal }) => api.pqError(signal),
    staleTime: Infinity,
    retry: false,
    refetchInterval: (query) => (query.state.data && 'status' in query.state.data ? 500 : false),
  })
  const data = q.data && !('status' in q.data) ? q.data : null
  const running = q.data && 'status' in q.data ? q.data : null

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Quantization</h1>
        <p className="text-sm text-ink-2">
          How far the vectors {info.name} stores are from your raw vectors, and how that bends
          the distances it ranks by.
        </p>
      </header>

      {q.isPending && <Spinner label="Loading…" />}
      {q.isError && <Banner tone="error">{q.error.message}</Banner>}
      {running && (
        <Card>
          <div className="flex justify-center py-6">
            <Progress value={running.progress} message={running.message} />
          </div>
        </Card>
      )}
      {data && !data.available && (
        <Card>
          <EmptyState title={data.reason ?? 'Not available'}>
            <p>{data.hint}</p>
            <pre className="mt-3 rounded-lg bg-surface-2 px-3 py-2 text-left text-xs text-ink">
              faissight serve {info.name} --vectors vectors.npy
            </pre>
          </EmptyState>
        </Card>
      )}
      {data?.available && <Report data={data} />}
    </div>
  )
}

function Report({ data }: { data: PqError }) {
  const compression = data.code_size && data.raw_bytes ? data.raw_bytes / data.code_size : null
  const d = data.distortion!
  const corrTone = d.near_correlation >= 0.99 ? 'good' : d.near_correlation < 0.9 ? 'bad' : 'default'

  return (
    <>
      {data.has_transform && (
        <Banner>
          This index applies a dimension-reducing transform before storing vectors, so the error
          below includes the transform's loss, not just the codes'.
        </Banner>
      )}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="Mean squared error"
          value={sig(data.mean ?? 0)}
          caption={`median ${sig(data.median ?? 0)} · p95 ${sig(data.p95 ?? 0)}`}
        />
        <StatTile
          label="Relative error"
          value={`${((data.relative_mean ?? 0) * 100).toFixed(2)}%`}
          caption="‖x − x̂‖² / ‖x‖², mean"
        />
        <StatTile
          label="Compression"
          value={compression ? `${sig(compression, 3)}×` : '—'}
          caption={data.code_size ? `${data.code_size} B per vector vs ${data.raw_bytes} B raw` : undefined}
        />
        <StatTile
          label="Distance fidelity (near pairs)"
          value={d.near_correlation.toFixed(3)}
          tone={corrTone}
          caption={`correlation; ${d.correlation.toFixed(4)} over all pairs`}
        />
      </div>
      <p className="-mt-1 text-sm text-ink-2">
        Near-pair fidelity is what decides nearest-neighbour ranking. Far pairs are easy to tell
        apart, so the all-pairs figure flatters lossy codes.
      </p>

      <div className="grid gap-4 xl:grid-cols-2">
        <DistortionChart data={data} />
        <HistogramChart data={data} />
      </div>
      {data.per_list && <PerListChart perList={data.per_list} />}
      <WorstTable data={data} />
    </>
  )
}

function HistogramChart({ data }: { data: PqError }) {
  const h = data.histogram!
  const bins = h.counts.map((count, i) => ({
    label: `${sig(h.edges[i])}–${sig(h.edges[i + 1])}`,
    mid: (h.edges[i] + h.edges[i + 1]) / 2,
    count,
  }))
  return (
    <Card title="Error distribution" subtitle={`Squared reconstruction error of all ${fmtNum(data.n ?? 0)} vectors`}>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 20, left: 0 }} barCategoryGap={1}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="mid"
              tick={AXIS}
              tickLine={false}
              axisLine={{ stroke: 'var(--axis)' }}
              tickFormatter={(v: number) => sig(v, 2)}
              interval="preserveStartEnd"
              label={{ value: '‖x − x̂‖²', position: 'insideBottom', offset: -12, ...AXIS }}
            />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={44} allowDecimals={false} />
            <Tooltip
              cursor={{ fill: 'var(--accent-wash)' }}
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                    <div className="text-ink-2">error {payload[0].payload.label}</div>
                    <div className="font-medium text-ink">{fmtNum(payload[0].payload.count)} vectors</div>
                  </div>
                ) : null
              }
            />
            <Bar dataKey="count" fill="var(--series-1)" radius={[3, 3, 0, 0]} maxBarSize={24} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function DistortionChart({ data }: { data: PqError }) {
  const d = data.distortion!
  const ip = data.metric === 'IP'
  // Random far pairs stretch the axes and squash the near pairs into a corner, so the
  // default view is near pairs only, with axes fitted to them.
  const [scope, setScope] = useState<'near' | 'all'>('near')
  const { near, far, lo, hi } = useMemo(() => {
    const pts = d.true.map((t, i) => ({ t, a: d.approx[i], near: d.near[i] }))
    const nearPts = pts.filter((p) => p.near)
    const shown = scope === 'near' ? nearPts : pts
    const vals = shown.flatMap((p) => [p.t, p.a])
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const pad = (max - min || 1) * 0.04
    return {
      near: nearPts,
      far: scope === 'near' ? [] : pts.filter((p) => !p.near),
      lo: min - pad,
      hi: max + pad,
    }
  }, [d, scope])
  const label = ip ? 'inner product' : 'squared L2'
  return (
    <Card
      title="True vs approximate distance"
      subtitle={`Sampled query–vector pairs (${label}); points on the diagonal are ranked exactly`}
      actions={
        <Segmented
          label="Pairs shown"
          value={scope}
          onChange={setScope}
          options={[
            { value: 'near', label: 'Near neighbours' },
            { value: 'all', label: 'All pairs' },
          ]}
        />
      }
    >
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 20, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" />
            <XAxis
              dataKey="t"
              type="number"
              domain={[lo, hi]}
              tick={AXIS}
              tickLine={false}
              axisLine={{ stroke: 'var(--axis)' }}
              tickFormatter={(v: number) => sig(v, 2)}
              label={{ value: `true ${label}`, position: 'insideBottom', offset: -12, ...AXIS }}
            />
            <YAxis
              dataKey="a"
              type="number"
              domain={[lo, hi]}
              tick={AXIS}
              tickLine={false}
              axisLine={false}
              width={48}
              tickFormatter={(v: number) => sig(v, 2)}
            />
            <ZAxis range={[14, 14]} />
            <Tooltip
              content={({ active, payload }) => {
                const p = active ? (payload?.[0]?.payload as { t: number; a: number; near: boolean } | undefined) : undefined
                return p ? (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                    <div className="text-ink">{p.near ? 'near pair' : 'random pair'}</div>
                    <div className="tabular text-ink-2">true {sig(p.t, 4)} · approx {sig(p.a, 4)}</div>
                  </div>
                ) : null
              }}
            />
            <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 11, color: 'var(--ink-2)' }} />
            <ReferenceLine
              segment={[
                { x: lo, y: lo },
                { x: hi, y: hi },
              ]}
              stroke="var(--ink-2)"
              ifOverflow="extendDomain"
            />
            {scope === 'all' && (
              <Scatter name="random pairs" data={far} fill="var(--muted)" fillOpacity={0.5} isAnimationActive={false} />
            )}
            <Scatter name="near neighbours" data={near} fill="var(--series-1)" fillOpacity={0.8} isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

type Sort = 'list' | 'error' | 'size'

function PerListChart({ perList }: { perList: NonNullable<PqError['per_list']> }) {
  const [sort, setSort] = useState<Sort>('error')
  const rows = useMemo(() => {
    const r = [...perList]
    if (sort === 'error') r.sort((a, b) => b.mean_error - a.mean_error)
    if (sort === 'size') r.sort((a, b) => b.size - a.size)
    if (sort === 'list') r.sort((a, b) => a.list_no - b.list_no)
    return r.map((x) => ({ ...x, key: `#${x.list_no}` }))
  }, [perList, sort])
  return (
    <Card
      title="Mean error per inverted list"
      subtitle="Lists whose vectors compress worst; click a bar to see the list on the map"
      actions={
        <Segmented
          label="Sort lists"
          value={sort}
          onChange={setSort}
          options={[
            { value: 'error', label: 'By error' },
            { value: 'size', label: 'By size' },
            { value: 'list', label: 'By list' },
          ]}
        />
      }
    >
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }} barCategoryGap={1}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis dataKey="key" tick={AXIS} tickLine={false} axisLine={{ stroke: 'var(--axis)' }} interval="preserveStartEnd" />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} width={48} tickFormatter={(v: number) => sig(v, 2)} />
            <Tooltip
              cursor={{ fill: 'var(--accent-wash)' }}
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                    <div className="font-medium text-ink">List #{payload[0].payload.list_no}</div>
                    <div className="tabular text-ink-2">
                      mean error {sig(payload[0].payload.mean_error, 4)} · {fmtNum(payload[0].payload.size)} vectors
                    </div>
                  </div>
                ) : null
              }
            />
            <Bar
              dataKey="mean_error"
              fill="var(--series-1)"
              radius={[3, 3, 0, 0]}
              maxBarSize={24}
              isAnimationActive={false}
              cursor="pointer"
              onClick={(entry: { list_no?: number }) => {
                if (entry?.list_no !== undefined) navigate('map', { list: entry.list_no })
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  )
}

function WorstTable({ data }: { data: PqError }) {
  const rows = data.worst ?? []
  return (
    <Card
      title="Worst-reconstructed vectors"
      subtitle="The 50 stored vectors furthest from their raw version"
      actions={
        <div className="flex gap-2 print:hidden">
          <button
            className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink"
            onClick={() => downloadCSV('faissight-quantization-worst.csv', rows as never)}
          >
            CSV
          </button>
          <button
            className="rounded-md border border-line px-2.5 py-1 text-xs text-ink-2 hover:text-ink"
            onClick={() => downloadJSON('faissight-quantization.json', data)}
          >
            JSON
          </button>
        </div>
      }
    >
      <div className="max-h-96 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-surface text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">Id</th>
              <th className="py-1 pr-3 text-right font-normal">‖x − x̂‖²</th>
              <th className="py-1 pr-3 text-right font-normal">Relative</th>
              <th className="py-1 pr-3 text-right font-normal">List</th>
              <th className="py-1 pr-3 font-normal">Text</th>
              <th className="py-1 font-normal" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className="border-t border-line align-top">
                <td className="tabular py-1.5 pr-3 text-ink">{r.id}</td>
                <td className="tabular py-1.5 pr-3 text-right text-ink">{sig(r.error, 4)}</td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{(r.relative * 100).toFixed(2)}%</td>
                <td className="tabular py-1.5 pr-3 text-right text-ink-2">{r.list_no ?? '—'}</td>
                <td className="py-1.5 pr-3 text-xs"><SnippetText snippet={r.snippet} /></td>
                <td className="py-1.5 text-right">
                  <button
                    onClick={() => navigate('query', { id: r.id })}
                    className="text-xs whitespace-nowrap text-series-1 hover:underline"
                    title="See how this vector's own neighbours are ranked"
                  >
                    Query →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
