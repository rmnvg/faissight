import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { useIvfLists } from '../api/hooks'
import type { Info, ListSizes } from '../api/types'
import { Banner, Card, Spinner, StatTile } from '../components/ui'
import { fmtNum } from '../lib/format'
import { navigate } from '../lib/route'

const IVF_KINDS = new Set(['IVF_FLAT', 'IVF_PQ', 'IVF_SQ'])

export function Overview({ info }: { info: Info }) {
  const isIvf = IVF_KINDS.has(info.kind)
  const lists = useIvfLists(isIvf && info.supported)

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 p-6">
      <header>
        <h1 className="text-xl font-semibold text-ink">Overview</h1>
        <p className="text-sm text-ink-2">
          {info.name} · {info.class_chain.join(' → ')}
        </p>
      </header>

      {!info.supported && (
        <Banner tone="warning">
          <strong>Unsupported index.</strong> {info.unsupported_reason} faissight can only show the
          basic stats below.
        </Banner>
      )}
      {info.supported && info.ground_truth_source === 'reconstructed' && (
        <Banner tone="warning">
          Ground truth computed on reconstructed vectors; PQ/SQ error is not measured. Pass{' '}
          <code>--vectors</code> for exact ground truth.
        </Banner>
      )}
      {info.transforms.length > 0 && (
        <Banner>
          Queries pass through {info.transforms.map((t) => `${t.name} (${t.d_in}→${t.d_out})`).join(', ')}{' '}
          before the index. Centroids and maps live in the transformed {info.core_d}-d space.
        </Banner>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Vectors" value={fmtNum(info.ntotal)} />
        <StatTile
          label="Dimension"
          value={info.d}
          caption={info.core_d !== info.d ? `${info.core_d} after transform` : undefined}
        />
        <StatTile
          label="Metric"
          value={info.metric}
          caption={info.higher_is_closer ? 'higher = closer' : 'lower = closer'}
        />
        {isIvf ? (
          <StatTile
            label="Inverted lists"
            value={fmtNum(Number(info.params.nlist))}
            caption={`default nprobe ${info.params.nprobe}`}
          />
        ) : (
          <StatTile label="Kind" value={info.kind} />
        )}
      </div>

      {isIvf && info.supported && (
        <>
          {lists.isPending && <Spinner label="Reading inverted lists…" />}
          {lists.data && <IvfHealth lists={lists.data} ntotal={info.ntotal} />}
        </>
      )}

      {!isIvf && info.supported && (
        <Card title="Parameters">
          <ParamTable params={info.params} />
          <p className="mt-3 text-sm text-ink-2">
            List-balance views need an IVF index. The cluster map and query explorer work for this
            index.
          </p>
        </Card>
      )}

      <InputsCard info={info} />
    </div>
  )
}

function verdict(l: ListSizes): { label: string; tone: 'good' | 'bad' | 'default'; detail: string } {
  const nTop = Math.max(1, Math.ceil(0.05 * l.nlist))
  const fair = nTop / l.nlist
  const detail = `Top 5% of lists (${nTop}) hold ${(l.top_5pct_share * 100).toFixed(1)}% of vectors; ${(
    fair * 100
  ).toFixed(1)}% if perfectly balanced.`
  if (l.imbalance_factor <= 1.2) return { label: 'Healthy', tone: 'good', detail }
  if (l.imbalance_factor <= 2) return { label: 'Mildly skewed', tone: 'default', detail }
  return { label: 'Skewed', tone: 'bad', detail }
}

function histogram(sizes: number[], maxBins = 30) {
  const max = Math.max(...sizes, 0)
  const binWidth = Math.max(1, Math.ceil((max + 1) / maxBins))
  const bins = Array.from({ length: Math.ceil((max + 1) / binWidth) }, (_, i) => ({
    lo: i * binWidth,
    hi: (i + 1) * binWidth - 1,
    count: 0,
  }))
  for (const s of sizes) bins[Math.floor(s / binWidth)].count++
  return bins.map((b) => ({ ...b, label: binWidth === 1 ? `${b.lo}` : `${b.lo}–${b.hi}` }))
}

function IvfHealth({ lists, ntotal }: { lists: ListSizes; ntotal: number }) {
  const v = verdict(lists)
  const bins = useMemo(() => histogram(lists.sizes), [lists.sizes])
  const maxTop = lists.top_lists[0]?.size ?? 1

  return (
    <>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile
          label="Balance"
          value={v.label}
          tone={v.tone}
          caption={`imbalance factor ${lists.imbalance_factor.toFixed(2)} (1.00 = perfect)`}
        />
        <StatTile label="Empty lists" value={fmtNum(lists.n_empty)} caption={`of ${fmtNum(lists.nlist)}`} />
        <StatTile label="Median list size" value={fmtNum(lists.median)} caption={`min ${fmtNum(lists.min)}`} />
        <StatTile label="Largest list" value={fmtNum(lists.max)} caption={`${((lists.max / Math.max(ntotal, 1)) * 100).toFixed(1)}% of vectors`} />
      </div>
      <p className="-mt-1 text-sm text-ink-2">{v.detail}</p>

      <div className="grid gap-4 lg:grid-cols-5">
        <Card
          className="lg:col-span-3"
          title="List size distribution"
          subtitle="Number of inverted lists by how many vectors they hold"
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 20, left: 0 }} barCategoryGap={2}>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: 'var(--muted)', fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--axis)' }}
                  interval="preserveStartEnd"
                  label={{ value: 'vectors in list', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 11 }}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fill: 'var(--muted)', fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  width={36}
                />
                <Tooltip
                  cursor={{ fill: 'var(--accent-wash)' }}
                  content={({ active, payload }) =>
                    active && payload?.[0] ? (
                      <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                        <div className="text-ink-2">{payload[0].payload.label} vectors</div>
                        <div className="font-medium text-ink">
                          {fmtNum(payload[0].payload.count)} lists
                        </div>
                      </div>
                    ) : null
                  }
                />
                <Bar dataKey="count" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="lg:col-span-2" title="Largest lists" subtitle="Click a list to see it on the map">
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs text-muted">
                <tr>
                  <th className="py-1 font-normal">List</th>
                  <th className="py-1 text-right font-normal">Vectors</th>
                  <th className="py-1 pl-3 font-normal">Share</th>
                </tr>
              </thead>
              <tbody>
                {lists.top_lists.map((t) => (
                  <tr
                    key={t.list_no}
                    onClick={() => navigate('map', { list: t.list_no })}
                    className="cursor-pointer border-t border-line hover:bg-surface-2"
                  >
                    <td className="tabular py-1.5 text-ink">#{t.list_no}</td>
                    <td className="tabular py-1.5 text-right text-ink">{fmtNum(t.size)}</td>
                    <td className="py-1.5 pl-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 rounded-full bg-surface-2">
                          <div
                            className="h-full rounded-full bg-series-1"
                            style={{ width: `${(t.size / maxTop) * 100}%` }}
                          />
                        </div>
                        <span className="tabular w-12 text-right text-xs text-ink-2">
                          {((t.size / Math.max(ntotal, 1)) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  )
}

function ParamTable({ params }: { params: Info['params'] }) {
  const rows = Object.entries(params)
  if (rows.length === 0) return <p className="text-sm text-ink-2">No tunable parameters.</p>
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-2">{k}</dt>
          <dd className="tabular text-ink">{String(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

function InputsCard({ info }: { info: Info }) {
  const i = info.inputs
  const rows: [string, string][] = [
    ['Raw vectors', i.raw_vectors ? 'loaded (exact ground truth)' : 'not given (reconstructed)'],
    [
      'Metadata',
      i.metadata_rows === null
        ? 'not given'
        : `${fmtNum(i.metadata_rows)} rows · ${i.metadata_columns.join(', ') || 'no columns'}${
            i.metadata_coverage !== null && i.metadata_coverage < 1
              ? ` · covers ${(i.metadata_coverage * 100).toFixed(0)}% of ids`
              : ''
          }`,
    ],
    ['Text queries', i.embedder ? `${i.embedder} (${i.embedder_status})` : 'no embedder'],
    ['Query set', i.queries === null ? 'not given' : `${fmtNum(i.queries)} queries`],
  ]
  return (
    <Card title="Inputs" subtitle="What faissight was started with">
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-ink-2">{k}</dt>
            <dd className="text-ink">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  )
}
