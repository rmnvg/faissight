import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { SweepParam, SweepPoint } from '../api/types'
import { Card } from '../components/ui'
import { fmtNum } from '../lib/format'
import { queryRefParams } from '../lib/ids'
import { navigate } from '../lib/route'
import { fractionBelow } from '../lib/tuner'

const AXIS = { fill: 'var(--muted)', fontSize: 11 }

/** How per-query recall is spread at one setting: the mean hides a tail of bad queries. */
export function RecallBreakdown({
  point,
  param,
  k,
  target,
}: {
  point: SweepPoint
  param: SweepParam
  k: number
  target: number
}) {
  const bars = point.recall_distribution.map(([recall, n]) => ({ recall, n, label: recall.toFixed(2) }))
  const below = fractionBelow(point, target)
  const total = bars.reduce((s, b) => s + b.n, 0)
  return (
    <Card
      title={`Per-query recall@${k} at ${param} ${point.value}`}
      subtitle={
        below === null
          ? 'Number of queries at each recall'
          : `${(below * 100).toFixed(0)}% of ${fmtNum(total)} queries are below the target ${target.toFixed(2)}`
      }
    >
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bars} margin={{ top: 8, right: 8, bottom: 20, left: 0 }} barCategoryGap={2}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="label"
              tick={AXIS}
              tickLine={false}
              axisLine={{ stroke: 'var(--axis)' }}
              interval="preserveStartEnd"
              label={{ value: 'recall of the query', position: 'insideBottom', offset: -12, ...AXIS }}
            />
            <YAxis allowDecimals={false} tick={AXIS} tickLine={false} axisLine={false} width={36} />
            <Tooltip
              cursor={{ fill: 'var(--accent-wash)' }}
              content={({ active, payload }) =>
                active && payload?.[0] ? (
                  <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-lg">
                    <div className="text-ink-2">recall {payload[0].payload.label}</div>
                    <div className="font-medium text-ink">{fmtNum(payload[0].payload.n)} queries</div>
                  </div>
                ) : null
              }
            />
            <Bar dataKey="n" radius={[4, 4, 0, 0]} maxBarSize={32} isAnimationActive={false}>
              {bars.map((b) => (
                <Cell key={b.label} fill={b.recall < target - 1e-9 ? 'var(--series-2)' : 'var(--series-1)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-muted">
        <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-series-2" />
        below target
        <span className="mr-1 ml-3 inline-block h-2 w-2 rounded-sm bg-series-1" />
        meets target
      </p>
    </Card>
  )
}

/** The lowest-recall queries at one setting, linked to the Query Explorer to see why. */
export function WorstQueries({
  point,
  param,
  k,
  target,
}: {
  point: SweepPoint
  param: SweepParam
  k: number
  target: number
}) {
  const hasCoverage = point.worst_queries.some((w) => w.probe_coverage !== null)
  return (
    <Card
      title={`Worst queries at ${param} ${point.value}`}
      subtitle="Open one in the Query Explorer to see why its neighbours were missed"
    >
      {point.worst_queries.length === 0 ? (
        <p className="text-sm text-ink-2">No per-query results in this sweep.</p>
      ) : (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted">
            <tr>
              <th className="py-1 pr-3 font-normal">Query</th>
              <th className="py-1 pr-3 text-right font-normal">Recall@{k}</th>
              {hasCoverage && (
                <th
                  className="py-1 pr-3 text-right font-normal"
                  title="Share of its true neighbours in the probed lists; the rest of its misses were ranked out"
                >
                  In probed lists
                </th>
              )}
              <th className="py-1 font-normal" />
            </tr>
          </thead>
          <tbody>
            {point.worst_queries.map((w) => (
              <tr key={w.query_no} className="border-t border-line">
                <td className="tabular py-1.5 pr-3 text-ink">
                  {w.id !== null ? `id ${w.id}` : `query #${w.query_no}`}
                </td>
                <td
                  className={`tabular py-1.5 pr-3 text-right ${w.recall < target - 1e-9 ? 'text-critical' : 'text-ink'}`}
                >
                  {w.recall.toFixed(2)}
                </td>
                {hasCoverage && (
                  <td className="tabular py-1.5 pr-3 text-right text-ink-2">
                    {w.probe_coverage === null ? '—' : w.probe_coverage.toFixed(2)}
                  </td>
                )}
                <td className="py-1.5 text-right">
                  <button
                    className="rounded-md border border-line px-2 py-0.5 text-xs text-ink-2 hover:text-ink"
                    onClick={() =>
                      navigate('query', {
                        ...queryRefParams(w),
                        k,
                        [param === 'nprobe' ? 'nprobe' : 'ef']: point.value,
                      })
                    }
                  >
                    Explain
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  )
}
