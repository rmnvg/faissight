import { CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import type { IvfTrace, ProbeRow } from '../../api/types'
import { Card } from '../../components/ui'
import { fmtDist, fmtNum } from '../../lib/format'

export function ProbeChart({ trace }: { trace: IvfTrace }) {
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
